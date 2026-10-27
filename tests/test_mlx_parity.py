"""MLX model (fp16, fused depth kernels) against the torch model in fp32 on the CPU, on the real weights.

Needs the checkpoint: MAYA_MODEL_DIR=<dir holding maya2-native/> pytest tests/test_mlx_parity.py (~2 min, ~16 GB RAM).
Covers the prompt composition, a backbone prefill and decode step, the cb0 head, all 31 depth levels teacher-forced,
and the feedback embedding.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
MODELS = os.environ.get("MAYA_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODELS, reason="set MAYA_MODEL_DIR to run the parity test")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("bits,tol", [(None, 1e-2), (8, 6e-2)])
def test_parity(bits, tol):
    import torch
    import maya_mac.mlx_model as MM
    from maya_mac import reference as S
    from maya_mac.engine import HERE
    from maya_mac.prompt import DEFAULT_BUILDER, Voice
    from maya_mac.tts.data.packing_v2 import PackerV2, collate_v2
    from maya_mac.tts.model.depth_v2 import GROUP_OF_LEVEL
    from maya_mac.tts.model.load_s3 import load_model_for_eval, truncate_prompt_batch

    md = Path(MODELS) / "maya2-native"
    mcfg = {"model": {"backbone": str(md / "backbone"), "attn": "eager"}, "data": {"max_tokens": 8192}}
    model, _ = load_model_for_eval(str(md), device="cpu", dtype=torch.float32, cfg=mcfg,
                                   config=str(HERE / "configs/s3_fresh_v1.yaml"), amortize_frac=1.0,
                                   require_anchor=True, verbose=False)
    model.audio_in.norm.eps = 2.0 ** -10       # the fp16 engines' RMSNorm epsilon (torch takes it from the dtype)
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    rng = np.random.default_rng(0)
    ecapa = rng.normal(size=192).astype(np.float32)
    ecapa /= np.linalg.norm(ecapa)
    clip = {"text": "this is the reference clip", "codes": rng.integers(0, 2048, (32, 40)), "lang": "en"}
    v = Voice("t", "icl_spk", clips=[clip], ecapa=ecapa)
    v.prefix, v.spk_refs = DEFAULT_BUILDER.prefix(packer, v), ecapa.reshape(1, 192)
    ex = DEFAULT_BUILDER.example(packer, v, "नमस्ते, आप कैसे हैं? Testing the fast engine.", "hi")

    w = MM.load_weights(md)
    f16 = mx.float16
    bb = MM.Backbone(w, json.loads((md / "backbone/config.json").read_text()), dtype=f16, bits=bits, capacity=512)
    step = MM.FrameStep(w, dtype=f16, bits=bits, fb_dtype=f16)
    assert step.fused
    b = collate_v2([ex], q_choices=(32,), q_rng=np.random.default_rng(0))
    a = (b["kind"][0] == 1).nonzero().flatten()
    pm = MM.Composer(w, dtype=f16)(truncate_prompt_batch(b, int(a[a >= ex["target_start"]][0]), True))

    with torch.inference_mode():
        pt, _ = S.prompt_from_example(ex, model, "cpu", torch.float32, 32)
        hs, kv = model.backbone(pt, use_cache=True)
        h_t = hs[:, -1]
        lt = model.backbone.cb0_logits(h_t).numpy()[0]
    assert rel(pm, pt.numpy()) < 2e-3
    h_m = bb(pm)
    assert rel(h_m.astype(mx.float32), h_t.numpy()) < tol
    lm = np.array((h_m @ step.cb0_head.T).astype(mx.float32)[0])
    assert rel(lm, lt) < tol and lm.argmax() == lt.argmax()

    cb0 = int(lt[:2048].argmax())
    with torch.inference_mode():           # torch depth, teacher-forced on its own greedy codes
        mem = model.depth.spk_memory(torch.tensor(ecapa)[None, None], "ecapa")
        codes, tl = [torch.tensor([cb0])], []
        for level in range(1, 32):
            x = model.depth._prefix(h_t, torch.stack(codes, 1), level)
            lg = model.depth.heads[level - 1](model.depth._run(x, GROUP_OF_LEVEL[level], mem,
                                                               torch.ones(1, 1, dtype=torch.bool)))[0]
            tl.append(lg.numpy())
            codes.append(lg.argmax()[None])
    forced = [int(c) for c in torch.cat(codes)]
    ml = []

    def forced_sample(logits, gumbel, level, table, prefix, inv_temp, top_k, embed=True):
        ml.append(np.array(logits.astype(mx.float32))[0])
        c = mx.array([forced[level]], mx.int32)
        return c, (mx.concatenate([prefix, table[level][c]]) if embed else prefix)

    real = MM.kernels.sample
    MM.kernels.sample = forced_sample
    try:
        out, fb, _, _ = step._step(mx.array(h_t.numpy()).astype(f16), step.cross(ecapa.reshape(1, 192)),
                                   mx.full((12,), -1, mx.int32), mx.array(0.0), mx.array(False), mx.array(0.0),
                                   mx.array(0.9), mx.array(64), mx.array(1.0), mx.array(0.8), mx.random.key(0))
        mx.eval(out, fb)
    finally:
        MM.kernels.sample = real
    assert max(rel(m, t) for m, t in zip(ml, tl)) < tol
    assert sum(int(m.argmax() == t.argmax()) for m, t in zip(ml, tl)) >= (31 if bits is None else 29)
    with torch.inference_mode():
        fbt = model.audio_in.frames(torch.tensor(forced)[None], 32)[0]
        hs2, _ = model.backbone(fbt[None, None], past_key_values=kv, use_cache=True)
    assert rel(fb.astype(mx.float32), fbt.numpy()) < 2e-3
    assert rel(bb(fb[None, None]).astype(mx.float32), hs2[:, -1].numpy()) < tol
