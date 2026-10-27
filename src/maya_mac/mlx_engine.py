"""MayaTTS on MLX: the same engine (prompt, sampler, stop rule, guards, voices, text splitting, streaming Mimi), with
the model run by maya_mac.mlx_model instead of torch MPS.

What makes it faster than the torch engine, per 80 ms frame:
  * the cb0 sampler, the 31-level depth decoder and the feedback embedding are one compiled Metal graph
    (FrameStep.step): no per-op dispatch from Python, no host round trip between the backbone and depth;
  * the frame loop is pipelined: frame t+1 is queued on the GPU before the host reads frame t (stop flag + codes, one
    33-int read), so the host-side work (guards, Mimi hand-off) overlaps the GPU instead of stalling it. A
    speculative frame is thrown away when frame t stops, or recomputed when the loop guard changes the AEOS bias;
  * a voice's reference prefix (the ICL clip) is prefilled once and its KV reused by every later request;
  * optional 8 / 4-bit weights (EngineConfig.backbone_bits / depth_bits).

Sampling draws from the same distributions as the torch engine but with MLX's RNG, so a given seed yields different
(equally likely) audio on the two backends.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from maya_mac import guards as G
from maya_mac import reference as S
from maya_mac.engine import HERE, EngineConfig, MayaTTS
from maya_mac.mlx_model import DTYPES, STOP_REASONS, Backbone, Composer, FrameStep, load_weights
from maya_mac.prompt import DEFAULT_BUILDER, MODES, Voice, VoiceRegistry, load_named_voices
from maya_mac.tts.tokens import N_Q


class MayaMLX(MayaTTS):
    def __init__(self, model_dir, mimi_dir, ecapa_dir=None, device=None, config: EngineConfig = None, log=print):
        from transformers import AutoTokenizer, MimiModel
        from maya_mac.codec import StreamingMimi
        from maya_mac.tts.data.packing_v2 import PackerV2
        from maya_mac.tts.tokens import extend_tokenizer

        self.device = "mlx"
        self.cfg = cfg = config or EngineConfig()
        self.log = log
        t0 = time.perf_counter()
        dtype = DTYPES[cfg.backbone_dtype]
        self.dtype = dtype
        bb = Path(model_dir) / "backbone"
        # MLX streams are thread-local: every MLX call (load, compile, voices, frames) runs on this one thread, so the
        # engine can be built on one thread and driven from another (the server loads on one and serves on another)
        self._mlx = ThreadPoolExecutor(1, thread_name_prefix="maya-mlx")
        self._mlx_thread = None

        def build():
            self._mlx_thread = threading.current_thread()
            w = load_weights(model_dir)
            self.backbone = Backbone(w, json.loads((bb / "config.json").read_text()), dtype=dtype,
                                     bits=cfg.backbone_bits, capacity=cfg.capacity)
            self.step = FrameStep(w, dtype=DTYPES[cfg.depth_dtype], bits=cfg.depth_bits, depth_temp=cfg.depth_temp,
                                  depth_top_k=cfg.depth_top_k, rep_win=cfg.cb0_rep_win, fb_dtype=dtype)
            self.compose = Composer(w, dtype=dtype)
        self._on_mlx(build)
        tokenizer, token_ids = extend_tokenizer(AutoTokenizer.from_pretrained(str(bb)))
        self.packer = PackerV2(tokenizer, token_ids, max_tokens=8192)
        self.prior = S.load_prior(str(HERE / "configs/text_frames_prior.json"))
        self.dur = G.DurationPrior()
        self.mimi = MimiModel.from_pretrained(str(mimi_dir)).to(cfg.codec_device).eval()
        self.codec = StreamingMimi(self.mimi)
        self._codec_pool = ThreadPoolExecutor(1, thread_name_prefix="maya-mimi")
        self.ecapa_dir = ecapa_dir
        self._ecapa = None
        self._ecapa_device = "cpu"
        self.registry = VoiceRegistry()
        voices_json = Path(model_dir) / "voices" / "voices.json"
        if voices_json.is_file():
            load_named_voices(self.registry, str(voices_json))
        if cfg.warmup:      # compile the frame graph and the Metal kernels at load, not on the first request
            list(self.stream("Hello, this is a warmup.", "en", max_frames=6, seed=1))
        self.load_s = time.perf_counter() - t0
        self._seed_rng = np.random.default_rng(cfg.seed)
        self.last_info = None

    def _on_mlx(self, fn, *args):
        if threading.current_thread() is self._mlx_thread:
            return fn(*args)
        return self._mlx.submit(fn, *args).result()

    def close(self):
        super().close()
        self._mlx.shutdown(wait=True)

    # ------------------------------------------------------------------ voices
    def prepare_voice(self, voice: Voice):
        """Pack the voice's reference prefix and compute its depth cross-attention constants (once per voice)."""
        return self._on_mlx(self._prepare_voice, voice)

    def _prepare_voice(self, voice):
        with voice.lock:
            if voice.mem is not None:
                return voice
            _, use_mem = MODES[voice.mode]
            builder = voice.builder or DEFAULT_BUILDER
            voice.prefix = builder.prefix(self.packer, voice)
            voice.spk_refs = (np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None
                              else np.zeros((1, 192), np.float32))
            voice.mem = self.step.cross(voice.spk_refs if use_mem else None)
            mx.eval(voice.mem)
            voice.ts = int(voice.prefix["tokens"].shape[0]) if voice.prefix is not None else 0
            voice.kv = voice.emb = None
            return voice

    def _embed(self, w):
        if self._ecapa is None:
            if self.ecapa_dir is None:
                raise ValueError("ecapa_dir is required for reference-audio voices")
            from maya_mac.engine import Ecapa
            self._ecapa = Ecapa(self.ecapa_dir, self._ecapa_device)
        return self._ecapa(w)

    # ------------------------------------------------------------------ synthesis
    def _plan(self, text, lang, voice, max_frames):
        from maya_mac.tts.data.packing_v2 import collate_v2
        from maya_mac.tts.model.load_s3 import truncate_prompt_batch
        cfg, g = self.cfg, self.cfg.guards
        builder = voice.builder or DEFAULT_BUILDER
        ex = builder.example(self.packer, voice, text, lang)
        # reference.prompt_from_example, composed by MLX
        b = collate_v2([ex], q_choices=(N_Q,), q_rng=np.random.default_rng(0))
        kind = b["kind"][0]
        a = (kind == 1).nonzero().flatten()
        a = a[a >= int(ex.get("target_start", 0))]
        assert a.numel(), "prompt has no target audio position"
        prompt = self.compose(truncate_prompt_batch(b, int(a[0]), require_target_span=self.compose.anchor is not None))
        ef, er = S.expected_frames(self.prior, lang, len(text))
        loose = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        cap = loose
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        if max_frames:
            cap = min(cap, max(1, int(max_frames)))
        minimum = max(4, int(0.5 * ef))
        suspend = G.suspend_windows_v2(text, lang, self.dur) if (g.enabled and g.loop and g.aware_v2) else ()
        return prompt, cap, loose, minimum, suspend

    def _prefill(self, prompt, voice):
        """Run the prompt through the backbone, reusing the voice's cached reference-prefix KV when the prompt starts
        with exactly that prefix."""
        bb, ts = self.backbone, voice.ts
        bb.reset()
        if ts and ts < prompt.shape[1]:
            head = prompt[:, :ts]
            if voice.kv is not None and mx.array_equal(head, voice.emb).item():
                bb.restore(voice.kv)
                return bb(prompt[:, ts:])
            bb(head)
            voice.kv, voice.emb = bb.snapshot(ts), head
            mx.eval(voice.kv, voice.emb)   # materialise now, so later in-place cache writes don't have to copy the cache
            return bb(prompt[:, ts:])
        return bb(prompt)

    def _frames(self, text, lang, voice, sampling, max_frames, info):
        """Yield the (32,) code tensor of each frame of one text piece; info receives the stop record. Each frame is
        produced on the MLX thread."""
        gen = self._frames_mlx(text, lang, voice, sampling, max_frames, info)
        try:
            while True:
                try:
                    codes = self._on_mlx(next, gen)
                except StopIteration:
                    return
                yield codes
        finally:
            self._on_mlx(gen.close)

    def _frames_mlx(self, text, lang, voice, sampling, max_frames, info):
        cfg, g = self.cfg, self.cfg.guards
        prompt, cap, loose, minimum, suspend = self._plan(text, lang, voice, max_frames)
        seed = sampling.get("seed")
        seed = int(seed) if seed is not None else int(self._seed_rng.integers(1, 2 ** 31 - 1))
        if prompt.shape[1] + cap + 1 > cfg.capacity:
            raise ValueError(f"prompt of {prompt.shape[1]} tokens + {cap} frames exceeds KV capacity {cfg.capacity}")
        rp = sampling.get("repetition_penalty")
        knobs = (mx.array(float(sampling.get("temperature") or cfg.cb0_temp)),
                 mx.array(min(int(sampling.get("top_k") or cfg.cb0_top_k), 2049)),
                 mx.array(float(sampling.get("top_p") or cfg.cb0_top_p)),
                 mx.array(float(rp if rp is not None else cfg.cb0_rep)))
        cross = voice.mem
        keys = iter(mx.random.split(mx.random.key(seed), cap + 1))
        backbone = self.backbone

        def issue(h, hazard, hist, frame, bias, key):
            """Queue one frame on the GPU; nothing is read back here."""
            out, fb, hazard, hist = self.step.step(h, cross, hist, hazard, mx.array(frame >= minimum),
                                                   mx.array(bias, mx.float32), *knobs, key)
            h = backbone(fb[None, None])
            mx.async_eval(out, h, hazard, hist)
            return out, h, hazard, hist, key

        h = self._prefill(prompt, voice)
        hist0 = mx.full((max(1, cfg.cb0_rep_win),), -1, mx.int32)
        cur = issue(h, mx.array(0.0), hist0, 0, 0.0, next(keys))
        cb0_hist = np.full(g.hist, -1, np.int64)
        frames, reason, boost_at, n_boost, bias = 0, "cap", None, 0, 0.0
        while True:
            nxt = issue(*cur[1:4], frames + 1, bias, next(keys)) if frames + 1 < cap else None
            out = np.array(cur[0])                      # waits for this frame only; the next one is already queued
            stop = int(out[0])
            if stop:
                reason = STOP_REASONS[stop]
                break
            codes = torch.from_numpy(out[1:].astype(np.int64))
            yield codes
            frames += 1
            cb0_hist[:-1] = cb0_hist[1:]
            cb0_hist[-1] = int(out[1])
            if frames >= cap:
                reason = "max_frames" if max_frames and frames >= max_frames else (
                    "cap_guard" if g.enabled and cap < loose else "cap")
                break
            new_bias = bias
            if (g.enabled and g.loop and frames >= g.start and frames % g.every == 0
                    and not any(a <= frames <= b for a, b in suspend)):
                hit = G.loop_detect_batch(cb0_hist[None], np.array([frames]), g)[0]
                if hit is not None and g.mode == "boost":
                    if boost_at is None:
                        boost_at, n_boost = frames, n_boost + 1
                        new_bias = g.boost
                    if frames - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif hit is not None:
                    reason = "loop_" + hit.split(":")[0]
                    break
                elif boost_at is not None:
                    new_bias, boost_at = 0.0, None
            if new_bias != bias:      # the queued frame used the old AEOS bias: redo it (it rewrites the same KV slot)
                bias = new_bias
                nxt = issue(*cur[1:4], frames, bias, nxt[4])
            cur = nxt
        info.update({"reason": reason, "frames": frames, "cap": cap, "n_boost": n_boost})
