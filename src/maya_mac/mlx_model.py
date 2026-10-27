"""Maya model on MLX (Apple's C++/Metal array framework), loaded straight from the release safetensors.

No torch model is built: the weights are read once with mx.load, cast (or quantized) and kept as plain arrays.

* Backbone   : Qwen3-1.7B decode with a preallocated KV cache, fused q|k|v and gate|up GEMMs and MLX's fused Metal
               kernels (rms_norm, rope, scaled_dot_product_attention). MLX SDPA is accurate here, unlike torch MPS SDPA.
* FrameStep  : everything between two backbone steps as ONE compiled graph: cb0 sampler (presence penalty,
               temperature, top-k, top-p, controlled stop), the 31-level depth decoder (per-band KV, constant speaker
               cross-attention, last-position-only final layer), and the feedback embedding of the frame's 32 codes.
               mx.compile fuses the elementwise work and removes the per-op Python cost; nothing leaves the GPU.
* compose    : the backbone prompt embeddings from a collate_v2 batch (text, audio frames, <aeos>, speaker anchor).

Optional weight quantization (8 or 4 bit, group 64) through mx.quantized_matmul cuts the bytes read per frame, which is
what bounds single-stream decode.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from maya_mac import mlx_kernels as kernels
from maya_mac.tts.model.depth_v2 import D, GROUP_OF_LEVEL, H, LAYERS
from maya_mac.tts.tokens import AEOS_CLASS, N_CODES, N_Q

HEAD = D // H
DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}
KIND_AUDIO, KIND_AEOS = 1, 3
STOP_REASONS = {1: "aeos_argmax", 2: "aeos_tau", 3: "aeos_hazard"}


def load_weights(model_dir):
    """{name: array} for every tensor of the release export (lazy: read from disk when first evaluated)."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else [
        "model.safetensors"]
    w = {}
    for f in files:
        w.update(mx.load(str(d / f)))
    return w


def _eps(dtype):
    """torch nn.RMSNorm(eps=None) uses the input dtype's machine epsilon; audio_in.norm is built that way, and the
    epsilon is ~35 % of a feedback frame's mean square, so it is part of the function, not a rounding detail."""
    return {mx.float16: 2.0 ** -10, mx.bfloat16: 2.0 ** -7}.get(dtype, 2.0 ** -23)


class Linear:
    """y = x @ W.T, dense in `dtype` or quantized (bits 4 | 8, group_size 64)."""

    def __init__(self, w, dtype, bits=None, group_size=64):
        self.bits, self.group_size = bits, group_size
        w = w.astype(dtype)
        if bits:
            self.wq, self.scales, self.biases = mx.quantize(w, group_size=group_size, bits=bits)
        else:
            self.w = w

    def __call__(self, x):
        if self.bits:
            return mx.quantized_matmul(x, self.wq, self.scales, self.biases, transpose=True,
                                       group_size=self.group_size, bits=self.bits)
        return x @ self.w.T

    def arrays(self):
        return [self.wq, self.scales, self.biases] if self.bits else [self.w]


class Backbone:
    """Qwen3 decoder over a preallocated KV cache; one piece of text at a time (reset() between pieces)."""

    def __init__(self, w, config, dtype=mx.float16, bits=None, capacity=4096):
        P = "backbone.lm.model."
        self.dtype = dtype
        self.n_heads, self.n_kv = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        self.theta = float((config.get("rope_parameters") or {}).get("rope_theta", config.get("rope_theta", 1e4)))
        self.scale = self.head_dim ** -0.5
        self.capacity = int(capacity)
        self.layers = []
        for i in range(config["num_hidden_layers"]):
            p = f"{P}layers.{i}."
            a, m = p + "self_attn.", p + "mlp."
            self.layers.append({
                "ln1": w[p + "input_layernorm.weight"].astype(dtype),
                "ln2": w[p + "post_attention_layernorm.weight"].astype(dtype),
                "qkv": Linear(mx.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"],
                                              w[a + "v_proj.weight"]]), dtype, bits),
                "qn": w[a + "q_norm.weight"].astype(dtype), "kn": w[a + "k_norm.weight"].astype(dtype),
                "o": Linear(w[a + "o_proj.weight"], dtype, bits),
                "gu": Linear(mx.concatenate([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]]), dtype, bits),
                "down": Linear(w[m + "down_proj.weight"], dtype, bits),
            })
            mx.eval(*[v for L in self.layers[-1:] for x in L.values()
                      for v in (x.arrays() if isinstance(x, Linear) else [x])])
        self.norm = w[P + "norm.weight"].astype(dtype)
        shape = (1, self.n_kv, self.capacity, self.head_dim)
        self.k = [mx.zeros(shape, dtype) for _ in self.layers]
        self.v = [mx.zeros(shape, dtype) for _ in self.layers]
        mx.eval(self.norm, *self.k, *self.v)
        self.length = 0

    def reset(self, length=0):
        self.length = int(length)

    def __call__(self, x):
        """x: (1, T, D) input embeddings appended at the current position -> final-normed last hidden state (1, D)."""
        T = x.shape[1]
        p0, p1 = self.length, self.length + T
        if p1 > self.capacity:
            raise RuntimeError(f"backbone KV capacity {self.capacity} exceeded")
        H, KV, Dh = self.n_heads, self.n_kv, self.head_dim
        h = x.astype(self.dtype)
        last = len(self.layers) - 1
        for i, L in enumerate(self.layers):
            y = mx.fast.rms_norm(h, L["ln1"], self.eps)
            q, k, v = mx.split(L["qkv"](y), [H * Dh, (H + KV) * Dh], axis=-1)
            q = mx.fast.rms_norm(q.reshape(1, T, H, Dh), L["qn"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k.reshape(1, T, KV, Dh), L["kn"], self.eps).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, Dh, traditional=False, base=self.theta, scale=1.0, offset=p0)
            k = mx.fast.rope(k, Dh, traditional=False, base=self.theta, scale=1.0, offset=p0)
            self.k[i][:, :, p0:p1] = k
            self.v[i][:, :, p0:p1] = v.reshape(1, T, KV, Dh).transpose(0, 2, 1, 3)
            if i == last and T > 1:       # only the last position's output is read
                q, h, T = q[:, :, -1:], h[:, -1:], 1
                mask = None
            else:
                mask = "causal" if T > 1 else None
            a = mx.fast.scaled_dot_product_attention(q, self.k[i][:, :, :p1], self.v[i][:, :, :p1],
                                                     scale=self.scale, mask=mask)
            h = h + L["o"](a.transpose(0, 2, 1, 3).reshape(1, T, H * Dh))
            g, u = mx.split(L["gu"](mx.fast.rms_norm(h, L["ln2"], self.eps)), 2, axis=-1)
            h = h + L["down"](g * mx.sigmoid(g) * u)
        self.length = p1
        return mx.fast.rms_norm(h[:, -1], self.norm, self.eps)

    def snapshot(self, n):
        """Copy of the first n KV positions (a reusable prompt prefix)."""
        return [(k[:, :, :n] + 0, v[:, :, :n] + 0) for k, v in zip(self.k, self.v)]

    def restore(self, snap):
        n = snap[0][0].shape[2]
        for i, (k, v) in enumerate(snap):
            self.k[i][:, :, :n] = k
            self.v[i][:, :, :n] = v
        self.length = n


def _layer_norm(x, w, b):
    return mx.fast.layer_norm(x, w, b, 1e-5)


def _gelu(x):            # exact (erf) GELU, as torch F.gelu
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


class FrameStep:
    """One frame minus the backbone: (h, sampler state) -> codes, stop, feedback embedding. Call through .step, the
    compiled graph (one compilation; every per-request knob is an array input, so nothing recompiles)."""

    def __init__(self, w, dtype=mx.float16, bits=None, depth_temp=0.7, depth_top_k=64, rep_win=12,
                 fb_dtype=mx.float16, fused=True):
        self.dtype = dtype
        # the custom kernels read fp16 activations and dense fp16 or 8-bit weights; anything else takes the op path
        self.fused = bool(fused) and dtype == mx.float16 and bits in (None, 8)
        self.fused_max_t = 4
        dp = "depth."
        f32 = mx.float32
        self.cb0_head = w["backbone.cb0_head.weight"].astype(fb_dtype)       # never quantized: it decides the stop
        self.in_proj = Linear(w[dp + "in_proj.weight"], dtype)
        self.level0 = w[dp + "level_emb.weight"][0].astype(dtype)
        # slot k+1 holds code cb_k: code_up(code_lo) + level_emb, as one (31, 2048, D) table (FastDepth's fusion)
        lo = w[dp + "code_lo.weight"].astype(f32).reshape(N_Q - 1, N_CODES, -1)
        self.embed = (lo @ w[dp + "code_up.weight"].astype(f32).T
                      + w[dp + "level_emb.weight"].astype(f32)[1:N_Q, None]).astype(dtype)
        mx.eval(self.embed)
        self.groups = []
        for g in range(max(GROUP_OF_LEVEL[1:]) + 1):
            blocks = []
            for li in range(LAYERS):
                p = f"{dp}groups.{g}.{li}."
                blocks.append({
                    "n1w": w[p + "n1.weight"].astype(dtype), "n1b": w[p + "n1.bias"].astype(dtype),
                    "n2w": w[p + "n2.weight"].astype(dtype), "n2b": w[p + "n2.bias"].astype(dtype),
                    "qkv": Linear(w[p + "qkv.weight"], dtype, bits), "o": Linear(w[p + "o.weight"], dtype, bits),
                    "f1": Linear(w[p + "f1.weight"], dtype, bits), "f2": Linear(w[p + "f2.weight"], dtype, bits),
                })
            self.groups.append(blocks)
            mx.eval(*[v for b in blocks for x in b.values() for v in (x.arrays() if isinstance(x, Linear) else [x])])
        self.norm_w, self.norm_b = w[dp + "norm.weight"].astype(dtype), w[dp + "norm.bias"].astype(dtype)
        self.heads = [Linear(w[f"{dp}heads.{i}.weight"], dtype, bits) for i in range(N_Q - 1)]
        # speaker path (per voice, not per frame)
        self.spk_proj = w[dp + "spk_proj_ecapa.weight"].astype(f32)
        self.null_mem = w[dp + "null_mem"].astype(f32).reshape(1, D)
        self.cross_kv = [w[f"{dp}cross.{i}.kv.weight"].astype(f32) for i in range(LAYERS)]
        self.cross_o = [w[f"{dp}cross.{i}.o.weight"].astype(f32) for i in range(LAYERS)]
        # feedback: the frame's 32 codes -> one backbone input vector (AudioInputEmbed.frames at q_use = 32)
        self.fb_table = w["audio_in.table.weight"]
        self.fb_norm = w["audio_in.norm.weight"].astype(fb_dtype)
        self.fb_dtype, self.fb_eps = fb_dtype, _eps(fb_dtype)
        self.fb_offsets = mx.arange(N_Q, dtype=mx.int32) * N_CODES
        self.depth_temp, self.depth_top_k, self.rep_win = float(depth_temp), int(depth_top_k), int(rep_win)
        mx.eval(self.cb0_head, self.level0, self.norm_w, self.norm_b, self.fb_norm, self.spk_proj, self.null_mem,
                *self.cross_kv, *self.cross_o, *[a for h in self.heads for a in h.arrays()], *self.in_proj.arrays())
        self.step = mx.compile(self._step)

    def cross(self, ecapa=None):
        """(LAYERS, 1, D) speaker cross-attention outputs. With one unmasked reference the softmax is exactly one, so
        layer i adds o(v(mem)) whatever the query: a per-voice constant. ecapa None -> the learned null memory."""
        mem = self.null_mem if ecapa is None else mx.array(np.asarray(ecapa, np.float32).reshape(1, -1)) @ self.spk_proj.T
        mem = mem.astype(self.dtype).astype(mx.float32)        # the torch path runs the cross block in depth dtype
        out = [(mx.split(mem @ kv.T, 2, axis=-1)[1]) @ o.T for kv, o in zip(self.cross_kv, self.cross_o)]
        return mx.stack(out).astype(self.dtype)

    def _block(self, x, blk, cross, attend):
        y = blk["qkv"](_layer_norm(x, blk["n1w"], blk["n1b"]))
        T = x.shape[1]
        y = y.reshape(1, T, 3, H, HEAD).transpose(2, 0, 3, 1, 4)          # (3, 1, H, T, HEAD)
        a, x = attend(y[0], y[1], y[2], x)
        x = x + blk["o"](a.transpose(0, 2, 1, 3).reshape(1, x.shape[1], D))
        return x + blk["f2"](_gelu(blk["f1"](_layer_norm(x, blk["n2w"], blk["n2b"])))) + cross

    def _sample(self, logits, key):
        kth = mx.min(mx.topk(logits, self.depth_top_k, axis=-1), axis=-1, keepdims=True)
        return mx.random.categorical(mx.where(logits < kth, -mx.inf, logits), key=key)

    def _cb0(self, h, hist, hazard, armed, bias, temp, top_k, top_p, rep, key):
        """Upstream Engine._cb0 for one stream (see engine.Cb0Sampler), on the GPU."""
        n = N_CODES + 1
        logits = (h.astype(self.fb_dtype) @ self.cb0_head.T).astype(mx.float32)[0] / temp
        cls = mx.arange(n)
        logits = logits + mx.where(cls == AEOS_CLASS, bias, 0.0)
        if self.rep_win > 0:
            seen = (hist[:, None] == cls[None]).any(0) & (cls != AEOS_CLASS)
            logits = logits - mx.where(seen, rep, 0.0)
        kth = mx.take(mx.sort(logits), n - top_k)
        logits = mx.where(logits < kth, -mx.inf, logits)
        sp = -mx.sort(-mx.softmax(logits))
        keep_min = mx.min(mx.where((mx.cumsum(sp) - sp) > top_p, mx.inf, sp))
        logits = mx.where((top_p < 1.0) & (mx.softmax(logits) < keep_min), -mx.inf, logits)
        probs = mx.softmax(logits)
        pa = probs[AEOS_CLASS]
        hazard = mx.where(armed, 1 - (1 - hazard) * (1 - pa), hazard)
        stop = mx.where(mx.argmax(probs) == AEOS_CLASS, 1, mx.where(pa >= 0.35, 2, mx.where(hazard >= 0.6, 3, 0)))
        stop = mx.where(armed, stop, 0)
        cb0 = mx.random.categorical(mx.where(cls == AEOS_CLASS, -mx.inf, logits), key=key).astype(mx.int32)
        hist = mx.concatenate([hist[1:], cb0[None]]) if self.rep_win > 0 else hist
        return cb0, stop, hazard, hist

    def _depth_ops(self, h, cb0, cross, keys):
        """Levels 1..31 from MLX ops (any dtype, any bits) -> list of (1,) int32 codes."""
        prefix = [self.in_proj(h.astype(self.dtype)) + self.level0, self.embed[0][cb0][None]]   # (1, D) each
        codes = []
        cache, previous = None, None
        for level in range(1, N_Q):
            group, n = GROUP_OF_LEVEL[level], level + 1
            blocks = self.groups[group]
            if group == previous:                     # same trunk as the level before: one new position
                x = prefix[level][None]
                for li, blk in enumerate(blocks):
                    def attend(q, k, v, x, li=li):
                        k = mx.concatenate([cache[li][0], k], axis=2)
                        v = mx.concatenate([cache[li][1], v], axis=2)
                        cache[li] = (k, v)
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD ** -0.5), x
                    x = self._block(x, blk, cross[li], attend)
            else:                                     # first level of a trunk: full causal pass over the prefix
                reused = level + 1 < N_Q and GROUP_OF_LEVEL[level + 1] == group
                cache = [None] * LAYERS
                x = mx.concatenate(prefix[:n])[None]
                for li, blk in enumerate(blocks):
                    def attend(q, k, v, x, li=li):
                        if reused:
                            cache[li] = (k, v)
                        if li == LAYERS - 1:          # nothing reads the other positions' last-layer output
                            return mx.fast.scaled_dot_product_attention(q[:, :, -1:], k, v, scale=HEAD ** -0.5), \
                                x[:, -1:]
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD ** -0.5, mask="causal"), x
                    x = self._block(x, blk, cross[li], attend)
            hn = _layer_norm(x[:, -1], self.norm_w, self.norm_b)
            logits = self.heads[level - 1](hn).astype(mx.float32) / self.depth_temp
            code = self._sample(logits, keys[level]).astype(mx.int32)              # (1,)
            codes.append(code)
            if n < N_Q:
                prefix.append(self.embed[level][code])
            previous = group
        return codes

    def _linear(self, x, lin, ln=None, gelu=False, res=None, add=None):
        """kernels.linear for few positions; MLX's tiled GEMM (+ separate LayerNorm / GELU / adds) for a long prefix,
        where the one-pass kernel would re-read every input row in every simdgroup."""
        if x.shape[0] <= self.fused_max_t:
            return kernels.linear(x, lin, ln=ln, gelu=gelu, res=res, add=None if add is None else add.reshape(-1))
        y = lin(x if ln is None else _layer_norm(x, *ln))
        if gelu:
            y = _gelu(y)
        if res is not None:
            y = y + res
        return y if add is None else y + add

    def _depth_fused(self, h, cb0, cross, keys):
        """Levels 1..31 through the custom Metal kernels (fp16, dense or 8-bit): 5 kernels per layer pass, 2 per
        level for the head and the draw. Same function as _depth_ops."""
        K = kernels
        gumbel = mx.random.gumbel((N_Q, N_CODES), key=keys[1])
        prefix = mx.concatenate([self.in_proj(h.astype(self.dtype)) + self.level0, self.embed[0][cb0][None]])
        codes = []
        cache, previous = None, None
        for level in range(1, N_Q):
            group = GROUP_OF_LEVEL[level]
            incremental = group == previous
            reused = level + 1 < N_Q and GROUP_OF_LEVEL[level + 1] == group
            if not incremental:
                cache = [None] * LAYERS
            x = prefix[-1:] if incremental else prefix
            for li, blk in enumerate(self.groups[group]):
                last = not incremental and li == LAYERS - 1
                qkv = self._linear(x, blk["qkv"], ln=(blk["n1w"], blk["n1b"]))
                a, kv = K.attend(qkv, cache[li], H, HEAD, last=last)
                if incremental or reused:
                    cache[li] = kv
                if last:
                    x = x[-1:]
                x = self._linear(a, blk["o"], res=x)
                f = self._linear(x, blk["f1"], ln=(blk["n2w"], blk["n2b"]), gelu=True)
                x = self._linear(f, blk["f2"], res=x, add=cross[li])
            logits = self._linear(x, self.heads[level - 1], ln=(self.norm_w, self.norm_b))
            code, prefix = K.sample(logits, gumbel, level, self.embed, prefix, 1.0 / self.depth_temp,
                                    self.depth_top_k, embed=level + 1 < N_Q)
            codes.append(code)
            previous = group
        return codes

    def _step(self, h, cross, hist, hazard, armed, bias, temp, top_k, top_p, rep, key):
        keys = mx.random.split(key, N_Q)
        cb0, stop, hazard, hist = self._cb0(h, hist, hazard, armed, bias, temp, top_k, top_p, rep, keys[0])
        depth = self._depth_fused if self.fused else self._depth_ops
        codes = mx.concatenate([cb0[None]] + depth(h, cb0, cross, keys))           # (32,)
        e = self.fb_table[codes + self.fb_offsets].astype(mx.float32).sum(0) / N_Q ** 0.5
        fb = mx.fast.rms_norm(e.astype(self.fb_dtype), self.fb_norm, self.fb_eps)
        out = mx.concatenate([stop.astype(mx.int32)[None], codes])                 # one host read per frame
        return out, fb, hazard, hist


class Composer:
    """TTSModelV2.compose_inputs for one collated, truncated prompt, from the release weights."""

    def __init__(self, w, dtype=mx.float16):
        self.dtype = dtype
        self.embed = w["backbone.lm.model.embed_tokens.weight"]
        self.table = w["audio_in.table.weight"]
        self.norm = w["audio_in.norm.weight"].astype(dtype)
        self.eps = _eps(dtype)
        self.aeos = mx.fast.rms_norm(w["audio_in.aeos_in"].astype(dtype), self.norm, self.eps)
        a = "speaker_anchor."
        self.anchor = None
        if a + "fc1.weight" in w:
            self.anchor = {k: w[a + k].astype(mx.float32) for k in
                           ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "null_vec")}

    def __call__(self, pb):
        """pb: truncate_prompt_batch output (torch CPU tensors, batch of one) -> (1, T, D) embeddings."""
        tokens = pb["tokens"][0].numpy()
        kind = pb["kind"][0].numpy()
        x = self.embed[mx.array(tokens.astype(np.int32))].astype(self.dtype)
        audio = np.flatnonzero(kind == KIND_AUDIO)
        if audio.size:
            q = int(pb["q_use"])
            codes = pb["codes"][0].numpy()[audio, :q].astype(np.int32)
            if codes.min() < 0 or codes.max() >= N_CODES:
                raise ValueError("audio code outside [0, n_codes) in the prompt")
            e = self.table[mx.array(codes + np.arange(q, dtype=np.int32) * N_CODES)].astype(self.dtype)
            x[mx.array(audio)] = mx.fast.rms_norm(e.sum(1) / q ** 0.5, self.norm, self.eps)
        aeos = np.flatnonzero(kind == KIND_AEOS)
        if aeos.size:
            x[mx.array(aeos)] = mx.broadcast_to(self.aeos, (aeos.size, self.aeos.shape[0]))
        if self.anchor is not None and "span_sentinel" in pb and pb["span_sentinel"].numel():
            x = x + self._anchor(pb, x.shape[0])
        return x[None]

    def _anchor(self, pb, T):
        """SpeakerAnchor (eval: no dropout) added at every span's <|audio|> sentinel."""
        A = self.anchor
        refs = pb["spk_refs"][0].float().numpy()[:, :A["null_vec"].shape[0]]
        mask = pb["spk_ref_mask"][0].numpy() if "spk_ref_mask" in pb else np.ones(refs.shape[0], bool)
        if bool(pb["spk_null"][0]) or not mask.any():
            v = A["null_vec"]
        else:
            v = mx.array((refs * mask[:, None]).sum(0) / max(1.0, float(mask.sum())))
        y = _gelu(v @ A["fc1.weight"].T + A["fc1.bias"]) @ A["fc2.weight"].T + A["fc2.bias"]
        add = np.zeros((T, 1), np.float32)
        for col in pb["span_sentinel"].numpy():
            add[int(col)] += 1.0
        return (mx.array(add) * y[None]).astype(self.dtype)
