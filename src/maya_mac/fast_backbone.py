"""Lean Qwen3 backbone for single-stream decoding (MPS / CPU / CUDA), sharing the HF model's weights.

Same function as Backbone.forward over a DynamicCache, with the per-step overhead removed:
* a preallocated KV cache [layer][k|v][kv_heads, capacity, head_dim]; no Cache objects, no per-step masks;
* q|k|v and gate|up concatenated into one GEMM each (the only extra memory: one copy of those weights);
* RoPE cos/sin precomputed for the whole capacity;
* attention written out as grouped matmuls: MPS SDPA is wrong for this model in fp16/bf16 through HF, and wrong even in
  fp32 when K/V are broadcast over the query groups (torch 2.8.0).
RMSNorms use the fused F.rms_norm kernel: on MPS it is ~20x cheaper than HF Qwen3RMSNorm's seven elementwise ops
(6 vs 133 us per call, identical output on a test vector) and the backbone runs 4 of them per layer.
"""
import torch
import torch.nn.functional as F


def _rms(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


class LeanBackbone:
    def __init__(self, backbone, capacity=4096):
        lm = backbone.lm
        cfg = lm.config
        self.model = lm.model
        p = next(lm.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.groups = self.n_heads // self.n_kv
        self.eps = cfg.rms_norm_eps
        self.capacity = int(capacity)
        self.scale = self.head_dim ** -0.5
        with torch.inference_mode():
            self.layers = []
            for layer in self.model.layers:
                a, m = layer.self_attn, layer.mlp
                self.layers.append({
                    "ln1": layer.input_layernorm.weight, "ln2": layer.post_attention_layernorm.weight,
                    "qkv": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous(),
                    "qn": a.q_norm.weight, "kn": a.k_norm.weight, "o": a.o_proj.weight,
                    "gu": torch.cat([m.gate_proj.weight, m.up_proj.weight]).contiguous(), "down": m.down_proj.weight,
                })
            self.norm = self.model.norm.weight
            rot = self.model.rotary_emb
            pos = torch.arange(self.capacity, device=self.device, dtype=torch.float32)
            freqs = pos[:, None] * rot.inv_freq.to(self.device).float()[None]
            emb = torch.cat((freqs, freqs), -1)
            scaling = float(getattr(rot, "attention_scaling", 1.0))
            self.cos = (emb.cos() * scaling).to(self.dtype)
            self.sin = (emb.sin() * scaling).to(self.dtype)
            n = len(self.layers)
            self.k = torch.zeros(n, self.n_kv, self.capacity, self.head_dim, device=self.device, dtype=self.dtype)
            self.v = torch.zeros_like(self.k)
        self.length = 0

    def reset(self):
        self.length = 0

    @torch.inference_mode()
    def forward(self, x):
        """x: [1, T, D] input embeddings appended at the current position -> last hidden state [1, D]."""
        T = x.shape[1]
        p0, p1 = self.length, self.length + T
        if p1 > self.capacity:
            raise RuntimeError(f"backbone KV capacity {self.capacity} exceeded")
        H, KV, Dh = self.n_heads, self.n_kv, self.head_dim
        cos, sin = self.cos[p0:p1][None], self.sin[p0:p1][None]
        mask = None
        if T > 1:
            mask = torch.ones(T, p1, dtype=torch.bool, device=self.device).tril(p0)
        h = x[0].to(self.dtype)                                         # [T, D]
        for i, L in enumerate(self.layers):
            y = _rms(h, L["ln1"], self.eps)
            q, k, v = F.linear(y, L["qkv"]).split((H * Dh, KV * Dh, KV * Dh), -1)
            q = _rope(_rms(q.view(T, H, Dh), L["qn"], self.eps).transpose(0, 1), cos, sin)     # [H, T, Dh]
            k = _rope(_rms(k.view(T, KV, Dh), L["kn"], self.eps).transpose(0, 1), cos, sin)    # [KV, T, Dh]
            self.k[i, :, p0:p1] = k
            self.v[i, :, p0:p1] = v.view(T, KV, Dh).transpose(0, 1)
            K, V = self.k[i, :, :p1], self.v[i, :, :p1]                # [KV, p1, Dh]
            qg = q.reshape(KV, self.groups * T, Dh) * self.scale       # GQA without repeat_kv
            s = qg @ K.transpose(1, 2)                                  # [KV, G*T, p1]; q, k are RMS-normed
            if mask is not None:
                s = s.view(KV, self.groups, T, p1).masked_fill(~mask, float("-inf")).view(KV, self.groups * T, p1)
            a = (s.softmax(-1) @ V).reshape(H, T, Dh).transpose(0, 1).reshape(T, H * Dh)
            h = h + F.linear(a, L["o"])
            g, u = F.linear(_rms(h, L["ln2"], self.eps), L["gu"]).chunk(2, -1)
            h = h + F.linear(F.silu(g) * u, L["down"])
        self.length = p1
        return _rms(h[-1:], self.norm, self.eps)
