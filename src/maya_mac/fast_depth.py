"""Whole-frame depth decoder for one request, eager PyTorch (MPS / CPU / CUDA).

Port of upstream serve/fast_depth.FastDepth._compute without CUDA graphs, seat slots or Triton. Same real-valued
function as DepthV2.generate (kernel order differs, so not bit-exact):

* a private lower-precision deep copy of the depth stack (the original is untouched);
* speaker cross-attention hoisted to a per-voice constant (single unmasked reference => softmax is exactly one =>
  the output is o(v), independent of the query);
* prefix embeddings from a fused `code_up(code_lo) + level_emb` table (position k depends on slot k alone);
* within-group KV cache: levels 9-16 / 17-24 / 25-31 share a trunk, so each band is one full pass + 1-token steps;
* the last trunk layer of a full pass evaluates only the final position.
"""
import copy

import torch
import torch.nn.functional as F

from maya_mac.tts.model.depth_v2 import GROUP_OF_LEVEL, LAYERS, D, H

HEAD = D // H
N_LEVELS = 32


class FastDepth:
    def __init__(self, depth, dtype=torch.float16, temp=0.7, top_k=64, seed=1234):
        if depth.n_q != N_LEVELS:
            raise ValueError("FastDepth expects the 32-codebook depth stack")
        self.dtype = dtype
        self.temp = float(temp)
        self.top_k = int(top_k)
        with torch.inference_mode():
            d = copy.deepcopy(depth).to(dtype).eval().requires_grad_(False)
            self.depth = d
            self.device = d.in_proj.weight.device
            self.n_codes = d.n_codes
            table = d.code_up(d.code_lo.weight).view(N_LEVELS - 1, self.n_codes, D)
            self.embed = (table + d.level_emb.weight[1:N_LEVELS, None]).contiguous()
            self.level0 = d.level_emb.weight[0].clone()
            self.cross = torch.zeros(LAYERS, 1, 1, D, device=self.device, dtype=dtype)
            self.oT = [[b.o.weight.t() for b in g] for g in d.groups]
            self.f2T = [[b.f2.weight.t() for b in g] for g in d.groups]
            self.kv = [torch.empty(1, N_LEVELS, 2, H, HEAD, device=self.device, dtype=dtype) for _ in range(LAYERS)]
            self.prefix = torch.empty(1, N_LEVELS, D, device=self.device, dtype=dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def set_speaker(self, mem, seed=None):
        """mem: (1, 1, D) depth speaker memory of the voice (one reference, unmasked)."""
        if mem.ndim != 3 or mem.shape[:2] != (1, 1):
            raise ValueError(f"FastDepth serves exactly one reference; got mem {tuple(mem.shape)}")
        mem = mem.to(self.device, self.dtype)
        for layer, block in enumerate(self.depth.cross):
            _, v = block.kv(mem).chunk(2, -1)
            self.cross[layer].copy_(block.o(v))
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.inference_mode()
    def __call__(self, h, cb0):
        """h: (1, D_backbone) backbone hidden state; cb0: int -> (32,) long codes on device."""
        d, kv, prefix = self.depth, self.kv, self.prefix
        prefix[:, 0] = d.in_proj(h.to(self.dtype)) + self.level0
        prefix[0, 1] = self.embed[0, cb0]
        codes = [torch.full((1,), cb0, device=self.device, dtype=torch.long)]
        previous = None
        for level in range(1, N_LEVELS):
            group = GROUP_OF_LEVEL[level]
            n = level + 1
            incremental = group == previous
            reused_later = level + 1 < N_LEVELS and GROUP_OF_LEVEL[level + 1] == group
            x = prefix[:, level:n] if incremental else prefix[:, :n]
            for layer, block in enumerate(d.groups[group]):
                T = x.shape[1]
                qkv = block.qkv(block.n1(x)).view(1, T, 3, H, HEAD)
                q = qkv[:, :, 0].transpose(1, 2)
                if incremental:
                    kv[layer][:, level:n].copy_(qkv[:, :, 1:])
                    k = kv[layer][:, :n, 0].transpose(1, 2)
                    v = kv[layer][:, :n, 1].transpose(1, 2)
                    a = F.scaled_dot_product_attention(q, k, v)
                else:
                    if reused_later:
                        kv[layer][:, :n].copy_(qkv[:, :, 1:])
                    k = qkv[:, :, 1].transpose(1, 2)
                    v = qkv[:, :, 2].transpose(1, 2)
                    if layer == LAYERS - 1:
                        a = F.scaled_dot_product_attention(q[:, :, -1:], k, v)
                        x = x[:, -1:]
                    else:
                        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                # residual adds folded into addmm (MPS pays per kernel): x + o(a), then (x + cross) + f2(ffn)
                x = torch.addmm(x[0], a.transpose(1, 2).reshape(x.shape[1], D), self.oT[group][layer])
                x = torch.addmm(x + self.cross[layer][0], F.gelu(block.f1(block.n2(x))), self.f2T[group][layer])[None]
            logits = d.heads[level - 1](d.norm(x[:, -1])).float() / self.temp
            kth = logits.topk(self.top_k, -1).values[:, -1:]
            probs = logits.masked_fill(logits < kth, -float("inf")).softmax(-1)
            race = torch.empty_like(probs).exponential_(1, generator=self.generator)
            code = (probs / race).argmax(-1)                     # (1,) stays on device: no host sync per level
            codes.append(code)
            if n < N_LEVELS:
                prefix[:, n] = F.embedding(code, self.embed[level])
            previous = group
        return torch.cat(codes)
