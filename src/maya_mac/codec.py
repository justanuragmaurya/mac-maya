"""Streaming Mimi decoder for one stream: feed any number of frames at a time, get exactly their audio back.

Arithmetic of upstream serve/fast_codec.StaticMimi (itself a port of the verified streaming reference) without CUDA
graphs or seat slots: causal-convolution history, transposed-convolution overlap tails (bias removed from the stored
tail), and sliding-window attention with absolute RoPE positions (a query at position p sees keys with
p - sliding_window < key_pos <= p). HF sub-modules are only called for their weights.
"""
import torch
import torch.nn.functional as F
from transformers.models.mimi.modeling_mimi import (MimiConv1d, MimiConvTranspose1d, MimiResnetBlock,
                                                    apply_rotary_pos_emb, repeat_kv)


class StreamingMimi:
    def __init__(self, mimi):
        cfg = mimi.config
        if not cfg.use_causal_conv or cfg.pad_mode != "constant" or cfg.trim_right_ratio != 1:
            raise ValueError("streaming decode requires causal constant padding and trim_right_ratio=1")
        self.m = mimi
        self.window = int(cfg.sliding_window)
        attn = mimi.decoder_transformer.layers[0].self_attn
        self.kv_groups, self.head_dim = attn.num_key_value_groups, attn.head_dim
        rotary = attn.rotary_emb
        self._inv_freq = rotary.inv_freq.detach().float().clone()
        self._rope_scale = float(rotary.attention_scaling)
        self.reset()

    def reset(self):
        self.state = {}
        self.kv = [None] * len(self.m.decoder_transformer.layers)   # [(k, v)] post-RoPE, last window-1 tokens
        self.tokens = 0

    def _layer(self, module, x, key):
        if isinstance(module, MimiConv1d):
            if module.conv.stride != (1,):
                raise ValueError("only stride-one decoder convolutions are supported")
            pad = (module.conv.kernel_size[0] - 1) * module.conv.dilation[0]
            if pad:
                prev = self.state.get(key)
                if prev is None:
                    prev = x.new_zeros(x.shape[0], x.shape[1], pad)
                x = torch.cat((prev, x), dim=-1)
                self.state[key] = x[..., -pad:]
            return module.conv(x)
        if isinstance(module, MimiConvTranspose1d):
            y = module.conv(x)
            produced = x.shape[-1] * module.conv.stride[0]
            if module.padding_right:
                tail = y[..., produced:]
                if module.conv.bias is not None:
                    tail = tail - module.conv.bias[None, :, None]
                prev = self.state.get(key)
                y = y[..., :produced].clone()
                if prev is not None:
                    y[..., :prev.shape[-1]] += prev
                self.state[key] = tail
                return y
            return y[..., :produced]
        if isinstance(module, MimiResnetBlock):
            shortcut = self._layer(module.shortcut, x, key + ".shortcut")
            for i, child in enumerate(module.block):
                x = self._layer(child, x, key + f".block.{i}")
            return shortcut + x
        return module(x)

    @staticmethod
    def _rvq(rvq, codes):
        total = None
        for i in range(codes.shape[1]):
            e = F.embedding(codes[:, i], rvq.layers[i].codebook.embed).permute(0, 2, 1)
            total = e if total is None else total + e
        return rvq.output_proj(total) if rvq.output_proj is not None else total

    def _quantizer(self, codes):
        q = self.m.quantizer
        n = q.num_semantic_quantizers
        out = self._rvq(q.semantic_residual_vector_quantizer, codes[:, :n])
        if codes.shape[1] > n:
            out = out + self._rvq(q.acoustic_residual_vector_quantizer, codes[:, n:])
        return out

    def _transformer(self, h):
        B, T, _ = h.shape
        pos = torch.arange(self.tokens, self.tokens + T, device=h.device)
        freqs = pos[:, None].float() * self._inv_freq.to(h.device)[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self._rope_scale).to(h.dtype)[None]
        sin = (emb.sin() * self._rope_scale).to(h.dtype)[None]
        n_past = 0 if self.kv[0] is None else self.kv[0][0].shape[2]
        key_pos = torch.arange(self.tokens - n_past, self.tokens + T, device=h.device)
        mask = (key_pos[None] <= pos[:, None]) & (key_pos[None] > pos[:, None] - self.window)
        for li, layer in enumerate(self.m.decoder_transformer.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.kv[li] is not None:
                k = torch.cat((self.kv[li][0], k), dim=2)
                v = torch.cat((self.kv[li][1], v), dim=2)
            keep = self.window - 1
            self.kv[li] = (k[:, :, -keep:], v[:, :, -keep:])
            a = F.scaled_dot_product_attention(q, repeat_kv(k, self.kv_groups), repeat_kv(v, self.kv_groups),
                                               attn_mask=mask[None, None])
            a = attn.o_proj(a.transpose(1, 2).reshape(B, T, -1))
            h = h + layer.self_attn_layer_scale(a)
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        self.tokens += T
        return h

    @torch.inference_mode()
    def decode(self, codes):
        """codes: (32, T) long (one stream, T frames) -> float32 numpy (T * 1920,) PCM in [-1, 1]."""
        x = self._quantizer(codes[None].long())
        x = self._layer(self.m.upsample, x, "upsample")
        x = self._transformer(x.transpose(1, 2)).transpose(1, 2)
        for i, layer in enumerate(self.m.decoder.layers):
            x = self._layer(layer, x, f"decoder.{i}")
        return x[0, 0].float().clamp(-1, 1).cpu().numpy()
