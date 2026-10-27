"""Custom Metal kernels for the depth decoder (mx.fast.metal_kernel, JIT-compiled once per shape).

The depth decoder runs 124 tiny layer passes per frame (one or a few positions each), so its cost is the latency of
dependent GPU kernels, not FLOPs: every LayerNorm, GELU, residual add, KV concatenate and transpose is one more
~5-10 us dispatch on the critical path. These kernels cut a layer from ~12 dependent kernels to 5:

  linear   LayerNorm prologue (optional) + x @ W.T for T <= 32 rows + GELU / residual / constant-add epilogue;
           W dense fp16/bf16 or MLX 8-bit (affine, group 64)
  attend   append this pass's K/V to the band's cache and run causal attention over <= 32 positions
  sample   top-k (exact k-th largest by radix select, ties kept) + Gumbel-max draw + the drawn code's embedding,
           appended to the prefix
"""
from __future__ import annotations

import functools

import mlx.core as mx

SIMDS = 8                  # simdgroups per threadgroup in `linear` (one output row each)

_ERF = """
// erf with |error| < 1.2e-7 (Numerical Recipes erfc, Chebyshev fit): the exact-GELU torch uses
inline float erf_(float x) {
    float z = metal::fabs(x), t = 1.0f / (1.0f + 0.5f * z);
    float r = t * metal::exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
              t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
              t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}
"""


@functools.cache
def _linear_kernel(ln, act, res, add, bits, rows, split):
    """Each simdgroup owns `rows` output rows and 1/split of the K range (lane l reads k = 8l, 8l + 256, ... of its
    part) and accumulates the LayerNorm sums of x alongside the dot products: no normalisation pass before the weight
    stream starts. LN(x).w = rstd * (sum (x - x0) g w - (mean - x0) sum g w) + sum b w, the shift x0 = x[t, 0]
    guarding the single-pass variance against cancellation. With split > 1 the parts meet in threadgroup memory."""
    ins = ["x"] + (["wq", "scales", "biases"] if bits else ["w"]) + (["lnw", "lnb"] if ln else []) + \
          (["res"] if res else []) + (["cst"] if add else [])
    if bits == 8:
        load_w = """{
            const device uint* p = wq + ((row0 + r) * K + k) / 4;
            uint a = p[0], b = p[1];
            float sc = float(scales[((row0 + r) * K + k) / 64]), bi = float(biases[((row0 + r) * K + k) / 64]);
            for (int j = 0; j < 4; ++j) { wv[r][j] = float((a >> (8 * j)) & 0xff) * sc + bi;
                                          wv[r][4 + j] = float((b >> (8 * j)) & 0xff) * sc + bi; } }"""
    else:
        load_w = """{
            const device half4* p = (const device half4*)(w + (row0 + r) * K + k);
            half4 a = p[0], b = p[1];
            for (int j = 0; j < 4; ++j) { wv[r][j] = float(a[j]); wv[r][4 + j] = float(b[j]); } }"""
    ln_w = """
        for (int j = 0; j < 8; ++j) {
            float g = float(lnw[k + j]), b = float(lnb[k + j]);
            for (int r = 0; r < R; ++r) { gw[r] += g * wv[r][j]; bw[r] += b * wv[r][j]; wv[r][j] *= g; }
        }""" if ln else ""
    ln_x = """
            for (int j = 0; j < 8; ++j) { xv[j] -= x0[t]; sx[t] += xv[j]; sxx[t] += xv[j] * xv[j]; }""" if ln else ""
    ln_out = """
            float m = SUM(sx[t]) / K, var = SUM(sxx[t]) / K - m * m;
            float rstd = metal::rsqrt(metal::max(var, 0.0f) + 1e-5f);
            for (int r = 0; r < R; ++r) out[r] = rstd * (out[r] - m * gws[r]) + bws[r];""" if ln else ""
    epilogue = "".join([
        "v = 0.5f * v * (1.0f + erf_(v * 0.70710678f));" if act else "",
        "v += float(res[t * N + row0 + r]);" if res else "",
        "v += float(cst[row0 + r]);" if add else ""])
    reduce = f"""
    threadgroup float part_acc[{SIMDS}][R][T], part_g[{SIMDS}][R], part_b[{SIMDS}][R], part_x[{SIMDS}][T], part_xx[{SIMDS}][T];
    for (int r = 0; r < R; ++r) {{ for (int t = 0; t < T; ++t) {{ float v = simd_sum(acc[r][t]); if (lane == 0) part_acc[sg][r][t] = v; }}
                                 float a = simd_sum(gw[r]), b = simd_sum(bw[r]); if (lane == 0) {{ part_g[sg][r] = a; part_b[sg][r] = b; }} }}
    for (int t = 0; t < T; ++t) {{ float a = simd_sum(sx[t]), b = simd_sum(sxx[t]); if (lane == 0) {{ part_x[sg][t] = a; part_xx[sg][t] = b; }} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part != 0) return;
    for (int r = 0; r < R; ++r) {{
        float a = 0.0f, b = 0.0f;
        for (int q = 0; q < S; ++q) {{ a += part_g[sg + q][r]; b += part_b[sg + q][r]; }}
        gw[r] = a; bw[r] = b;
        for (int t = 0; t < T; ++t) {{ float c = 0.0f; for (int q = 0; q < S; ++q) c += part_acc[sg + q][r][t]; acc[r][t] = c; }}
    }}
    for (int t = 0; t < T; ++t) {{ float a = 0.0f, b = 0.0f; for (int q = 0; q < S; ++q) {{ a += part_x[sg + q][t]; b += part_xx[sg + q][t]; }}
                                 sx[t] = a; sxx[t] = b; }}
    #define SUM(v) (v)""" if split > 1 else "#define SUM(v) simd_sum(v)"
    src = f"""
    constexpr int R = {rows}, S = {split};
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, part = sg % S;
    uint row0 = (threadgroup_position_in_grid.x * ({SIMDS} / S) + sg / S) * R;
    float acc[R][T], wv[R][8], gw[R], bw[R], x0[T], sx[T], sxx[T];
    for (int r = 0; r < R; ++r) {{ gw[r] = 0.0f; bw[r] = 0.0f; for (int t = 0; t < T; ++t) acc[r][t] = 0.0f; }}
    for (int t = 0; t < T; ++t) {{ x0[t] = {"float(x[t * K])" if ln else "0.0f"}; sx[t] = 0.0f; sxx[t] = 0.0f; }}
    for (uint k = part * (K / S) + lane * 8; k < (part + 1) * (K / S); k += 256) {{
        for (int r = 0; r < R; ++r) {load_w}
        {ln_w}
        for (int t = 0; t < T; ++t) {{
            const device half4* xp = (const device half4*)(x + t * K + k);
            half4 xa = xp[0], xb = xp[1];
            float xv[8] = {{float(xa[0]), float(xa[1]), float(xa[2]), float(xa[3]),
                           float(xb[0]), float(xb[1]), float(xb[2]), float(xb[3])}};
            {ln_x}
            for (int r = 0; r < R; ++r) {{ float s = 0.0f; for (int j = 0; j < 8; ++j) s += xv[j] * wv[r][j]; acc[r][t] += s; }}
        }}
    }}
    {reduce}
    float gws[R], bws[R];
    for (int r = 0; r < R; ++r) {{ gws[r] = SUM(gw[r]); bws[r] = SUM(bw[r]); }}
    for (int t = 0; t < T; ++t) {{
        float out[R];
        for (int r = 0; r < R; ++r) out[r] = SUM(acc[r][t]);
        {ln_out}
        if (lane == 0) for (int r = 0; r < R; ++r) {{ float v = out[r]; {epilogue} y[t * N + row0 + r] = v; }}
    }}
    """
    return mx.fast.metal_kernel(name=f"maya_linear_{int(ln)}{int(act)}{int(res)}{int(add)}_{bits or 16}_{rows}_{split}",
                                input_names=ins, output_names=["y"], source=src, header=_ERF)


# Launch shape, swept on an M5 Pro over a chain of real-size depth layers: one row per simdgroup (most threadgroups in
# flight) everywhere; 8-bit weights also split the K range of the 1024-row matrices across two simdgroups.
# fp16 129 us/layer (vs 199 with two rows per simdgroup), 8-bit 94 us/layer.
def _shape(T, K, N, bits):
    return 1, (2 if bits and N <= 1024 else 1)


def linear(x, lin, ln=None, gelu=False, res=None, add=None, rows=None, split=None):
    """x (T, K) -> (T, N) = [res +] [add +] act(LN(x) @ W.T); lin: mlx_model.Linear (dense or 8-bit)."""
    T, K = x.shape
    bits = lin.bits
    if bits not in (None, 8):
        raise ValueError("fused depth kernels support dense or 8-bit weights")
    N = (lin.wq if bits else lin.w).shape[0]
    rows = rows or _shape(T, K, N, bits)[0]
    split = split or _shape(T, K, N, bits)[1]
    while (K // split) % 256 or N % (SIMDS // split * rows):
        split //= 2
    ins = [x] + ([lin.wq, lin.scales, lin.biases] if bits else [lin.w]) + (list(ln) if ln is not None else []) + \
          ([res] if res is not None else []) + ([add] if add is not None else [])
    k = _linear_kernel(ln is not None, gelu, res is not None, add is not None, bits, rows, split)
    groups = N // (SIMDS // split * rows)
    return k(inputs=ins, template=[("T", T), ("K", K), ("N", N)], grid=(groups * 32 * SIMDS, 1, 1),
             threadgroup=(32 * SIMDS, 1, 1), output_shapes=[(T, N)], output_dtypes=[x.dtype])[0]


_ATTEND = """
    // one simdgroup per (head, new position t); lane j <-> key j (<= 32 keys), lanes d, d+32 <-> head dims
    uint h = threadgroup_position_in_grid.x, t = threadgroup_position_in_grid.y, lane = thread_index_in_simdgroup;
    const uint D = H * HD, L = P + T;
    for (uint d = lane; d < HD; d += 32) {           // this pass's K/V rows -> the cache
        kv_out[(h * L + P + t) * HD + d] = qkv[t * 3 * D + D + h * HD + d];
        kv_out[((H + h) * L + P + t) * HD + d] = qkv[t * 3 * D + 2 * D + h * HD + d];
    }
    if (t == 0) for (uint j = 0; j < P; ++j) for (uint d = lane; d < HD; d += 32) {
        kv_out[(h * L + j) * HD + d] = kv[(h * P + j) * HD + d];
        kv_out[((H + h) * L + j) * HD + d] = kv[((H + h) * P + j) * HD + d];
    }
    if (LAST && t != T - 1) return;
    uint pos = P + t;                                  // causal: keys 0..pos
    float s = -INFINITY;
    if (lane <= pos) {
        const device half* q = qkv + t * 3 * D + h * HD;
        const device half* k = lane < P ? kv + (h * P + lane) * HD : qkv + (lane - P) * 3 * D + D + h * HD;
        float a = 0.0f;
        for (uint d = 0; d < HD; ++d) a += float(q[d]) * float(k[d]);
        s = a * SCALE;
    }
    float m = simd_max(s);
    float p = lane <= pos ? metal::exp(s - m) : 0.0f;
    p /= simd_sum(p);
    uint qi = LAST ? 0 : t;
    for (uint d = lane; d < HD; d += 32) {
        float o = 0.0f;
        for (uint j = 0; j <= pos; ++j) {
            float v = j < P ? float(kv[((H + h) * P + j) * HD + d]) : float(qkv[(j - P) * 3 * D + 2 * D + h * HD + d]);
            o += simd_shuffle(p, (ushort)j) * v;
        }
        out[qi * D + h * HD + d] = o;
    }
"""


@functools.cache
def _attend_kernel(scale):
    return mx.fast.metal_kernel(name="maya_attend", input_names=["qkv", "kv"], output_names=["out", "kv_out"],
                                source=_ATTEND.replace("SCALE", repr(float(scale)) + "f"))


def attend(qkv, kv, heads, head_dim, last=False):
    """qkv (T, 3*D) of the new positions, kv (2, H, P, HD) the band's cache or None -> (attention output (T or 1, D),
    cache (2, H, P + T, HD)). Causal; last=True computes only the last position's output."""
    T = qkv.shape[0]
    P = 0 if kv is None else kv.shape[2]
    if P + T > 32:
        raise ValueError("attend handles at most 32 positions")
    D = heads * head_dim
    if kv is None:          # nothing cached yet: a placeholder the kernel never reads (P = 0)
        kv = mx.zeros((2, heads, 1, head_dim), qkv.dtype)
    out, kv_out = _attend_kernel(head_dim ** -0.5)(
        inputs=[qkv, kv], template=[("H", heads), ("HD", head_dim), ("P", P), ("T", T), ("LAST", int(last))],
        grid=(heads * 32, T, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1 if last else T, D), (2, heads, P + T, head_dim)], output_dtypes=[qkv.dtype, qkv.dtype])
    return out, kv_out


_SAMPLE = """
    // one threadgroup of 1024 threads, 2 logits each: exact k-th largest (radix select on order-preserving keys),
    // then Gumbel-max over the kept set (ties with the k-th value are kept, as torch's `logits < kth` mask does)
    uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup atomic_uint hist[256];
    threadgroup uint sh_prefix, sh_need, sh_code;
    threadgroup float best_s[32];
    threadgroup uint best_i[32];
    uint key[2]; float val[2];
    for (int e = 0; e < 2; ++e) {
        float f = float(logits[tid + e * 1024]);
        uint u = as_type<uint>(f);
        key[e] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        val[e] = f;
    }
    if (tid == 0) { sh_prefix = 0; sh_need = TOPK; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = sh_prefix;
        for (int e = 0; e < 2; ++e)
            if (shift == 24 || (key[e] >> (shift + 8)) == prefix)
                atomic_fetch_add_explicit(&hist[(key[e] >> shift) & 0xff], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {                     // digit holding the need-th largest, scanning bins from the top
            uint c[8], tot = 0;
            for (int i = 0; i < 8; ++i) { c[i] = atomic_load_explicit(&hist[255 - (lane * 8 + i)], memory_order_relaxed); tot += c[i]; }
            uint before = simd_prefix_exclusive_sum(tot), need = sh_need;
            if (before < need && need <= before + tot) {
                uint run = before;
                for (int i = 0; i < 8; ++i) {
                    if (need <= run + c[i]) { sh_prefix = (prefix << 8) | (255 - (lane * 8 + i)); sh_need = need - run; break; }
                    run += c[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint kth = sh_prefix;
    float bs = -INFINITY; uint bi = 0xffffffffu;
    for (int e = 0; e < 2; ++e) {
        uint i = tid + e * 1024;
        if (key[e] >= kth) {
            float s = val[e] * INV_TEMP + gumbel[LEVEL * 2048 + i];
            if (s > bs || (s == bs && i < bi)) { bs = s; bi = i; }
        }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float os = simd_shuffle_down(bs, off); uint oi = simd_shuffle_down(bi, off);
        if (os > bs || (os == bs && oi < bi)) { bs = os; bi = oi; }
    }
    if (lane == 0) { best_s[sg] = bs; best_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bs = best_s[lane]; bi = best_i[lane];
        for (ushort off = 16; off > 0; off >>= 1) {
            float os = simd_shuffle_down(bs, off); uint oi = simd_shuffle_down(bi, off);
            if (os > bs || (os == bs && oi < bi)) { bs = os; bi = oi; }
        }
        if (lane == 0) { sh_code = bi; code[0] = int(bi); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = sh_code;
    for (uint r = 0; r < N_PREV; ++r) prefix_out[r * 1024 + tid] = prefix[r * 1024 + tid];
    if (EMBED) prefix_out[N_PREV * 1024 + tid] = table[(LEVEL * 2048 + c) * 1024 + tid];
"""


@functools.cache
def _sample_kernel(inv_temp, top_k):
    src = _SAMPLE.replace("INV_TEMP", repr(float(inv_temp)) + "f").replace("TOPK", f"{int(top_k)}u")
    return mx.fast.metal_kernel(name=f"maya_sample_{top_k}", input_names=["logits", "gumbel", "table", "prefix"],
                                output_names=["code", "prefix_out"], source=src)


def sample(logits, gumbel, level, table, prefix, inv_temp, top_k, embed=True):
    """logits (1, 2048) of depth level `level`; gumbel (32, 2048) float32 noise of this frame (row = level); table
    (31, 2048, D) code
    embeddings; prefix (n, D) -> (code (1,) int32, prefix + [table[level, code]] (n+1, D), or prefix if not embed)."""
    n, D = prefix.shape
    if logits.shape[-1] != 2048 or D != 1024:
        raise ValueError("sample is built for 2048 codes and D = 1024")
    return _sample_kernel(inv_temp, top_k)(
        inputs=[logits, gumbel, table, prefix], template=[("LEVEL", level), ("N_PREV", n), ("EMBED", int(embed))],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(1,), (n + int(embed), D)],
        output_dtypes=[mx.int32, prefix.dtype])
