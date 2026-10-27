"""The custom Metal kernels against plain MLX ops (random weights, no checkpoint needed)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from maya_mac import mlx_kernels as K  # noqa: E402
from maya_mac.mlx_model import Linear, _gelu  # noqa: E402

F16 = mx.float16


def rel(a, b):
    a, b = np.array(a.astype(mx.float32), np.float64), np.array(b.astype(mx.float32), np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("bits", [None, 8])
@pytest.mark.parametrize("T,Kd,N", [(1, 1024, 3072), (3, 1024, 1024), (1, 3072, 1024), (4, 1024, 2048)])
def test_linear(bits, T, Kd, N):
    mx.random.seed(0)
    lin = Linear(mx.random.normal((N, Kd)) * 0.03, F16, bits)
    w = (mx.dequantize(lin.wq, lin.scales, lin.biases, 64, 8) if bits else lin.w).astype(mx.float32)
    x = (mx.random.normal((T, Kd)) * 2 + 0.5).astype(F16)
    g, b = (1 + 0.1 * mx.random.normal((Kd,))).astype(F16), (0.1 * mx.random.normal((Kd,))).astype(F16)
    res, c = mx.random.normal((T, N)).astype(F16), mx.random.normal((N,)).astype(F16)
    ln = mx.fast.layer_norm(x.astype(mx.float32), g.astype(mx.float32), b.astype(mx.float32), 1e-5)
    assert rel(K.linear(x, lin, ln=(g, b), gelu=True, res=res, add=c), _gelu(ln @ w.T) + res + c) < 2e-3
    assert rel(K.linear(x, lin), x.astype(mx.float32) @ w.T) < 2e-3


@pytest.mark.parametrize("P,T,last", [(0, 9, False), (0, 9, True), (12, 1, False), (0, 1, False), (31, 1, False)])
def test_attend(P, T, last):
    H, HD = 16, 64
    mx.random.seed(1)
    qkv = mx.random.normal((T, 3 * H * HD)).astype(F16)
    kv = mx.random.normal((2, H, P, HD)).astype(F16) if P else None
    out, kv_out = K.attend(qkv, kv, H, HD, last)

    def heads(a):
        return a.reshape(T, H, HD).transpose(1, 0, 2)[None]
    q, k, v = heads(qkv[:, :1024]), heads(qkv[:, 1024:2048]), heads(qkv[:, 2048:])
    if P:
        k, v = mx.concatenate([kv[0][None], k], 2), mx.concatenate([kv[1][None], v], 2)
    mask = mx.arange(P + T)[None] <= (P + mx.arange(T))[:, None]
    f32 = mx.float32
    ref = mx.fast.scaled_dot_product_attention(q.astype(f32), k.astype(f32), v.astype(f32), scale=HD ** -0.5,
                                               mask=mask)[0].transpose(1, 0, 2).reshape(T, H * HD)
    assert rel(out, ref[-1:] if last else ref) < 2e-3
    assert mx.array_equal(kv_out, mx.stack([k[0], v[0]])).item()


def test_sample_distribution():
    """Draws stay inside the top-k (ties with the k-th value kept) and follow softmax(logits / temp) over it."""
    mx.random.seed(2)
    logits = (mx.random.normal((1, 2048)) * 3).astype(F16)
    table = mx.random.normal((31, 2048, 1024)).astype(F16)
    prefix = mx.random.normal((3, 1024)).astype(F16)
    inv_t, k, n = 1 / 0.7, 64, 20000
    lf = np.array(logits.astype(mx.float32))[0]
    p = np.where(lf >= np.sort(lf)[-k], np.exp((lf - lf.max()) * inv_t), 0.0)
    p /= p.sum()
    draw = mx.compile(lambda key: K.sample(logits, mx.random.gumbel((32, 2048), key=key), 4, table, prefix,
                                           inv_t, k)[0])
    keys = mx.random.split(mx.random.key(1), n)
    codes = np.array(mx.concatenate([draw(keys[i]) for i in range(n)]))
    freq = np.bincount(codes, minlength=2048) / n
    assert (freq[p == 0] == 0).all()
    assert 0.5 * np.abs(freq - p).sum() < 0.03
    code, out = K.sample(logits, mx.random.gumbel((32, 2048), key=keys[0]), 4, table, prefix, inv_t, k)
    assert mx.array_equal(out[:3], prefix).item() and mx.array_equal(out[3], table[4, code[0]]).item()
