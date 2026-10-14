import mlx.core as mx
import pytest

from tests.util import causal_leak
from vm.model.layers import KDA, Attention, ShortConv, SwiGLU, ZRMSNorm

B, T, D = 2, 70, 128


@pytest.fixture
def x():
    mx.random.seed(1)
    return mx.random.normal((B, T, D))


def test_causal_leak_has_teeth(x):
    """The checker must catch a function that looks ahead."""
    assert causal_leak(lambda y: mx.broadcast_to(y.mean(axis=1, keepdims=True), y.shape), x, 30) > 1e-3


@pytest.mark.parametrize("make", [
    lambda: ShortConv(D),
    lambda: KDA(D, 2, 64),
    lambda: Attention(D, 2, 1, 64),
])
def test_layers_are_causal_and_keep_shape(make, x):
    layer = make()
    mx.eval(layer.parameters())
    assert layer(x).shape == (B, T, D)
    assert causal_leak(layer, x, 30) < 1e-5


def test_norm_and_mlp_shapes(x):
    assert ZRMSNorm(D)(x).shape == (B, T, D)
    assert SwiGLU(D, 256)(x).shape == (B, T, D)


def test_zero_centred_norm_starts_as_plain_rms_norm(x):
    out = ZRMSNorm(D)(x)
    expected = x * mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
    assert mx.abs(out - expected).max().item() < 1e-4


def test_kda_gates_stay_in_range(x):
    layer = KDA(D, 2, 64, gate_lower=-1.0)
    log_g, beta = layer.gates(x)
    assert log_g.shape == (B, T, 2, 64) and beta.shape == (B, T, 2)
    assert log_g.max().item() <= 0.0 and log_g.min().item() >= -1.0
    assert beta.min().item() >= 0.0 and beta.max().item() <= 1.0


def naive_attention(layer, x):
    Bn, Tn, _ = x.shape
    H, Hk, Dh = layer.n_heads, layer.n_kv_heads, layer.head_dim
    q = layer.q_norm(layer.q_proj(x).reshape(Bn, Tn, H, Dh)).transpose(0, 2, 1, 3)
    k = layer.k_norm(layer.k_proj(x).reshape(Bn, Tn, Hk, Dh)).transpose(0, 2, 1, 3)
    v = layer.v_proj(x).reshape(Bn, Tn, Hk, Dh).transpose(0, 2, 1, 3)
    k, v = mx.repeat(k, H // Hk, axis=1), mx.repeat(v, H // Hk, axis=1)
    s = (q @ k.swapaxes(-1, -2)) * Dh**-0.5
    mask = mx.arange(Tn)[:, None] >= mx.arange(Tn)[None, :]
    p = mx.softmax(mx.where(mask, s, -1e9), axis=-1)
    o = (p @ v).transpose(0, 2, 1, 3).reshape(Bn, Tn, -1) * mx.sigmoid(layer.g_proj(x))
    return layer.o_proj(o)


def test_attention_matches_naive_softmax(x):
    layer = Attention(D, 4, 2, 32)
    mx.eval(layer.parameters())
    assert mx.abs(layer(x) - naive_attention(layer, x)).max().item() < 1e-4
