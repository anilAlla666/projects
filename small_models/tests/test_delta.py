import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_ops

from vm.model.delta import chunked_delta


def inputs(B=2, T=128, H=3, Dk=32, Dv=48, lo=-1.0, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, H, Dk))
    k = mx.random.normal((B, T, H, Dk))
    v = mx.random.normal((B, T, H, Dv))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * Dk**-0.5
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    log_g = mx.random.uniform(lo, 0.0, (B, T, H, Dk))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    return q, k, v, log_g, beta


def reference(q, k, v, log_g, beta, state=None):
    ref_state = None if state is None else state.swapaxes(-1, -2)
    o, s = gated_delta_ops(q, k, v, mx.exp(log_g), beta, ref_state)
    return o, s.swapaxes(-1, -2)


def maxdiff(a, b):
    return mx.abs(a - b).max().item()


@pytest.mark.parametrize("lo", [-0.1, -1.0])
def test_matches_reference(lo):
    x = inputs(lo=lo)
    o1, s1 = chunked_delta(*x)
    o2, s2 = reference(*x)
    assert maxdiff(o1, o2) < 1e-5
    assert maxdiff(s1, s2) < 1e-5


def test_padding_when_length_is_not_a_multiple_of_the_chunk():
    x = inputs(T=100)
    o1, s1 = chunked_delta(*x)
    o2, s2 = reference(*x)
    assert o1.shape == (2, 100, 3, 48)
    assert maxdiff(o1, o2) < 1e-5
    assert maxdiff(s1, s2) < 1e-5


def test_initial_state_carries_over():
    q, k, v, g, b = inputs(T=128)
    o_full, s_full = chunked_delta(q, k, v, g, b)
    o_a, s_a = chunked_delta(q[:, :64], k[:, :64], v[:, :64], g[:, :64], b[:, :64])
    o_b, s_b = chunked_delta(q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], b[:, 64:], state=s_a)
    assert maxdiff(mx.concatenate([o_a, o_b], axis=1), o_full) < 1e-5
    assert maxdiff(s_b, s_full) < 1e-5


def test_gradients_match_reference():
    x = inputs(T=64)
    f1 = lambda *a: chunked_delta(*a)[0].square().sum()  # noqa: E731
    f2 = lambda *a: reference(*a)[0].square().sum()  # noqa: E731
    g1 = mx.grad(f1, argnums=(0, 1, 2, 3, 4))(*x)
    g2 = mx.grad(f2, argnums=(0, 1, 2, 3, 4))(*x)
    for a, b in zip(g1, g2):
        assert maxdiff(a, b) < 1e-4 * max(1.0, mx.abs(b).max().item())


def test_comparison_has_teeth():
    """A 10% change in beta must break the comparison, so a pass above is not vacuous."""
    q, k, v, g, b = inputs()
    o1, _ = chunked_delta(q, k, v, g, b * 0.9)
    o2, _ = reference(q, k, v, g, b)
    assert maxdiff(o1, o2) > 1e-3
