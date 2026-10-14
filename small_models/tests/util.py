import mlx.core as mx


def causal_leak(fn, x, t0: int) -> float:
    """Largest change in outputs at positions < t0 when inputs at positions >= t0 are replaced."""
    noise = mx.random.normal(x[:, t0:].shape).astype(x.dtype)
    x2 = mx.concatenate([x[:, :t0], noise], axis=1)
    return mx.abs(fn(x)[:, :t0] - fn(x2)[:, :t0]).max().item()
