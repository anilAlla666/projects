import numpy as np
import torch

from vmt.data import MixedSampler, TokenStream


def stream(tmp_path, name, value, n=100_000):
    p = tmp_path / f"{name}.bin"
    np.full(n, value, dtype=np.uint16).tofile(p)
    return TokenStream(p)


def test_mixing_ratio_and_shapes(tmp_path):
    s = MixedSampler({"code": stream(tmp_path, "code", 7), "en": stream(tmp_path, "en", 9)}, {"code": 0.6, "en": 0.4}, seed=0, rank=0)
    b = s.batch(1000, 64)
    assert b.shape == (1000, 65) and b.dtype == torch.long
    frac_code = (b[:, 0] == 7).float().mean().item()
    assert 0.55 < frac_code < 0.65


def test_ranks_draw_different_batches(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "r.bin"
    rng.integers(0, 30000, 200_000).astype(np.uint16).tofile(p)
    st = {"x": TokenStream(p)}
    a = MixedSampler(st, {"x": 1.0}, seed=0, rank=0).batch(4, 32)
    b = MixedSampler(st, {"x": 1.0}, seed=0, rank=1).batch(4, 32)
    assert not torch.equal(a, b)
    assert torch.equal(a, MixedSampler(st, {"x": 1.0}, seed=0, rank=0).batch(4, 32))
