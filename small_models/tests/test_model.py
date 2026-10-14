import mlx.core as mx

from tests.util import causal_leak  # noqa: F401  (kept with the other causality tests)
from vm.model.config import DEV, MAIN, MID, TINY
from vm.model.model import Judge, n_params


def test_parameter_counts_match_the_spec():
    assert 20e6 <= n_params(Judge(DEV)) <= 30e6
    assert 40e6 <= n_params(Judge(MID)) <= 50e6
    assert 100e6 <= n_params(Judge(MAIN)) <= 125e6


def test_three_delta_layers_to_one_attention_layer():
    assert [i for i in range(MAIN.n_layers) if MAIN.is_attention(i)] == [3, 7, 11]
    assert [i for i in range(DEV.n_layers) if DEV.is_attention(i)] == [3]


def batch(B=2, T=40, seed=0):
    mx.random.seed(seed)
    tok = mx.random.randint(0, TINY.vocab_size, (B, T))
    seg = mx.random.randint(0, 4, (B, T))
    return tok, seg


def test_heads_have_the_right_shapes():
    m = Judge(TINY)
    tok, seg = batch()
    h = m(tok, seg)
    assert h.shape == (2, 40, TINY.d_model)
    assert m.lm_logits(h).shape == (2, 40, TINY.vocab_size)
    pos = mx.array([39, 20])
    assert m.verdict_logit(h, pos).shape == (2,)
    line_mask = mx.zeros((2, 40), dtype=mx.bool_)
    line_mask[:, 5] = True
    p = m.pointer_logits(h, pos, line_mask)
    assert p.shape == (2, 40)
    assert p[0, 6].item() == -1e9 and p[0, 5].item() > -1e9


def test_model_is_causal():
    m = Judge(TINY)
    tok, seg = batch()
    emb = lambda t: m(t, seg)  # noqa: E731
    noise = mx.random.randint(0, TINY.vocab_size, (2, 20))
    h1 = emb(tok)
    h2 = emb(mx.concatenate([tok[:, :20], noise], axis=1))
    assert mx.abs(h1[:, :20] - h2[:, :20]).max().item() < 1e-5


def test_segment_ids_change_the_representation():
    """The trust tag must reach the model: same tokens, different segments, different states."""
    m = Judge(TINY)
    tok, seg = batch()
    h1 = m(tok, mx.zeros_like(seg))
    h2 = m(tok, mx.full(seg.shape, 3, dtype=seg.dtype))
    assert mx.abs(h1 - h2).max().item() > 1e-3


def test_language_model_head_is_tied():
    names = [k for k, _ in __import__("mlx.utils", fromlist=["x"]).tree_flatten(Judge(TINY).parameters())]
    assert not any(k.startswith("lm_head") for k in names)
