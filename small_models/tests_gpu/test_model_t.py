import torch

from vm.model.config import DEV, MAIN, MID, TINY
from vmt.model import Judge, n_params

DEV_ = "cuda"


def test_parameter_counts_equal_the_mlx_model():
    assert n_params(Judge(TINY)) == 245_572
    assert n_params(Judge(DEV)) == 24_324_959
    assert n_params(Judge(MID)) == 44_509_873
    assert n_params(Judge(MAIN)) == 116_490_541


def test_layer_pattern():
    m = Judge(MAIN)
    kinds = [type(b.mixer).__name__ for b in m.layers]
    assert [i for i, k in enumerate(kinds) if k == "Attention"] == [3, 7, 11]


def batch(B=2, T=40):
    torch.manual_seed(0)
    return (torch.randint(0, TINY.vocab_size, (B, T), device=DEV_),
            torch.randint(0, 4, (B, T), device=DEV_))


def test_heads_and_masking():
    m = Judge(TINY).to(DEV_)
    tok, seg = batch()
    with torch.no_grad():
        h = m(tok, seg)
        assert h.shape == (2, 40, TINY.d_model)
        assert m.lm_logits(h).shape == (2, 40, TINY.vocab_size)
        pos = torch.tensor([39, 20], device=DEV_)
        assert m.verdict_logit(h, pos).shape == (2,)
        mask = torch.zeros(2, 40, dtype=torch.bool, device=DEV_)
        mask[:, 5] = True
        p = m.pointer_logits(h, pos, mask)
        assert p[0, 6].item() == -1e9 and p[0, 5].item() > -1e9


def test_model_is_causal_and_segments_matter():
    m = Judge(TINY).to(DEV_)
    tok, seg = batch()
    with torch.no_grad():
        h1 = m(tok, seg)
        h2 = m(torch.cat([tok[:, :20], torch.randint(0, TINY.vocab_size, (2, 20), device=DEV_)], 1), seg)
        assert (h1[:, :20] - h2[:, :20]).abs().max().item() < 1e-4
        h3 = m(tok, torch.full_like(seg, 3))
        assert (h1 - h3).abs().max().item() > 1e-3
