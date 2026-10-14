import torch

from vm.model.config import TINY
from vmt.model import Judge
from vmt.train import Trainer, build_optimizers, param_groups

DEV = "cuda"


def run(lr_muon, lr_adam, steps=60):
    torch.manual_seed(0)
    model = Judge(TINY).to(DEV)
    opts, scheds = build_optimizers(model, lr_muon, lr_adam, total_steps=steps, warmup=5)
    trainer = Trainer(model, opts, scheds)
    torch.manual_seed(0)
    tok = torch.randint(0, TINY.vocab_size, (4, 65), device=DEV)
    seg = torch.zeros_like(tok)
    return [trainer.step(tok, seg) for _ in range(steps)]


def test_muon_gets_exactly_the_hidden_weight_matrices():
    m = Judge(TINY)
    muon, adam = param_groups(m)
    names = {id(p): n for n, p in m.named_parameters()}
    muon_names = {names[id(p)] for p in muon}
    assert "layers.0.mixer.q_proj.weight" in muon_names
    assert not any(n.startswith(("embed", "segment", "verdict_head", "pointer_query")) for n in muon_names)
    assert all(p.ndim == 2 for p in muon)
    assert len(muon) + len(adam) == len(list(m.parameters()))


def test_training_fits_a_fixed_batch():
    losses = run(0.02, 3e-3)
    assert losses[-1] < 0.1 * losses[0]


def test_zero_learning_rate_learns_nothing():
    losses = run(0.0, 0.0, steps=10)
    assert abs(losses[-1] - losses[0]) < 1e-3
