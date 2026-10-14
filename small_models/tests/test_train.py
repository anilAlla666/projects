import mlx.core as mx
from mlx.utils import tree_flatten

from vm.model.config import TINY
from vm.model.model import Judge
from vm.train.optim import build_optimizer, muon_filter
from vm.train.step import make_train_step


def fixed_batch():
    mx.random.seed(0)
    tok = mx.random.randint(0, TINY.vocab_size, (4, 65))
    return tok, mx.zeros((4, 65), dtype=mx.int32)


def train(lr_muon, lr_adam, steps=60, compile=True):
    mx.random.seed(0)
    model = Judge(TINY)
    mx.eval(model.parameters())
    opt = build_optimizer(lr_muon, lr_adam, total_steps=steps, warmup=5)
    step, state = make_train_step(model, opt, compile=compile)
    tok, seg = fixed_batch()
    losses = []
    for _ in range(steps):
        loss = step(tok, seg)
        mx.eval(loss, state)
        losses.append(loss.item())
    return losses


def test_muon_gets_exactly_the_hidden_weight_matrices():
    params = dict(tree_flatten(Judge(TINY).parameters()))
    muon = {k for k, v in params.items() if muon_filter(k, v)}
    assert "layers.0.mixer.q_proj.weight" in muon
    assert "layers.0.mlp.down.weight" in muon
    assert not any(k.startswith(("embed", "segment", "verdict_head", "pointer_query")) for k in muon)
    assert all(params[k].ndim == 2 for k in muon)


def test_training_fits_a_fixed_batch():
    losses = train(0.02, 3e-3)
    assert losses[-1] < 0.1 * losses[0]


def test_zero_learning_rate_learns_nothing():
    """Tooth for the test above: with no learning rate the loss must not move."""
    losses = train(0.0, 0.0, steps=10)
    assert abs(losses[-1] - losses[0]) < 1e-5


def test_compiled_step_matches_eager():
    a = train(0.02, 3e-3, steps=10, compile=False)
    b = train(0.02, 3e-3, steps=10, compile=True)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4
