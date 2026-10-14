import pytest
import torch
import torch.nn.functional as F

from vmt.kda_ref import kda_recurrent
from vmt.layers import KDA, Attention, ShortConv, SwiGLU, ZRMSNorm
from fla.ops.kda import chunk_kda

DEV = "cuda"
B, T, D = 2, 70, 128


def causal_leak(fn, x, t0):
    x2 = torch.cat([x[:, :t0], torch.randn_like(x[:, t0:])], dim=1)
    with torch.no_grad():
        return (fn(x)[:, :t0] - fn(x2)[:, :t0]).abs().max().item()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D, device=DEV)


def kda_inputs(Bn=2, Tn=128, H=3, K=64, lo=-1.0):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(Bn, Tn, H, K, device=DEV), dim=-1)
    k = F.normalize(torch.randn(Bn, Tn, H, K, device=DEV), dim=-1)
    v = torch.randn(Bn, Tn, H, K, device=DEV)
    g = torch.rand(Bn, Tn, H, K, device=DEV) * lo
    b = torch.rand(Bn, Tn, H, device=DEV)
    return q, k, v, g, b


@pytest.mark.parametrize("lo", [-0.1, -1.0])
def test_fused_kernel_matches_plain_recurrence(lo):
    q, k, v, g, b = kda_inputs(lo=lo)
    o1, s1 = chunk_kda(q, k, v, g, b, scale=64**-0.5, output_final_state=True)
    o2, s2 = kda_recurrent(q, k, v, g, b, 64**-0.5)
    assert (o1.float() - o2).abs().max().item() < 2e-3 * o2.abs().max().item() + 1e-4
    assert (s1.float() - s2).abs().max().item() < 2e-3 * s2.abs().max().item() + 1e-3


def test_fused_kernel_gradients_match():
    xs = [t.clone().requires_grad_() for t in kda_inputs(Tn=64)]
    chunk_kda(*xs, scale=64**-0.5)[0].square().sum().backward()
    ys = [t.detach().clone().requires_grad_() for t in xs]
    kda_recurrent(*ys, 64**-0.5)[0].square().sum().backward()
    for a, c in zip(xs, ys):
        assert (a.grad - c.grad).abs().max().item() < 5e-3 * max(1.0, c.grad.abs().max().item())


def test_comparison_has_teeth():
    q, k, v, g, b = kda_inputs()
    o1, _ = chunk_kda(q, k, v, g, b * 0.9, scale=64**-0.5)
    o2, _ = kda_recurrent(q, k, v, g, b, 64**-0.5)
    assert (o1.float() - o2).abs().max().item() > 1e-2 * o2.abs().max().item()


def test_causal_leak_has_teeth(x):
    assert causal_leak(lambda y: y.mean(dim=1, keepdim=True).expand_as(y), x, 30) > 1e-3


@pytest.mark.parametrize("make", [lambda: ShortConv(D), lambda: KDA(D, 2, 64), lambda: Attention(D, 2, 1, 64)])
def test_layers_are_causal_and_keep_shape(make, x):
    layer = make().to(DEV)
    assert layer(x).shape == (B, T, D)
    assert causal_leak(layer, x, 30) < 1e-4


def test_kda_layer_fused_equals_reference_path(x):
    layer = KDA(D, 2, 64).to(DEV)
    with torch.no_grad():
        fused = layer(x)
        layer.use_reference = True
        ref = layer(x)
    assert (fused - ref).abs().max().item() < 2e-3 * ref.abs().max().item() + 1e-4


def test_kda_gates_stay_in_range(x):
    log_g, beta = KDA(D, 2, 64).to(DEV).gates(x)
    assert log_g.max().item() <= 0.0 and log_g.min().item() >= -1.0
    assert 0.0 <= beta.min().item() and beta.max().item() <= 1.0


def test_norm_and_mlp(x):
    out = ZRMSNorm(D).to(DEV)(x)
    assert (out - x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).abs().max().item() < 1e-4
    assert SwiGLU(D, 256).to(DEV)(x).shape == (B, T, D)


def test_attention_matches_naive_softmax(x):
    layer = Attention(D, 4, 2, 32).to(DEV)
    H, Hk, Dh = 4, 2, 32
    with torch.no_grad():
        q = layer.q_norm(layer.q_proj(x).view(B, T, H, Dh)).transpose(1, 2)
        k = layer.k_norm(layer.k_proj(x).view(B, T, Hk, Dh)).transpose(1, 2).repeat_interleave(H // Hk, 1)
        v = layer.v_proj(x).view(B, T, Hk, Dh).transpose(1, 2).repeat_interleave(H // Hk, 1)
        s = (q @ k.transpose(-1, -2)) * Dh**-0.5
        s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(layer.g_proj(x))
        assert (layer(x) - layer.o_proj(o)).abs().max().item() < 1e-4
