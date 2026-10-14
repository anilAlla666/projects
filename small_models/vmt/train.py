"""Muon for hidden weight matrices, AdamW for the rest, warm-up then cosine decay to 10%; bfloat16 autocast;
fused linear cross-entropy so the [tokens x vocabulary] logits are never materialised."""
from __future__ import annotations

import math

import torch
from fla.modules import FusedLinearCrossEntropyLoss

_NOT_MUON = ("embed", "segment", "verdict_head", "pointer_query")
_flce = FusedLinearCrossEntropyLoss()


def param_groups(model):
    muon, adam = [], []
    for name, p in model.named_parameters():
        (muon if p.ndim == 2 and not name.startswith(_NOT_MUON) else adam).append(p)
    return muon, adam


def _lr_lambda(total_steps: int, warmup: int):
    if warmup < 1 or total_steps <= warmup:
        raise ValueError("need 1 <= warmup < total_steps")

    def f(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = min(1.0, (step - warmup) / (total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    return f


def build_optimizers(model, lr_muon, lr_adam, total_steps, warmup, weight_decay=0.01):
    muon_p, adam_p = param_groups(model)
    opts = [torch.optim.Muon(muon_p, lr=lr_muon, weight_decay=weight_decay),
            torch.optim.AdamW(adam_p, lr=lr_adam, weight_decay=weight_decay, betas=(0.9, 0.95))]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, _lr_lambda(total_steps, warmup)) for o in opts]
    return opts, scheds


def lm_loss_with(run_model, model, tokens, segments):
    """Next-token loss where run_model (possibly torch.compile'd) runs the forward and model owns the tied weight."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = run_model(tokens[:, :-1], segments[:, :-1])
    flat = hidden.reshape(-1, hidden.shape[-1])
    return _flce(flat, tokens[:, 1:].reshape(-1), model.lm_weight().to(flat.dtype))


def lm_loss(model, tokens, segments):
    return lm_loss_with(model, model, tokens, segments)


class Trainer:
    def __init__(self, model, optimizers, schedulers, loss_fn=lm_loss):
        self.model, self.optimizers, self.schedulers, self.loss_fn = model, optimizers, schedulers, loss_fn

    def step(self, *batch) -> float:
        loss = self.loss_fn(self.model, *batch)
        loss.backward()
        for o in self.optimizers:
            o.step()
            o.zero_grad(set_to_none=True)
        for s in self.schedulers:
            s.step()
        return loss.item()
