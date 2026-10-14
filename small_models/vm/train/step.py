from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def lm_loss(model, tokens, segments):
    hidden = model(tokens[:, :-1], segments[:, :-1])
    return nn.losses.cross_entropy(model.lm_logits(hidden), tokens[:, 1:]).mean()


def make_train_step(model, optimizer, loss_fn=lm_loss, compile: bool = True):
    """Returns (step, state). Call loss = step(*batch), then mx.eval(loss, state)."""
    optimizer.init(model.trainable_parameters())
    value_and_grad = nn.value_and_grad(model, loss_fn)

    def step(*batch):
        loss, grads = value_and_grad(model, *batch)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state, mx.random.state]
    if compile:
        step = mx.compile(step, inputs=state, outputs=state)
    return step, state
