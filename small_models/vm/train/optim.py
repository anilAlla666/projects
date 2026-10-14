"""Muon for hidden weight matrices, AdamW for everything else (embeddings, heads, norms, gates,
convolutions), both with linear warm-up and cosine decay to 10%: the Moonlight recipe."""
from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim

_NOT_MUON = ("embed", "segment", "verdict_head", "pointer_query")


def muon_filter(path: str, weight: mx.array) -> bool:
    return weight.ndim == 2 and not path.startswith(_NOT_MUON)


def _schedule(base: float, total_steps: int, warmup: int):
    if warmup < 1 or total_steps <= warmup:
        raise ValueError("need 1 <= warmup < total_steps")
    return optim.join_schedules(
        [optim.linear_schedule(0.0, base, warmup), optim.cosine_decay(base, total_steps - warmup, base * 0.1)],
        [warmup],
    )


def build_optimizer(lr_muon: float, lr_adam: float, total_steps: int, warmup: int,
                    weight_decay: float = 0.01) -> optim.MultiOptimizer:
    muon = optim.Muon(learning_rate=_schedule(lr_muon, total_steps, warmup), weight_decay=weight_decay)
    adam = optim.AdamW(learning_rate=_schedule(lr_adam, total_steps, warmup), weight_decay=weight_decay)
    return optim.MultiOptimizer([muon, adam], [muon_filter])
