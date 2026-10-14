"""Evaluation statistics: balanced accuracy, expected calibration error, bootstrap intervals resampled by library,
the gap-relative bar, and accept rates per attempt kind."""
from __future__ import annotations

import random
from collections import defaultdict


def balanced_accuracy(labels, preds) -> float:
    pos = [p for l, p in zip(labels, preds) if l == 1]
    neg = [p for l, p in zip(labels, preds) if l == 0]
    tpr = sum(pos) / len(pos) if pos else 0.0
    tnr = sum(1 - p for p in neg) / len(neg) if neg else 0.0
    return (tpr + tnr) / 2


def ece(labels, probs, bins: int = 10) -> float:
    n = len(labels)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(conf - acc)
    return total


def gap_closed(ba: float, baseline_ba: float) -> float:
    return (ba - baseline_ba) / (1 - baseline_ba) if baseline_ba < 1 else 0.0


def bootstrap_by_library(rows, stat, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """rows carry 'library'; stat(rows) -> float. Resamples whole libraries, since examples within one are not
    independent. Returns the (alpha/2, 1 - alpha/2) percentile interval."""
    by = defaultdict(list)
    for r in rows:
        by[r["library"]].append(r)
    libs = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [r for lib in (rng.choice(libs) for _ in libs) for r in by[lib]]
        vals.append(stat(sample))
    vals.sort()
    return vals[int(alpha / 2 * n)], vals[int((1 - alpha / 2) * n) - 1]


def per_kind(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["attempt_kind"]].append(r["pred"])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())}
