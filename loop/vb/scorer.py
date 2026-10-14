"""Score a verifier's verdicts against known labels. An error counts as fail and is reported as flakiness."""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

LABELS = ("pass", "fail")
VERDICTS = ("pass", "fail", "error")


@dataclass(frozen=True)
class Verdict:
    entry_id: str
    source: str
    label: str      # truth: pass | fail
    verdict: str    # the verifier: pass | fail | error


def score_task(verdicts: list[Verdict]) -> dict:
    for v in verdicts:
        if v.label not in LABELS:
            raise ValueError(f"{v.entry_id}: label must be pass/fail, got {v.label!r}")
        if v.verdict not in VERDICTS:
            raise ValueError(f"{v.entry_id}: verdict must be pass/fail/error, got {v.verdict!r}")
    correct = [v for v in verdicts if v.label == "pass"]
    wrong = [v for v in verdicts if v.label == "fail"]
    tpr = sum(v.verdict == "pass" for v in correct) / len(correct) if correct else None
    tnr = sum(v.verdict != "pass" for v in wrong) / len(wrong) if wrong else None
    by_source: dict[str, list[Verdict]] = {}
    for v in wrong:
        by_source.setdefault(v.source, []).append(v)
    return {
        "n_correct": len(correct), "n_wrong": len(wrong), "tpr": tpr, "tnr": tnr,
        "ba": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "tnr_by_source": {s: sum(v.verdict != "pass" for v in vs) / len(vs) for s, vs in sorted(by_source.items())},
        "flakiness": sum(v.verdict == "error" for v in verdicts) / len(verdicts) if verdicts else 0.0,
    }


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap of the mean. With 5 tasks x 2 attempts per writer the interval is wide on purpose:
    a model or harness effect is called only when two writers' intervals do not overlap."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    lo, hi = means[int((1 - level) / 2 * n)], means[int((1 + level) / 2 * n) - 1]
    return (lo, hi)


def macro(scores: list[dict]) -> dict:
    def avg(k):
        vals = [s[k] for s in scores if s.get(k) is not None]
        return mean(vals) if vals else None
    return {"n_tasks": len(scores), "tpr": avg("tpr"), "tnr": avg("tnr"), "ba": avg("ba"), "flakiness": avg("flakiness"),
            "ba_ci": bootstrap_ci([s["ba"] for s in scores if s.get("ba") is not None])}


def check_baselines(always_pass: dict, always_fail: dict, original: dict, network_probe: dict | None = None) -> list[str]:
    """Problems with the baselines; an empty list means the pool, the swap, the offline rule and the scorer look right."""
    problems = []
    probes = [("always_pass", always_pass, {"tpr": 1.0, "tnr": 0.0, "ba": 0.5}),
              ("always_fail", always_fail, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5})]
    if network_probe is not None:
        probes.append(("network_probe", network_probe, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5}))
    for name, s, want in probes:
        for k, v in want.items():
            if s.get(k) is None or abs(s[k] - v) > 1e-9:
                problems.append(f"{name}: {k} = {s.get(k)} (expected {v})")
    if original.get("ba") is None or abs(original["ba"] - 1.0) > 1e-9:
        problems.append(f"original tests: ba = {original.get('ba')} (expected 1.0 on the kept pool)")
    return problems
