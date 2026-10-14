"""Choose tasks by rule (spec §5.1), spread across categories."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from .index import load_attempts, task_counts
from .util import ROOT, TB_ROOT, read_json, write_json

RULES = {"min_each": 10, "max_verifier_timeout": 900}


def reasons_to_reject(task_dir: Path, counts: dict, excluded: dict, *, min_each: int = 10,
                      max_verifier_timeout: int = 900) -> list[str]:
    task_dir = Path(task_dir)
    t = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    ver, env = t.get("verifier") or {}, t.get("environment") or {}
    why = []
    if counts.get("pass", 0) < min_each or counts.get("fail", 0) < min_each:
        why.append(f"needs >= {min_each} pass and fail attempts, has {counts.get('pass', 0)}/{counts.get('fail', 0)}")
    if (env.get("gpus") or 0) > 0 or ((ver.get("environment") or {}).get("gpus") or 0) > 0:
        why.append("needs a GPU")
    if ver.get("environment_mode") != "separate":
        why.append("verifier is not separate")
    if any(not isinstance(a, str) for a in t.get("artifacts", [])) or \
            any(c.get("service", "main") != "main" for c in ver.get("collect", [])):
        why.append("artifacts off the main container")
    if (task_dir / "environment" / "docker-compose.yaml").exists():
        why.append("multi-container environment")
    if float(ver.get("timeout_sec") or 0) > max_verifier_timeout:
        why.append(f"verifier timeout {ver.get('timeout_sec')} s > {max_verifier_timeout} s")
    if task_dir.name in excluded:
        why.append(f"community-reported test problem: {excluded[task_dir.name]}")
    return why


def select(tb_root: Path, counts: dict, excluded: dict, n: int = 5) -> tuple[list[str], dict[str, list[str]]]:
    ok, rejected = {}, {}
    for d in sorted(p for p in Path(tb_root).iterdir() if (p / "task.toml").is_file()):
        why = reasons_to_reject(d, counts.get(d.name, {}), excluded, **RULES)
        if why:
            rejected[d.name] = why
            continue
        meta = tomllib.loads((d / "task.toml").read_text(encoding="utf-8")).get("metadata") or {}
        c = counts[d.name]
        ok[d.name] = (meta.get("category") or "Other", -min(c["pass"], c["fail"]),
                      float(meta.get("expert_time_estimate_hours") or 99), d.name)
    by_cat: dict[str, list[str]] = {}
    for name, key in sorted(ok.items(), key=lambda kv: kv[1]):
        by_cat.setdefault(key[0], []).append(name)
    # Categories in order of their strongest pool (most attempts on the smaller side), then by name.
    cat_order = sorted(by_cat, key=lambda c: (ok[by_cat[c][0]][1], c))
    chosen = []
    while len(chosen) < n and any(by_cat.values()):
        for cat in cat_order:
            if by_cat[cat] and len(chosen) < n:
                chosen.append(by_cat[cat].pop(0))
    return chosen, rejected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    excluded = read_json(ROOT / "configs" / "excluded_tasks.json")
    chosen, rejected = select(TB_ROOT, task_counts(load_attempts()), excluded, a.n)
    write_json(ROOT / "configs" / "m1.json", {"tasks": chosen, "rules": RULES, "rejected": rejected,
                                              "excluded": excluded})
    print("chosen:", chosen)
