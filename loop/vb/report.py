"""summary.md from scores.json."""
from __future__ import annotations


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ba(m: dict) -> str:
    ci = m.get("ba_ci")
    return _p(m.get("ba")) + (f" [{_p(ci[0])}, {_p(ci[1])}]" if ci else "")


def render(scores: dict) -> str:
    lines = [f"# {scores['name']}", ""]
    if scores["baseline_problems"]:
        lines += ["**Baselines failed; these results are NOT VALID:**", ""] + [f"- {p}" for p in scores["baseline_problems"]]
    else:
        lines += ["Baselines: all hold (original tests BA 100; always pass, always fail and network probe as required "
                  "on every task)."]
    writers = scores.get("writers") or {}
    if writers:
        harnesses = sorted({w["harness"] for w in writers.values()})
        models = sorted({w["model_id"] for w in writers.values()}, reverse=True)
        cell = {(w["model_id"], w["harness"]): scores["macro"].get(name) for name, w in writers.items()}
        lines += ["", "## Model × harness (BA with 95% bootstrap interval, macro over tasks and attempts)", "",
                  "| model | " + " | ".join(harnesses) + " |", "|---|" + "---:|" * len(harnesses)]
        lines += [f"| {mo} | " + " | ".join(_ba(cell[(mo, h)]) if cell.get((mo, h)) else "n/a" for h in harnesses) + " |"
                  for mo in models]
        lines += ["", "A model or harness effect is stated only where two intervals do not overlap."]
    lines += ["", "## Writers (macro over tasks and attempts)", "", "| writer | task-attempts | TPR | TNR | BA | flaky | cost |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for w, m in sorted(scores["macro"].items()):
        cost = scores.get("writer_cost_usd", {}).get(w)
        lines.append(f"| {w} | {m['n_tasks']} | {_p(m['tpr'])} | {_p(m['tnr'])} | {_ba(m)} | {_p(m['flakiness'])} | "
                     f"{'n/a' if cost is None else f'${cost:.2f}'} |")
    missing = sorted(f"{t}: {n}" for t, per in scores["per_task"].items() for n, sc in per.items() if sc.get("no_verifier"))
    if missing:
        lines += ["", "## Writer attempts that produced no verifier (scored as rejecting everything)", ""]
        lines += [f"- {m}" for m in missing]
    if scores.get("leaks"):
        lines += ["", "## Verifiers excluded: copied or fetched the official tests", ""]
        lines += [f"- `{name}`: " + "; ".join(found) for name, found in sorted(scores["leaks"].items())]
    lines += ["", "## By task", "", "| task | verifier | correct/wrong | TPR | TNR | BA | TNR near miss | TNR real fail | TNR nop | flaky |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, per in sorted(scores["per_task"].items()):
        for name, s in sorted(per.items()):
            by = s.get("tnr_by_source", {})
            lines.append(f"| {task} | {name} | {s['n_correct']}/{s['n_wrong']} | {_p(s['tpr'])} | {_p(s['tnr'])} | {_p(s['ba'])} | "
                         f"{_p(by.get('near_miss'))} | {_p(by.get('real_fail'))} | {_p(by.get('nop'))} | {_p(s['flakiness'])} |")
    return "\n".join(lines) + "\n"
