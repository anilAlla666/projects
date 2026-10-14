"""The real-attempts test set: agent patches from nebius/SWE-agent-trajectories (CC-BY-4.0) on SWE-rebench instances
(prebuilt images, CC-BY-4.0), every label re-derived by running the instance's oracle tests ourselves.

    python scripts/build_real.py select RAW OUT --instances N --per-label K
    python scripts/build_real.py run OUT [--workers W]
    python scripts/build_real.py assemble OUT TOKENIZER

Selection is fixed before any patch is looked at: permissive licence, pytest log parser, 1 to 10 failing-to-passing
tests and 10 to 1,500 passing-to-passing tests, at least one resolved and one unresolved non-empty patch, one
instance per repository, and no repository whose name is among the 15,000 PyPI candidates we screened (so no
real-attempts code was in pretraining or judge training) or among the SWE-bench Lite repositories.
Per instance, the gold patch must pass every oracle test and the unpatched code must fail a failing-to-passing
test, or the instance is dropped. The judge sees exactly the training format: the named failing test, the
function the reference fix changes (never the fix), the attempt's diff, and the evidence from visible tests."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.data.example import bucket, pointer_for  # noqa: E402
from vm.harness.evidence import diff_shape, split_untrusted, touches_tests, unified_diff, visible_lines  # noqa: E402
from vm.libs.candidates import EXCLUDED, normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["sudo", "-n", "docker"]
_PERMISSIVE = re.compile(r"\b(mit|apache|bsd|isc)\b", re.I)


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def _list(v):
    return json.loads(v) if isinstance(v, str) else list(v or [])


def _first_changed(patch: str):
    """(path, old line number) of the first changed line in the first non-test Python file of a patch."""
    path, old = None, None
    for ln in patch.splitlines():
        if ln.startswith("--- a/"):
            path = ln[6:].split("\t")[0].strip()
            path = path if path.endswith(".py") and not touches_tests({path: ""}, set()) else None
        elif path and ln.startswith("@@"):
            old = int(re.match(r"@@ -(\d+)", ln).group(1))
        elif path and old is not None:
            if ln.startswith("-") or ln.startswith("+"):
                return path, max(1, old if ln.startswith("-") else old - 1)
            if ln.startswith(" "):
                old += 1
    return None


def select(raw: Path, out: Path, n: int, per_label: int, per_wrong: int | None = None) -> None:
    import pyarrow.parquet as pq
    cols = ["instance_id", "repo", "docker_image", "license_name", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config",
            "test_patch", "patch", "base_commit", "problem_statement"]
    reb = {r["instance_id"]: r for f in sorted(raw.glob("*SWE-rebench*.parquet")) for r in pq.read_table(f, columns=cols).to_pylist()
           if r["docker_image"]}
    screened = {json.loads(l)["name"] for l in (ROOT / "data" / "candidates.jsonl").read_text().splitlines()}
    trajs = {}
    for f in sorted(raw.glob("traj_*.parquet")):
        for t in pq.read_table(f, columns=["instance_id", "model_name", "target", "generated_patch"]).to_pylist():
            if t["instance_id"] in reb and (t["generated_patch"] or "").strip():
                trajs.setdefault(t["instance_id"], []).append(t)
    picked, repos, reasons = [], set(), {}

    def drop(why):
        reasons[why] = reasons.get(why, 0) + 1

    for iid in sorted(trajs, key=lambda i: _h(f"vm-real:{i}")):
        r = reb[iid]
        cfg = r["install_config"] or {}
        f2p, p2p = _list(r["FAIL_TO_PASS"]), _list(r["PASS_TO_PASS"])
        name = normalise(r["repo"].split("/")[-1])
        if r["repo"] in repos:
            drop("repository already picked"); continue
        if name in screened or name in EXCLUDED:
            drop("repository among screened PyPI packages or SWE-bench Lite"); continue
        if not _PERMISSIVE.search(r["license_name"] or ""):
            drop("licence not permissive"); continue
        if cfg.get("log_parser") != "parse_log_pytest" or not str(cfg.get("test_cmd", "")).startswith("pytest"):
            drop("not a plain pytest instance"); continue
        if not (1 <= len(f2p) <= 10 and 10 <= len(p2p) <= 1500):
            drop("test counts outside 1-10 / 10-1500"); continue
        if not _first_changed(r["patch"]):
            drop("reference fix changes no Python function"); continue
        seen, good, bad = set(), [], []
        for t in sorted(trajs[iid], key=lambda t: _h(f"vm-real-attempt:{t['generated_patch']}")):
            if t["generated_patch"] in seen:
                continue
            seen.add(t["generated_patch"])
            (good if t["target"] else bad).append(t)
        if not good or not bad:
            drop("no resolved and unresolved pair"); continue
        repos.add(r["repo"])
        picked.append({**{k: r[k] for k in ("instance_id", "repo", "docker_image", "license_name", "test_patch", "patch", "base_commit")},
                       "f2p": f2p, "p2p": p2p,
                       "attempts": [{"model": t["model_name"], "target": bool(t["target"]), "patch": t["generated_patch"]}
                                    for t in good[:per_label] + bad[:per_wrong or per_label]]})
        if len(picked) >= n:
            break
    out.mkdir(parents=True, exist_ok=True)
    (out / "instances.json").write_text(json.dumps(picked))
    print(json.dumps({"instances": len(picked), "attempts": sum(len(p["attempts"]) for p in picked), "dropped": reasons}))


def _visible(inst: dict, setting: str = "strict") -> tuple[str, list, list]:
    """strict: the named failing test and a fixed half of the existing tests, the training rule. realistic: the
    named failing test and every existing test, as a grader running a repository's own suite would have; only the
    reference fix's other new tests stay hidden."""
    named = sorted(inst["f2p"], key=lambda t: _h(f"vm-named:{t}"))[0]
    if setting == "realistic":
        vis = sorted({named} | set(inst["p2p"]))
        return named, vis, sorted(set(inst["f2p"]) - set(vis))
    vis = sorted({named} | {t for t in inst["p2p"] if _h(f"vm-visible:{inst['repo']}:{t}") % 2 == 0})
    oracle_only = sorted((set(inst["f2p"]) | set(inst["p2p"])) - set(vis))
    return named, vis, oracle_only


def _run_one(inst: dict, key: str, patch: str, base: Path, timeout: int) -> None:
    d = base / inst["instance_id"] / key
    if (d / "out" / "result.json").is_file():
        return
    if d.exists():
        subprocess.run(DOCKER[:2] + ["rm", "-rf", str(d)], capture_output=True)
    d.mkdir(parents=True)
    named, _, _ = _visible(inst)
    path, line = _first_changed(inst["patch"])
    (d / "spec.json").write_text(json.dumps({"mode": key if key in ("base", "gold") else "attempt", "base_commit": inst["base_commit"],
                                             "named": named, "gold": {"path": path, "line": line}, "timeout": timeout}))
    (d / "attempt.patch").write_text(patch if patch.endswith("\n") else patch + "\n")
    (d / "test.patch").write_text(inst["test_patch"])
    (d / "ids.json").write_text(json.dumps(sorted(set(inst["f2p"]) | set(inst["p2p"]))))
    shutil.copy(ROOT / "vm" / "real" / "in_container.py", d / "in_container.py")
    shutil.copy(ROOT / "vm" / "sandbox" / "vm_results_plugin.py", d / "vm_results_plugin.py")
    (d / "out").mkdir()
    d.chmod(0o777)
    (d / "out").chmod(0o777)
    cmd = DOCKER + ["run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "2", "-v", f"{d}:/work", "-w", "/testbed",
                    inst["docker_image"], "bash", "-c",
                    "P=/opt/conda/envs/testbed/bin/python; [ -x $P ] || P=$(command -v python); exec $P /work/in_container.py"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 300)
        if not (d / "out" / "result.json").is_file():
            (d / "out" / "error.txt").write_text((r.stdout + r.stderr)[-3000:])
    except subprocess.TimeoutExpired:
        (d / "out" / "error.txt").write_text("container timed out")


def _run_instance(inst: dict, base: Path, timeout: int) -> str:
    p = subprocess.run(DOCKER + ["pull", "-q", inst["docker_image"]], capture_output=True, text=True)
    if p.returncode:
        return f"{inst['instance_id']}: pull failed {p.stderr[-200:]}"
    _run_one(inst, "gold", inst["patch"], base, timeout)
    _run_one(inst, "base", "", base, timeout)
    for i, a in enumerate(inst["attempts"]):
        _run_one(inst, f"a{i}", a["patch"], base, timeout)
    return f"{inst['instance_id']}: done"


def run(out: Path, workers: int, timeout: int = 600) -> None:
    insts = json.loads((out / "instances.json").read_text())
    with cf.ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(lambda i: _run_instance(i, out / "runs", timeout), insts):
            print(msg, flush=True)


def _strip_doctests(text: str) -> str:
    from vm.tasks.task import _strip_doctests as s
    return s(text)


def assemble(out: Path, tokenizer: Path, setting: str = "strict") -> None:
    from vm.corpus.tok import load
    from vm.data.example import serialise
    tok = load(tokenizer)
    insts = json.loads((out / "instances.json").read_text())
    examples, report = [], {"instances": len(insts), "instances_dropped": {}, "attempts_dropped": {}, "agree": 0, "disagree": 0,
                            "disagreements": []}

    def res(iid, key):
        p = out / "runs" / iid / key / "out" / "result.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def drop(kind, why):
        report[kind][why] = report[kind].get(why, 0) + 1

    for inst in insts:
        iid = inst["instance_id"]
        named, vis, oracle_only = _visible(inst, setting)
        oracle = sorted(set(inst["f2p"]) | set(inst["p2p"]))
        gold, base = res(iid, "gold"), res(iid, "base")
        if not gold or not base or "outcomes" not in gold or "outcomes" not in base:
            drop("instances_dropped", "control did not run"); continue
        if any(gold["outcomes"].get(t) != "passed" for t in oracle):
            drop("instances_dropped", "reference fix fails an oracle test in our run"); continue
        if all(base["outcomes"].get(t) == "passed" for t in inst["f2p"]):
            drop("instances_dropped", "unpatched code already passes"); continue
        gf = gold.get("gold_function")
        if not gf or not gold.get("named_source"):
            drop("instances_dropped", "function or named test not found"); continue
        fn_lines = _strip_doctests("\n".join(gf["lines"])).splitlines()
        what = f"The code it exercises ({gf['name']}):" if gf.get("module_level") else f"The function `{gf['name']}` it exercises:"
        text = f"In `{gf['path']}`, the test `{named}` fails.\n\nThe test:\n{gold['named_source']}\n\n{what}\n"
        test_paths = {t.split("::")[0] for t in oracle}
        for i, a in enumerate(inst["attempts"]):
            r = res(iid, f"a{i}")
            if r is None or "apply_failed" in r or "test_patch_failed" in r or "outcomes" not in r:
                drop("attempts_dropped", "patch did not apply or did not run" if r else "no result"); continue
            failing = [t for t in oracle if r["outcomes"].get(t) != "passed"]
            lab = int(not failing and not r.get("timed_out"))
            if lab == int(a["target"]):
                report["agree"] += 1
            else:
                report["disagree"] += 1
                report["disagreements"].append({"instance": iid, "attempt": i, "dataset": a["target"], "ours": lab, "failing": failing[:5]})
            lines = visible_lines(r["outcomes"], vis) if not r.get("timed_out") else ["visible tests: timed out"]
            edited = sorted(p for p in r["files"] if touches_tests({p: ""}, test_paths))
            lines.append("test files edited: " + (", ".join(edited) if edited else "none"))
            lines.append("fuzz: not applicable")
            diff = unified_diff(r["old"], r["new"])
            lines.append(diff_shape(diff))
            evidence = [{"id": k, "text": t} for k, t in enumerate(lines)]
            shown, moved = split_untrusted(diff)
            n_hidden = sum(t in set(oracle_only) for t in failing)
            ex = {"example_id": f"real:{iid}/a{i}/{hashlib.sha256(shown.encode()).hexdigest()[:10]}", "library": inst["repo"],
                  "split": "real", "operator": "real", "function": gf["name"], "attempt_kind": "real",
                  "task_text": text, "function_lines": fn_lines,
                  "bug_line": max(0, min(len(fn_lines) - 1, gf["changed_line"] - gf["start"])),
                  "diff": shown, "evidence": evidence, "untrusted": list(dict.fromkeys(moved)), "label": lab,
                  "hidden_bucket": bucket(n_hidden), "n_hidden_failing": n_hidden, "pointer": pointer_for(lab, evidence),
                  "model": a["model"], "dataset_target": a["target"], "attempt_files": r["new"]}
            examples.append(ex)
    suffix = "" if setting == "strict" else f"_{setting}"
    (out / f"examples{suffix}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    ser, too_long = [], 0
    for e in examples:
        try:
            s = serialise(e, tok)
        except ValueError:
            too_long += 1
            continue
        ser.append({**s, "example_id": e["example_id"], "attempt_kind": "real", "operator": "real", "library": e["library"]})
    (out / f"serialised_real{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ser))
    report.update({"examples": len(examples), "serialised": len(ser), "too_long": too_long,
                   "labels": {"correct": sum(e["label"] for e in examples), "wrong": sum(1 - e["label"] for e in examples)},
                   "repositories": len({e["library"] for e in examples})})
    report["disagreements"] = report["disagreements"][:30]
    (out / f"real_report{suffix}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--per-label", type=int, default=3)
    ap.add_argument("--per-wrong", type=int)
    ap.add_argument("--setting", default="strict", choices=["strict", "realistic"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "select":
        select(a.paths[0], a.paths[1], a.instances, a.per_label, a.per_wrong)
    elif a.stage == "run":
        run(a.paths[0], a.workers)
    elif a.stage == "assemble":
        assemble(a.paths[0], a.paths[1], a.setting)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
