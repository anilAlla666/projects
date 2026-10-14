"""Build the judging data on the instance, in stages, each resumable (finished outputs are skipped).

    python scripts/build_judge_data.py tasks ROOT                 # tasks for every qualified library
    python scripts/build_judge_data.py model ROOT MODEL [N] [TP]  # open-model attempts (GPU, vLLM, TP GPUs)
    python scripts/build_judge_data.py examples ROOT              # attempts, evidence, labels, per task
    python scripts/build_judge_data.py hard ROOT                  # more wrong attempts that pass every visible test
    python scripts/build_judge_data.py checks ROOT                # the four pipeline checks
    python scripts/build_judge_data.py serialise ROOT TOKENIZER   # token sequences per split

ROOT is the qualification root (it holds qualified.jsonl and one folder per library).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.data.checks import check_fake_output_fails, check_labels, check_no_oracle_leak, check_split  # noqa: E402
from vm.data.example import label, make_example, serialise  # noqa: E402
from vm.harness.evidence import (_changed_functions, _literals, _run_fuzz, diff_shape, fuzz_lines,  # noqa: E402
                                 touches_tests, unified_diff, visible_lines)
from vm.data.example import bucket, pointer_for  # noqa: E402
from vm.sandbox.runner import run_pytest  # noqa: E402
from vm.tasks.attempts import Attempt, rule_attempts  # noqa: E402
from vm.tasks.mutate import mutants, pick_heldout  # noqa: E402
from vm.tasks.task import _test_source, make_tasks  # noqa: E402

WORKERS = int(__import__("os").environ.get("VM_WORKERS", "80"))


def qualified(root: Path) -> list:
    return [r for r in (json.loads(l) for l in (root / "qualified.jsonl").read_text().splitlines()) if r["ok"]]


def all_tasks(root: Path) -> list:
    out = []
    for lib in qualified(root):
        p = root / lib["name"] / "tasks.json"
        if p.is_file():
            out += json.loads(p.read_text())
    return out


def _tasks_one(lib, root):
    p = root / lib["name"] / "tasks.json"
    if not p.is_file():
        try:
            p.write_text(json.dumps(make_tasks(lib, root)))
        except Exception as e:  # noqa: BLE001  one library must not stop the stage
            print("tasks failed", lib["name"], repr(e)[:200], flush=True)
            return lib["name"], 0
    return lib["name"], len(json.loads(p.read_text()))


def stage_tasks(root: Path) -> None:
    # processes, not threads: finding candidate bugs is pure-Python parsing that the GIL serialises
    libs = qualified(root)
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        counts = dict(ex.map(_tasks_one, libs, [root] * len(libs)))
    print(json.dumps({"libraries": len(counts), "tasks": sum(counts.values())}))


def stage_model(root: Path, model: str, n: int, tp: int = 1) -> None:
    from vm.tasks.model_attempts import generate
    out = root / "model_attempts.jsonl"
    if out.is_file():
        print("exists", out)
        return
    tasks = all_tasks(root)
    rows = generate(tasks, model, n=n, tp=tp)
    with open(out, "w") as f:
        for task_id, a in rows:
            f.write(json.dumps({"task_id": task_id, "kind": a.kind, "files": a.files, "untrusted": a.untrusted}) + "\n")
    print(json.dumps({"tasks": len(tasks), "attempts": len(rows)}))


def _visible_only(task: dict, root: Path, held: dict, limit: int = 2, tries: int = 6) -> list:
    """Other single-site changes to the broken function that make the named test pass; labelled later."""
    src, venv = root / task["library"] / "src", root / task["library"] / "venv"
    cands = [m for m in mutants(task["broken_source"], task["path"])
             if m.function == task["function"] and m.new_source != task["original_source"]
             and (task["split"] == "test" or m.operator != held["operator"])][:tries]
    out = []
    for m in cands:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "src"
            shutil.copytree(src, work)
            (work / task["path"]).write_text(m.new_source)
            r = run_pytest(work, venv, node_ids=[task["named_test"]], timeout=60)
        if r.outcomes.get(task["named_test"]) == "passed":
            out.append(Attempt("visible_only", {task["path"]: m.new_source}))
            if len(out) == limit:
                break
    return out


def _module_of(path: str) -> str:
    return path[:-3].replace("/", ".").removeprefix("src.")


def _evaluate(task: dict, files: dict, root: Path, fuzz_broken: dict):
    """One run of the whole suite (original test files) on the patched code gives both the evidence's visible
    outcomes and the label; one fuzz run on the patched code is compared with the task's cached broken run."""
    src, venv = root / task["library"] / "src", root / task["library"] / "venv"
    tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
    test_paths = {t.split("::")[0] for t in tests}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "src"
        shutil.copytree(src, work)
        (work / task["path"]).write_text(task["broken_source"])
        for p, text in files.items():
            if touches_tests({p: ""}, test_paths):
                continue
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(text)
        r = run_pytest(work, venv, node_ids=tests, timeout=150)
        fns = _changed_functions(task["broken_source"], files.get(task["path"], task["broken_source"]))
        patched_fuzz = _run_fuzz(work, venv, _module_of(task["path"]), fns, _literals([task["named_test_source"]])) if fns else {}
    lines = visible_lines(r.outcomes, task["visible"]) if r.outcomes else ["visible tests: timed out"]
    edited = sorted(p for p in files if touches_tests({p: ""}, test_paths))
    lines.append("test files edited: " + (", ".join(edited) if edited else "none"))
    fl = fuzz_lines({k: v for k, v in fuzz_broken.items() if k in fns}, patched_fuzz) if fns else []
    lines += fl or ["fuzz: not applicable"]
    old = {task["path"]: task["broken_source"]}
    for p in files:
        if p != task["path"]:
            old[p] = (src / p).read_text() if (src / p).is_file() else ""
    lines.append(diff_shape(unified_diff(old, files)))
    failing = [t for t in tests if r.outcomes.get(t) != "passed"]
    n_hidden = sum(t in set(task["oracle_only"]) for t in failing)
    lab = {"label": int(not failing and not r.timed_out), "n_hidden_failing": n_hidden, "hidden_bucket": bucket(n_hidden),
           "timed_out": r.timed_out}
    return [{"id": i, "text": t} for i, t in enumerate(lines)], lab


def _task_examples(task: dict, root: Path, model_rows: list, held: dict) -> list:
    out_path = root / task["library"] / "examples" / f"{task['task_id'].split(':')[1]}.jsonl"
    if out_path.is_file():
        return [json.loads(l) for l in out_path.read_text().splitlines()]
    src = root / task["library"] / "src"
    rng = random.Random(f"vm-attempts:{task['task_id']}")
    others = [m for m in mutants(task["broken_source"], task["path"]) if m.function != task["function"]
              and (task["split"] == "test" or m.operator != held["operator"])]
    others.sort(key=lambda m: hashlib.sha256(f"{task['task_id']}:{m.line}:{m.operator}".encode()).hexdigest())
    test_file = task["named_test"].split("::")[0]
    attempts = rule_attempts(task, others, rng, test_file_text=(src / test_file).read_text())
    attempts += _visible_only(task, root, held)
    attempts += [Attempt(r["kind"], r["files"], r["untrusted"]) for r in model_rows]
    if task["split"] != "test":
        attempts = [a for a in attempts if a.kind != held["wrong_kind"]]
    src_root, venv = root / task["library"] / "src", root / task["library"] / "venv"
    fns = sorted({f for a in attempts for f in _changed_functions(task["broken_source"], a.files.get(task["path"], task["broken_source"]))})
    fuzz_broken = {}
    if fns:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "src"
            shutil.copytree(src_root, work)
            (work / task["path"]).write_text(task["broken_source"])
            fuzz_broken = _run_fuzz(work, venv, _module_of(task["path"]), fns, _literals([task["named_test_source"]]))
    examples = []
    for a in attempts:
        try:
            evidence, lab = _evaluate(task, a.files, root, fuzz_broken)
        except Exception as e:  # noqa: BLE001  one broken attempt must not lose the task
            print("attempt failed", task["task_id"], a.kind, repr(e)[:200], flush=True)
            continue
        examples.append(make_example(task, a, evidence, lab, root))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(e) + "\n" for e in examples))
    return examples


def _visible_pass(evidence: list) -> bool:
    import re as _re
    m = _re.match(r"visible tests: (\d+) of (\d+) pass", evidence[0]["text"])
    return bool(m and m.group(1) == m.group(2))


def _task_hard(task: dict, root: Path, held: dict, partial_tries: int = 5, alt_tries: int = 12, skip: int = 0, suffix: str = ".hard.jsonl") -> list:
    """Hard negatives (and the occasional correct alternative) that pass every visible test: the real fix plus one
    more single-site change in the same function ("partial"), and more single-site changes to the broken function
    that make the named test pass ("visible_only"). Written beside the task's examples as <n>.hard.jsonl."""
    base_path = root / task["library"] / "examples" / f"{task['task_id'].split(':')[1]}.jsonl"
    out_path = base_path.with_suffix(suffix)
    if out_path.is_file():
        return [json.loads(l) for l in out_path.read_text().splitlines()]
    if not base_path.is_file():
        return []
    seen = {json.dumps(e["attempt_files"], sort_keys=True) for p in base_path.parent.glob(base_path.stem + ".*jsonl")
            for e in map(json.loads, p.read_text().splitlines())}
    partial = [m for m in mutants(task["original_source"], task["path"])
               if m.function == task["function"] and m.new_source not in (task["broken_source"], task["original_source"])
               and (task["split"] == "test" or m.operator != held["operator"])]
    partial.sort(key=lambda m: hashlib.sha256(f"vm-partial:{task['task_id']}:{m.line}:{m.operator}".encode()).hexdigest())
    attempts = [Attempt("partial", {task["path"]: m.new_source}) for m in partial[skip : skip + partial_tries]]
    attempts += _visible_only(task, root, held, limit=3, tries=alt_tries) if alt_tries else []
    attempts = [a for a in attempts if json.dumps(a.files, sort_keys=True) not in seen]
    if task["split"] != "test":
        attempts = [a for a in attempts if a.kind != held["wrong_kind"]]
    src_root, venv = root / task["library"] / "src", root / task["library"] / "venv"
    fns = sorted({f for a in attempts for f in _changed_functions(task["broken_source"], a.files.get(task["path"], task["broken_source"]))})
    fuzz_broken = {}
    if fns:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "src"
            shutil.copytree(src_root, work)
            (work / task["path"]).write_text(task["broken_source"])
            fuzz_broken = _run_fuzz(work, venv, _module_of(task["path"]), fns, _literals([task["named_test_source"]]))
    examples, visible_fail_kept = [], 0
    for a in attempts:
        try:
            evidence, lab = _evaluate(task, a.files, root, fuzz_broken)
        except Exception as e:  # noqa: BLE001
            print("attempt failed", task["task_id"], a.kind, repr(e)[:200], flush=True)
            continue
        if a.kind == "partial" and not _visible_pass(evidence):
            if visible_fail_kept:  # easy negatives are plentiful already; keep one so the kind is not always visible-pass
                continue
            visible_fail_kept += 1
        examples.append(make_example(task, a, evidence, lab, root))
    out_path.write_text("".join(json.dumps(e) + "\n" for e in examples))
    return examples


def stage_hard(root: Path, second: bool = False) -> None:
    """second: a further pass over the next mutations of each task's function (partial only), written as .hard2.jsonl."""
    held = pick_heldout()
    tasks = all_tasks(root)
    kw = {"partial_tries": 8, "alt_tries": 0, "skip": 5, "suffix": ".hard2.jsonl"} if second else {}
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_task_hard, t, root, held, **kw) for t in tasks]
        n = Counter()
        for i, f in enumerate(cf.as_completed(futs)):
            for e in f.result():
                n[(e["split"], e["attempt_kind"], e["label"], _visible_pass(e["evidence"]))] += 1
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} tasks", flush=True)
    print(json.dumps({f"{s}/{k}/{l}/visible_pass={v}": c for (s, k, l, v), c in sorted(n.items())}))


def _task_fix_plus_test(task: dict, root: Path) -> list:
    """The real fix plus a new test appended to the named test's file: correct by construction, and its evidence
    is exact without a run, because the original tests (all the oracle uses) are untouched, the new test is in no
    visible list, and the code change is the revert's. Only top-level named tests are copied, so the appended
    function cannot break the file's indentation."""
    base_path = root / task["library"] / "examples" / f"{task['task_id'].split(':')[1]}.jsonl"
    out_path = base_path.with_suffix(".fpt.jsonl")
    if out_path.is_file() or not base_path.is_file():
        return []
    revert = next((e for e in map(json.loads, base_path.read_text().splitlines()) if e["attempt_kind"] == "revert" and e["label"] == 1), None)
    named_src = task["named_test_source"]
    if revert is None or not named_src.startswith(("def test", "async def test")):
        out_path.write_text("")
        return []
    test_file = task["named_test"].split("::")[0]
    fn = task["named_test"].split("::")[-1].split("[")[0]
    text = (root / task["library"] / "src" / test_file).read_text()
    added = named_src.replace(f"def {fn}(", f"def {fn}_added(", 1)
    files = {task["path"]: task["original_source"], test_file: text.rstrip("\n") + "\n\n\n" + added + "\n"}
    old = {task["path"]: task["broken_source"], test_file: text}
    lines = [x["text"] for x in revert["evidence"]]
    lines = [f"test files edited: {test_file}" if t.startswith("test files edited:") else t for t in lines]
    lines = [diff_shape(unified_diff(old, files)) if t.startswith("diff:") else t for t in lines]
    evidence = [{"id": i, "text": t} for i, t in enumerate(lines)]
    lab = {"label": 1, "n_hidden_failing": 0, "hidden_bucket": 0}
    ex = make_example(task, Attempt("fix_plus_test", files), evidence, lab, root)
    out_path.write_text(json.dumps(ex) + "\n")
    return [ex]


def stage_fix_plus_test(root: Path) -> None:
    n = Counter()
    for t in all_tasks(root):
        for e in _task_fix_plus_test(t, root):
            n[e["split"]] += 1
    print(json.dumps(n))


def _task_more_model(task: dict, root: Path, rows: list) -> list:
    """Further open-model attempts for a task (a second sampling seed), evaluated exactly like the first ones and
    written as <n>.ma2.jsonl; attempts identical to any already built for the task are skipped."""
    base_path = root / task["library"] / "examples" / f"{task['task_id'].split(':')[1]}.jsonl"
    out_path = base_path.with_suffix(".ma2.jsonl")
    if out_path.is_file() or not base_path.is_file():
        return []
    seen = {json.dumps(e["attempt_files"], sort_keys=True) for p in base_path.parent.glob(base_path.stem + ".*jsonl")
            for e in map(json.loads, p.read_text().splitlines())}
    attempts, keys = [], set()
    for r in rows:
        k = json.dumps(r["files"], sort_keys=True)
        if k not in seen and k not in keys:
            keys.add(k)
            attempts.append(Attempt(r["kind"], r["files"], r["untrusted"]))
    src_root, venv = root / task["library"] / "src", root / task["library"] / "venv"
    fns = sorted({f for a in attempts for f in _changed_functions(task["broken_source"], a.files.get(task["path"], task["broken_source"]))})
    fuzz_broken = {}
    if fns:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "src"
            shutil.copytree(src_root, work)
            (work / task["path"]).write_text(task["broken_source"])
            fuzz_broken = _run_fuzz(work, venv, _module_of(task["path"]), fns, _literals([task["named_test_source"]]))
    examples = []
    for a in attempts:
        try:
            evidence, lab = _evaluate(task, a.files, root, fuzz_broken)
        except Exception as e:  # noqa: BLE001
            print("attempt failed", task["task_id"], a.kind, repr(e)[:200], flush=True)
            continue
        examples.append(make_example(task, a, evidence, lab, root))
    out_path.write_text("".join(json.dumps(e) + "\n" for e in examples))
    return examples


def stage_more_model(root: Path, attempts_file: str) -> None:
    rows = {}
    for l in (root / attempts_file).read_text().splitlines():
        r = json.loads(l)
        if r["kind"] != "none":
            rows.setdefault(r["task_id"], []).append(r)
    tasks = [t for t in all_tasks(root) if t["task_id"] in rows]
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_task_more_model, t, root, rows[t["task_id"]]) for t in tasks]
        n = Counter()
        for i, f in enumerate(cf.as_completed(futs)):
            for e in f.result():
                n[(e["split"], e["label"], _visible_pass(e["evidence"]))] += 1
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} tasks", flush=True)
    print(json.dumps({f"{s}/{l}/visible_pass={v}": c for (s, l, v), c in sorted(n.items())}))


def _realistic_visible(task: dict) -> tuple[list, set]:
    """The general-purpose setting: a grader can run every test the repository already has, so the only hidden tests
    are the other ones the bug broke (the tests a fix must also make pass, standing in for the new tests a real fix
    would add). Returns (visible, hidden)."""
    hidden = set(task["failing"]) - {task["named_test"]}
    tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
    return [t for t in tests if t not in hidden], hidden


def _run_suite(task: dict, files: dict, root: Path):
    """One run of the whole suite with the original test files on the patched code (test edits are not applied),
    exactly as _evaluate runs it."""
    src, venv = root / task["library"] / "src", root / task["library"] / "venv"
    tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
    test_paths = {t.split("::")[0] for t in tests}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "src"
        shutil.copytree(src, work)
        (work / task["path"]).write_text(task["broken_source"])
        for p, text in files.items():
            if touches_tests({p: ""}, test_paths):
                continue
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(text)
        return run_pytest(work, venv, node_ids=tests, timeout=150), tests


def _realistic_example(e: dict, task: dict, outcomes: dict, timed_out: bool) -> dict:
    vis, hidden = _realistic_visible(task)
    old = [x["text"] for x in e["evidence"]]
    lines = visible_lines(outcomes, vis) if outcomes and not timed_out else ["visible tests: timed out"]
    lines += [t for t in old if t.startswith("test files edited:")]
    lines += [t for t in old if t.startswith("fuzz")]
    lines += [t for t in old if t.startswith("diff:")]
    evidence = [{"id": i, "text": t} for i, t in enumerate(lines)]
    tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
    n_hidden = sum(t in hidden and outcomes.get(t) != "passed" for t in tests)
    return {**e, "evidence": evidence, "hidden_bucket": bucket(n_hidden), "n_hidden_failing": n_hidden,
            "pointer": pointer_for(e["label"], evidence), "setting": "realistic"}


def _task_realistic(task: dict, root: Path) -> dict:
    """Re-derives every example of a task in the realistic setting: one fresh run of the whole suite per attempt
    gives the evidence over all existing tests and re-checks the label; an attempt whose label differs from the
    stored one is flaky and dropped. A fix-plus-test attempt takes its revert's outcomes (its code is the revert's
    and its test edit is never applied). Written to <lib>/examples_realistic/ under the same file names."""
    n = task["task_id"].split(":")[1]
    src_dir, out_dir = root / task["library"] / "examples", root / task["library"] / "examples_realistic"
    stats = {"examples": 0, "flaky": 0}
    files = sorted(src_dir.glob(f"{n}.jsonl")) + sorted(src_dir.glob(f"{n}.*.jsonl"))
    revert_outcomes = None
    for f in files:
        dst = out_dir / f.name
        if dst.is_file():
            continue
        rows, out = [json.loads(l) for l in f.read_text().splitlines()], []
        for e in rows:
            if e["attempt_kind"] == "fix_plus_test":
                if revert_outcomes is None:
                    rv = next((r for r in map(json.loads, (src_dir / f"{n}.jsonl").read_text().splitlines()) if r["attempt_kind"] == "revert"), None)
                    if rv is None:
                        continue
                    r, _ = _run_suite(task, rv["attempt_files"], root)
                    revert_outcomes = (r.outcomes, r.timed_out)
                outcomes, timed_out = revert_outcomes
            else:
                r, tests = _run_suite(task, e["attempt_files"], root)
                outcomes, timed_out = r.outcomes, r.timed_out
                if e["attempt_kind"] == "revert":
                    revert_outcomes = (outcomes, timed_out)
            tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
            lab = int(not timed_out and all(outcomes.get(t) == "passed" for t in tests))
            if lab != e["label"]:
                stats["flaky"] += 1
                continue
            out.append(_realistic_example(e, task, outcomes, timed_out))
        out_dir.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(json.dumps(x) + "\n" for x in out))
        stats["examples"] += len(out)
    return stats


def stage_realistic(root: Path) -> None:
    tasks = all_tasks(root)
    tot = Counter()
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_task_realistic, t, root) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            tot.update(f.result())
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} tasks", dict(tot), flush=True)
    print(json.dumps(dict(tot)))


def stage_examples(root: Path) -> None:
    held = pick_heldout()
    model_rows = {}
    mp = root / "model_attempts.jsonl"
    if mp.is_file():
        for l in mp.read_text().splitlines():
            r = json.loads(l)
            if r["kind"] != "none":  # "none" marks a task whose replies all failed to parse
                model_rows.setdefault(r["task_id"], []).append(r)
    tasks = all_tasks(root)
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_task_examples, t, root, model_rows.get(t["task_id"], []), held) for t in tasks]
        n = Counter()
        for i, f in enumerate(cf.as_completed(futs)):
            for e in f.result():
                n[(e["split"], e["attempt_kind"], e["label"])] += 1
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} tasks", flush=True)
    print(json.dumps({f"{s}/{k}/{l}": c for (s, k, l), c in sorted(n.items())}))


def all_examples(root: Path) -> list:
    from vm.data.load import load_examples
    return load_examples(root)


def doctest_outputs(source: str) -> list:
    lines = source.splitlines()
    return [ln.strip() for prev, ln in zip(lines, lines[1:])
            if ln.strip() and not ln.strip().startswith((">>>", "...")) and prev.strip().startswith((">>>", "..."))]


def _leaks_of(t: dict, group: list, root: Path) -> list:
    src = root / t["library"] / "src"
    oracle_src = [s for s in (_test_source(src, o) for o in t["oracle_only"]) if s != t["named_test_source"]]
    doct = doctest_outputs(t["original_source"])
    return [v for e in group for v in check_no_oracle_leak(e, oracle_src, doct)]


def _lib_leaks(items: list, root: Path) -> list:
    return [v for t, group in items for v in _leaks_of(t, group, root)]


def stage_checks(root: Path) -> None:
    held = pick_heldout()
    exs = all_examples(root)
    tasks = {t["task_id"]: t for t in all_tasks(root)}
    by_task = {}
    for e in exs:
        by_task.setdefault(e["example_id"].rsplit("/", 2)[0], []).append(e)
    # grouped by library so each worker parses a library's test files once
    by_lib = {}
    for tid, group in by_task.items():
        by_lib.setdefault(tasks[tid]["library"], []).append((tasks[tid], group))
    leak = []
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        for found in ex.map(_lib_leaks, by_lib.values(), [root] * len(by_lib)):
            leak += found

    def relabel(e):
        return label(tasks[e["example_id"].rsplit("/", 2)[0]], e["attempt_files"], root)["label"]

    sample = random.Random(0).sample(exs, min(50, len(exs)))
    with cf.ThreadPoolExecutor(25) as ex:
        fresh = dict(zip([e["example_id"] for e in sample], ex.map(relabel, sample)))
    labels = check_labels(sample, lambda e: fresh[e["example_id"]], k=len(sample))
    report = {"examples": len(exs), "n_oracle_leak": len(leak), "oracle_leak": leak[:20],
              "split": check_split(exs, held)[:20], "fake_output": check_fake_output_fails(exs)[:20],
              "labels": labels}
    (root / "checks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if isinstance(v, int) else len(v)) for k, v in report.items()}))


def stage_serialise(root: Path, tokenizer: Path, setting: str = "strict", out_dir: Path | None = None) -> None:
    from vm.corpus.tok import load
    tok = load(tokenizer)
    from vm.data.load import load_examples
    exs, filt = load_examples(root, with_report=True, setting=setting)
    out_dir = out_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"filter": filt}))
    out, too_long = {}, 0
    for e in exs:
        try:
            s = serialise(e, tok)
        except ValueError:  # the task and evidence alone exceed the model's length; dropped and counted
            too_long += 1
            continue
        out.setdefault(e["split"], []).append({**s, "example_id": e["example_id"], "attempt_kind": e["attempt_kind"],
                                                "operator": e["operator"], "library": e["library"]})
    for split, rows in out.items():
        with open(out_dir / f"serialised_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({**{s: len(r) for s, r in out.items()}, "too_long": too_long}))


if __name__ == "__main__":
    stage, root = sys.argv[1], Path(sys.argv[2])
    if stage == "tasks":
        stage_tasks(root)
    elif stage == "model":
        stage_model(root, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4, int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif stage == "examples":
        stage_examples(root)
    elif stage == "fix_plus_test":
        stage_fix_plus_test(root)
    elif stage == "hard":
        stage_hard(root)
    elif stage == "realistic":
        stage_realistic(root)
    elif stage == "more_model":
        stage_more_model(root, sys.argv[3])
    elif stage == "hard2":
        stage_hard(root, second=True)
    elif stage == "checks":
        stage_checks(root)
    elif stage == "serialise":
        stage_serialise(root, Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "strict",
                        Path(sys.argv[5]) if len(sys.argv) > 5 else None)
    else:
        raise SystemExit(f"unknown stage {stage}")
