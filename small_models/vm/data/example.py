"""Labels from the oracle, example records, and serialisation into the four segments the judge reads."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from vm.harness.evidence import split_untrusted, touches_tests, unified_diff
from vm.model.config import SEGMENTS
from vm.sandbox.runner import run_pytest

BUCKETS = (0, 1, 2, 6)  # 0, 1, 2 to 5, 6 or more hidden failures


def bucket(n: int) -> int:
    return max(i for i, b in enumerate(BUCKETS) if n >= b)


def label(task: dict, attempt_files: dict, lib_root: Path) -> dict:
    """Run the full suite on the patched code with the ORIGINAL tests: an attempt that edits tests is judged
    by the tests it did not write. Correct iff every test passes."""
    src, venv = lib_root / task["library"] / "src", lib_root / task["library"] / "venv"
    tests = sorted(set(task["visible"]) | set(task["oracle_only"]) | set(task["failing"]))
    test_paths = {t.split("::")[0] for t in tests}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "src"
        shutil.copytree(src, work)
        (work / task["path"]).write_text(task["broken_source"])
        for p, text in attempt_files.items():
            if touches_tests({p: ""}, test_paths):
                continue
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(text)
        r = run_pytest(work, venv, node_ids=tests, timeout=120)
    failing = [t for t in tests if r.outcomes.get(t) != "passed"]
    n_hidden = sum(t in set(task["oracle_only"]) for t in failing)
    return {"label": int(not failing and not r.timed_out), "n_hidden_failing": n_hidden, "hidden_bucket": bucket(n_hidden),
            "timed_out": r.timed_out}


def pointer_for(label_value: int, evidence: list) -> int:
    texts = [e["text"] for e in evidence]
    if label_value == 1:
        return 0
    for i, t in enumerate(texts):
        if t.startswith("test files edited:") and not t.endswith("none"):
            return i
    for i, t in enumerate(texts[1:], 1):
        if t.endswith(": failed") or t.endswith(": not run") or t.endswith(": skipped") or "timed out" in t:
            return i
    for i, t in enumerate(texts):
        if t.startswith("fuzz ") and "differ" in t:
            return i
    return 0


def make_example(task: dict, attempt, evidence: list, lab: dict, lib_root: Path) -> dict:
    src = lib_root / task["library"] / "src"
    old = {task["path"]: task["broken_source"]}
    for p in attempt.files:
        if p != task["path"]:
            old[p] = (src / p).read_text() if (src / p).is_file() else ""
    shown, moved = split_untrusted(unified_diff(old, attempt.files))
    untrusted = list(dict.fromkeys(moved + [u for u in attempt.untrusted if u]))
    return {"example_id": f"{task['task_id']}/{attempt.kind}/{hashlib.sha256(shown.encode()).hexdigest()[:10]}", "library": task["library"],
            "split": task["split"], "operator": task["operator"], "function": task["function"], "attempt_kind": attempt.kind,
            "task_text": task["task_text"], "function_lines": task["function_lines"], "bug_line": task["bug_line"],
            "diff": shown, "evidence": evidence, "untrusted": untrusted, "label": lab["label"],
            "hidden_bucket": lab["hidden_bucket"], "n_hidden_failing": lab["n_hidden_failing"],
            "pointer": pointer_for(lab["label"], evidence), "attempt_files": attempt.files}


def serialise(ex: dict, tok, max_len: int = 8192) -> dict:
    """Tokens and segment ids for <|task|> task text + numbered function lines, <|diff|> diff, <|evidence|> lines,
    <|untrusted|> prose, then <|verdict|>. Each function line and each evidence line starts with <|line|>.
    When too long, the diff is cut first, then the untrusted prose; the task and evidence are never cut."""
    ids = lambda s: tok.encode(s).ids  # noqa: E731
    line = tok.token_to_id("<|line|>")
    task = [tok.token_to_id("<|task|>")] + ids(ex["task_text"])
    code_starts = []
    for i, ln in enumerate(ex["function_lines"]):
        code_starts.append(len(task))
        task += [line] + ids(f"{i + 1:4d}  {ln}\n")
    evid, ev_starts = [tok.token_to_id("<|evidence|>")], []
    for e in ex["evidence"]:
        ev_starts.append(len(evid))
        evid += [line] + ids(e["text"] + "\n")
    diff = [tok.token_to_id("<|diff|>")] + ids(ex["diff"])
    untr = [tok.token_to_id("<|untrusted|>")] + ids("\n".join(ex["untrusted"]))
    room = max_len - len(task) - len(evid) - 1
    if room < 2:
        raise ValueError("task and evidence alone exceed max_len")
    if len(diff) + len(untr) > room:
        untr = untr[: max(1, min(len(untr), room // 4))]
        diff = diff[: room - len(untr)]
    parts = [(task, SEGMENTS["task"]), (diff, SEGMENTS["diff"]), (evid, SEGMENTS["evidence"]), (untr, SEGMENTS["untrusted"])]
    tokens, segs, offsets = [], [], {}
    for name, (p, s) in zip(("task", "diff", "evidence", "untrusted"), parts):
        offsets[name] = len(tokens)
        tokens += p
        segs += [s] * len(p)
    tokens.append(tok.token_to_id("<|verdict|>"))
    segs.append(SEGMENTS["task"])
    return {"tokens": tokens, "segments": segs, "verdict_pos": len(tokens) - 1,
            "line_starts": [offsets["evidence"] + i for i in ev_starts],
            "code_line_starts": code_starts, "label": ex["label"], "pointer": ex["pointer"],
            "hidden_bucket": ex["hidden_bucket"], "bug_line": ex["bug_line"]}
