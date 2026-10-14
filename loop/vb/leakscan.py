"""Signs that a written verifier copied or fetches the official tests instead of writing its own.

Verifiers grade offline (vb.taskdir.set_verifier_offline, proven by the network_probe baseline), but a Dockerfile
builds with network, so a fetch at build time is caught here by its text. A disguised fetch (e.g. a package that
bundles the tests) is a residual this scan does not claim to catch; it is disclosed in the report.

Not flagged, by design (both were false alarms on the first real verifier, 26 Sep):
- the task's own environment image, which the writer is told it may build FROM (it is named
  harborframework/terminal-bench:...); any OTHER image of that name, such as the official verifier image that
  holds the tests, is still flagged;
- the benchmark's canary marker, because the environment's own files carry it too, so a verifier that copies
  an environment file legitimately contains it. Copies of official test files are caught by content instead."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

FETCH = re.compile(r"github\.com|githubusercontent|huggingface\.co|\bhf\.co\b|harbor-framework|harborframework|"
                   r"terminal-bench|tbench\.ai", re.I)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan(tests_dir: Path, official_tests: Path, allowed: tuple[str, ...] = (), inputs: Path | None = None) -> list[str]:
    """Findings, empty when none. `allowed` are exact strings (the environment image) removed before matching.
    `inputs` is the task's environment/ folder: an official test file that is byte-identical to one of the task's
    own inputs (e.g. uefi-bootkit's infected firmware, which the solver is given) is not evidence of a copy."""
    given = {_sha(p) for p in Path(inputs).rglob("*") if p.is_file()} if inputs and Path(inputs).is_dir() else set()
    official = {_sha(p): p.name for p in Path(official_tests).rglob("*")
                if p.is_file() and p.stat().st_size > 64 and _sha(p) not in given}
    found = []
    for p in sorted(q for q in Path(tests_dir).rglob("*") if q.is_file()):
        rel = p.relative_to(tests_dir).as_posix()
        text = p.read_text(encoding="utf-8", errors="ignore")
        for a in allowed:
            text = text.replace(a, "")
        if p.stat().st_size > 64 and _sha(p) in official:
            found.append(f"{rel}: byte-identical to official {official[_sha(p)]}")
        m = FETCH.search(text)
        if m:
            found.append(f"{rel}: mentions {m.group(0)}")
    return found
