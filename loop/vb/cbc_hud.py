"""Correct-by-construction pilot on HUD: pools from solver attempts plus rule-based near misses, and two writer
conditions per task (docs/research_intent_goal.md, round 6).

  blind        the task text and the broken machine (the same contract as the published taskset)
  constructor  the same, plus what the task's constructor knows: the original file and the diff of the break

The hidden grader is the taskset's own (templates/hud/grader.py); the grading image is the oracle's image with the
library's tests removed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import shutil
import tomllib
from pathlib import Path

from .cbc import CBC, SRC
from .hudtask import HUD_PROFILES, TEMPLATES, image_config
from .util import DATA, ROOT, write_json
from .writer import dump_toml

OUT = DATA / "cbc_hud"
MIN_EACH = 5

CONSTRUCTOR = """
## You built this task

You are the task's constructor. You made it by taking a correct version of the library and breaking it with one
small change, so you know exactly what a correct library looks like:

- `/construction/original/{name}` is the file as it was before you broke it;
- `/construction/break.diff` is the change you made.

A solver's fix does not have to match your original file: any change that makes the library behave correctly again
is a correct solution.
"""


def pool_entries(k: int) -> list[dict]:
    """Every attempt for task k: real solver attempts (labelled by their own oracle run) and rule-based near misses."""
    name = f"cbc-portion-{k}"
    rows = []
    for rj in sorted(DATA.glob("runs/cbc_solvers/*/*/*/result.json")):
        r = json.loads(rj.read_text())
        if r["task_name"].split("/")[-1] != name:
            continue
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        art = rj.parent / "artifacts" / "app" / "portion"
        if rew is None or not art.is_dir():
            continue
        rows.append({"id": rj.parent.name, "label": "pass" if rew >= 1.0 else "fail", "source": "solver",
                     "solver": rj.parent.parent.name, "dir": art})
    ms = json.loads((CBC / "mutants.json").read_text())
    m = ms[k - 1]
    nms = json.loads((CBC / "near_misses.json").read_text()).get(str(k), [])
    for i, nm in enumerate(nms):
        d = CBC / "nm_attempts" / name / f"nm{i}" / "portion"
        if d.exists():
            shutil.rmtree(d.parent)
        shutil.copytree(SRC / "portion", d)
        (d / Path(m["file"]).name).write_text(nm["source"])
        rows.append({"id": f"{name}-nm{i}", "label": "fail", "source": "near_miss_" + nm["origin"], "dir": d})
    return rows


def differs_from_original(d: Path, k: int) -> bool:
    ms = json.loads((CBC / "mutants.json").read_text())
    f = Path(ms[k - 1]["file"]).name
    return (d / f).read_text() != (SRC / "portion" / f).read_text()


def build(k: int, cond: str) -> Path | None:
    name = f"cbc-portion-{k}"
    src = CBC / "tasks" / name
    entries = pool_entries(k)
    npass = sum(e["label"] == "pass" for e in entries)
    nfail = len(entries) - npass
    if npass < MIN_EACH or nfail < MIN_EACH:
        print(f"{name}: dropped by rule ({npass} correct, {nfail} wrong)")
        return None
    out = OUT / cond / f"write-verifier-{name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(src / "environment", out / "environment")
    ms = json.loads((CBC / "mutants.json").read_text())
    m = ms[k - 1]
    fname = Path(m["file"]).name
    extra = ""
    if cond == "constructor":
        c = out / "environment" / "_construction"
        (c / "original").mkdir(parents=True)
        shutil.copy2(SRC / m["file"], c / "original" / fname)
        diff = difflib.unified_diff((SRC / m["file"]).read_text().splitlines(keepends=True),
                                    m["source"].splitlines(keepends=True), f"a/{m['file']}", f"b/{m['file']}")
        (c / "break.diff").write_text("".join(diff))
        df = out / "environment" / "Dockerfile"
        df.write_text(df.read_text().rstrip() + "\nCOPY _construction /construction\n")
        extra = CONSTRUCTOR.format(name=fname)

    # grading image: the oracle's image with the library's tests removed
    tests = out / "tests"
    grader = tests / "grader"
    (grader / "pool" / "blobs").mkdir(parents=True)
    rng = random.Random(f"{name}:0")
    manifest, key = [], []
    for e in entries:
        pid = "%016x" % rng.getrandbits(64)
        files = {}
        for f in sorted(Path(e["dir"]).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                if not (grader / "pool" / "blobs" / h).exists():
                    shutil.copyfile(f, grader / "pool" / "blobs" / h)
                files["/app/portion/" + f.relative_to(e["dir"]).as_posix()] = [h, f.stat().st_mode & 0o777]
        manifest.append({"id": pid, "label": e["label"], "files": files, "dirs": ["/app/portion"]})
        key.append({"id": pid, "label": e["label"], "source": e["source"], "entry": e["id"],
                    "differs_from_original": e["label"] == "pass" and differs_from_original(Path(e["dir"]), k)})
    order = list(range(len(manifest)))
    rng.shuffle(order)
    write_json(grader / "pool" / "manifest.json", [manifest[i] for i in order])
    write_json(DATA / "cbc_pool_keys" / f"{name}.json", [key[i] for i in order])
    oracle_df = (src / "tests" / "Dockerfile").read_text()
    shutil.copytree(src / "tests" / "app", tests / "app")
    (tests / "Dockerfile").write_text(
        oracle_df.replace("COPY test.sh /tests/test.sh\n", "") +
        "RUN rm -rf /app/tests /tests && mkdir -p /tests /logs/verifier \\\n"
        " && apt-get update && apt-get install -y --no-install-recommends bubblewrap && rm -rf /var/lib/apt/lists/*\n"
        "COPY test.sh /tests/test.sh\nCOPY grader /grader\nRUN chmod -R go-rwx /grader && chmod 755 /tests/test.sh\n")
    (tests / "test.sh").write_text("#!/bin/bash\nexec python3 /grader/grader.py\n")
    (tests / "test.sh").chmod(0o755)
    shutil.copy2(TEMPLATES / "grader.py", grader / "grader.py")
    write_json(grader / "config.json", {"task": name, "artifacts": ["/app/portion"], "attempt_timeout_sec": 120.0,
                                        "env": {"PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"},
                                        "workdir": "/app", "allow_flat": False})
    # the writer's view of the grading image: the oracle Dockerfile without the tests
    shown = "\n".join(l for l in oracle_df.splitlines() if "test.sh" not in l)
    shown += "\n# the library's own tests are removed; a solver's /app/portion replaces the library before your tests run"
    t = tomllib.loads((src / "task.toml").read_text())
    template = (ROOT / "templates" / "writer_instruction_hud.md").read_text()
    text = template.format(instruction=(src / "instruction.md").read_text().strip(), grading_dockerfile=shown,
                           artifacts="- `/app/portion/`", timeout=120)
    if extra:
        text = text.replace("## What to write", extra.strip() + "\n\n## What to write", 1)
    (out / "instruction.md").write_text(text)
    cpus, mem = HUD_PROFILES[0]
    doc = {"schema_version": "2.0", "artifacts": ["/verifier_out/"],
           "task": {"name": f"vbench-cbc/{cond}-{name}", "description": ""},
           "metadata": {"source_task": name, "condition": cond, "pool_size": len(manifest)},
           "agent": {"timeout_sec": 3600.0, "network_mode": "no-network"},
           "verifier": {"timeout_sec": 120.0 * len(manifest) + 300.0, "environment_mode": "separate",
                        "network_mode": "no-network"},
           "environment": {"build_timeout_sec": 600.0, "cpus": cpus, "memory_mb": mem}}
    (out / "task.toml").write_text(dump_toml(doc))
    print(f"{name} {cond}: {npass} correct, {nfail} wrong -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1,2,3,4,5")
    a = ap.parse_args()
    for k in [int(x) for x in a.tasks.split(",")]:
        for cond in ("blind", "constructor"):
            build(k, cond)
