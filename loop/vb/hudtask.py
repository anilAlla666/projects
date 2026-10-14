"""HUD taskset: writing the verifier is the task, and the reward is how well it grades real attempts.

Each task is a Harbor task directory, adapted to HUD with ``hud.integrations.harbor.adapt``:

- the agent works in the source task's own environment (the solver's starting machine), with no network,
  and writes /verifier_out/tests/test.sh;
- a separate grading image, the source task's official verifier image with its tests removed, holds a hidden
  grader and the task's pool (real leaderboard attempts, labelled by the official tests and re-checked);
- the grader runs the agent's tests on every attempt in a nested sandbox and scores the agreement
  (templates/hud/grader.py): reward = max(0, 2 * balanced accuracy - 1).

Unlike milestone 1, the agent cannot supply a Dockerfile: on HUD the grading image is built before the run.
Its tests run in the official grading image instead, which the instruction shows it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tomllib
from pathlib import Path

from .pool import load_manifest, pool_dir
from .util import DATA, ROOT, TB_ROOT, write_json
from .writer import _without_canary, dump_toml

HUD_TASKS = DATA / "hud_tasks"
AGENT_TIMEOUT_S = 3600.0
# HUD's hosted runtime accepts fixed compute profiles only (2 CPU / 4 GB is not one); one container runs both the
# agent's environment and the grader, so it gets the larger of the two task sizes, rounded up to a profile.
HUD_PROFILES = [(4, 8192), (16, 16384)]     # Modal CPU-only profiles (docs.hud.ai, 27 Sep 2026)
TEMPLATES = ROOT / "templates" / "hud"


def official_image(task_dir: Path) -> str:
    from .taskdir import official_verifier_image
    return official_verifier_image(task_dir)


def image_config(image: str) -> dict:
    """Env and working directory of an image (docker inspect), so the agent's tests run as the official ones do."""
    out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{json .Config}}"],
                         capture_output=True, text=True, check=True).stdout
    c = json.loads(out)
    env = dict(e.split("=", 1) for e in (c.get("Env") or []) if "=" in e)
    return {"env": env, "workdir": c.get("WorkingDir") or "/"}


def grading_dockerfile(task_dir: Path) -> str:
    """The official tests' Dockerfile as the agent is shown it: minus the canary and the lines that copy the
    official test files in (those are what the agent is asked to replace)."""
    keep = []
    for line in (task_dir / "tests" / "Dockerfile").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if "harbor-" + "canary" in s or "BENCHMARK DATA" in s:
            continue
        if s.upper().startswith("COPY") and "/tests" in s:
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def build_pool(task: str, out: Path, seed: int = 0) -> list[dict]:
    """Each kept attempt under an opaque id, in shuffled order. Files are stored once by content (blobs/<sha256>)
    and each manifest entry maps the attempt's artifact files to them: uefi-bootkit's 21 attempts carry the same
    622 MB disk image, which made the upload too large for HUD when stored per attempt."""
    src = pool_dir(task) / "kept_job"
    artifacts = tomllib.loads((TB_ROOT / task / "task.toml").read_text(encoding="utf-8")).get("artifacts", [])
    rng = random.Random(f"{task}:{seed}")
    blobs = out / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)

    def store(f: Path) -> list:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if not (blobs / h).exists():
            shutil.copyfile(f, blobs / h)
        return [h, f.stat().st_mode & 0o777]

    rows = []
    for e in load_manifest(task):
        if not e["kept"]:
            continue
        pid = "%016x" % rng.getrandbits(64)
        files, dirs = {}, []
        for a in artifacts:
            p = a if isinstance(a, str) else a["source"]
            s = src / e["entry_id"] / "artifacts" / p.lstrip("/")
            if s.is_dir():
                dirs.append(p)
                for f in sorted(s.rglob("*")):
                    if f.is_dir():
                        dirs.append(p.rstrip("/") + "/" + f.relative_to(s).as_posix())
                    elif f.is_file():
                        files[p.rstrip("/") + "/" + f.relative_to(s).as_posix()] = store(f)
            elif s.is_file():
                files[p] = store(s)
        rows.append({"id": pid, "label": e["label"], "files": files, "dirs": dirs,
                     "entry_id": e["entry_id"], "source": e["source"]})
    rng.shuffle(rows)
    write_json(out / "manifest.json", [{k: r[k] for k in ("id", "label", "files", "dirs")} for r in rows])
    return rows


def build_task(task: str, out_root: Path = HUD_TASKS, allow_flat: bool = False, no_pool: bool = False) -> Path:
    """A task in Harbor's layout (TB_ROOT/<task>: instruction.md, task.toml with its artifacts and a separate
    verifier environment, environment/, tests/Dockerfile) as a grader-writing task. With no_pool, there are no
    labelled attempts: a grader can be written, probed and repaired (vb make), but not scored."""
    task_dir = TB_ROOT / task
    t = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    out = Path(out_root) / f"write-verifier-{task}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(task_dir / "environment", out / "environment")

    artifacts = [a if isinstance(a, str) else a["source"] for a in t.get("artifacts", [])]
    attempt_timeout = float(t["verifier"]["timeout_sec"])
    image = official_image(task_dir)
    ic = image_config(image)

    tests = out / "tests"
    grader = tests / "grader"
    if no_pool:
        (grader / "pool" / "blobs").mkdir(parents=True)
        write_json(grader / "pool" / "manifest.json", [])
        rows = []
    else:
        rows = build_pool(task, grader / "pool")
    write_json(grader / "config.json", {"task": task, "artifacts": artifacts, "attempt_timeout_sec": attempt_timeout,
                                        "env": ic["env"], "workdir": ic["workdir"], "allow_flat": allow_flat})
    shutil.copy2(TEMPLATES / "grader.py", grader / "grader.py")
    (tests / "test.sh").write_text("#!/bin/bash\nexec python3 /grader/grader.py\n", encoding="utf-8")
    (tests / "test.sh").chmod(0o755)
    (tests / "Dockerfile").write_text(
        f"FROM {image}\n"
        "RUN rm -rf /tests && mkdir -p /tests /logs/verifier \\\n"
        " && apt-get update && apt-get install -y --no-install-recommends bubblewrap \\\n"
        " && (command -v python3 || apt-get install -y --no-install-recommends python3) \\\n"
        " && rm -rf /var/lib/apt/lists/*\n"
        "COPY test.sh /tests/test.sh\n"
        "COPY grader /grader\n"
        "RUN chmod -R go-rwx /grader && chmod 755 /tests/test.sh\n", encoding="utf-8")
    # Which leaderboard attempt each opaque id is: kept outside the task directory, so no image can carry it.
    write_json(DATA / "hud_pool_keys" / f"{task}.json", rows)

    # Oracle: the official tests themselves. Harbor uploads solution/ only for the oracle agent; HUD's adapter
    # never packages it.
    sol = out / "solution"
    shutil.copytree(task_dir / "tests", sol / "official_tests")
    (sol / "official_tests" / "Dockerfile").unlink()
    (sol / "solve.sh").write_text("#!/bin/bash\nset -e\nmkdir -p /verifier_out\nrm -rf /verifier_out/tests\n"
                                  "cp -r /solution/official_tests /verifier_out/tests\n", encoding="utf-8")
    (sol / "solve.sh").chmod(0o755)

    venv = t["verifier"].get("environment") or {}
    env = {k: v for k, v in (t.get("environment") or {}).items()
           if k in ("docker_image", "build_timeout_sec", "gpus", "os")}
    need_cpu = max(float((t.get("environment") or {}).get("cpus") or 1), float(venv.get("cpus") or 1))
    need_mem = max(int((t.get("environment") or {}).get("memory_mb") or 0), int(venv.get("memory_mb") or 0))
    env["cpus"], env["memory_mb"] = next((c, m) for c, m in HUD_PROFILES if c >= need_cpu and m >= need_mem)
    doc = {"schema_version": "2.0", "artifacts": ["/verifier_out/"],
           "task": {"name": f"vbench/write-verifier-{task}", "description": ""},
           "metadata": {"source_task": t["task"]["name"], "pool_size": len(rows),
                        "reward": "max(0, 2 * balanced_accuracy - 1)"},
           "agent": {"timeout_sec": AGENT_TIMEOUT_S, "network_mode": "no-network"},
           "verifier": {"timeout_sec": attempt_timeout * len(rows) + 300.0, "environment_mode": "separate",
                        "network_mode": "no-network"},
           "environment": env}
    # No [verifier.environment]: on HUD, a verifier with its own resources gets its own machine, where the
    # agent's session files (its /verifier_out) do not exist. The grader shares the agent's container instead,
    # sized above for both.
    (out / "task.toml").write_text(dump_toml(doc), encoding="utf-8")

    template = (ROOT / "templates" / "writer_instruction_hud.md").read_text(encoding="utf-8")
    (out / "instruction.md").write_text(template.format(
        instruction=_without_canary((task_dir / "instruction.md").read_text(encoding="utf-8")),
        grading_dockerfile=grading_dockerfile(task_dir),
        artifacts="\n".join(f"- `{a}`" for a in artifacts), timeout=int(attempt_timeout)), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(HUD_TASKS))
    ap.add_argument("--allow-flat", action="store_true", help="local checks on a host without user namespaces")
    ap.add_argument("--no-pool", action="store_true", help="a task with no labelled attempts (vb make still runs)")
    a = ap.parse_args()
    print(build_task(a.task, Path(a.out), a.allow_flat, a.no_pool))
