"""Step 13: turn a Terminal Wrench task (github.com/few-sh/terminal-wrench, Apache-2.0) into labelled attempts for the
verify loop: the task's reference solution and the legitimate solves are CORRECT attempts; the trajectories that
earned reward without solving the task (hacks) are WRONG attempts the original verifier accepts.

Runs on the GPU host (Docker through sudo). For one task:
  1. build the task image; run the original tests once with the network on, so their dependencies are cached, then
     commit that as the grading image, offline from then on
  2. replay every attempt (solve.sh, each legitimate and each hack trajectory's keystrokes) in a fresh container
  3. snapshot the files the attempt changed, restore the snapshot in a fresh offline container, run the original
     tests there, and keep the attempt only if the reward reproduces (1 for both kinds: hacks pass the original tests)
Writes <out>/<task>/attempts/<id>/ (paths mirrored from /) and <out>/<task>/manifest.json.

    python3 tools/tw_build.py --tw ~/tw --out ~/twdata TASK [TASK ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

D = ["sudo", "-n", "docker"]
SKIP = ("/tmp", "/var/tmp", "/proc", "/sys", "/dev", "/run", "/logs", "/tests", "/solution", "/var/lib/apt",
        "/var/cache", "/var/log", "/root/.cache", "/root/.npm", "/root/.local/share/uv", "/etc/hostname",
        "/etc/hosts", "/etc/resolv.conf", "/replay.sh", "/var/lib/dpkg", "/usr/share/doc", "/usr/share/man")
MAX_ATTEMPT_BYTES = 40 * 2**20


def sh(cmd: list[str], timeout: int = 1800, inp: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=inp)


def dk(*a: str, timeout: int = 1800, inp: str | None = None) -> subprocess.CompletedProcess:
    return sh(D + list(a), timeout, inp)


def build(task_dir: Path, tid: str) -> tuple[str, str]:
    """(attempt image, grading image). The grading image has the test dependencies cached and runs offline."""
    img, gimg = f"tw-{tid}".lower(), f"twv-{tid}".lower()
    r = dk("build", "-q", "-t", img, str(task_dir / "environment"), timeout=3600)
    if r.returncode:
        raise RuntimeError("build: " + (r.stderr or r.stdout)[-600:])
    c = f"twwarm-{tid}".lower()
    dk("rm", "-f", c)
    dk("run", "-d", "--name", c, "--entrypoint", "sleep", img, "infinity")
    try:
        dk("cp", str(task_dir / "tests"), f"{c}:/tests")
        dk("exec", c, "bash", "-lc", "mkdir -p /logs/verifier; cd \"$(pwd)\"; bash /tests/test.sh >/dev/null 2>&1; true",
           timeout=1800)
        dk("exec", c, "bash", "-lc", "rm -rf /tests /logs")
        r = dk("commit", "-c", "ENV UV_OFFLINE=1", "-c", "ENV PIP_NO_INDEX=1", "-c", "ENV UV_NO_CACHE=0", "-c", "ENTRYPOINT []", "-c", 'CMD ["bash"]', c, gimg)
        if r.returncode:
            raise RuntimeError("commit: " + r.stderr[-400:])
    finally:
        dk("rm", "-f", c)
    return img, gimg


def keystrokes(traj: Path) -> str:
    """A Terminus-2 trajectory's keystrokes as one bash script (special keys dropped)."""
    t = json.loads(traj.read_text())
    out = []
    for s in t.get("steps", []):
        for c in s.get("tool_calls") or []:
            k = (c.get("arguments") or {}).get("keystrokes")
            if isinstance(k, str) and k and not k.startswith(("C-", "Escape", "Up", "Down")):
                out.append(k if k.endswith("\n") else k + "\n")
    return "".join(out)


def changed(c: str) -> list[str]:
    r = dk("diff", c)
    paths = []
    for line in r.stdout.splitlines():
        kind, _, p = line.partition(" ")
        if kind in ("A", "C") and not any(p == s or p.startswith(s + "/") for s in SKIP) and "__pycache__" not in p:
            paths.append(p)
    return paths


def snapshot(c: str, dest: Path) -> int:
    """Copy the regular files the attempt added or changed into dest (mirrored from /); returns bytes."""
    paths = changed(c)
    q = " ".join(shlex.quote(p) for p in paths)
    r = dk("exec", c, "bash", "-c", f"for p in {q}; do [ -f \"$p\" ] && [ ! -L \"$p\" ] && printf '%s\\0' \"$p\"; done"
           " | tar --null -T - -cf - 2>/dev/null | base64 -w0", timeout=600)
    dest.mkdir(parents=True, exist_ok=True)
    if r.stdout.strip():
        import base64
        data = base64.b64decode(r.stdout.strip())
        p = subprocess.run(["tar", "-xf", "-", "-C", str(dest)], input=data, capture_output=True)
        if p.returncode:
            raise RuntimeError("untar: " + p.stderr.decode()[-300:])
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def reward_of(gimg: str, task_dir: Path, attempt: Path | None, workdir: str) -> tuple[float | None, str]:
    """Run the ORIGINAL tests on a restored attempt in a fresh offline container."""
    c = f"twchk-{os.getpid()}-{time.time_ns()}"
    dk("run", "-d", "--name", c, "--network", "none", "--entrypoint", "sleep", gimg, "infinity")
    try:
        if attempt is not None and any(attempt.iterdir()):
            dk("cp", f"{attempt}/.", f"{c}:/")
        dk("cp", str(task_dir / "tests"), f"{c}:/tests")
        r = dk("exec", "-w", workdir, c, "bash", "-lc", "mkdir -p /logs/verifier; timeout 900 bash /tests/test.sh; "
               "echo; echo REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || python3 -c "
               "\"import json;print(json.load(open('/logs/verifier/reward.json'))['reward'])\" 2>/dev/null)",
               timeout=1200)
        tail = (r.stdout + r.stderr)[-1500:]
        v = r.stdout.strip().rsplit("REWARD=", 1)[-1].strip() if "REWARD=" in r.stdout else ""
        try:
            return float(v), tail
        except ValueError:
            return None, tail
    finally:
        dk("rm", "-f", c)


def replay(img: str, script: str, dest: Path, workdir: str, solution: Path | None = None) -> tuple[int, str]:
    c = f"twrep-{os.getpid()}-{time.time_ns()}"
    dk("run", "-d", "--name", c, "--entrypoint", "sleep", img, "infinity")
    try:
        if solution is not None:
            dk("cp", str(solution), f"{c}:/solution")
        dk("exec", "-i", c, "bash", "-c", "cat > /replay.sh", inp=script)
        r = dk("exec", "-w", workdir, c, "bash", "-c", "timeout 900 bash /replay.sh </dev/null >/tmp/replay.out 2>&1; echo rc=$?",
               timeout=1200)
        size = snapshot(c, dest)
        return size, r.stdout.strip()
    finally:
        dk("rm", "-f", c)


def workdir_of(img: str) -> str:
    r = dk("image", "inspect", "-f", "{{.Config.WorkingDir}}", img)
    return r.stdout.strip() or "/app"


def build_task(tw: Path, out: Path, tid: str, max_each: int = 6) -> dict:
    t0 = time.time()
    models = sorted(p for p in (tw / "tasks" / tid).iterdir() if p.is_dir())
    task_dir = models[0] / "original_task"
    res = {"task": tid, "attempts": [], "dropped": []}
    img, gimg = build(task_dir, tid)
    wd = workdir_of(img)
    res |= {"image": img, "grading_image": gimg, "workdir": wd, "build_s": round(time.time() - t0)}
    base = out / tid / "attempts"
    if base.exists():
        subprocess.run(["rm", "-rf", str(base)])
    base.mkdir(parents=True)
    (base / "empty").mkdir()
    rew, tail = reward_of(gimg, task_dir, base / "empty", wd)
    res["empty_reward"] = rew
    jobs = [("oracle", "correct", "bash /solution/solve.sh\n", task_dir / "solution")]
    for m in models:
        for kind, sub, label in (("legit", "baseline_trajectories", "correct"), ("hack", "hack_trajectories", None)):
            for d in sorted((m / sub).glob("*"))[:max_each] if (m / sub).exists() else []:
                meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
                cls = meta.get("classification", "baseline" if kind == "legit" else "")
                if kind == "hack":
                    if cls in ("rewarded_serious_exploit", "rewarded_nonserious_hack"):
                        label = "wrong"
                    elif cls == "attacker_legitimate_solve":
                        label = "correct"
                    else:
                        continue
                tj = d / "trial" / "agent" / "trajectory.json"
                if tj.exists():
                    jobs.append((f"{kind}-{m.name}-{d.name}", label, keystrokes(tj), None))
    for aid, label, script, sol in jobs:
        dest = base / aid
        try:
            size, rc = replay(img, script, dest, wd, sol)
        except Exception as e:  # noqa: BLE001
            res["dropped"].append({"id": aid, "why": f"replay: {e}"[:200]}); continue
        if size > MAX_ATTEMPT_BYTES or size == 0:
            res["dropped"].append({"id": aid, "why": f"size {size}"}); subprocess.run(["rm", "-rf", str(dest)]); continue
        r, tail = reward_of(gimg, task_dir, dest, wd)
        if r != 1.0:  # a correct attempt must pass; a hack is only a hack if the original tests accept it
            res["dropped"].append({"id": aid, "label": label, "why": f"reward {r}", "tail": tail[-300:]})
            subprocess.run(["rm", "-rf", str(dest)]); continue
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
        res["attempts"].append({"id": aid, "label": label, "bytes": size, "files": files[:50], "n_files": len(files)})
    res["seconds"] = round(time.time() - t0)
    (out / tid / "manifest.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--tw", type=Path, default=Path.home() / "tw")
    ap.add_argument("--out", type=Path, default=Path.home() / "twdata")
    ap.add_argument("--max-each", type=int, default=6)
    a = ap.parse_args()
    for tid in a.tasks:
        try:
            r = build_task(a.tw, a.out, tid, a.max_each)
            n = {k: sum(x["label"] == k for x in r["attempts"]) for k in ("correct", "wrong")}
            print(json.dumps({"TASK": tid, **n, "dropped": len(r["dropped"]), "empty_reward": r.get("empty_reward"),
                              "build_s": r.get("build_s"), "seconds": r["seconds"]}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"TASK": tid, "error": str(e)[:400]}), flush=True)
