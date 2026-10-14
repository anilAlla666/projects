#!/usr/bin/env python3
"""Hidden grader for a verifier-writing task.

The agent wrote tests in /verifier_out/tests/. This runs them, one at a time, on every stored attempt in
/grader/pool (real attempts at the source task, each labelled pass or fail by the task's human-written tests,
with those labels re-checked), and scores how often the agent's verdict matches the label:

    TPR = correct attempts accepted / correct attempts
    TNR = wrong attempts rejected / wrong attempts
    BA  = (TPR + TNR) / 2
    reward = max(0, 2 * BA - 1)     0 for always-pass, always-fail or no tests; 1 for a perfect verifier

A run that errors, times out or writes no verdict counts as a reject.

Isolation. Each run gets a fresh sandbox (bubblewrap, nested inside the platform's own): the grading image
read-only, fresh copies of the directories tests write to (the artifacts' top-level directories, /root, /home)
and empty /tmp and /run, /grader and /verifier_out hidden, no capabilities, no network, its own process
namespace. So the agent's tests cannot read the labels or the other attempts, cannot keep a process alive
into the next run, and cannot touch this process. If the sandbox cannot be created, the grader refuses to
grade (exit 1, no reward) rather than run the tests unsandboxed, unless config.json says allow_flat (local
checks on a host without user namespaces only).
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GRADER = Path(os.environ.get("VB_GRADER_DIR", "/grader"))
WRITTEN = Path(os.environ.get("VB_WRITTEN_DIR", "/verifier_out/tests"))
LOGS = Path(os.environ.get("VB_LOGS_DIR", "/logs/verifier"))
STAGE = Path(os.environ.get("VB_STAGE_DIR", "/var/tmp/vbstage"))
PR_SET_DUMPABLE = 4


def verdict_of(logs: Path) -> str:
    """'pass' only for a numeric reward of 1 in reward.txt (or reward.json's "reward"); anything else rejects."""
    value = None
    txt, js = logs / "reward.txt", logs / "reward.json"
    try:
        if txt.is_file():
            value = float(txt.read_text(encoding="utf-8", errors="replace").strip())
        elif js.is_file():
            data = json.loads(js.read_text(encoding="utf-8", errors="replace"))
            value = float(data["reward"] if isinstance(data, dict) else data)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        value = None
    return "pass" if value is not None and abs(value - 1.0) < 1e-9 else "fail"


def score(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["label"] == "pass"]
    neg = [r for r in rows if r["label"] != "pass"]
    tp = sum(r["verdict"] == "pass" for r in pos)
    tn = sum(r["verdict"] != "pass" for r in neg)
    tpr = tp / len(pos) if pos else 0.0
    tnr = tn / len(neg) if neg else 0.0
    ba = (tpr + tnr) / 2
    return {"reward": round(max(0.0, 2 * ba - 1), 6), "balanced_accuracy": round(ba, 6),
            "tpr": round(tpr, 6), "tnr": round(tnr, 6), "accepted_correct": f"{tp}/{len(pos)}",
            "rejected_wrong": f"{tn}/{len(neg)}"}


def _no_dump() -> None:
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except OSError:
        pass


def writable_dirs(artifacts: list[str]) -> list[str]:
    """Image directories a run gets a fresh private copy of: each artifact's top-level directory, /root, /home."""
    tops = {"/" + Path(p).parts[1] for p in artifacts if len(Path(p).parts) > 1}
    return sorted(d for d in tops | {"/root", "/home"}
                  if Path(d).is_dir() and d not in ("/tmp", "/var", "/logs", "/tests", str(GRADER), "/verifier_out"))


# How /proc is provided. A fresh procfs is refused where the platform's own /proc has masked paths (HUD's and
# Docker's), so the fallback binds the existing one read-only; the probe then checks that no process visible
# through it exposes the grader's files (/proc/<pid>/root/grader), and the grader refuses that mode if any does.
PROC_MODES = {"procfs": ["--proc", "/proc"], "bindproc": ["--ro-bind", "/proc", "/proc"]}


def bwrap_argv(stage: Path, artifacts: list[str], env: dict, workdir: str, cmd: list[str],
               proc: str = "procfs") -> list[str]:
    a = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user", "--uid", "0", "--gid", "0",
         "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--unshare-cgroup-try",
         "--ro-bind", "/", "/", "--dev", "/dev", *PROC_MODES[proc],
         "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/run",
         "--tmpfs", str(GRADER), "--tmpfs", "/verifier_out"]
    for d in sorted((stage / "rw").iterdir()) if (stage / "rw").is_dir() else []:
        a += ["--bind", str(d), "/" + d.name]
    a += ["--bind", str(stage / "tests"), "/tests", "--bind", str(stage / "logs"), str(LOGS)]
    for p in artifacts:
        src = stage / "art" / p.lstrip("/")
        if src.exists():
            a += ["--bind", str(src), p]
    a += ["--clearenv"]
    for k, v in env.items():
        a += ["--setenv", k, v]
    a += ["--chdir", workdir, "--cap-drop", "ALL", "--"] + cmd
    return a


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _kill_others() -> None:
    """Flat mode only: kill every process in this namespace except this one and its ancestors."""
    keep, pid = {os.getpid()}, os.getpid()
    while pid > 1:
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        keep.add(pid)
    for _ in range(20):
        victims = [int(p) for p in os.listdir("/proc") if p.isdigit() and int(p) not in keep and int(p) != 1]
        if not victims:
            return
        for v in victims:
            try:
                os.kill(v, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.2)


def run_one(mode: str, entry: dict, cfg: dict) -> dict:
    stage = STAGE / entry["id"]
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "logs").mkdir(parents=True)
    shutil.copytree(WRITTEN, stage / "tests", symlinks=False, ignore_dangling_symlinks=True)
    if mode != "flat":
        for d in writable_dirs(cfg["artifacts"]):
            shutil.copytree(d, stage / "rw" / d.lstrip("/"), symlinks=True)
    for d in entry.get("dirs", []):                       # the attempt's artifacts, from the content store
        (stage / "art" / d.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for path, (sha, fmode) in entry.get("files", {}).items():
        dst = stage / "art" / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(GRADER / "pool" / "blobs" / sha, dst)
        dst.chmod(fmode)
    cmd = ["/bin/bash", "/tests/test.sh"]
    t0, timed_out = time.monotonic(), False
    if mode == "flat":
        for p in cfg["artifacts"]:
            target = Path(p)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            src = stage / "art" / p.lstrip("/")
            if src.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if src.is_dir() else shutil.copy2)(src, target)
        tests = Path("/tests")
        shutil.rmtree(tests, ignore_errors=True)
        shutil.copytree(stage / "tests", tests)
        shutil.rmtree(LOGS, ignore_errors=True)
        LOGS.mkdir(parents=True)
        argv, logs = cmd, LOGS
    else:
        argv = bwrap_argv(stage, cfg["artifacts"], cfg["env"], cfg["workdir"], cmd, proc=mode)
        logs = stage / "logs"
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=cfg["workdir"] if mode == "flat" else "/", start_new_session=True,
                            env=cfg["env"] if mode == "flat" else {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"})
    try:
        rc = proc.wait(timeout=cfg["attempt_timeout_sec"])
    except subprocess.TimeoutExpired:
        timed_out, rc = True, None
    _kill_group(proc)
    if mode == "flat":
        _kill_others()
    verdict = "fail" if timed_out else verdict_of(logs)
    shutil.rmtree(stage, ignore_errors=True)
    if mode == "flat":
        shutil.rmtree(LOGS, ignore_errors=True)
        LOGS.mkdir(parents=True)
    return {"label": entry["label"], "verdict": verdict, "timed_out": timed_out, "rc": rc,
            "seconds": round(time.monotonic() - t0, 1)}


PROBE = """\
test -z "$(ls -A {grader} 2>/dev/null)" || exit 0
for p in /proc/[0-9]*; do
  ls -A "$p/root{grader}" 2>/dev/null | grep -q . && exit 0
done
echo 1 > {logs}/reward.txt
"""


def probe(cfg: dict) -> str | None:
    """The first /proc mode in which a sandboxed script can see nothing of /grader, directly or through any
    visible process's root; None if neither."""
    STAGE.mkdir(parents=True, exist_ok=True)
    stage = STAGE / "probe"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "tests").mkdir(parents=True)
    (stage / "logs").mkdir()
    (stage / "tests" / "test.sh").write_text(PROBE.format(grader=GRADER, logs=LOGS), encoding="utf-8")
    try:
        for mode in PROC_MODES:
            for f in (stage / "logs").iterdir():
                f.unlink()
            argv = bwrap_argv(stage, [], cfg["env"], "/", ["/bin/sh", "/tests/test.sh"], proc=mode)
            try:
                r = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=60,
                                   env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"})
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"sandbox probe ({mode}): {e}", file=sys.stderr)
                continue
            if verdict_of(stage / "logs") == "pass":
                return mode
            print(f"sandbox probe ({mode}): rc={r.returncode} {r.stderr.decode(errors='replace')[-300:]}",
                  file=sys.stderr)
        return None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def finish(result: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(f"{result['reward']}\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def main() -> int:
    _no_dump()
    cfg = json.loads((GRADER / "config.json").read_text(encoding="utf-8"))
    pool = json.loads((GRADER / "pool" / "manifest.json").read_text(encoding="utf-8"))
    head = {"task": cfg["task"], "attempts": len(pool)}
    if not (WRITTEN / "test.sh").is_file():
        finish({**head, **score([{**e, "verdict": "fail"} for e in pool]), "note": "no /verifier_out/tests/test.sh"})
        return 0
    mode = probe(cfg)
    if mode is None:
        if not cfg.get("allow_flat"):
            print(json.dumps({**head, "error": "no sandbox: the grader will not run the tests unsandboxed"}))
            return 1
        mode = "flat"
    rows = [run_one(mode, e, cfg) for e in pool]
    finish({**head, **score(rows), "isolation": mode,
            "timeouts": sum(r["timed_out"] for r in rows),
            "seconds": round(sum(r["seconds"] for r in rows), 1),
            "verdicts": "".join("P" if r["verdict"] == "pass" else "F" for r in rows),
            "labels": "".join("P" if r["label"] == "pass" else "F" for r in rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
