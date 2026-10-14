"""Run untrusted code in Docker: no network, 1 GB, one CPU, a wall-clock limit. The source tree is mounted at
/work/src (the editable install points there), the environment read-only at /work/venv, and outcomes come from
the results plugin's JSON in a separate mount."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DOCKER = ["sudo", "-n", "docker"]
IMAGE = "python:3.12-slim"
# run as the invoking user, so every file a container writes can be cleaned up without root
AS_USER = ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
PLUGIN_DIR = Path(__file__).resolve().parent


@dataclass
class RunResult:
    outcomes: dict = field(default_factory=dict)
    exit_code: int = -1
    timed_out: bool = False
    seconds: float = 0.0


def _docker(src_dir: Path, venv_dir: Path, out_dir: Path, cmd: list[str], timeout: int):
    args = [*DOCKER, "run", "--rm", *AS_USER, "--network", "none", "--memory", "1g", "--memory-swap", "1g", "--cpus", "1",
            "--pids-limit", "256", "-e", "VM_RESULTS=/work/out/results.tsv", "-e", "PYTHONPATH=/work/tools",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{src_dir}:/work/src", "-v", f"{venv_dir}:/work/venv:ro", "-v", f"{out_dir}:/work/out",
            "-v", f"{PLUGIN_DIR}:/work/tools:ro", "-w", "/work/src", IMAGE, "timeout", "-s", "KILL", str(timeout), *cmd]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout + 60)


def run_pytest(src_dir: Path, venv_dir: Path, node_ids=None, timeout: int = 90, maxfail: int | None = None) -> RunResult:
    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        out_dir.chmod(0o777)
        cmd = ["/work/venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "vm_results_plugin",
               "-o", "timeout=20", *([f"--maxfail={maxfail}"] if maxfail else []), *(node_ids or [])]
        t = time.perf_counter()
        try:
            p = _docker(Path(src_dir), Path(venv_dir), out_dir, cmd, timeout)
        except subprocess.TimeoutExpired:
            return RunResult(timed_out=True, seconds=time.perf_counter() - t)
        res = out_dir / "results.tsv"
        outcomes = {}
        if res.is_file():
            for ln in res.read_text().splitlines():
                nodeid, _, outcome = ln.rpartition("\t")
                if nodeid:
                    outcomes[nodeid] = outcome
        return RunResult(outcomes, p.returncode, p.returncode == 137, time.perf_counter() - t)


def run_python(src_dir: Path, venv_dir: Path, script: str, timeout: int = 60) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as out:
        out_dir = Path(out)
        out_dir.chmod(0o777)
        (out_dir / "script.py").write_text(script)
        try:
            p = _docker(Path(src_dir), Path(venv_dir), out_dir, ["/work/venv/bin/python", "/work/out/script.py"], timeout)
        except subprocess.TimeoutExpired:
            return -1, ""
        return p.returncode, p.stdout
