"""Where the Docker work runs. By default, on this machine. With VB_HOST=user@host, over ssh on that machine (how
the 9 Oct runs were driven: the model API key stays on the laptop, the containers run on a Linux server).

    VB_HOST           unset: local; otherwise an ssh target
    VB_DATA           the data directory on the Docker machine (task bundles, pools, candidates, run folders)
    VB_REMOTE_REPO    the repository's path on the Docker machine when VB_HOST is set (default /root/vbench/repo)
    VB_REMOTE_PYTHON  the Python there (default /root/venv/bin/python)
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .util import DATA, ROOT

HOST = os.environ.get("VB_HOST", "").strip()
REMOTE = bool(HOST)
REPO = os.environ.get("VB_REMOTE_REPO", "/root/vbench/repo") if REMOTE else str(ROOT)
PYTHON = os.environ.get("VB_REMOTE_PYTHON", "/root/venv/bin/python") if REMOTE else sys.executable
DATA_DIR = os.environ.get("VB_DATA", "/root/vbench") if REMOTE else str(DATA)


def run(cmd: str, stdin: str | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    """A bash command on the Docker machine. Over ssh, a failed CONNECTION (exit 255 before the command ran) is
    retried for up to 2 hours: on 12 Oct a dropped laptop network killed every loop at once (step 13)."""
    argv = (["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", HOST, cmd]
            if REMOTE else ["bash", "-c", cmd])
    for attempt in range(240):
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, errors="replace", timeout=timeout)
        if not (REMOTE and r.returncode == 255 and r.stderr.startswith("ssh:")):
            return r
        import time
        time.sleep(30)
    return r


def put(local_dir: Path, remote_dir: str) -> None:
    """Copy a local folder's contents to a folder on the Docker machine."""
    if REMOTE:
        subprocess.run(["rsync", "-a", "--delete", f"{local_dir}/", f"{HOST}:{remote_dir}/"], check=True)
    else:
        run(f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)} && "
            f"cp -a {shlex.quote(str(local_dir))}/. {shlex.quote(remote_dir)}/").check_returncode()


def get(remote_dir: str, local_dir: Path, max_size: str | None = None, exclude: tuple[str, ...] = ()) -> None:
    """Copy a folder on the Docker machine back here (no-op locally when the paths are the same)."""
    if not REMOTE and Path(remote_dir).resolve() == Path(local_dir).resolve():
        return
    args = ["rsync", "-a"] + ([f"--max-size={max_size}"] if max_size else []) + sum((["--exclude", e] for e in exclude), [])
    src = f"{HOST}:{remote_dir}/" if REMOTE else f"{remote_dir}/"
    subprocess.run(args + [src, f"{local_dir}/"], capture_output=True)


def data(*parts: str) -> str:
    return "/".join([DATA_DIR.rstrip("/"), *parts])
