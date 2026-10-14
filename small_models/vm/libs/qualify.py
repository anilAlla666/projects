"""Which candidate libraries can carry tasks: pure Python, a test suite of at least 100 tests that all pass (or
skip) on the original code in under 60 s with no network."""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from vm.corpus.code import MAX_ARCHIVE, _download
from vm.libs.candidates import _fetch_json
from vm.sandbox.runner import AS_USER, DOCKER, IMAGE, run_pytest

COMPILED = {".c", ".cc", ".cpp", ".pyx", ".pxd", ".rs", ".f", ".f90"}
TEST_PLUGINS = "pytest pytest-timeout hypothesis pytest-asyncio pytest-mock mock freezegun"
_GH = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/|$)")


def has_tests(src: Path) -> bool:
    return any(p.name.startswith("test_") or p.name.endswith("_test.py") for p in src.rglob("*.py"))


def github_repo(name: str) -> str | None:
    """The project's GitHub repository from its PyPI metadata (project URLs first, then the home page)."""
    try:
        info = _fetch_json(f"https://pypi.org/pypi/{name}/json")["info"]
    except Exception:  # noqa: BLE001
        return None
    urls = list((info.get("project_urls") or {}).values()) + [info.get("home_page") or ""]
    for u in urls:
        m = _GH.match(u or "")
        if m:
            return f"https://github.com/{m.group(1)}/{m.group(2)}"
    return None


def clone_at_version(repo: str, version: str, name: str, dest: Path) -> bool:
    """Shallow clone at the release tag; nothing in the repository is run."""
    for tag in (f"v{version}", version, f"{name}-{version}", f"release-{version}"):
        r = subprocess.run(["git", "-c", "core.symlinks=false", "clone", "-q", "--depth", "1", "--branch", tag, repo, str(dest)],
                           capture_output=True, text=True, timeout=180, env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"})
        if r.returncode == 0:
            shutil.rmtree(dest / ".git", ignore_errors=True)
            return True
        shutil.rmtree(dest, ignore_errors=True)
    return False


def has_compiled_sources(src: Path) -> bool:
    return any(p.suffix in COMPILED for p in src.rglob("*") if p.is_file())


def visible_half(node_ids, name: str) -> list[str]:
    key = lambda t: hashlib.sha256(f"vm-visible:{name}:{t}".encode()).hexdigest()  # noqa: E731
    ordered = sorted(node_ids, key=key)
    return sorted(ordered[: len(ordered) // 2])


def _unpack(archive: Path, dest: Path) -> Path | None:
    """Extract without links, absolute paths or '..'; returns the single top directory."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for i in z.infolist():
                p = PurePosixPath(i.filename)
                if p.is_absolute() or ".." in p.parts or (i.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                z.extract(i, dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    tops = [p for p in dest.iterdir() if p.is_dir()]
    return tops[0] if len(tops) == 1 else None


def prepare(row: dict, root: Path) -> Path | None:
    lib = root / row["name"]
    if (lib / "venv" / "bin" / "python").exists():
        return lib
    if lib.exists():
        shutil.rmtree(lib)
    cache = Path(os.environ.get("VM_SDIST_CACHE", root / "_sdists"))
    cache.mkdir(parents=True, exist_ok=True)
    archive = _download(row, cache)
    if archive is None:
        return None
    top = _unpack(archive, lib / "_unpacked")
    if top is None or has_compiled_sources(top):
        return None
    if has_tests(top):
        top.rename(lib / "src")
    else:
        repo = github_repo(row["name"])
        if repo is None or not clone_at_version(repo, row["version"], row["name"], lib / "src") or has_compiled_sources(lib / "src"):
            return None
    shutil.rmtree(lib / "_unpacked", ignore_errors=True)
    r = subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "--memory", "4g", "-v", f"{lib}:/work", "-w", "/work/src", IMAGE,
                        "timeout", "600", "sh", "-c",
                        f"python -m venv /work/venv && /work/venv/bin/pip install -q -U pip && /work/venv/bin/pip install -q -e . {TEST_PLUGINS} "
                        "&& (for f in requirements-test*.txt requirements_test*.txt test-requirements.txt requirements/test*.txt "
                        "requirements-dev.txt tests/requirements*.txt; do [ -f $f ] && /work/venv/bin/pip install -q -r $f; done; true) "
                        "&& (for g in test tests dev; do /work/venv/bin/pip install -q --group $g 2>/dev/null; done; true) "
                        "&& (/work/venv/bin/pip install -q -e '.[test]' || /work/venv/bin/pip install -q -e '.[tests]' "
                        "|| /work/venv/bin/pip install -q -e '.[testing]' || true)"],
                       capture_output=True, text=True)
    return lib if r.returncode == 0 else None


_MISSING = re.compile(r"No module named '([A-Za-z_][\w]*)")


def _install_missing(lib: Path, rounds: int = 3) -> None:
    """Collect the suite; when collection fails on a missing module, install it (network on) and try again."""
    for _ in range(rounds):
        r = subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "--network", "none", "-v", f"{lib}/src:/work/src", "-v", f"{lib}/venv:/work/venv",
                            "-w", "/work/src", IMAGE, "timeout", "120", "/work/venv/bin/python", "-m", "pytest", "--collect-only", "-q",
                            "-p", "no:cacheprovider"], capture_output=True, text=True)
        missing = sorted(set(_MISSING.findall(r.stdout + r.stderr)))
        if not missing:
            return
        subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "-v", f"{lib}:/work", IMAGE, "timeout", "300", "/work/venv/bin/pip", "install", "-q",
                        *[m.replace("_", "-") for m in missing]], capture_output=True, text=True)


def qualify(row: dict, root: Path) -> dict:
    out = {"name": row["name"], "split": row["split"], "ok": False}
    try:
        lib = prepare(row, root)
    except Exception as e:  # noqa: BLE001
        return {**out, "reason": f"prepare: {e!r}"[:200]}
    if lib is None:
        return {**out, "reason": "no archive, compiled sources or install failed"}
    r = run_pytest(lib / "src", lib / "venv", timeout=90)
    if len(r.outcomes) < 100:
        _install_missing(lib)
        r = run_pytest(lib / "src", lib / "venv", timeout=90)
    tests = sorted(r.outcomes)
    bad = [t for t, o in r.outcomes.items() if o not in ("passed", "skipped")]
    if r.timed_out or r.seconds > 60:
        return {**out, "reason": "slow", "seconds": r.seconds}
    if len(tests) < 100:
        return {**out, "reason": f"{len(tests)} tests"}
    if len(bad) > 0.05 * len(tests):
        return {**out, "reason": f"{len(bad)} tests fail on the original code"}
    passing = [t for t in tests if r.outcomes[t] == "passed"]
    if len(passing) < 100:
        return {**out, "reason": f"{len(passing)} passing tests"}
    # tests that already fail on the untouched code (at most 5%) are left out of the suite, and recorded
    return {**out, "ok": True, "n_tests": len(tests), "deselected": bad, "seconds": round(r.seconds, 1), "tests": passing,
            "visible": visible_half(passing, row["name"])}


def run(candidates: Path, root: Path, targets=None, workers: int = 80) -> list[dict]:
    targets = targets or {"train": 160, "dev": 20, "test": 20}
    rows = [json.loads(l) for l in candidates.read_text().splitlines()]
    rows = [r for r in rows if r.get("licence") and r.get("sdist_url") and not r["excluded"]
            and (r.get("sdist_size") or 0) <= MAX_ARCHIVE]
    rows.sort(key=lambda r: hashlib.sha256(f"vm-qualify:{r['name']}".encode()).hexdigest())
    done = {s: 0 for s in targets}
    results = []
    root.mkdir(parents=True, exist_ok=True)
    out = open(root / "qualified.jsonl", "a")
    with cf.ThreadPoolExecutor(workers) as ex:
        pending = set()
        it = iter(rows)
        while True:
            while len(pending) < workers:
                r = next(it, None)
                if r is None:
                    break
                if done.get(r["split"], 0) < targets.get(r["split"], 0):
                    pending.add(ex.submit(qualify, r, root))
            if not pending:
                break
            finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                q = f.result()
                results.append(q)
                out.write(json.dumps(q) + "\n")
                out.flush()
                if q["ok"] and done[q["split"]] < targets[q["split"]]:
                    done[q["split"]] += 1
            # a systematic error (every library failing the same way) stops the run instead of consuming the list
            if len(results) >= 100 and not any(r["ok"] for r in results):
                top, n = collections.Counter((r.get("reason") or "")[:60] for r in results).most_common(1)[0]
                if n >= 0.9 * len(results):
                    for f in pending:
                        f.cancel()
                    raise RuntimeError(f"systematic failure: {n} of {len(results)} libraries failed with {top!r}")
            if all(done[s] >= targets[s] for s in targets):
                break
    out.close()
    return results
