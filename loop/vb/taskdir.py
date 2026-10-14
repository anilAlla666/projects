"""A copy of a task whose verifier is a different tests/ folder.

The published Terminal-Bench 4.0 package pins a prebuilt verifier image ([verifier.environment] docker_image)
with the original tests inside it, so a replaced tests/ folder is ignored unless that pin is removed."""
from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path

from .util import ROOT

BASELINES = {name: ROOT / "templates" / "baselines" / name for name in ("always_pass", "always_fail", "network_probe")}
_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")
_PIN = re.compile(r"^\s*docker_image\s*=")


def set_verifier_offline(text: str) -> str:
    """[verifier] network_mode = "no-network": a verifier may not fetch anything (e.g. the public official
    tests) while it grades. Applied to every copy, the original-tests copy included, so all run alike."""
    out, section, done = [], None, False
    for line in text.splitlines(keepends=True):
        m = _HEADER.match(line)
        if m:
            if section == "verifier" and not done:
                out.append('network_mode = "no-network"\n')
                done = True
            section = m.group(1)
        elif section == "verifier" and re.match(r"^\s*(network_mode|allowed_hosts)\s*=", line):
            continue
        out.append(line)
    if section == "verifier" and not done:
        out.append(("" if not out or out[-1].endswith("\n") else "\n") + 'network_mode = "no-network"\n')
    result = "".join(out)
    if (tomllib.loads(result).get("verifier") or {}).get("network_mode") != "no-network":
        raise ValueError("could not set the verifier offline (no [verifier] table?)")
    return result


def unpin_verifier_image(text: str) -> str:
    out, section = [], None
    for line in text.splitlines(keepends=True):
        m = _HEADER.match(line)
        if m:
            section = m.group(1)
        if section == "verifier.environment" and _PIN.match(line):
            continue
        out.append(line)
    result = "".join(out)
    parsed = tomllib.loads(result)
    if "docker_image" in ((parsed.get("verifier") or {}).get("environment") or {}):
        raise ValueError("a verifier image pin survived unpinning (inline table?); refusing to build a copy")
    return result


def pin_verifier_image(text: str, image: str) -> str:
    """Point [verifier.environment] at an image we built ourselves (or the official one), so Harbor builds
    nothing while grading. Harbor's own build reused stale files when two test folders had files of equal size
    and timestamp, which made 'always fail' pass (swap proof, 26 Sep)."""
    out, placed = [], False
    for line in unpin_verifier_image(text).splitlines(keepends=True):
        out.append(line)
        m = _HEADER.match(line)
        if m and m.group(1) == "verifier.environment" and not placed:
            out.append(f'docker_image = "{image}"\n')
            placed = True
    result = "".join(out)
    if not placed:  # no [verifier.environment] table: add one
        result = result.rstrip("\n") + f'\n\n[verifier.environment]\ndocker_image = "{image}"\n'
    if ((tomllib.loads(result).get("verifier") or {}).get("environment") or {}).get("docker_image") != image:
        raise ValueError("could not pin the verifier image (no [verifier.environment] table?)")
    return result


def official_verifier_image(task_dir: Path) -> str:
    """The pinned official verifier image, or, where the task no longer pins one (the Terminal-Bench checkout
    updated in October dropped the pins), one we build from the task's own tests/ folder."""
    t = tomllib.loads((Path(task_dir) / "task.toml").read_text(encoding="utf-8"))
    pinned = ((t.get("verifier") or {}).get("environment") or {}).get("docker_image")
    if pinned:
        return pinned
    from .regrade import build_image
    image = build_image(Path(task_dir) / "tests")
    if image is None:
        raise RuntimeError(f"{task_dir}: the official tests' image did not build")
    return image


def make_verifier_task(task_dir: Path, tests_dir: Path, out_dir: Path, image: str | None = None) -> Path:
    """A copy of the task graded by tests_dir. With `image`, the verifier runs that prebuilt image (built from
    tests_dir by vb.regrade.build_image, or the official one); without it, Harbor builds from tests_dir."""
    task_dir, tests_dir, out_dir = Path(task_dir), Path(tests_dir), Path(out_dir)
    if not (tests_dir / "Dockerfile").is_file():
        raise ValueError(f"{tests_dir}: a verifier needs tests/Dockerfile")
    if not (tests_dir / "test.sh").is_file():
        raise ValueError(f"{tests_dir}: a verifier needs tests/test.sh")
    text = (task_dir / "task.toml").read_text(encoding="utf-8")
    if (tomllib.loads(text).get("verifier") or {}).get("environment_mode") != "separate":
        raise ValueError(f"{task_dir}: only tasks with a separate verifier environment are supported")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(task_dir, out_dir, ignore=shutil.ignore_patterns("tests", ".git"))
    shutil.copytree(tests_dir, out_dir / "tests")
    (out_dir / "tests" / "test.sh").chmod(0o755)
    toml = pin_verifier_image(text, image) if image else unpin_verifier_image(text)
    (out_dir / "task.toml").write_text(set_verifier_offline(toml), encoding="utf-8")
    return out_dir
