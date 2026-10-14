"""Candidate libraries: the most-downloaded PyPI packages, their licences, and the split each one belongs to.

The split is decided from the name alone, before anything about a library is looked at, so no later choice can
move a library between train and test."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import urllib.request
from pathlib import Path

TOP_URL = "https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json"
SPLIT_SEED = "vm-split-2026-10-08:"
EXCLUDED = frozenset({"astropy", "django", "matplotlib", "seaborn", "flask", "requests", "xarray", "pylint",
                      "pytest", "scikit-learn", "sphinx", "sympy"})
_SPDX_OK = {"MIT", "BSD-2-CLAUSE", "BSD-3-CLAUSE", "0BSD", "APACHE-2.0", "ISC"}
_CLASSIFIER_OK = {"License :: OSI Approved :: MIT License": "MIT", "License :: OSI Approved :: BSD License": "BSD",
                  "License :: OSI Approved :: Apache Software License": "Apache-2.0",
                  "License :: OSI Approved :: ISC License (ISCL)": "ISC"}
_TEXT_OK = [(re.compile(r"^\s*(the\s+)?mit(\s+license)?\s*$", re.I), "MIT"),
            (re.compile(r"^\s*(new\s+|modified\s+|simplified\s+)?bsd([\s-]*[23][\s-]*clause)?(\s+license)?\s*$", re.I), "BSD"),
            (re.compile(r"^\s*apache(\s+software)?(\s+license)?[\s,-]*(v(ersion)?\s*)?2(\.0)?\s*$", re.I), "Apache-2.0"),
            (re.compile(r"^\s*isc(\s+license)?\s*$", re.I), "ISC")]


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def split_of(name: str) -> str:
    h = int(hashlib.sha256((SPLIT_SEED + normalise(name)).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "dev" if h < 90 else "test"


def permissive(info: dict) -> str | None:
    expr = (info.get("license_expression") or "").strip()
    if expr:
        terms = [t.strip("() ").upper() for t in re.split(r"\s+(?:AND|OR|WITH)\s+", expr)]
        return expr if terms and all(t in _SPDX_OK for t in terms) else None
    found = {_CLASSIFIER_OK.get(c) for c in info.get("classifiers") or [] if c.startswith("License ::")}
    if found:
        return found.pop() if len(found) == 1 and None not in found else None
    text = (info.get("license") or "").strip()
    if 0 < len(text) <= 40:
        for pattern, name in _TEXT_OK:
            if pattern.match(text):
                return name
    return None


def _fetch_json(url: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "verifier-model-research (contact: anil@hud.ai)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _one(name: str) -> dict:
    row = {"name": normalise(name), "split": split_of(name), "excluded": normalise(name) in EXCLUDED}
    try:
        d = _fetch_json(f"https://pypi.org/pypi/{name}/json")
    except Exception as e:  # noqa: BLE001  a missing or unreachable package is recorded, not fatal
        return {**row, "error": repr(e)[:200]}
    sdists = [u for u in d.get("urls", []) if u.get("packagetype") == "sdist"]
    row.update(version=d["info"].get("version"), licence=permissive(d["info"]),
               sdist_url=sdists[0]["url"] if sdists else None, sdist_size=sdists[0]["size"] if sdists else None)
    return row


def build(out_path: Path, top_n: int = 15000, workers: int = 32) -> dict:
    names = [r["project"] for r in _fetch_json(TOP_URL)["rows"][:top_n]]
    with cf.ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, names))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    usable = [r for r in rows if r.get("licence") and r.get("sdist_url")]
    return {"candidates": len(rows), "permissive_with_sdist": len(usable),
            "by_split": {s: sum(r["split"] == s for r in usable) for s in ("train", "dev", "test")}}


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl"))))
