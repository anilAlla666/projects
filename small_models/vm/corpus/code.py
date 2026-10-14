"""Python source from permissively licensed, train-split PyPI packages. Archives are untrusted: they are read
member by member, never extracted to disk, installed or executed."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MAX_FILE = 1_000_000
MAX_ARCHIVE = 60_000_000


def _clean(name: str) -> str | None:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not name.endswith(".py"):
        return None
    parts = p.parts[1:]  # drop the archive's top directory, e.g. "pkg-1.0/"
    return "/".join(parts) if parts else None


def safe_members(path: Path):
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                rel = _clean(info.filename)
                if rel and not info.is_dir() and info.file_size <= MAX_FILE and (info.external_attr >> 16) & 0o170000 != 0o120000:
                    yield rel, z.read(info)
        return
    with tarfile.open(path) as t:
        for m in t:
            rel = _clean(m.name)
            if rel and m.isreg() and m.size <= MAX_FILE:
                f = t.extractfile(m)
                if f is not None:
                    yield rel, f.read()


def top_level_names(paths) -> set[str]:
    names = set()
    for p in paths:
        parts = PurePosixPath(p).parts
        if parts[0] in ("tests", "test", "docs", "examples", "src"):
            if parts[0] == "src" and len(parts) > 1:
                names.add(parts[1].removesuffix(".py"))
            continue
        names.add(parts[0].removesuffix(".py"))
    return names


def filter_files(files, held_out_imports: set[str]):
    """files: (package, path, text). Drops files under a directory named like a held-out import (vendored
    copies) and exact duplicates (first occurrence kept)."""
    seen, kept = set(), []
    for pkg, path, text in files:
        dirs = PurePosixPath(path).parts[:-1]
        if any(d in held_out_imports for d in dirs) or PurePosixPath(path).stem in held_out_imports and len(dirs) > 1:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append((pkg, path, text))
    return kept


def _download(row: dict, cache: Path) -> Path | None:
    if not row.get("sdist_url") or (row.get("sdist_size") or 0) > MAX_ARCHIVE:
        return None
    dest = cache / row["sdist_url"].rsplit("/", 1)[1]
    if not dest.is_file():
        req = urllib.request.Request(row["sdist_url"], headers={"User-Agent": "verifier-model-research (contact: anil@hud.ai)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def _read(row: dict, cache: Path):
    try:
        path = _download(row, cache)
        if path is None:
            return row["name"], []
        return row["name"], [(rel, data.decode("utf-8")) for rel, data in safe_members(path) if _utf8(data)]
    except Exception:  # noqa: BLE001  a broken archive drops that package; counted in the manifest
        return row["name"], None


def _utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def build(candidates: Path, out_dir: Path, workers: int = 96) -> dict:
    rows = [json.loads(l) for l in candidates.read_text().splitlines()]
    usable = [r for r in rows if r.get("licence") and r.get("sdist_url")]
    cache = out_dir / "sdists"
    cache.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        read = dict(ex.map(lambda r: _read(r, cache), usable))
    split = {r["name"]: r for r in usable}
    held_out = set()
    for name, files in read.items():
        if files and (split[name]["split"] != "train" or split[name]["excluded"]):
            held_out |= top_level_names(p for p, _ in files)
    held_out |= {n.replace("-", "_") for n in rows_names(rows, lambda r: r["split"] != "train" or r["excluded"])}
    train = [(name, p, t) for name, files in read.items() if files and split[name]["split"] == "train"
             and not split[name]["excluded"] for p, t in files]
    kept = filter_files(train, held_out)
    shard, n = 0, 0
    out = open(out_dir / f"code_{shard:03d}.jsonl", "w")
    for pkg, path, text in kept:
        out.write(json.dumps({"package": pkg, "path": path, "text": text}) + "\n")
        n += len(text)
        if n > 500_000_000:
            out.close()
            shard, n = shard + 1, 0
            out = open(out_dir / f"code_{shard:03d}.jsonl", "w")
    out.close()
    manifest = {"usable_packages": len(usable), "read_ok": sum(f is not None for f in read.values()),
                "train_packages_used": len({p for p, _, _ in kept}), "files": len(kept),
                "characters": sum(len(t) for _, _, t in kept), "held_out_import_names": sorted(held_out)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {k: v for k, v in manifest.items() if k != "held_out_import_names"}


def rows_names(rows, pred):
    return [r["name"] for r in rows if pred(r)]


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2]))))
