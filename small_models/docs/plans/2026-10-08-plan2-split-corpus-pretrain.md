# Plan 2: Library split, pretraining corpus, tokenizer and pretraining Implementation Plan

**Goal:** Freeze the library split, build a licence-clean pretraining corpus (Python from permissively licensed PyPI packages, English from FineWeb-Edu) that excludes every dev and test library, train the 32k tokenizer, and pretrain the MAIN config to 20 tokens per parameter on 4 H100s.

**Architecture:** Candidates come from the 15,000 most-downloaded PyPI packages; each gets a split from a seeded hash of its normalised name, decided before any library is looked at. Only train-split, permissive packages enter the corpus, and vendored copies of dev or test libraries are filtered by import name. Tokenised text is stored as uint16 streams that a seeded sampler reads in windows. Pretraining is `torchrun` data-parallel with `torch.compile`, bfloat16 autocast, Muon and AdamW, checkpoints and a held-out validation loss.

**Tech Stack:** Python 3.12, PyTorch 2.11, flash-linear-attention 0.5.2, Hugging Face `tokenizers`, `huggingface_hub`, `pyarrow`, `zstandard`, pytest. Instance: 4x H100 80GB, 104 vCPUs, `ubuntu@192.222.52.170`.

**Spec:** `docs/specs/2026-10-08-verifier-judge-design.md` (section 5, sources and splits; section 6, stages 1 and 2). MAIN config chosen by the user on 8 Oct after `docs/stage0_gpu_report.md`.

## Global Constraints

- Hard cap $100 (`docs/cost_ledger.md`). The 4x H100 instance costs about $12 per hour; nothing GPU-heavy idles.
- Split rule, fixed now: `sha256("vm-split-2026-10-08:" + normalised name)` read as an integer, modulo 100: below 80 train, 80 to 89 dev, 90 and above test.
- Permissive licences only: MIT, BSD (2- and 3-clause, 0BSD), Apache-2.0, ISC. Anything else, any compound expression containing anything else, or an empty licence is rejected.
- Excluded from the corpus whatever their split: the real-attempts repositories (SWE-bench Lite's twelve: astropy, django, matplotlib, seaborn, flask, requests, xarray, pylint, pytest, scikit-learn, sphinx, sympy) and every dev- and test-split package.
- Downloaded archives are untrusted: extracted with a path-traversal and symlink guard, never installed or executed in this plan.
- Tokenizer: byte-level BPE, vocabulary exactly 32,768, special tokens `<|eos|> <|pad|> <|task|> <|diff|> <|evidence|> <|untrusted|> <|line|> <|verdict|>`.
- Pretraining: MAIN (116,490,541 parameters), 2.33B tokens, 60% code and 40% English, sequence 2,048 for the first 95% of tokens and 8,192 for the last 5%.

## File structure

| File | Responsibility |
|---|---|
| `vm/libs/candidates.py` | candidate list, licence check, split rule, exclusions |
| `vm/corpus/code.py` | download sdists, safe extraction, vendored and duplicate filters, code shards |
| `vm/corpus/english.py` | FineWeb-Edu parquet to text shards |
| `vm/corpus/tok.py` | tokenizer training, parallel tokenisation to uint16 streams |
| `vmt/data.py` | `TokenStream`, `MixedSampler` |
| `vmt/pretrain.py` | the `torchrun` pretraining script |
| `configs/pretrain_main.toml`, `configs/pretrain_dev_lr*.toml` | run configs |
| `tests/test_candidates.py`, `tests/test_code_corpus.py`, `tests/test_tok.py` | CPU tests (run anywhere) |
| `tests_gpu/test_data_t.py`, `tests_gpu/test_pretrain_t.py` | tests on the instance |

---

### Task 1: Candidates, licences and the split

**Files:** Create `vm/libs/__init__.py`, `vm/libs/candidates.py`. Test `tests/test_candidates.py`.

**Interfaces:**
- Produces: `normalise(name) -> str` (PEP 503); `split_of(name) -> "train" | "dev" | "test"`; `permissive(info: dict) -> str | None` (the licence it accepted, or None); `EXCLUDED: frozenset[str]`; `build(out_path, top_n=15000, workers=32) -> dict` writing JSON lines `{name, version, licence, split, excluded, sdist_url, sdist_size}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_candidates.py`:
```python
from vm.libs.candidates import EXCLUDED, normalise, permissive, split_of


def info(license=None, expr=None, classifiers=()):
    return {"license": license, "license_expression": expr, "classifiers": list(classifiers)}


def test_normalise_follows_pep_503():
    assert normalise("Scikit_Learn") == "scikit-learn"
    assert normalise("zope.interface") == "zope-interface"


def test_permissive_licences_are_accepted():
    assert permissive(info(expr="MIT")) == "MIT"
    assert permissive(info(expr="Apache-2.0")) == "Apache-2.0"
    assert permissive(info(expr="BSD-3-Clause OR MIT")) is not None
    assert permissive(info(classifiers=["License :: OSI Approved :: BSD License"])) == "BSD"
    assert permissive(info(license="MIT License")) == "MIT"


def test_anything_else_is_rejected():
    assert permissive(info(expr="GPL-3.0-only")) is None
    assert permissive(info(expr="MIT AND GPL-2.0-or-later")) is None
    assert permissive(info(classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])) is None
    assert permissive(info(license="Proprietary")) is None
    assert permissive(info()) is None


def test_split_is_deterministic_and_proportioned():
    names = [f"pkg{i}" for i in range(20000)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]
    assert 0.78 < splits.count("train") / len(names) < 0.82
    assert 0.08 < splits.count("dev") / len(names) < 0.12
    assert split_of("Foo_Bar") == split_of("foo-bar")


def test_real_attempt_repositories_are_excluded():
    assert {"django", "requests", "sympy", "scikit-learn"} <= EXCLUDED
```

- [ ] **Step 2: Run to verify failure:** `.venv/bin/pytest tests/test_candidates.py -q` on the Mac. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`vm/libs/__init__.py`: empty.

`vm/libs/candidates.py`:
```python
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
```

- [ ] **Step 4: Run tests:** `.venv/bin/pytest tests/test_candidates.py -q`. Expected: `5 passed`.

- [ ] **Step 5: Build the candidate list on the instance:** `sh scripts/sync.sh` (with `HOST=ubuntu@192.222.52.170`), then `ssh ... 'cd verifier_model && ~/vmenv/bin/python -m vm.libs.candidates ~/data/candidates.jsonl'`. Expected: a JSON summary; copy `~/data/candidates.jsonl` back to `data/candidates.jsonl` and commit it, so the split is on record before any library is opened.

- [ ] **Step 6: Commit:** `git add vm/libs tests/test_candidates.py data/candidates.jsonl` with message "Candidate libraries and the split, frozen before any library is opened".

---

### Task 2: The code corpus

**Files:** Create `vm/corpus/__init__.py`, `vm/corpus/code.py`. Test `tests/test_code_corpus.py`.

**Interfaces:**
- Consumes: `data/candidates.jsonl`.
- Produces: `safe_members(archive_path) -> iterator of (relative_path, bytes)` (Python files only, no absolute paths, no `..`, no links, at most 1 MB each); `top_level_names(paths) -> set[str]`; `build(candidates, out_dir, workers=96) -> dict` writing `out_dir/code_*.jsonl` lines `{"package", "path", "text"}` for train-split, permissive, non-excluded packages, dropping exact duplicates and any file whose path has a directory component equal to an import name of a dev-, test-split or excluded package; plus `out_dir/manifest.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_code_corpus.py`:
```python
import io
import json
import tarfile

from vm.corpus.code import filter_files, safe_members, top_level_names


def make_tar(tmp_path, files, name="pkg-1.0.tar.gz", link=None):
    p = tmp_path / name
    with tarfile.open(p, "w:gz") as t:
        for path, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(path)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        if link:
            ti = tarfile.TarInfo(link)
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/passwd"
            t.addfile(ti)
    return p


def test_extraction_refuses_traversal_links_and_non_python(tmp_path):
    p = make_tar(tmp_path, {"pkg-1.0/pkg/a.py": "x = 1\n", "../evil.py": "bad\n", "/abs.py": "bad\n",
                            "pkg-1.0/README.md": "readme"}, link="pkg-1.0/pkg/link.py")
    got = dict(safe_members(p))
    assert list(got) == ["pkg/a.py"]


def test_top_level_names():
    assert top_level_names(["six.py", "pkg/__init__.py", "pkg/sub/x.py", "tests/t.py"]) == {"six", "pkg"}


def test_filter_drops_vendored_held_out_code_and_duplicates():
    files = [("good", "good/a.py", "a = 1\n"), ("good", "good/_vendor/six/x.py", "s = 1\n"),
             ("other", "other/b.py", "a = 1\n"), ("other", "other/c.py", "c = 2\n")]
    kept = filter_files(files, held_out_imports={"six"})
    assert [(p, path) for p, path, _ in kept] == [("good", "good/a.py"), ("other", "other/c.py")]


def test_filter_has_teeth():
    """Without the held-out name, the vendored file stays: the filter is what removes it."""
    files = [("good", "good/_vendor/six/x.py", "s = 1\n")]
    assert filter_files(files, held_out_imports=set()) == files
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`vm/corpus/__init__.py`: empty.

`vm/corpus/code.py`:
```python
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
```

- [ ] **Step 4: Run tests.** Expected: `4 passed`.

- [ ] **Step 5: Build on the instance:** `ssh ... 'cd verifier_model && ~/vmenv/bin/python -m vm.corpus.code ~/data/candidates.jsonl ~/data/code'`. Copy `manifest.json` back to `data/code_manifest.json`.

- [ ] **Step 6: Commit** `vm/corpus tests/test_code_corpus.py data/code_manifest.json`: "Code corpus from train-split permissive packages; vendored held-out libraries and duplicates dropped".

---

### Task 3: English text, tokenizer and token streams

**Files:** Create `vm/corpus/english.py`, `vm/corpus/tok.py`. Test `tests/test_tok.py`.

**Interfaces:**
- Produces: `english.build(parquet_dir, out_dir, max_chars) -> dict` writing `english_*.jsonl` (`{"text"}`); `tok.SPECIAL` (the eight special tokens, ids 0 to 7); `tok.train_tokenizer(texts_iter, out_path, vocab=32768)`; `tok.encode_jsonl(tokenizer_path, jsonl_paths, out_bin, val_bin, val_fraction=0.005, workers=96) -> dict` writing uint16 token streams with `<|eos|>` after each document; validation documents chosen by a seeded hash of the document.

- [ ] **Step 1: Write the failing tests**

`tests/test_tok.py`:
```python
import json

import numpy as np

from vm.corpus.tok import SPECIAL, encode_jsonl, load, train_tokenizer

import random

random.seed(0)
WORDS = ["cat", "mat", "sat", "value", "return", "class", "def", "import", "numpy", "result", "token", "data"]
TEXTS = [f"def f{i}(x{i}):\n    return x{i} + {i}\n" if i % 2 else
         " ".join(random.choice(WORDS) + str(random.randint(0, 999)) for _ in range(12)) + ".\n" for i in range(3000)]


def test_tokenizer_round_trips_code_and_has_exact_special_ids(tmp_path):
    train_tokenizer(iter(TEXTS), tmp_path / "tok.json", vocab=600)
    tok = load(tmp_path / "tok.json")
    for s in ("def f(x):\n    return x + 1\n", "  indented\ttabs\n"):
        assert tok.decode(tok.encode(s).ids) == s
    assert [tok.token_to_id(s) for s in SPECIAL] == list(range(len(SPECIAL)))
    assert tok.get_vocab_size() == 600


def test_encode_writes_streams_with_eos(tmp_path):
    train_tokenizer(iter(TEXTS), tmp_path / "tok.json", vocab=600)
    src = tmp_path / "a.jsonl"
    src.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS[:50]))
    stats = encode_jsonl(tmp_path / "tok.json", [src], tmp_path / "a.bin", tmp_path / "a_val.bin", val_fraction=0.1, workers=2)
    train = np.fromfile(tmp_path / "a.bin", dtype=np.uint16)
    val = np.fromfile(tmp_path / "a_val.bin", dtype=np.uint16)
    assert stats["documents"] == 50
    assert (train == 0).sum() + (val == 0).sum() == 50  # one <|eos|> per document
    assert len(val) > 0 and len(train) > len(val)
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`vm/corpus/english.py`:
```python
"""FineWeb-Edu (ODC-By) parquet files to JSON lines of text, up to a character budget."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def build(parquet_dir: Path, out_dir: Path, max_chars: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chars = n_docs = 0
    with open(out_dir / "english_000.jsonl", "w") as out:
        for f in sorted(parquet_dir.rglob("*.parquet")):
            for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    out.write(json.dumps({"text": text}) + "\n")
                    n_chars += len(text)
                    n_docs += 1
                    if n_chars >= max_chars:
                        return {"documents": n_docs, "characters": n_chars}
    return {"documents": n_docs, "characters": n_chars}


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2]), int(float(sys.argv[3])))))
```

`vm/corpus/tok.py`:
```python
"""The 32k byte-level BPE tokenizer and parallel encoding into uint16 token streams."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<|eos|>", "<|pad|>", "<|task|>", "<|diff|>", "<|evidence|>", "<|untrusted|>", "<|line|>", "<|verdict|>"]


def train_tokenizer(texts, out_path: Path, vocab: int = 32768) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))


def load(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def _is_val(text: str, val_fraction: float) -> bool:
    return int(hashlib.sha256(text[:4096].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_fraction


def _encode_chunk(args):
    tok_path, lines, val_fraction = args
    tok = load(Path(tok_path))
    texts = [json.loads(l)["text"] for l in lines]
    enc = tok.encode_batch(texts)
    train, val = [], []
    for text, e in zip(texts, enc):
        (val if _is_val(text, val_fraction) else train).append(np.array(e.ids + [0], dtype=np.uint16))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.uint16)  # noqa: E731
    return cat(train), cat(val), len(texts)


def encode_jsonl(tok_path: Path, jsonl_paths, out_bin: Path, val_bin: Path, val_fraction: float = 0.005,
                 workers: int = 96, chunk: int = 2000) -> dict:
    def chunks():
        for p in jsonl_paths:
            buf = []
            with open(p) as f:
                for line in f:
                    buf.append(line)
                    if len(buf) == chunk:
                        yield (str(tok_path), buf, val_fraction)
                        buf = []
            if buf:
                yield (str(tok_path), buf, val_fraction)

    n_docs = n_train = n_val = 0
    with open(out_bin, "wb") as ft, open(val_bin, "wb") as fv, mp.get_context("spawn").Pool(workers) as pool:
        for train, val, docs in pool.imap(_encode_chunk, chunks()):
            ft.write(train.tobytes())
            fv.write(val.tobytes())
            n_docs, n_train, n_val = n_docs + docs, n_train + len(train), n_val + len(val)
    return {"documents": n_docs, "train_tokens": n_train, "val_tokens": n_val}
```

- [ ] **Step 4: Run tests.** Expected: `2 passed`.

- [ ] **Step 5: Build on the instance.**
  1. English: `python -m vm.corpus.english ~/data/fineweb_edu ~/data/english 4.5e9` (about 1.0B tokens at roughly 4.5 characters per token).
  2. Tokenizer: train on the first 400 MB of code and 400 MB of English (a short driver in `scripts/build_tokenizer.py` that reads both JSON-lines sets and calls `train_tokenizer`), saving `~/data/tokenizer.json`; copy it to `data/tokenizer.json` and commit it.
  3. Encode: `encode_jsonl` for code (`~/data/code/code_*.jsonl` to `code.bin`, `code_val.bin`) and English (`english.bin`, `english_val.bin`). Record the token counts in `data/corpus_stats.json`.
  4. If code has fewer than 1.4B training tokens, record how many epochs of code the run will see; do not add unlicensed sources.

- [ ] **Step 6: Commit** `vm/corpus tests/test_tok.py scripts/build_tokenizer.py data/tokenizer.json data/corpus_stats.json`: "English shards, the 32k tokenizer and token streams; corpus statistics recorded".

---

### Task 4: Sampling and the pretraining script

**Files:** Create `vmt/data.py`, `vmt/pretrain.py`, `configs/pretrain_main.toml`, `configs/pretrain_tiny.toml`. Tests `tests_gpu/test_data_t.py`, `tests_gpu/test_pretrain_t.py`.

**Interfaces:**
- Consumes: `Judge`, `build_optimizers`, `lm_loss_with`, `vm.model.config`.
- Produces: `TokenStream(path)` (memory-mapped uint16); `MixedSampler(streams: dict[str, TokenStream], weights: dict[str, float], seed: int, rank: int)` with `batch(batch_size, seq) -> LongTensor [B, seq + 1]`; `vmt/pretrain.py CONFIG` run under `torchrun`, writing `experiments/pretrain/<name>/<stamp>/` with `log.jsonl` (step, tokens, loss, lr, tokens per second), `val.jsonl` (step, loss per source), `ckpt_<step>.pt`, `final.pt`; `--resume <run dir>` continues from the newest checkpoint.

- [ ] **Step 1: Write the failing tests**

`tests_gpu/test_data_t.py`:
```python
import numpy as np
import torch

from vmt.data import MixedSampler, TokenStream


def stream(tmp_path, name, value, n=100_000):
    p = tmp_path / f"{name}.bin"
    np.full(n, value, dtype=np.uint16).tofile(p)
    return TokenStream(p)


def test_mixing_ratio_and_shapes(tmp_path):
    s = MixedSampler({"code": stream(tmp_path, "code", 7), "en": stream(tmp_path, "en", 9)}, {"code": 0.6, "en": 0.4}, seed=0, rank=0)
    b = s.batch(1000, 64)
    assert b.shape == (1000, 65) and b.dtype == torch.long
    frac_code = (b[:, 0] == 7).float().mean().item()
    assert 0.55 < frac_code < 0.65


def test_ranks_draw_different_batches(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "r.bin"
    rng.integers(0, 30000, 200_000).astype(np.uint16).tofile(p)
    st = {"x": TokenStream(p)}
    a = MixedSampler(st, {"x": 1.0}, seed=0, rank=0).batch(4, 32)
    b = MixedSampler(st, {"x": 1.0}, seed=0, rank=1).batch(4, 32)
    assert not torch.equal(a, b)
    assert torch.equal(a, MixedSampler(st, {"x": 1.0}, seed=0, rank=0).batch(4, 32))
```

`tests_gpu/test_pretrain_t.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_run_learns_checkpoints_and_resumes(tmp_path):
    rng = np.random.default_rng(0)
    pattern = np.tile(np.arange(1, 50, dtype=np.uint16), 20_000)
    for name in ("code", "english"):
        pattern.tofile(tmp_path / f"{name}.bin")
        rng.integers(1, 50, 5000).astype(np.uint16).tofile(tmp_path / f"{name}_val.bin")
    cfg = (ROOT / "configs/pretrain_tiny.toml").read_text().replace("DATA_DIR", str(tmp_path)).replace("OUT_DIR", str(tmp_path / "out"))
    (tmp_path / "tiny.toml").write_text(cfg)
    run = lambda *extra: subprocess.run(  # noqa: E731
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(ROOT / "vmt/pretrain.py"), str(tmp_path / "tiny.toml"), *extra],
        cwd=ROOT, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stderr[-3000:]
    run_dir = next((tmp_path / "out").glob("*/*"))
    log = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < 0.5 * log[0]["loss"]
    assert (run_dir / "final.pt").is_file() and list(run_dir.glob("ckpt_*.pt"))
    assert (run_dir / "val.jsonl").read_text().strip()
    r2 = run("--resume", str(run_dir), "--extra-steps", "5")
    assert r2.returncode == 0, r2.stderr[-3000:]
    log2 = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    assert log2[-1]["step"] == log[-1]["step"] + 5
```

- [ ] **Step 2: Sync and run to verify failure.** Expected: `ModuleNotFoundError: vmt.data`.

- [ ] **Step 3: Implement**

`vmt/data.py`:
```python
"""Token streams (memory-mapped uint16) and a seeded sampler that mixes sources by weight. Each sequence is a
random window of one source; documents inside it are separated by <|eos|> (id 0)."""
from __future__ import annotations

import numpy as np
import torch


class TokenStream:
    def __init__(self, path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.tokens)


class MixedSampler:
    def __init__(self, streams: dict, weights: dict, seed: int, rank: int):
        self.names = sorted(streams)
        self.streams = streams
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng([seed, rank])

    def batch(self, batch_size: int, seq: int) -> torch.Tensor:
        src = self.rng.choice(len(self.names), size=batch_size, p=self.p)
        out = np.empty((batch_size, seq + 1), dtype=np.int64)
        for i, s in enumerate(src):
            toks = self.streams[self.names[s]].tokens
            start = self.rng.integers(0, len(toks) - seq - 1)
            out[i] = toks[start : start + seq + 1]
        return torch.from_numpy(out)
```

`configs/pretrain_tiny.toml`:
```toml
name = "tiny"
model = "TINY"
data_dir = "DATA_DIR"
out_dir = "OUT_DIR"
sources = {code = 0.6, english = 0.4}
tokens = 2_000_000
batch_per_gpu = 8
seq = 128
long_seq = 256
long_frac = 0.1
lr_muon = 0.02
lr_adam = 0.003
warmup_frac = 0.05
weight_decay = 0.01
grad_clip = 1.0
compile = false
log_every = 5
val_every = 20
val_batches = 2
ckpt_every = 40
seed = 0
```

`configs/pretrain_main.toml`:
```toml
name = "main"
model = "MAIN"
data_dir = "/home/ubuntu/data/tokens"
out_dir = "/home/ubuntu/verifier_model/experiments/pretrain"
sources = {code = 0.6, english = 0.4}
tokens = 2_330_000_000
batch_per_gpu = 32
seq = 2048
long_seq = 8192
long_frac = 0.05
lr_muon = 0.02
lr_adam = 0.003
warmup_frac = 0.02
weight_decay = 0.01
grad_clip = 1.0
compile = true
log_every = 20
val_every = 500
val_batches = 20
ckpt_every = 1000
seed = 0
```

`vmt/pretrain.py`:
```python
"""Pretraining under torchrun: data-parallel, bfloat16 autocast, Muon and AdamW, held-out validation per source,
checkpoints and resume. The last long_frac of tokens use long_seq with the same tokens per step.

    torchrun --nproc_per_node 4 vmt/pretrain.py configs/pretrain_main.toml [--resume RUN_DIR --extra-steps N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.model import config as model_config  # noqa: E402
from vmt.data import MixedSampler, TokenStream  # noqa: E402
from vmt.model import Judge  # noqa: E402
from vmt.train import build_optimizers, lm_loss_with  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--extra-steps", type=int, default=0)
    a = ap.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    distributed.init_process_group("nccl")
    rank, world = distributed.get_rank(), distributed.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(cfg["seed"])

    mcfg = getattr(model_config, cfg["model"])
    model = Judge(mcfg).cuda()
    # the verdict and pointer heads are trained later; frozen here so data-parallel sees no unused parameters
    for p in [*model.verdict_head.parameters(), *model.pointer_query.parameters()]:
        p.requires_grad_(False)
    tokens_per_step = cfg["batch_per_gpu"] * cfg["seq"] * world
    total_steps = cfg["tokens"] // tokens_per_step + a.extra_steps
    long_from = int(total_steps * (1 - cfg["long_frac"]))
    opts, scheds = build_optimizers(model, cfg["lr_muon"], cfg["lr_adam"], total_steps,
                                    max(1, int(total_steps * cfg["warmup_frac"])), cfg["weight_decay"])
    step = 0
    if a.resume:
        run_dir = a.resume
        ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        state = torch.load(ckpts[-1], map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(scheds, state["scheds"]):
            sc.load_state_dict(s)
        step = state["step"]
        total_steps = step + a.extra_steps if a.extra_steps else total_steps
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(cfg["out_dir"]) / cfg["name"] / stamp
        if rank == 0:
            run_dir.mkdir(parents=True)
            (run_dir / "config.toml").write_text(a.config.read_text())
    ddp = DDP(model, device_ids=[local])
    run_model = torch.compile(ddp) if cfg["compile"] else ddp

    data = Path(cfg["data_dir"])
    train = MixedSampler({n: TokenStream(data / f"{n}.bin") for n in cfg["sources"]}, cfg["sources"], cfg["seed"] + step, rank)
    vals = {n: MixedSampler({n: TokenStream(data / f"{n}_val.bin")}, {n: 1.0}, 12345, rank) for n in cfg["sources"]}
    log = open(run_dir / "log.jsonl", "a") if rank == 0 else None
    t0, tok0 = time.perf_counter(), 0

    def save(name):
        if rank == 0:
            torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                        "scheds": [s.state_dict() for s in scheds], "step": step, "config": cfg}, run_dir / name)

    while step < total_steps:
        seq = cfg["long_seq"] if step >= long_from else cfg["seq"]
        bsz = max(1, cfg["batch_per_gpu"] * cfg["seq"] // seq)
        batch = train.batch(bsz, seq).cuda(non_blocking=True)
        seg = torch.zeros_like(batch)
        loss = lm_loss_with(run_model, model, batch, seg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        for s in scheds:
            s.step()
        step += 1
        tok0 += bsz * seq * world
        if step % cfg["log_every"] == 0 or step == total_steps:
            lv = loss.detach()
            distributed.all_reduce(lv, op=distributed.ReduceOp.AVG)
            if rank == 0:
                el = time.perf_counter() - t0
                log.write(json.dumps({"step": step, "tokens": step * tokens_per_step, "loss": round(lv.item(), 4),
                                      "lr_muon": scheds[0].get_last_lr()[0], "seq": seq,
                                      "tokens_per_s": round(tok0 / el)}) + "\n")
                log.flush()
                t0, tok0 = time.perf_counter(), 0
        if step % cfg["val_every"] == 0 or step == total_steps:
            model.eval()
            res = {}
            with torch.no_grad():
                for n, v in vals.items():
                    ls = [lm_loss_with(model, model, v.batch(max(1, bsz // 4), cfg["seq"]).cuda(), torch.zeros(max(1, bsz // 4), cfg["seq"] + 1, dtype=torch.long, device="cuda"))
                          for _ in range(cfg["val_batches"])]
                    x = torch.stack(ls).mean()
                    distributed.all_reduce(x, op=distributed.ReduceOp.AVG)
                    res[n] = round(x.item(), 4)
            model.train()
            if rank == 0:
                with open(run_dir / "val.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, **res}) + "\n")
        if step % cfg["ckpt_every"] == 0:
            save(f"ckpt_{step}.pt")
    save("final.pt")
    if rank == 0:
        save(f"ckpt_{step}.pt")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Sync and run** `tests_gpu/test_data_t.py tests_gpu/test_pretrain_t.py`. Expected: `3 passed`.

- [ ] **Step 5: Commit** `vmt/data.py vmt/pretrain.py configs tests_gpu/test_data_t.py tests_gpu/test_pretrain_t.py`: "Mixed sampler and the torchrun pretraining script: checkpoints, per-source validation, resume".

---

### Task 5: Learning-rate check on DEV, then pretrain MAIN

- [ ] **Step 1: LR check.** Four single-GPU DEV runs in parallel (`CUDA_VISIBLE_DEVICES=0..3`, `--nproc_per_node 1`), 150M tokens each, `batch_per_gpu = 64`, sequence 2,048, compiled: Muon learning rate 0.01, 0.02 and 0.04 with AdamW 0.003, and Muon 0.02 with AdamW 0.001. Configs `configs/pretrain_dev_lr{1,2,3,4}.toml`. Pick the lowest final validation loss averaged over the two sources; record all four in `docs/pretrain_report.md`.
- [ ] **Step 2: Pretrain MAIN** with the chosen learning rates on all four GPUs: `torchrun --nproc_per_node 4 vmt/pretrain.py configs/pretrain_main.toml`, in the background with its log written to a file. Watch for the first 200 steps (loss falling, tokens per second near 4 x the single-H100 rate, no out-of-memory), then check on it by its log file, never by matching a process name.
- [ ] **Step 3: After the run,** copy `log.jsonl`, `val.jsonl` and `final.pt` back, write the loss curves into `docs/pretrain_report.md`, update the cost ledger, commit, and report to the user. The instance is then used for plan 3 (judge data) or terminated, as the user decides.
