# Verifier Bench M1 (Terminal-Bench 4.0, five tasks) Implementation Plan

**Goal:** Score verifiers written by four writers (GPT-6 Luna and Claude Sonnet 5, each in its native harness and both in Harbor's shared `terminus-2` harness) on five Terminal-Bench 4.0 tasks, so that the model effect and the harness effect can be read separately. Each verifier is graded against leaderboard attempts whose pass/fail label comes from the task's own human-written tests.

**Architecture:**
- Everything runs on the Hetzner server through the Harbor CLI.
- Pools are downloaded leaderboard attempts (plus Harbor's `oracle` and `nop` runs), checked by regrading them with the original tests.
- A verifier is a replacement `tests/` folder. It is scored by `harbor job regrade` on a copy of the task whose pinned verifier image has been removed.
- Writers run as Harbor jobs through the HUD gateway, with the solver machine's network limited to the gateway. There are four: Luna in Codex, Sonnet in Claude Code, and both in `terminus-2`.

**Tech Stack:** Python 3.12 standard library on the server (`/usr/bin/python3`), pytest locally, Harbor 0.23.0 (`/root/venv/bin/harbor`), Docker, HUD inference gateway (`https://inference.hud.ai`).

**Spec:** `docs/design.md` (revision 2)

## Global Constraints

- **Repository and server:**
  - Repo: `/Users/anilkumar/verifier_bench`, synced to `root@178.156.248.194:/root/vbench/repo` by `tools/sync.sh`.
  - Data: `/root/vbench/{meta,pools,vtasks,writer_tasks,verifiers,runs}`, never inside the repo.
- **Commits:** authored `Anil Kumar <anil@hud.ai>` (already set in the repo config), no `Co-Authored-By` line, nothing pushed.
- **Dependencies:** `vb/` uses the standard library only (the server's `/usr/bin/python3` is 3.12 with `tomllib`); pytest runs locally.
- **Task sources:** Terminal-Bench 4.0 tasks at `/root/tb4/terminal-bench/<task>/`, never modified. The leaderboard is `9f966760-00f1-424e-90f5-c964fb6f6091`.
- **Verdicts:** reward 1 is `pass`; any other number is `fail`; no reward, an exception or a failed build is `error`, which counts as `fail`.
- **Pool per task:** 10 `pass` from at least 4 leaderboard runs; 10 fails, `near_miss` first (a failed attempt passing at least 80% of official tests); 1 `reference` (oracle); 1 `nop`.
  - Only entries whose recorded verdict the unpinned original tests reproduce are kept.
- **Writers:** four model × harness pairs (spec §6.1):

  | Writer | Harness | Model | Settings |
  |---|---|---|---|
  | `sonnet-claude-code` | `claude-code` (inside the container) | `anthropic/claude-sonnet-5` | `disallowed_tools=WebFetch,WebSearch`, `max_budget_usd=5`, gateway via `--ae` |
  | `luna-codex` | `codex` (inside the container) | `openai/gpt-6-luna` | `web_search=disabled`, gateway via `--ae` |
  | `sonnet-terminus` | `terminus-2` (drives the container from the host) | `anthropic/claude-sonnet-5` | `api_base=https://inference.hud.ai`, key in the host process environment |
  | `luna-terminus` | `terminus-2` | `openai/gpt-6-luna` | `api_base=https://inference.hud.ai/v1`, key in the host process environment |

  Each writer makes 2 attempts per task. The solver machine's network allowlist is `inference.hud.ai` only.
- **Verifiers grade offline:** every task copy sets `[verifier] network_mode = "no-network"`, including the original-tests copy. A written verifier found by the leak scan to copy or fetch the official tests is excluded from its writer's numbers.
- **Baselines** on every evaluation, or the run is void:
  - original tests (unpinned, offline): BA = 1;
  - always pass: TPR 1, TNR 0, BA 0.5;
  - always fail: TPR 0, TNR 1, BA 0.5 (proves the swap took effect);
  - network probe: passes nothing (proves grading is offline).
- **Intervals:** every writer's BA carries a 95% bootstrap interval over task-attempts. A model or harness effect is stated only where intervals do not overlap.
- **Spend cap for M1 writer runs:** $100 of the $500. A one-task smoke test of all four writers (under $8) comes before any batch.
- **Secrets:** the HUD key lives only in `/root/vbench/.env` (mode 600). It is passed to agents as the template `${HUD_API_KEY}`, never as a literal in argv, a file in the repo or chat.

---

## File Structure

```
verifier_bench/
  .gitignore  pytest.ini  README.md
  tools/sync.sh                      rsync repo to the server
  vb/__init__.py
  vb/util.py                         paths, JSON IO, run dirs, .env loading, Harbor calls
  vb/scorer.py                       TPR/TNR/BA, per-source TNR, flakiness, macro, baseline checks
  vb/taskdir.py                      unpin the verifier image; build a task copy with a replacement tests/
  vb/regrade.py                      run `harbor job regrade`; read rewards per source attempt
  vb/index.py                        index every leaderboard attempt with its reward
  vb/select.py                       choose tasks by rule
  vb/pool.py                         download attempts, oracle/nop runs, classify, label check
  vb/writer.py                       writer task dirs, writer runs, verifier collection
  vb/evaluate.py                     regrade every verifier on every pool, score, baselines
  vb/report.py                       summary.md
  templates/writer_instruction.md
  templates/baselines/{always_pass,always_fail}/{Dockerfile,test.sh}
  templates/netprobe/                one Harbor task that checks what the agent network can reach
  configs/m1.json                    selected tasks, rules, rejected tasks with reasons
  configs/excluded_tasks.json        tasks excluded for community-reported test problems
  tests/test_*.py
```

---

### Task 1: Scaffold, sync and the scorer

**Files:**
- Create: `.gitignore`, `pytest.ini`, `tools/sync.sh`, `vb/__init__.py`, `vb/util.py`, `vb/scorer.py`
- Test: `tests/test_scorer.py`, `tests/test_util.py`

**Interfaces:**
- Produces:
  - `vb.util`: `ROOT`, `DATA`, `TB_ROOT`, `HARBOR`
  - `vb.util`: `read_json(path)`, `write_json(path, obj)`, `git_sha() -> str`, `new_run_dir(name, root=DATA/"runs") -> Path`, `load_env(path=DATA/".env") -> list[str]` (returns the names loaded, never values), `harbor_json(*args) -> Any`
  - `vb.scorer`: `Verdict(entry_id, source, label, verdict)`, `score_task(list[Verdict]) -> dict` (keys `n_correct, n_wrong, tpr, tnr, ba, tnr_by_source, flakiness`), `bootstrap_ci(values) -> (lo, hi) | None`, `macro(list[dict]) -> dict` (adds `ba_ci`), `check_baselines(always_pass, always_fail, original, network_probe=None) -> list[str]`

- [ ] **Step 1: Write config files and the sync script**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
```

`tools/sync.sh`:
```sh
#!/bin/sh
# Copy the repo to the server. Data lives in /root/vbench, outside the synced folder.
set -e
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes root@178.156.248.194 'mkdir -p /root/vbench/repo'
rsync -az --delete --exclude .venv --exclude __pycache__ ./ root@178.156.248.194:/root/vbench/repo/
```

Run:
```bash
cd /Users/anilkumar/verifier_bench && chmod +x tools/sync.sh && (test -x .venv/bin/python || python3.12 -m venv .venv) && .venv/bin/pip install -q pytest && .venv/bin/python -c "import pytest; print('ok')"
```
Expected: `ok`

- [ ] **Step 2: Write the failing tests**

`tests/test_util.py`:
```python
import os

from vb.util import load_env, new_run_dir, read_json, write_json


def test_json_round_trip(tmp_path):
    write_json(tmp_path / "a" / "b.json", {"x": "é", "n": [1, 2]})
    assert read_json(tmp_path / "a" / "b.json") == {"x": "é", "n": [1, 2]}


def test_run_dirs_are_never_reused(tmp_path):
    a = new_run_dir("exp", root=tmp_path)
    b = new_run_dir("exp", root=tmp_path)
    assert a != b and a.parent == b.parent == tmp_path / "exp"


def test_load_env_sets_values_and_returns_only_names(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_TEST_SECRET", raising=False)
    (tmp_path / ".env").write_text("# comment\nVB_TEST_SECRET=abc=def\n\n")
    assert load_env(tmp_path / ".env") == ["VB_TEST_SECRET"]
    assert os.environ["VB_TEST_SECRET"] == "abc=def"
    assert load_env(tmp_path / "missing.env") == []
```

`tests/test_scorer.py`:
```python
import pytest

from vb.scorer import Verdict, bootstrap_ci, check_baselines, macro, score_task


def table(rows):
    return [Verdict(f"e{i}", src, lab, ver) for i, (src, lab, ver) in enumerate(rows)]


def test_mixed_table():
    s = score_task(table([("pass", "pass", "pass"), ("pass", "pass", "fail"), ("reference", "pass", "pass"),
                          ("near_miss", "fail", "pass"), ("real_fail", "fail", "fail"), ("nop", "fail", "error")]))
    assert (s["n_correct"], s["n_wrong"]) == (3, 3)
    assert s["tpr"] == pytest.approx(2 / 3) and s["tnr"] == pytest.approx(2 / 3)
    assert s["ba"] == pytest.approx(2 / 3)
    assert s["tnr_by_source"] == {"near_miss": 0.0, "nop": 1.0, "real_fail": 1.0}
    assert s["flakiness"] == pytest.approx(1 / 6)


def test_error_on_a_correct_entry_is_a_rejection():
    s = score_task(table([("pass", "pass", "error"), ("real_fail", "fail", "fail")]))
    assert s["tpr"] == 0.0 and s["tnr"] == 1.0 and s["flakiness"] == 0.5


def test_empty_side_and_bad_values():
    assert score_task(table([("real_fail", "fail", "fail")]))["ba"] is None
    with pytest.raises(ValueError):
        score_task([Verdict("x", "pass", "pass", "maybe")])
    with pytest.raises(ValueError):
        score_task([Verdict("x", "pass", "unknown", "pass")])


def test_macro():
    m = macro([{"tpr": 1.0, "tnr": 0.5, "ba": 0.75, "flakiness": 0.0},
               {"tpr": 0.5, "tnr": 1.0, "ba": 0.75, "flakiness": 0.2}])
    assert {k: m[k] for k in ("n_tasks", "tpr", "tnr", "ba", "flakiness")} == \
        {"n_tasks": 2, "tpr": 0.75, "tnr": 0.75, "ba": 0.75, "flakiness": 0.1}
    assert m["ba_ci"] == (0.75, 0.75)


def test_bootstrap_ci_widens_with_spread_and_is_deterministic():
    tight, wide = bootstrap_ci([0.8] * 10), bootstrap_ci([0.5, 1.0] * 5)
    assert tight == (0.8, 0.8) and wide[0] < 0.75 < wide[1]
    assert bootstrap_ci([0.5, 1.0] * 5) == wide and bootstrap_ci([]) is None


def test_baselines():
    ap, af, orig = {"tpr": 1.0, "tnr": 0.0, "ba": 0.5}, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5}, {"tpr": 1.0, "tnr": 1.0, "ba": 1.0}
    assert check_baselines(ap, af, orig) == []
    assert check_baselines(ap, af, orig, network_probe=af) == []
    assert check_baselines(ap, af, orig, network_probe={"tpr": 1.0, "tnr": 0.0, "ba": 0.5})   # a verifier reached GitHub
    assert check_baselines({**ap, "tnr": 0.1, "ba": 0.55}, af, orig)   # always-pass rejected something
    assert check_baselines(ap, {**af, "tnr": 0.9, "ba": 0.45}, orig)   # always-fail passed something: the pin is back
    assert check_baselines(ap, af, {**orig, "ba": 0.95})
    assert check_baselines(ap, af, {**orig, "ba": None})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb'`

- [ ] **Step 4: Implement `vb/__init__.py`, `vb/util.py` and `vb/scorer.py`**

`vb/__init__.py`:
```python
"""Verifier Bench: how well do agents write verifiers?"""
```

`vb/util.py`:
```python
"""Paths, JSON IO, run directories, .env loading and Harbor calls."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VB_DATA", "/root/vbench"))
TB_ROOT = Path(os.environ.get("VB_TB_ROOT", "/root/tb4/terminal-bench"))
HARBOR = os.environ.get("VB_HARBOR", "/root/venv/bin/harbor")


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "vb", "configs", "templates"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def new_run_dir(name: str, root: Path | None = None) -> Path:
    """<root>/<name>/<UTC stamp>_<git sha>[_n]/, created fresh; an existing path is never reused."""
    root = Path(root) if root is not None else DATA / "runs"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / name / f"{stamp}_{git_sha()}"
    d, n = base, 1
    while d.exists():
        d = base.with_name(f"{base.name}_{n}")
        n += 1
    d.mkdir(parents=True)
    return d


def load_env(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE lines into os.environ. Returns the names loaded, never the values."""
    path = Path(path) if path is not None else DATA / ".env"
    if not path.is_file():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()
        names.append(key.strip())
    return names


def harbor_json(*args: str) -> Any:
    out = subprocess.run([HARBOR, *args, "--json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)
```

`vb/scorer.py`:
```python
"""Score a verifier's verdicts against known labels. An error counts as fail and is reported as flakiness."""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

LABELS = ("pass", "fail")
VERDICTS = ("pass", "fail", "error")


@dataclass(frozen=True)
class Verdict:
    entry_id: str
    source: str
    label: str      # truth: pass | fail
    verdict: str    # the verifier: pass | fail | error


def score_task(verdicts: list[Verdict]) -> dict:
    for v in verdicts:
        if v.label not in LABELS:
            raise ValueError(f"{v.entry_id}: label must be pass/fail, got {v.label!r}")
        if v.verdict not in VERDICTS:
            raise ValueError(f"{v.entry_id}: verdict must be pass/fail/error, got {v.verdict!r}")
    correct = [v for v in verdicts if v.label == "pass"]
    wrong = [v for v in verdicts if v.label == "fail"]
    tpr = sum(v.verdict == "pass" for v in correct) / len(correct) if correct else None
    tnr = sum(v.verdict != "pass" for v in wrong) / len(wrong) if wrong else None
    by_source: dict[str, list[Verdict]] = {}
    for v in wrong:
        by_source.setdefault(v.source, []).append(v)
    return {
        "n_correct": len(correct), "n_wrong": len(wrong), "tpr": tpr, "tnr": tnr,
        "ba": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "tnr_by_source": {s: sum(v.verdict != "pass" for v in vs) / len(vs) for s, vs in sorted(by_source.items())},
        "flakiness": sum(v.verdict == "error" for v in verdicts) / len(verdicts) if verdicts else 0.0,
    }


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap of the mean. With 5 tasks x 2 attempts per writer the interval is wide on purpose:
    a model or harness effect is called only when two writers' intervals do not overlap."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    lo, hi = means[int((1 - level) / 2 * n)], means[int((1 + level) / 2 * n) - 1]
    return (lo, hi)


def macro(scores: list[dict]) -> dict:
    def avg(k):
        vals = [s[k] for s in scores if s.get(k) is not None]
        return mean(vals) if vals else None
    return {"n_tasks": len(scores), "tpr": avg("tpr"), "tnr": avg("tnr"), "ba": avg("ba"), "flakiness": avg("flakiness"),
            "ba_ci": bootstrap_ci([s["ba"] for s in scores if s.get("ba") is not None])}


def check_baselines(always_pass: dict, always_fail: dict, original: dict, network_probe: dict | None = None) -> list[str]:
    """Problems with the baselines; an empty list means the pool, the swap, the offline rule and the scorer look right."""
    problems = []
    probes = [("always_pass", always_pass, {"tpr": 1.0, "tnr": 0.0, "ba": 0.5}),
              ("always_fail", always_fail, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5})]
    if network_probe is not None:
        probes.append(("network_probe", network_probe, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5}))
    for name, s, want in probes:
        for k, v in want.items():
            if s.get(k) is None or abs(s[k] - v) > 1e-9:
                problems.append(f"{name}: {k} = {s.get(k)} (expected {v})")
    if original.get("ba") is None or abs(original["ba"] - 1.0) > 1e-9:
        problems.append(f"original tests: ba = {original.get('ba')} (expected 1.0 on the kept pool)")
    return problems
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `9 passed`

- [ ] **Step 6: Sync to the server, put the HUD key there, check the server Python**

The key is piped straight from the local HUD settings into the server file and never printed:
```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench/repo && /usr/bin/python3 -c "import vb.util, vb.scorer, tomllib; print(\"server ok\")"'
/Users/anilkumar/privacybench_hud/.venv/bin/python -c "from hud.settings import settings; print('HUD_API_KEY=' + settings.api_key)" | ssh root@178.156.248.194 'umask 077; cat > /root/vbench/.env; wc -c < /root/vbench/.env'
```
Expected: `server ok`, then a byte count above 20.

- [ ] **Step 7: Commit**

```bash
git add .gitignore pytest.ini tools vb tests
git commit -m "Scaffold for the Terminal-Bench run: server sync, paths and run dirs, scorer with baseline checks"
```

---

### Task 2: A task copy with a replacement verifier, and the regrade runner

**Files:**
- Create: `vb/taskdir.py`, `vb/regrade.py`, `templates/baselines/{always_pass,always_fail,network_probe}/{Dockerfile,test.sh}`
- Test: `tests/test_taskdir.py`, `tests/test_regrade.py`

**Interfaces:**
- Consumes: `vb.util.HARBOR`
- Produces:
  - `vb.taskdir.unpin_verifier_image(text: str) -> str`: removes `docker_image` from `[verifier.environment]` only; raises `ValueError` if a pin survives.
  - `vb.taskdir.set_verifier_offline(text: str) -> str`: sets `[verifier] network_mode = "no-network"`; raises `ValueError` if it cannot.
  - `vb.taskdir.make_verifier_task(task_dir: Path, tests_dir: Path, out_dir: Path) -> Path`: a copy of the task with `tests/` replaced, unpinned and offline; requires `tests/Dockerfile`, `tests/test.sh` and a separate verifier.
  - `vb.taskdir.BASELINES: dict[str, Path]` (`always_pass`, `always_fail`, `network_probe`).
  - `vb.regrade.run_regrade(job_dir, task_dir, out_root, job_name, n_concurrent=4, timeout_s=21600) -> Path` (the regraded job dir).
  - `vb.regrade.read_rewards(regraded_job: Path) -> dict[str, float | None]`, keyed by the **source attempt's directory name**.
  - `vb.regrade.verdict(reward: float | None) -> str`.

- [ ] **Step 1: Write the baseline verifiers**

Both `templates/baselines/always_pass/Dockerfile` and `templates/baselines/always_fail/Dockerfile`:
```dockerfile
FROM debian:bookworm-slim
COPY . /tests/
RUN mkdir -p /logs/verifier
```

`templates/baselines/always_pass/test.sh`:
```sh
#!/bin/bash
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
```

`templates/baselines/always_fail/test.sh`:
```sh
#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
```

`templates/baselines/network_probe/Dockerfile` is the same as the two above. Its `test.sh` passes an attempt only if the verifier can reach GitHub while grading. It must pass nothing, which proves verifiers run offline, the same way `always_fail` proves the swap:
```sh
#!/bin/bash
mkdir -p /logs/verifier
if timeout 5 bash -c '</dev/tcp/github.com/443' 2>/dev/null; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
```

- [ ] **Step 2: Write the failing tests**

`tests/test_taskdir.py`:
```python
import tomllib
from pathlib import Path

import pytest

from vb.taskdir import BASELINES, make_verifier_task, set_verifier_offline, unpin_verifier_image

TOML = '''schema_version = "2.0"
artifacts = ["/app/"]

[task]
name = "terminal-bench/demo"

[verifier]
timeout_sec = 300.0
environment_mode = "separate"

[verifier.environment]
build_timeout_sec = 600.0
cpus = 2
docker_image = "harborframework/terminal-bench:demo-verifier@sha256:abc"

[agent]
timeout_sec = 100.0

[environment]
cpus = 2
docker_image = "harborframework/terminal-bench:demo-environment@sha256:def"
'''


def task(tmp_path: Path, toml: str = TOML) -> Path:
    d = tmp_path / "demo"
    (d / "tests").mkdir(parents=True)
    (d / "environment").mkdir()
    (d / "task.toml").write_text(toml)
    (d / "instruction.md").write_text("Do the thing.")
    (d / "tests" / "Dockerfile").write_text("FROM python:3.13-slim\nCOPY . /tests/\n")
    (d / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (d / "tests" / "test_outputs.py").write_text("SECRET = 1\n")
    return d


def test_unpin_removes_only_the_verifier_pin():
    out = tomllib.loads(unpin_verifier_image(TOML))
    assert "docker_image" not in out["verifier"]["environment"]
    assert out["environment"]["docker_image"].endswith("sha256:def")
    assert out["verifier"]["environment"]["cpus"] == 2
    assert unpin_verifier_image(unpin_verifier_image(TOML)) == unpin_verifier_image(TOML)


def test_unpin_refuses_a_pin_it_cannot_remove():
    inline = TOML.replace('[verifier.environment]\nbuild_timeout_sec = 600.0\ncpus = 2\n'
                          'docker_image = "harborframework/terminal-bench:demo-verifier@sha256:abc"\n', '')
    inline = inline.replace('environment_mode = "separate"',
                            'environment_mode = "separate"\nenvironment = { docker_image = "x@sha256:1" }')
    with pytest.raises(ValueError):
        unpin_verifier_image(inline)


def test_set_verifier_offline():
    out = tomllib.loads(set_verifier_offline(TOML))
    assert out["verifier"]["network_mode"] == "no-network" and out["verifier"]["timeout_sec"] == 300.0
    assert "network_mode" not in out["environment"] and "network_mode" not in out["agent"]
    public = TOML.replace('environment_mode = "separate"', 'environment_mode = "separate"\nnetwork_mode = "public"')
    assert tomllib.loads(set_verifier_offline(public))["verifier"]["network_mode"] == "no-network"
    assert set_verifier_offline(set_verifier_offline(TOML)) == set_verifier_offline(TOML)
    with pytest.raises(ValueError):
        set_verifier_offline('[task]\nname = "x"\n')


def test_make_verifier_task_replaces_tests_and_keeps_the_original(tmp_path):
    src = task(tmp_path)
    out = make_verifier_task(src, BASELINES["always_fail"], tmp_path / "copies" / "always_fail")
    assert sorted(p.name for p in (out / "tests").iterdir()) == ["Dockerfile", "test.sh"]
    assert "echo 0" in (out / "tests" / "test.sh").read_text()
    copied = tomllib.loads((out / "task.toml").read_text())
    assert "docker_image" not in copied["verifier"]["environment"]
    assert copied["verifier"]["network_mode"] == "no-network"
    assert (src / "tests" / "test_outputs.py").exists()                      # original untouched
    assert "sha256:abc" in (src / "task.toml").read_text()


def test_make_verifier_task_requires_the_contract(tmp_path):
    src = task(tmp_path)
    bad = tmp_path / "bad_tests"
    bad.mkdir()
    (bad / "test.sh").write_text("#!/bin/bash\n")
    with pytest.raises(ValueError, match="Dockerfile"):
        make_verifier_task(src, bad, tmp_path / "o1")
    (bad / "test.sh").unlink()
    (bad / "Dockerfile").write_text("FROM debian\n")
    with pytest.raises(ValueError, match="test.sh"):
        make_verifier_task(src, bad, tmp_path / "o2")
    same = task(tmp_path / "x", TOML.replace('environment_mode = "separate"', ''))
    with pytest.raises(ValueError, match="separate"):
        make_verifier_task(same, BASELINES["always_pass"], tmp_path / "o3")
```

`tests/test_regrade.py`:
```python
import json
from pathlib import Path

from vb.regrade import read_rewards, verdict


def trial(job: Path, name: str, source: str, reward=None, exc=None):
    d = job / name
    d.mkdir(parents=True)
    r = {"trial_name": name, "config": {"source_trial": {"trial_id": "hub-" + source,
                                                         "path": f"/root/vbench/pools/demo/job/{source}"}},
         "exception_info": exc, "verifier_result": None if reward is None else {"rewards": {"reward": reward}}}
    (d / "result.json").write_text(json.dumps(r))


def test_read_rewards_keys_by_source_attempt(tmp_path):
    job = tmp_path / "regraded" / "v1"
    trial(job, "demo__aaa", "demo__src1", reward=1.0)
    trial(job, "demo__bbb", "demo__src2", reward=0.0)
    trial(job, "demo__ccc", "demo__src3", reward=1.0, exc={"type": "BuildError"})
    trial(job, "demo__ddd", "demo__src4")
    (job / "config.json").write_text("{}")
    assert read_rewards(job) == {"demo__src1": 1.0, "demo__src2": 0.0, "demo__src3": None, "demo__src4": None}


def test_verdict():
    assert (verdict(1.0), verdict(0.0), verdict(0.5), verdict(None)) == ("pass", "fail", "fail", "error")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_taskdir.py tests/test_regrade.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb.taskdir'`

- [ ] **Step 4: Implement `vb/taskdir.py` and `vb/regrade.py`**

`vb/taskdir.py`:
```python
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


def make_verifier_task(task_dir: Path, tests_dir: Path, out_dir: Path) -> Path:
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
    (out_dir / "task.toml").write_text(set_verifier_offline(unpin_verifier_image(text)), encoding="utf-8")
    return out_dir
```

`vb/regrade.py`:
```python
"""Regrade recorded attempts with a task copy's verifier, and read the rewards back."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .util import HARBOR


def run_regrade(job_dir: Path, task_dir: Path, out_root: Path, job_name: str, n_concurrent: int = 4,
                timeout_s: int = 21600) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / f"{job_name}.log").open("w") as log:
        subprocess.run([HARBOR, "job", "regrade", str(job_dir), "-p", str(task_dir), "-o", str(out_root),
                        "--job-name", job_name, "-n", str(n_concurrent)],
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    return out_root / job_name


def read_rewards(regraded_job: Path) -> dict[str, float | None]:
    """{source attempt dir name: reward, or None when the verifier errored or wrote no reward}."""
    out: dict[str, float | None] = {}
    for rj in sorted(Path(regraded_job).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        src = ((r.get("config") or {}).get("source_trial") or {})
        key = Path(src.get("path") or "").name or r.get("trial_name") or rj.parent.name
        reward = (((r.get("verifier_result") or {}).get("rewards")) or {}).get("reward")
        out[key] = None if r.get("exception_info") or reward is None else float(reward)
    return out


def verdict(reward: float | None) -> str:
    if reward is None:
        return "error"
    return "pass" if reward >= 1.0 else "fail"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `16 passed`

- [ ] **Step 6: The swap proof on the server** (compute only, about 10 minutes)

This reuses the probe attempt downloaded earlier (`/root/vbench/probe/job1`, one Astra attempt at `photonic-waveguide-routing`, recorded reward 1.0). Three verifiers must give three different answers:

```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench/repo && /usr/bin/python3 - <<"EOF"
from pathlib import Path
from vb.taskdir import BASELINES, make_verifier_task
from vb.regrade import read_rewards, run_regrade
from vb.util import TB_ROOT
task = TB_ROOT / "photonic-waveguide-routing"
out = Path("/root/vbench/probe/swap")
res = {}
for name, tests in {"original": task / "tests", **BASELINES}.items():
    vt = make_verifier_task(task, tests, out / "vtasks" / name)
    res[name] = read_rewards(run_regrade(Path("/root/vbench/probe/job1"), vt, out / "regrades", name))
print(res)
assert list(res["original"].values()) == [1.0], res
assert list(res["always_pass"].values()) == [1.0], res
assert list(res["always_fail"].values()) == [0.0], res
assert list(res["network_probe"].values()) == [0.0], res
print("SWAP PROOF OK")
EOF'
```
Expected: `SWAP PROOF OK`.
- If `always_fail` gives 1.0, the pin was not removed; stop and inspect `out/vtasks/always_fail/task.toml`.
- If `network_probe` gives 1.0, verifiers are not offline while grading; stop, because every writer could fetch the official tests.
- If `original` now gives 0.0 or an error, this task's official tests need the network at grading time. The label check (Task 4) then drops its entries, and such tasks are excluded.

- [ ] **Step 7: Commit**

```bash
git add vb/taskdir.py vb/regrade.py templates/baselines tests/test_taskdir.py tests/test_regrade.py
git commit -m "Verifier swap: task copy with replacement tests and the verifier image pin removed; regrade runner; proven on a real attempt"
```

---

### Task 3: Index the leaderboard and select five tasks

**Files:**
- Create: `vb/index.py`, `vb/select.py`, `configs/excluded_tasks.json`
- Generated: `configs/m1.json`, `/root/vbench/meta/tb4_index.json`, `/root/vbench/meta/tb_issues.json`
- Test: `tests/test_select.py`

**Interfaces:**
- Consumes: `vb.util.harbor_json, read_json, write_json, DATA, TB_ROOT`
- Produces:
  - `vb.index.LEADERBOARD`
  - `vb.index.build_index(leaderboard) -> list[dict]` (rows with `job`, `rank`, `agent`, `model`, `trials`)
  - `vb.index.attempts(index) -> list[dict]`: flat rows `{trial_id, task, reward, error, job, rank, agent, model}`, with `task` as the short name
  - `vb.index.task_counts(attempts) -> dict[str, dict]` (`pass`, `fail`, `error`)
  - `vb.select.reasons_to_reject(task_dir: Path, counts: dict, excluded: dict, *, min_each=10, max_verifier_timeout=900) -> list[str]`
  - `vb.select.select(tb_root, counts, excluded, n=5) -> tuple[list[str], dict[str, list[str]]]`
  - `configs/m1.json`: `{"tasks": [...], "rules": {...}, "rejected": {...}}`

- [ ] **Step 1: Write the failing tests**

`tests/test_select.py`:
```python
from pathlib import Path

from vb.index import attempts, task_counts
from vb.select import reasons_to_reject, select

BASE = '''schema_version = "2.0"
artifacts = {artifacts}

[task]
name = "terminal-bench/{name}"

[metadata]
category = "{category}"
expert_time_estimate_hours = {hours}

[verifier]
timeout_sec = {vt}
environment_mode = "separate"

[verifier.environment]
cpus = 2
gpus = 0

[environment]
cpus = 2
gpus = {gpus}
'''


def make(root: Path, name, category="Software", hours=2, vt=300.0, gpus=0, artifacts='["/app/"]', compose=False):
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "task.toml").write_text(BASE.format(name=name, category=category, hours=hours, vt=vt, gpus=gpus, artifacts=artifacts))
    if compose:
        (d / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    return d


def test_attempts_and_counts():
    idx = [{"job": "j1", "rank": 1, "agent": "Codex", "model": "M", "trials": [
        {"id": "t1", "task_name": "terminal-bench/a", "reward": 1.0, "error_type": None},
        {"id": "t2", "task_name": "terminal-bench/a", "reward": 0, "error_type": None},
        {"id": "t3", "task_name": "terminal-bench/a", "reward": None, "error_type": "TimeoutError"}]}]
    atts = attempts(idx)
    assert [a["task"] for a in atts] == ["a", "a", "a"] and atts[0]["job"] == "j1"
    assert task_counts(atts) == {"a": {"pass": 1, "fail": 1, "error": 1}}


def test_reasons_to_reject(tmp_path):
    ok = {"pass": 12, "fail": 15, "error": 0}
    assert reasons_to_reject(make(tmp_path, "good"), ok, {}) == []
    assert reasons_to_reject(make(tmp_path, "few"), {"pass": 3, "fail": 40, "error": 0}, {})
    assert reasons_to_reject(make(tmp_path, "gpu", gpus=1), ok, {})
    assert reasons_to_reject(make(tmp_path, "slow", vt=1800.0), ok, {})
    assert reasons_to_reject(make(tmp_path, "side", artifacts='[{ source = "/x", service = "api" }]'), ok, {})
    assert reasons_to_reject(make(tmp_path, "multi", compose=True), ok, {})
    assert reasons_to_reject(make(tmp_path, "flagged"), ok, {"flagged": "issue #12: wrong expected output"})


def test_select_spreads_categories_and_is_deterministic(tmp_path):
    for i in range(3):
        make(tmp_path, f"sw{i}", category="Software")
    make(tmp_path, "sci0", category="Science")
    make(tmp_path, "ops0", category="Operations")
    counts = {p.name: {"pass": 20, "fail": 20 + i, "error": 0} for i, p in enumerate(sorted(tmp_path.iterdir()))}
    chosen, rejected = select(tmp_path, counts, {}, n=4)
    assert len(chosen) == 4 and {"sci0", "ops0"} <= set(chosen)
    assert select(tmp_path, counts, {}, n=4)[0] == chosen
    assert rejected == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_select.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb.index'`

- [ ] **Step 3: Implement `vb/index.py` and `vb/select.py`**

`vb/index.py`:
```python
"""Index every Terminal-Bench 4.0 leaderboard attempt with its reward (metadata only, no downloads)."""
from __future__ import annotations

import argparse
from collections import defaultdict

from .util import DATA, harbor_json, read_json, write_json

LEADERBOARD = "9f966760-00f1-424e-90f5-c964fb6f6091"
INDEX = DATA / "meta" / "tb4_index.json"


def build_index(leaderboard: str = LEADERBOARD) -> list[dict]:
    rows = []
    for r in harbor_json("hub", "leaderboard", "show", leaderboard)["rows"]:
        first = harbor_json("hub", "leaderboard", "row", "trial", "list", r["id"], "--limit", "1", "--page", "1")
        job = harbor_json("hub", "trial", "show", first["items"][0]["trial_id"])["job_id"]
        trials, page = [], 1
        while True:
            d = harbor_json("hub", "job", "trials", job, "--limit", "500", "--page", str(page))
            items = d.get("items") or []
            trials += items
            if not items or page >= (d.get("total_pages") or 1):
                break
            page += 1
        m = r["metadata"]
        rows.append({"row": r["id"], "rank": r["rank"], "agent": m["agent_display"]["label"],
                     "model": m["model_display"]["label"], "job": job, "trials": trials})
    return rows


def attempts(index: list[dict]) -> list[dict]:
    out = []
    for row in index:
        for t in row["trials"]:
            out.append({"trial_id": t["id"], "task": t["task_name"].split("/")[-1], "reward": t.get("reward"),
                        "error": t.get("error_type"), "job": row["job"], "rank": row["rank"],
                        "agent": row["agent"], "model": row["model"]})
    return out


def task_counts(atts: list[dict]) -> dict[str, dict]:
    counts: dict[str, dict] = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0})
    for a in atts:
        if a["error"] or a["reward"] is None:
            counts[a["task"]]["error"] += 1
        elif float(a["reward"]) >= 1.0:
            counts[a["task"]]["pass"] += 1
        else:
            counts[a["task"]]["fail"] += 1
    return dict(counts)


def load_attempts() -> list[dict]:
    return attempts(read_json(INDEX))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh or not INDEX.exists():
        write_json(INDEX, build_index())
    atts = load_attempts()
    print(len(atts), "attempts;", len(task_counts(atts)), "tasks")
```

`vb/select.py`:
```python
"""Choose tasks by rule (spec §5.1), spread across categories."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from .index import load_attempts, task_counts
from .util import ROOT, TB_ROOT, read_json, write_json

RULES = {"min_each": 10, "max_verifier_timeout": 900}


def reasons_to_reject(task_dir: Path, counts: dict, excluded: dict, *, min_each: int = 10,
                      max_verifier_timeout: int = 900) -> list[str]:
    task_dir = Path(task_dir)
    t = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    ver, env = t.get("verifier") or {}, t.get("environment") or {}
    why = []
    if counts.get("pass", 0) < min_each or counts.get("fail", 0) < min_each:
        why.append(f"needs >= {min_each} pass and fail attempts, has {counts.get('pass', 0)}/{counts.get('fail', 0)}")
    if (env.get("gpus") or 0) > 0 or ((ver.get("environment") or {}).get("gpus") or 0) > 0:
        why.append("needs a GPU")
    if ver.get("environment_mode") != "separate":
        why.append("verifier is not separate")
    if any(not isinstance(a, str) for a in t.get("artifacts", [])) or \
            any(c.get("service", "main") != "main" for c in ver.get("collect", [])):
        why.append("artifacts off the main container")
    if (task_dir / "environment" / "docker-compose.yaml").exists():
        why.append("multi-container environment")
    if float(ver.get("timeout_sec") or 0) > max_verifier_timeout:
        why.append(f"verifier timeout {ver.get('timeout_sec')} s > {max_verifier_timeout} s")
    if task_dir.name in excluded:
        why.append(f"community-reported test problem: {excluded[task_dir.name]}")
    return why


def select(tb_root: Path, counts: dict, excluded: dict, n: int = 5) -> tuple[list[str], dict[str, list[str]]]:
    ok, rejected = {}, {}
    for d in sorted(p for p in Path(tb_root).iterdir() if (p / "task.toml").is_file()):
        why = reasons_to_reject(d, counts.get(d.name, {}), excluded, **RULES)
        if why:
            rejected[d.name] = why
            continue
        meta = tomllib.loads((d / "task.toml").read_text(encoding="utf-8")).get("metadata") or {}
        c = counts[d.name]
        ok[d.name] = (meta.get("category") or "Other", -min(c["pass"], c["fail"]),
                      float(meta.get("expert_time_estimate_hours") or 99), d.name)
    by_cat: dict[str, list[str]] = {}
    for name, key in sorted(ok.items(), key=lambda kv: kv[1]):
        by_cat.setdefault(key[0], []).append(name)
    chosen = []
    while len(chosen) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(chosen) < n:
                chosen.append(by_cat[cat].pop(0))
    return chosen, rejected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    excluded = read_json(ROOT / "configs" / "excluded_tasks.json")
    chosen, rejected = select(TB_ROOT, task_counts(load_attempts()), excluded, a.n)
    write_json(ROOT / "configs" / "m1.json", {"tasks": chosen, "rules": RULES, "rejected": rejected,
                                              "excluded": excluded})
    print("chosen:", chosen)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `19 passed`

- [ ] **Step 5: Read the community's reports on the official tests**

Fetch every issue in the Terminal-Bench repository (public API, no key needed):
```bash
ssh root@178.156.248.194 'mkdir -p /root/vbench/meta && python3 - <<"EOF"
import json, urllib.request
out, page = [], 1
while True:
    url = f"https://api.github.com/repos/harbor-framework/terminal-bench/issues?state=all&per_page=100&page={page}"
    items = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vbench"})))
    if not items: break
    out += [{"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"], "body": (i.get("body") or "")[:3000]} for i in items]
    page += 1
json.dump(out, open("/root/vbench/meta/tb_issues.json", "w"))
print(len(out), "issues")
EOF'
```
Then list the issues that name a Terminal-Bench 4.0 task:
```bash
ssh root@178.156.248.194 'python3 - <<"EOF"
import json, os
issues = json.load(open("/root/vbench/meta/tb_issues.json"))
tasks = sorted(os.listdir("/root/tb4/terminal-bench"))
for t in tasks:
    hits = [i for i in issues if t in (i["title"] + i["body"])]
    for i in hits: print(t, "|", i["state"], "|", i["title"][:90], "|", i["url"])
EOF'
```
Read each hit. **A task goes into `configs/excluded_tasks.json` when an issue says its official tests reject a correct solution or accept a wrong one, and the fix is not in v4.0.0.** Record the reason and the issue URL, for example `{"some-task": "tests accept an empty output file: https://github.com/.../issues/123"}`. If nothing qualifies, the file is `{}`.

- [ ] **Step 6: Select the tasks on the server**

Run:
```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench/repo && python3 -m vb.index && python3 -m vb.select --n 5 && cat configs/m1.json | head -20' && scp root@178.156.248.194:/root/vbench/repo/configs/m1.json configs/m1.json
```
Expected: `8910 attempts; 66 tasks`, then `chosen:` with 5 task names from at least 3 categories. `configs/m1.json` lists every rejected task with its reasons.

- [ ] **Step 7: Check that each chosen task's environment does not contain its official tests**

The writer works inside the environment image, so that image must not hold the tests or the solution:
```bash
ssh root@178.156.248.194 'cd /root/vbench/repo && python3 - <<"EOF"
import json, subprocess, tomllib
from pathlib import Path
from vb.util import TB_ROOT
for t in json.load(open("configs/m1.json"))["tasks"]:
    img = tomllib.loads((TB_ROOT / t / "task.toml").read_text())["environment"]["docker_image"]
    names = [p.name for p in (TB_ROOT / t / "tests").iterdir() if p.is_file() and p.name not in ("Dockerfile", "test.sh")]
    pat = " -o ".join(f"-name {n!r}" for n in names) or "-name nothing"
    cmd = f"ls -d /tests /solution 2>/dev/null; find / -xdev \\( {pat} \\) 2>/dev/null | head -5"
    out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "sh", img, "-c", cmd], capture_output=True, text=True).stdout.strip()
    print(t, "LEAK: " + out if out else "clean")
EOF'
```
Expected: `clean` for all five. A task that leaks is added to `configs/excluded_tasks.json` with the reason "environment image contains the official tests", and Step 6 is rerun.

- [ ] **Step 8: Commit**

```bash
git add vb/index.py vb/select.py configs tests/test_select.py
git commit -m "Leaderboard index (8,910 attempts) and rule-based selection of five Terminal-Bench 4.0 tasks"
```

---

### Task 4: Build the pools and check every label

**Files:**
- Create: `vb/pool.py`
- Generated: `/root/vbench/pools/<task>/{job/, manifest.json, kept_job/}`
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes: `vb.index.load_attempts`, `vb.taskdir.make_verifier_task`, `vb.regrade.run_regrade`, `vb.regrade.read_rewards`, `vb.regrade.verdict`
- Produces:
  - `vb.pool.round_robin(atts: list[dict]) -> list[dict]`: interleaves attempts across leaderboard runs, deterministically.
  - `vb.pool.ctrf_counts(trial_dir: Path) -> tuple[int, int]` (passed, total).
  - `vb.pool.classify_fail(passed, total, threshold=0.8) -> str`.
  - `vb.pool.build_pool(task: str) -> Path` (the manifest path).
  - `vb.pool.check_labels(task: str) -> dict` (counts; sets `regraded_reward`, `kept`, `drop_reason`).
  - `vb.pool.kept_job(task: str) -> Path` (a job dir with kept entries only).
  - `vb.pool.load_manifest(task) -> list[dict]`.
  - Manifest entry: `{entry_id, trial_id, source, label, recorded_reward, model, agent, job, ctrf, regraded_reward, kept, drop_reason}`.
  - CLI: `python3 -m vb.pool --task <t> [--check]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pool.py`:
```python
import json

from vb.pool import classify_fail, ctrf_counts, round_robin


def att(job, i, rank=1):
    return {"trial_id": f"{job}-{i}", "job": job, "rank": rank}


def test_round_robin_interleaves_runs_in_rank_order():
    atts = [att("b", 1, 2), att("b", 2, 2), att("a", 1, 1), att("a", 2, 1), att("a", 3, 1), att("c", 1, 3)]
    assert [x["trial_id"] for x in round_robin(atts)] == ["a-1", "b-1", "c-1", "a-2", "b-2", "a-3"]
    assert round_robin(list(reversed(atts))) == round_robin(atts)


def test_ctrf_counts(tmp_path):
    (tmp_path / "verifier").mkdir()
    assert ctrf_counts(tmp_path) == (0, 0)
    (tmp_path / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"summary": {"tests": 10, "passed": 9, "failed": 1}}}))
    assert ctrf_counts(tmp_path) == (9, 10)


def test_classify_fail():
    assert classify_fail(9, 10) == "near_miss"
    assert classify_fail(8, 10) == "near_miss"
    assert classify_fail(7, 10) == "real_fail"
    assert classify_fail(0, 0) == "real_fail"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pool.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb.pool'`

- [ ] **Step 3: Implement `vb/pool.py`**

```python
"""Pools: leaderboard attempts plus oracle and nop runs; labels checked by regrading with the original tests."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .index import load_attempts
from .regrade import read_rewards, run_regrade, verdict
from .taskdir import make_verifier_task
from .util import DATA, HARBOR, TB_ROOT, read_json, write_json

N_PASS, N_FAIL, MIN_JOBS, FAIL_CANDIDATES = 10, 10, 4, 24


def pool_dir(task: str) -> Path:
    return DATA / "pools" / task


def load_manifest(task: str) -> list[dict]:
    return read_json(pool_dir(task) / "manifest.json")


def round_robin(atts: list[dict]) -> list[dict]:
    jobs: dict[str, list[dict]] = {}
    for a in sorted(atts, key=lambda a: (a["rank"], a["job"], a["trial_id"])):
        jobs.setdefault(a["job"], []).append(a)
    order = sorted(jobs, key=lambda j: (jobs[j][0]["rank"], j))
    out, i = [], 0
    while any(i < len(jobs[j]) for j in order):
        out += [jobs[j][i] for j in order if i < len(jobs[j])]
        i += 1
    return out


def ctrf_counts(trial_dir: Path) -> tuple[int, int]:
    p = Path(trial_dir) / "verifier" / "ctrf.json"
    if not p.is_file():
        return (0, 0)
    s = json.loads(p.read_text(encoding="utf-8"))["results"]["summary"]
    return (int(s.get("passed", 0)), int(s.get("tests", 0)))


def classify_fail(passed: int, total: int, threshold: float = 0.8) -> str:
    return "near_miss" if total and passed / total >= threshold else "real_fail"


def _download(trial_id: str, dest: Path) -> Path:
    tmp = dest.parent / "_dl" / trial_id
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run([HARBOR, "hub", "trial", "download", trial_id, "-o", str(tmp)], check=True, capture_output=True)
    (child,) = [p for p in tmp.iterdir() if p.is_dir()]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / child.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(child), str(target))
    shutil.rmtree(tmp)
    return target


def _run_agent(task: str, agent: str, dest: Path) -> tuple[Path, float | None]:
    runs = dest.parent / "_runs"
    subprocess.run([HARBOR, "run", "-p", str(TB_ROOT), "-i", task, "-a", agent, "-o", str(runs),
                    "--job-name", agent, "-y", "-q"], check=True, capture_output=True)
    (trial,) = [p for p in (runs / agent).iterdir() if (p / "result.json").is_file()]
    r = json.loads((trial / "result.json").read_text())
    reward = (((r.get("verifier_result") or {}).get("rewards")) or {}).get("reward")
    target = dest / f"{agent}__{trial.name}"
    shutil.copytree(trial, target)
    return target, reward


def _job_config(job_dir: Path, task: str) -> None:
    write_json(job_dir / "config.json", {"job_name": job_dir.name, "jobs_dir": str(job_dir.parent),
                                         "n_concurrent_trials": 1, "agents": [{"name": "codex"}],
                                         "datasets": [{"path": str(TB_ROOT), "task_names": [task]}]})


def build_pool(task: str) -> Path:
    root = pool_dir(task)
    job = root / "job"
    if root.exists():
        shutil.rmtree(root)
    job.mkdir(parents=True)
    atts = [a for a in load_attempts() if a["task"] == task and not a["error"] and a["reward"] is not None]
    passes = round_robin([a for a in atts if float(a["reward"]) >= 1.0])[:N_PASS]
    if len({a["job"] for a in passes}) < MIN_JOBS:
        raise RuntimeError(f"{task}: passes come from fewer than {MIN_JOBS} runs")
    entries = []

    def entry(a, d, source, label, reward):
        p, t = ctrf_counts(d)
        return {"entry_id": d.name, "trial_id": a.get("trial_id"), "source": source, "label": label,
                "recorded_reward": reward, "model": a.get("model"), "agent": a.get("agent"), "job": a.get("job"),
                "ctrf": [p, t], "regraded_reward": None, "kept": None, "drop_reason": None}

    for a in passes:
        entries.append(entry(a, _download(a["trial_id"], job), "pass", "pass", float(a["reward"])))
    cand_dir = root / "candidates"
    fails = []
    for a in round_robin([a for a in atts if float(a["reward"]) < 1.0])[:FAIL_CANDIDATES]:
        d = _download(a["trial_id"], cand_dir)
        fails.append((classify_fail(*ctrf_counts(d)) == "near_miss", a, d))
    fails.sort(key=lambda x: not x[0])                          # near misses first, stable otherwise
    for is_near, a, d in fails[:N_FAIL]:
        target = job / d.name
        shutil.move(str(d), str(target))
        entries.append(entry(a, target, "near_miss" if is_near else "real_fail", "fail", float(a["reward"])))
    shutil.rmtree(cand_dir)
    for agent, source, label in (("oracle", "reference", "pass"), ("nop", "nop", "fail")):
        d, reward = _run_agent(task, agent, job)
        entries.append(entry({"agent": agent, "model": None}, d, source, label, reward))
    _job_config(job, task)
    write_json(root / "manifest.json", entries)
    return root / "manifest.json"


def check_labels(task: str) -> dict:
    """Regrade the whole pool with the ORIGINAL tests, unpinned; keep an entry only when its label reproduces."""
    root = pool_dir(task)
    entries = load_manifest(task)
    vt = make_verifier_task(TB_ROOT / task, TB_ROOT / task / "tests", DATA / "vtasks" / task / "original")
    regraded = run_regrade(root / "job", vt, root / "label_check", "original")
    rewards = read_rewards(regraded)
    counts = {"kept": 0, "dropped": 0}
    for e in entries:
        e["regraded_reward"] = rewards.get(e["entry_id"])
        got = verdict(e["regraded_reward"])
        if e["entry_id"] not in rewards:
            e["kept"], e["drop_reason"] = False, "no regrade result"
        elif got == "error":
            e["kept"], e["drop_reason"] = False, "original tests errored on regrade"
        elif e["source"] in ("reference", "nop") and verdict(e["recorded_reward"]) != e["label"]:
            e["kept"], e["drop_reason"] = False, f"{e['source']} did not score as expected on its own run"
        elif got != e["label"]:
            e["kept"], e["drop_reason"] = False, f"label {e['label']} did not reproduce (regraded {got})"
        else:
            e["kept"], e["drop_reason"] = True, None
        counts["kept" if e["kept"] else "dropped"] += 1
    write_json(root / "manifest.json", entries)
    return counts


def kept_job(task: str) -> Path:
    root = pool_dir(task)
    kj = root / "kept_job"
    if kj.exists():
        shutil.rmtree(kj)
    kj.mkdir()
    for e in load_manifest(task):
        if e["kept"]:
            shutil.copytree(root / "job" / e["entry_id"], kj / e["entry_id"], symlinks=True)
    _job_config(kj, task)
    return kj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--check", action="store_true", help="only rerun the label check")
    a = ap.parse_args()
    if not a.check:
        print(build_pool(a.task))
    print(json.dumps(check_labels(a.task)))
    print(kept_job(a.task))
```

- [ ] **Step 4: Run the unit tests**

Run: `.venv/bin/python -m pytest -q`
Expected: `22 passed`

- [ ] **Step 5: Build one pool on the server and look at it before building the rest** (compute only)

```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench/repo && T=$(python3 -c "import json; print(json.load(open(\"configs/m1.json\"))[\"tasks\"][0])") && nohup python3 -u -m vb.pool --task $T > /root/vbench/pool_$T.log 2>&1 & echo started'
```
Wait with a single background command that exits when the log shows the kept-job path or a traceback:
```bash
ssh root@178.156.248.194 'until grep -qE "kept_job|Traceback|Error" /root/vbench/pool_*.log; do sleep 20; done; tail -5 /root/vbench/pool_*.log'
```
Then inspect the manifest:
```bash
ssh root@178.156.248.194 'cd /root/vbench/repo && python3 - <<"EOF"
import json, collections, glob
m = json.load(open(glob.glob("/root/vbench/pools/*/manifest.json")[0]))
print(collections.Counter((e["source"], e["label"], e["kept"]) for e in m))
for e in m:
    if not e["kept"]: print("dropped", e["entry_id"], e["drop_reason"])
EOF'
```
Expected: `kept` is True for most entries; at least 8 kept `pass`, 8 kept fails, and the `reference` kept.
- If `oracle` does not pass, the task's reference solution fails its own tests: exclude the task with that reason and reselect.
- If more than 3 entries drop with "label did not reproduce", the unpinned build differs from the pinned image. **Stop and report**: every writer score on that task would carry the same bias.
- If mixing `oracle`/`nop` trials into the leaderboard job breaks the regrade (entries report "no regrade result"), put them in a separate job dir, `job_local/`, with its own `config.json`. Change `check_labels` and `kept_job` to regrade both dirs and merge the rewards, then rerun.

- [ ] **Step 6: Build the other four pools**

```bash
ssh root@178.156.248.194 'cd /root/vbench/repo && for T in $(python3 -c "import json; print(\" \".join(json.load(open(\"configs/m1.json\"))[\"tasks\"][1:]))"); do python3 -u -m vb.pool --task $T > /root/vbench/pool_$T.log 2>&1; done; echo ALL_POOLS_DONE >> /root/vbench/pools.done' &
```
Wait for `/root/vbench/pools.done`, then rerun the Step 5 inspection over all five manifests. Record for each task: entries by source, kept, dropped with reasons.

- [ ] **Step 7: Commit the pool summaries** (the attempts themselves stay on the server)

```bash
mkdir -p runs && ssh root@178.156.248.194 'cd /root/vbench && python3 -c "
import json, glob, os
out = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in sorted(glob.glob(\"pools/*/manifest.json\"))}
print(json.dumps(out, indent=1))"' > runs/m1_pools.json
git add vb/pool.py tests/test_pool.py runs/m1_pools.json
git commit -m "Pools for five Terminal-Bench 4.0 tasks: leaderboard attempts, oracle and nop runs, labels checked against the unpinned original tests"
```

---

### Task 5: Writer tasks, the network check and the smoke test

**Files:**
- Create: `vb/writer.py`, `vb/leakscan.py`, `templates/writer_instruction.md`, `templates/netprobe/{task.toml,instruction.md,solution/solve.sh,tests/test.sh,environment/Dockerfile}`
- Test: `tests/test_writer.py`

**Interfaces:**
- Consumes: `vb.util`, `vb.select` output `configs/m1.json`
- Produces:
  - `vb.writer.WRITERS`: `{name: {model_id, harness, agent, model, env, kwargs, host_key}}`, with the four names from Global Constraints
  - `vb.writer.dump_toml(doc: dict) -> str`: tables of scalars and string lists only; round-trips through `tomllib`.
  - `vb.writer.make_writer_task(task_dir: Path, out_root: Path) -> Path`: writes `out_root/write-verifier-<task>/`.
  - `vb.writer.run_writer(writer: str, tasks_dir: Path, out_root: Path, attempts: int, n_concurrent: int = 2) -> Path`: the Harbor job dir.
  - `vb.writer.collect(job_dir: Path, writer: str, dest: Path = DATA/"verifiers") -> list[dict]`: one provenance row per attempt, with `leak` findings; verifiers land at `dest/<writer>/<task>/<trial name>/tests/`.
  - `vb.leakscan.scan(tests_dir: Path, official_tests: Path) -> list[str]`: signs that a verifier copied or fetches the official tests. Empty means none found.
  - CLI: `python3 -m vb.writer --make`, `python3 -m vb.writer --run <writer> --attempts N [--tasks a,b]`, `python3 -m vb.writer --collect <job dir> --writer <writer>`, where `<writer>` is one of the four names.

- [ ] **Step 1: Write the writer instruction**

`templates/writer_instruction.md`:
```markdown
# Write the verifier for a task

You are not solving this task. You are writing the **verifier**: the tests that decide whether someone
else's attempt at it succeeded. Your verifier will be run on many real attempts, some correct and some
subtly wrong. Accepting a wrong attempt and rejecting a correct one count equally against you.

## The task the solver was given

<task-instruction>
{instruction}
</task-instruction>

## Where you are

This machine is the solver's starting environment: the same files, tools and services the solver begins
with. Its image is `{env_image}`; your verifier's Dockerfile may start `FROM` it if your tests need what
it contains. Explore it and run anything you like to understand what a correct result looks like. Do not
attempt the whole task. A small hand-made output, in a scratch directory outside `/verifier_out`, is
enough to check that your tests run.

## What to write

Write a folder `/verifier_out/tests/` containing:

- `Dockerfile` — builds the container your tests run in. Install everything you need here; the
  container has no network once it runs. The folder is copied into the image with `COPY . /tests/`,
  so include that line.
- `test.sh` — the entry point, run as `bash /tests/test.sh`. It must write exactly `1` (the attempt fully
  meets the task) or `0` (anything else) to `/logs/verifier/reward.txt`.
- any other files your tests use.

## What your tests will see

Your tests run in a fresh container built from your Dockerfile, not on the solver's machine. The only
things carried over from a solver's machine are these paths, copied to the same locations:

{artifacts}

Nothing else from the solver's machine is available. Your tests run with no network access and must
finish within {timeout} seconds. Anything your tests need must be written by you or installed in your
Dockerfile.

When `/verifier_out/tests/` is complete, stop.
```

- [ ] **Step 2: Write the network probe task**

`templates/netprobe/task.toml`:
```toml
schema_version = "2.0"
artifacts = ["/out/"]

[task]
name = "vbench/netprobe"
description = ""

[agent]
timeout_sec = 120.0
network_mode = "allowlist"
allowed_hosts = ["inference.hud.ai"]

[verifier]
timeout_sec = 60.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 1024
storage_mb = 2048
```

`templates/netprobe/environment/Dockerfile`:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
```

`templates/netprobe/instruction.md`:
```markdown
Report which hosts are reachable.
```

`templates/netprobe/solution/solve.sh`:
```sh
#!/bin/sh
python3 - <<'EOF'
import json, os, urllib.error, urllib.request
res = {}
for name, url in {"gateway": "https://inference.hud.ai/v1/messages", "github": "https://github.com",
                  "raw_github": "https://raw.githubusercontent.com/harbor-framework/terminal-bench/v4.0.0/README.md",
                  "huggingface": "https://huggingface.co", "pypi": "https://pypi.org/simple/"}.items():
    try:
        urllib.request.urlopen(url, timeout=10)
        res[name] = "reachable"
    except urllib.error.HTTPError as e:
        res[name] = f"reachable (http {e.code})"
    except Exception as e:
        res[name] = f"blocked ({type(e).__name__})"
os.makedirs("/out", exist_ok=True)
json.dump(res, open("/out/net.json", "w"), indent=1)
EOF
```

`templates/netprobe/tests/test.sh`:
```sh
#!/bin/bash
mkdir -p /logs/verifier
test -f /out/net.json && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
```

- [ ] **Step 3: Run the network probe** (free: the oracle agent runs `solve.sh`)

```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench && rm -rf netprobe_runs && /root/venv/bin/harbor run -p /root/vbench/repo/templates/netprobe -a oracle -o netprobe_runs --job-name probe -y -q > netprobe.log 2>&1; cat netprobe_runs/probe/*/artifacts/out/net.json'
```
Expected: `gateway` reachable (an HTTP 401 or 405 counts as reachable), and `github`, `raw_github` and `huggingface` blocked.
- If github is reachable, the Docker environment does not enforce the agent allowlist. **Stop and report.** The writer would be able to download the official tests, so the network has to be closed another way before any writer runs.
- If the gateway is blocked, the allowlist is enforced but misses the gateway; check `harbor run --help` for `--allow-agent-host` semantics and rerun.

- [ ] **Step 4: Write the failing tests**

`tests/test_writer.py`:
```python
import tomllib
from pathlib import Path

import pytest

from vb.leakscan import scan
from vb.writer import WRITERS, dump_toml, make_writer_task


def test_leakscan_finds_copies_canaries_and_fetches(tmp_path: Path):
    official = tmp_path / "official"
    official.mkdir()
    body = "def test_x():\n    assert compute() == 42, 'the answer must be forty-two, as the task says'\n"
    (official / "test_outputs.py").write_text(body)
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "Dockerfile").write_text("FROM python:3.12-slim\nRUN pip install pytest\nCOPY . /tests/\n")
    (mine / "test.sh").write_text("#!/bin/bash\npytest /tests/test_mine.py && echo 1 > /logs/verifier/reward.txt\n")
    (mine / "test_mine.py").write_text("def test_mine():\n    assert True\n")
    assert scan(mine, official) == []
    (mine / "copied.py").write_text(body)
    (mine / "Dockerfile").write_text("FROM python:3.12-slim\nRUN git clone https://github.com/harbor-framework/terminal-bench\n")
    (mine / "test_mine.py").write_text("# harbor-" + "canary GUID 1234\n")
    found = " | ".join(scan(mine, official))
    assert "byte-identical" in found and "github.com" in found and "canary" in found


def test_writers_form_a_full_model_by_harness_grid():
    cells = {(w["model_id"], w["harness"]) for w in WRITERS.values()}
    assert cells == {("sonnet", "claude-code"), ("luna", "codex"), ("sonnet", "terminus-2"), ("luna", "terminus-2")}
    for name, w in WRITERS.items():
        assert name == f"{w['model_id']}-{w['harness'].split('-')[0]}" or name == f"{w['model_id']}-{w['harness']}"
        assert all("HUD_API_KEY" not in kw for kw in w["kwargs"])          # the key is never an argument
        assert all(v in ("${HUD_API_KEY}",) or "hud.ai" in v for v in w["env"].values())

TOML = '''schema_version = "2.0"
artifacts = ["/app/out/", "/results/answer.csv"]

[task]
name = "terminal-bench/demo"

[verifier]
timeout_sec = 300.0
environment_mode = "separate"

[verifier.environment]
docker_image = "x-verifier@sha256:1"

[agent]
timeout_sec = 28800.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
docker_image = "x-environment@sha256:2"
'''


def test_dump_toml_round_trips():
    doc = {"schema_version": "2.0", "artifacts": ["/a/", "/b"], "task": {"name": "x/y", "description": ""},
           "agent": {"timeout_sec": 10.0, "network_mode": "allowlist", "allowed_hosts": ["h.ai"]},
           "environment": {"cpus": 2, "docker_image": 'img"q'}}
    assert tomllib.loads(dump_toml(doc)) == doc
    with pytest.raises(TypeError):
        dump_toml({"t": {"nested": {"deep": 1}}})


def test_make_writer_task(tmp_path: Path):
    src = tmp_path / "demo"
    (src / "tests").mkdir(parents=True)
    (src / "environment").mkdir()
    (src / "solution").mkdir()
    (src / "task.toml").write_text(TOML)
    (src / "instruction.md").write_text("Produce /results/answer.csv with the totals.")
    (src / "tests" / "test_outputs.py").write_text("SECRET = 1\n")
    (src / "solution" / "solve.sh").write_text("echo secret solution\n")
    (src / "environment" / "Dockerfile").write_text("FROM debian\n")
    out = make_writer_task(src, tmp_path / "writer_tasks")
    assert out.name == "write-verifier-demo"
    t = tomllib.loads((out / "task.toml").read_text())
    assert t["artifacts"] == ["/verifier_out/"]
    assert t["environment"]["docker_image"] == "x-environment@sha256:2"
    assert t["agent"]["network_mode"] == "allowlist" and t["agent"]["allowed_hosts"] == ["inference.hud.ai"]
    assert "verifier" in t and t["verifier"].get("environment_mode") is None
    text = (out / "instruction.md").read_text()
    assert "Produce /results/answer.csv with the totals." in text
    assert "- `/app/out/`" in text and "- `/results/answer.csv`" in text and "300" in text
    assert "`x-environment@sha256:2`" in text and "{" not in text                 # every placeholder filled
    every = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert "SECRET" not in every and "secret solution" not in every
    assert not (out / "solution").exists()
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_writer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb.writer'`

- [ ] **Step 6: Implement `vb/leakscan.py` and `vb/writer.py`**

`vb/leakscan.py`:
```python
"""Signs that a written verifier copied or fetches the official tests instead of writing its own.

Verifiers run offline while grading (vb.taskdir.set_verifier_offline), but a Dockerfile builds with network,
so a fetch at build time is caught here by its text. A disguised fetch (e.g. a package that bundles the
tests) is a residual this scan does not claim to catch; it is disclosed in the report."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

FETCH = re.compile(r"github\.com|githubusercontent|huggingface\.co|\bhf\.co\b|harbor-framework|harborframework|"
                   r"terminal-bench|tbench\.ai", re.I)
CANARY = "harbor-" + "canary"          # built at runtime so this scanner never matches its own source


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scan(tests_dir: Path, official_tests: Path) -> list[str]:
    official = {_sha(p): p.name for p in Path(official_tests).rglob("*") if p.is_file() and p.stat().st_size > 64}
    found = []
    for p in sorted(q for q in Path(tests_dir).rglob("*") if q.is_file()):
        rel = p.relative_to(tests_dir).as_posix()
        text = p.read_text(encoding="utf-8", errors="ignore")
        if CANARY in text:
            found.append(f"{rel}: carries the official tests' canary string")
        if p.stat().st_size > 64 and _sha(p) in official:
            found.append(f"{rel}: byte-identical to official {official[_sha(p)]}")
        m = FETCH.search(text)
        if m:
            found.append(f"{rel}: mentions {m.group(0)}")
    return found
```

`vb/writer.py`:

```python
"""Writer tasks: the solver's environment, an instruction to write tests/, and only the model gateway reachable."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from .leakscan import scan
from .util import DATA, HARBOR, ROOT, TB_ROOT, load_env, read_json, write_json

GATEWAY_HOST = "inference.hud.ai"
# Four writers = 2 models x 2 harnesses. "env" goes into the container (harnesses that run inside it); "host_key"
# names the variable the host-side harness (terminus-2, via LiteLLM) reads, set from HUD_API_KEY in the
# subprocess environment only.
WRITERS = {
    "sonnet-claude-code": {"model_id": "sonnet", "harness": "claude-code", "agent": "claude-code",
                           "model": "anthropic/claude-sonnet-5",
                           "env": {"ANTHROPIC_BASE_URL": "https://inference.hud.ai", "ANTHROPIC_API_KEY": "${HUD_API_KEY}"},
                           "kwargs": ["disallowed_tools=WebFetch,WebSearch", "max_budget_usd=3"], "host_key": None},
    "luna-codex": {"model_id": "luna", "harness": "codex", "agent": "codex", "model": "openai/gpt-6-luna",
                   "env": {"OPENAI_BASE_URL": "https://inference.hud.ai/v1", "OPENAI_API_KEY": "${HUD_API_KEY}"},
                   "kwargs": ["web_search=disabled"], "host_key": None},
    "sonnet-terminus": {"model_id": "sonnet", "harness": "terminus-2", "agent": "terminus-2",
                        "model": "anthropic/claude-sonnet-5", "env": {},
                        "kwargs": ["api_base=https://inference.hud.ai", "max_turns=60"], "host_key": "ANTHROPIC_API_KEY"},
    "luna-terminus": {"model_id": "luna", "harness": "terminus-2", "agent": "terminus-2", "model": "openai/gpt-6-luna",
                      "env": {}, "kwargs": ["api_base=https://inference.hud.ai/v1", "max_turns=60"],
                      "host_key": "OPENAI_API_KEY"},
}
# Every writer is also bounded by the writer task's agent timeout (WRITER_TIMEOUT_S); Codex has no money setting,
# so for it the timeout is the only hard bound, and spend is checked between writers (Task 6 Step 5).
WRITER_TIMEOUT_S = 1800.0
WRITER_TASKS = DATA / "writer_tasks"


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "[" + ", ".join(json.dumps(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


def dump_toml(doc: dict) -> str:
    lines = [f"{k} = {_scalar(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for table, body in doc.items():
        if isinstance(body, dict):
            lines += ["", f"[{table}]"] + [f"{k} = {_scalar(v)}" for k, v in body.items()]
    return "\n".join(lines) + "\n"


def make_writer_task(task_dir: Path, out_root: Path = WRITER_TASKS) -> Path:
    task_dir = Path(task_dir)
    t = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    env = {k: v for k, v in (t.get("environment") or {}).items()
           if k in ("docker_image", "build_timeout_sec", "cpus", "memory_mb", "storage_mb", "gpus", "os")}
    out = Path(out_root) / f"write-verifier-{task_dir.name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(task_dir / "environment", out / "environment")
    doc = {"schema_version": "2.0", "artifacts": ["/verifier_out/"],
           "task": {"name": f"vbench/write-verifier-{task_dir.name}", "description": ""},
           "metadata": {"source_task": t["task"]["name"]},
           "agent": {"timeout_sec": WRITER_TIMEOUT_S, "network_mode": "allowlist", "allowed_hosts": [GATEWAY_HOST]},
           "verifier": {"timeout_sec": 60.0},
           "environment": env}
    (out / "task.toml").write_text(dump_toml(doc), encoding="utf-8")
    artifacts = "\n".join(f"- `{a}`" for a in t.get("artifacts", []))
    template = (ROOT / "templates" / "writer_instruction.md").read_text(encoding="utf-8")
    (out / "instruction.md").write_text(template.format(
        instruction=(task_dir / "instruction.md").read_text(encoding="utf-8").strip(), artifacts=artifacts,
        timeout=int(float((t.get("verifier") or {}).get("timeout_sec") or 300)),
        env_image=env.get("docker_image", "(built from the task's environment/Dockerfile)")), encoding="utf-8")
    (out / "tests").mkdir()
    (out / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        "if [ -s /verifier_out/tests/test.sh ] && [ -s /verifier_out/tests/Dockerfile ]; then\n"
        "  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n", encoding="utf-8")
    (out / "tests" / "test.sh").chmod(0o755)
    return out


def run_writer(writer: str, tasks_dir: Path, out_root: Path, attempts: int, n_concurrent: int = 2,
               include: list[str] | None = None) -> Path:
    w = WRITERS[writer]
    if "HUD_API_KEY" not in os.environ:
        raise RuntimeError("HUD_API_KEY not loaded (expected in /root/vbench/.env)")
    cmd = [HARBOR, "run", "-p", str(tasks_dir), "-a", w["agent"], "-m", w["model"], "-k", str(attempts),
           "-n", str(n_concurrent), "-o", str(out_root), "--job-name", writer, "-y", "-q",
           "--allow-agent-host", GATEWAY_HOST]
    for k, v in w["env"].items():
        cmd += ["--ae", f"{k}={v}"]
    for kw in w["kwargs"]:
        cmd += ["--ak", kw]
    for name in include or []:
        cmd += ["-i", name]
    env = os.environ.copy()
    if w["host_key"]:
        env[w["host_key"]] = os.environ["HUD_API_KEY"]      # host-side harness only; never in argv
    Path(out_root).mkdir(parents=True, exist_ok=True)
    with (Path(out_root) / f"{writer}.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return Path(out_root) / writer


def collect(job_dir: Path, writer: str, dest: Path | None = None) -> list[dict]:
    dest = Path(dest) if dest is not None else DATA / "verifiers"
    rows = []
    for rj in sorted(Path(job_dir).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        task = r["task_name"].split("/")[-1].removeprefix("write-verifier-")
        src = rj.parent / "artifacts" / "verifier_out" / "tests"
        ok = (src / "test.sh").is_file() and (src / "Dockerfile").is_file()
        target = dest / writer / task / rj.parent.name / "tests"
        if ok:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        rows.append({"writer": writer, "task": task, "trial": rj.parent.name, "ok": ok,
                     "verifier": str(target) if ok else None, "exception": r.get("exception_info"),
                     "leak": scan(target, TB_ROOT / task / "tests") if ok else [],
                     "agent_result": r.get("agent_result")})
    write_json(dest / writer / f"collect_{Path(job_dir).parent.name}.json", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--collect")
    ap.add_argument("--writer")
    a = ap.parse_args()
    tasks = read_json(ROOT / "configs" / "m1.json")["tasks"]
    if a.make:
        for t in tasks:
            print(make_writer_task(TB_ROOT / t))
    if a.run:
        print("loaded:", load_env())
        from .util import new_run_dir
        run_dir = new_run_dir(f"write_{a.run}")
        include = [f"write-verifier-{t}" for t in a.tasks.split(",") if t]
        job = run_writer(a.run, WRITER_TASKS, run_dir, a.attempts, include=include)
        rows = collect(job, a.run)
        print(json.dumps([{k: r[k] for k in ("task", "trial", "ok")} for r in rows], indent=1))
    if a.collect:
        print(json.dumps(collect(Path(a.collect), a.writer), indent=1)[:4000])
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `26 passed`

- [ ] **Step 8: Smoke test, one task, one attempt per writer** (all four writers, under $8)

```bash
tools/sync.sh && ssh root@178.156.248.194 'cd /root/vbench/repo && python3 -m vb.writer --make && T=$(python3 -c "import json; print(json.load(open(\"configs/m1.json\"))[\"tasks\"][0])") && for W in sonnet-claude-code luna-codex sonnet-terminus luna-terminus; do python3 -m vb.writer --run $W --attempts 1 --tasks $T; done'
```
Expected: for each of the four writers, one row with `"ok": true`, and a folder under `/root/vbench/verifiers/<writer>/<task>/`. For the two terminus writers, the solver machine needs no gateway access at all; they must still succeed with the allowlist in place.

Then check, in this order:
1. **The key never landed on disk:**
   ```bash
   ssh root@178.156.248.194 'cd /root/vbench/runs && grep -rl "$(sed -n "s/^HUD_API_KEY=//p" /root/vbench/.env)" . | head'
   ```
   Expected: no output.
2. **The model was reached through the gateway:** the log `runs/write_<writer>/*/<writer>.log` shows no authentication error. If `${HUD_API_KEY}` was not expanded by Harbor, the log shows a 401 and the agent made no progress. In that case pass the key with `--env-file /root/vbench/.env` plus `--ae OPENAI_API_KEY=${HUD_API_KEY}`, and recheck check 1.
3. **Spend:** `agent_result` in the provenance JSON holds token counts and, for some harnesses, cost; Codex may report none. So the owner also reads the HUD credit balance before and after the smoke test, and the larger of the two figures counts. Record the per-attempt cost for all four.
   - If any attempt cost more than $3, cap that writer before the batch: `max_budget_usd` for Claude Code, `max_turns` for terminus-2.
   - If the four smoke attempts together cost more than $8, the batch (40 attempts) would exceed the $100 cap. Stop and report the measured costs instead of starting it.
4. **The verifier works on a known pass:** build a verifier task from the Sonnet folder, regrade the task's kept pool and look at TPR and TNR. This is a preview of Task 6.

- [ ] **Step 9: Commit**

```bash
git add vb/writer.py templates tests/test_writer.py
git commit -m "Writer tasks: the solver's environment plus a verifier-writing instruction, gateway-only network proven by a probe; smoke test on one task"
```

---

### Task 6: Evaluation, the M1 run and the report

**Files:**
- Create: `vb/evaluate.py`, `vb/report.py`, `README.md`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `vb.pool.load_manifest, kept_job`, `vb.taskdir.make_verifier_task, BASELINES`, `vb.regrade.run_regrade, read_rewards, verdict`, `vb.scorer.*`
- Produces:
  - `vb.evaluate.verifiers_for(task) -> dict[str, Path]`: the baselines, the original tests, and every collected writer verifier as `<writer>/<attempt>`.
  - `vb.evaluate.evaluate(tasks, run_dir) -> dict`: writes `scores.json`, `verdicts.jsonl` and `summary.md`; writes `BASELINE_FAILED.txt` and exits 1 on a baseline problem.
  - `vb.report.render(scores: dict) -> str`.
  - CLI: `python3 -m vb.evaluate [--tasks a,b]`.

- [ ] **Step 1: Write the failing test for the report**

`tests/test_report.py`:
```python
from vb.report import render


def m(ba):
    return {"n_tasks": 5, "tpr": ba, "tnr": ba, "ba": ba, "flakiness": 0.0}


SCORES = {"name": "m1", "baseline_problems": [], "writer_cost_usd": {"sonnet-claude-code": 1.25},
          "writers": {"sonnet-claude-code": {"model_id": "sonnet", "harness": "claude-code"},
                      "sonnet-terminus": {"model_id": "sonnet", "harness": "terminus-2"},
                      "luna-codex": {"model_id": "luna", "harness": "codex"},
                      "luna-terminus": {"model_id": "luna", "harness": "terminus-2"}},
          "per_task": {"demo": {
              "always_pass": {"tpr": 1.0, "tnr": 0.0, "ba": 0.5, "flakiness": 0.0, "tnr_by_source": {"real_fail": 0.0}, "n_correct": 11, "n_wrong": 11},
              "sonnet-claude-code/t1": {"tpr": 0.9, "tnr": 0.8, "ba": 0.85, "flakiness": 0.05, "tnr_by_source": {"near_miss": 0.6, "real_fail": 1.0}, "n_correct": 11, "n_wrong": 11}}},
          "macro": {"sonnet-claude-code": m(0.85), "sonnet-terminus": m(0.7), "luna-codex": m(0.8), "luna-terminus": m(0.65)}}


def test_render_names_every_writer_and_the_baseline_status():
    md = render(SCORES)
    assert "sonnet-claude-code" in md and "85.0" in md and "60.0" in md and "$1.25" in md
    assert "Baselines: all hold" in md
    assert "NOT VALID" in render({**SCORES, "baseline_problems": ["demo: always_fail: tnr = 0.9"]})


def test_render_model_by_harness_table():
    md = render(SCORES)
    grid = md.split("## Model × harness")[1].split("##")[0]
    assert "| sonnet | 85.0 | n/a | 70.0 |" in grid          # columns: claude-code, codex, terminus-2
    assert "| luna | n/a | 80.0 | 65.0 |" in grid
    with_ci = {**SCORES, "macro": {**SCORES["macro"], "luna-codex": {**m(0.8), "ba_ci": (0.6, 0.9)}}}
    assert "80.0 [60.0, 90.0]" in render(with_ci)


def test_render_lists_leaks():
    md = render({**SCORES, "leaks": {"luna-codex/t1": ["Dockerfile: mentions github.com"]}})
    assert "excluded" in md and "luna-codex/t1" in md and "github.com" in md
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vb.report'`

- [ ] **Step 3: Implement `vb/report.py` and `vb/evaluate.py`**

`vb/report.py`:
```python
"""summary.md from scores.json."""
from __future__ import annotations


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ba(m: dict) -> str:
    ci = m.get("ba_ci")
    return _p(m.get("ba")) + (f" [{_p(ci[0])}, {_p(ci[1])}]" if ci else "")


def render(scores: dict) -> str:
    lines = [f"# {scores['name']}", ""]
    if scores["baseline_problems"]:
        lines += ["**Baselines failed; these results are NOT VALID:**", ""] + [f"- {p}" for p in scores["baseline_problems"]]
    else:
        lines += ["Baselines: all hold (original tests BA 100, always pass and always fail BA 50 on every task)."]
    writers = scores.get("writers") or {}
    if writers:
        harnesses = sorted({w["harness"] for w in writers.values()})
        models = sorted({w["model_id"] for w in writers.values()}, reverse=True)
        cell = {(w["model_id"], w["harness"]): scores["macro"].get(name) for name, w in writers.items()}
        lines += ["", "## Model × harness (BA with 95% bootstrap interval, macro over tasks and attempts)", "",
                  "| model | " + " | ".join(harnesses) + " |", "|---|" + "---:|" * len(harnesses)]
        lines += [f"| {mo} | " + " | ".join(_ba(cell[(mo, h)]) if cell.get((mo, h)) else "n/a" for h in harnesses) + " |"
                  for mo in models]
        lines += ["", "A model or harness effect is stated only where two intervals do not overlap."]
    lines += ["", "## Writers (macro over tasks and attempts)", "", "| writer | task-attempts | TPR | TNR | BA | flaky | cost |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for w, m in sorted(scores["macro"].items()):
        cost = scores.get("writer_cost_usd", {}).get(w)
        lines.append(f"| {w} | {m['n_tasks']} | {_p(m['tpr'])} | {_p(m['tnr'])} | {_ba(m)} | {_p(m['flakiness'])} | "
                     f"{'n/a' if cost is None else f'${cost:.2f}'} |")
    if scores.get("leaks"):
        lines += ["", "## Verifiers excluded: copied or fetched the official tests", ""]
        lines += [f"- `{name}`: " + "; ".join(found) for name, found in sorted(scores["leaks"].items())]
    lines += ["", "## By task", "", "| task | verifier | correct/wrong | TPR | TNR | BA | TNR near miss | TNR real fail | TNR nop | flaky |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, per in sorted(scores["per_task"].items()):
        for name, s in sorted(per.items()):
            by = s.get("tnr_by_source", {})
            lines.append(f"| {task} | {name} | {s['n_correct']}/{s['n_wrong']} | {_p(s['tpr'])} | {_p(s['tnr'])} | {_p(s['ba'])} | "
                         f"{_p(by.get('near_miss'))} | {_p(by.get('real_fail'))} | {_p(by.get('nop'))} | {_p(s['flakiness'])} |")
    return "\n".join(lines) + "\n"
```

`vb/evaluate.py`:
```python
"""Regrade every verifier on every kept pool, score against the labels, and check the baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pool import kept_job, load_manifest
from .regrade import read_rewards, run_regrade, verdict
from .report import render
from .scorer import Verdict, check_baselines, macro, score_task
from .taskdir import BASELINES, make_verifier_task
from .util import DATA, ROOT, TB_ROOT, new_run_dir, read_json, write_json
from .writer import WRITERS


def verifiers_for(task: str) -> dict[str, Path]:
    out = {"original": TB_ROOT / task / "tests", **BASELINES}
    for d in sorted((DATA / "verifiers").glob(f"*/{task}/*/tests")):
        out[f"{d.parents[2].name}/{d.parent.name}"] = d
    return out


def leaks() -> dict[str, list[str]]:
    """{'<writer>/<trial>': findings} for every collected verifier the leak scan flagged."""
    out = {}
    for f in (DATA / "verifiers").glob("*/collect_*.json"):
        for r in read_json(f):
            if r.get("leak"):
                out[f"{r['writer']}/{r['trial']}"] = r["leak"]
    return out


def writer_costs() -> dict[str, float]:
    costs: dict[str, float] = {}
    for f in (DATA / "verifiers").glob("*/collect_*.json"):
        for r in read_json(f):
            c = (r.get("agent_result") or {}).get("cost_usd")
            if c is not None:
                costs[r["writer"]] = costs.get(r["writer"], 0.0) + float(c)
    return costs


def evaluate(tasks: list[str], run_dir: Path) -> dict:
    per_task, rows, by_writer, flagged = {}, [], {}, leaks()
    for task in tasks:
        kept = [e for e in load_manifest(task) if e["kept"]]
        job = kept_job(task)
        per_task[task] = {}
        for name, tests in verifiers_for(task).items():
            vt = make_verifier_task(TB_ROOT / task, tests, run_dir / "vtasks" / task / name.replace("/", "__"))
            rewards = read_rewards(run_regrade(job, vt, run_dir / "regrades" / task, name.replace("/", "__")))
            verdicts = [Verdict(e["entry_id"], e["source"], e["label"], verdict(rewards.get(e["entry_id"]))) for e in kept]
            s = score_task(verdicts)
            per_task[task][name] = s
            rows += [{"task": task, "verifier": name, "entry_id": v.entry_id, "source": v.source, "label": v.label,
                      "verdict": v.verdict, "reward": rewards.get(v.entry_id)} for v in verdicts]
            if "/" in name and name in flagged:
                s["leak"] = flagged[name]                  # scored and shown, never counted in a writer's macro
            elif "/" in name:
                by_writer.setdefault(name.split("/")[0], []).append(s)
    problems = []
    for task, per in per_task.items():
        problems += [f"{task}: {p}" for p in check_baselines(per["always_pass"], per["always_fail"], per["original"],
                                                             per["network_probe"])]
    scores = {"name": run_dir.name, "tasks": tasks, "per_task": per_task,
              "macro": {w: macro(ss) for w, ss in by_writer.items()}, "baseline_problems": problems,
              "writer_cost_usd": writer_costs(), "leaks": flagged,
              "writers": {n: {"model_id": w["model_id"], "harness": w["harness"]} for n, w in WRITERS.items()}}
    write_json(run_dir / "scores.json", scores)
    (run_dir / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "summary.md").write_text(render(scores), encoding="utf-8")
    if problems:
        (run_dir / "BASELINE_FAILED.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        raise SystemExit(1)
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t] or read_json(ROOT / "configs" / "m1.json")["tasks"]
    run_dir = new_run_dir("m1_eval")
    evaluate(tasks, run_dir)
    print(run_dir)
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest -q`
Expected: `29 passed`

- [ ] **Step 5: Run all four writers on all five tasks, one writer at a time** (two attempts each; within the $100 cap)

Run one writer, then check spend before starting the next:
```bash
W=sonnet-claude-code; tools/sync.sh && ssh root@178.156.248.194 "cd /root/vbench/repo && (python3 -m vb.writer --run $W --attempts 2 > /root/vbench/write_$W.out 2>&1; echo DONE > /root/vbench/write_$W.done) > /dev/null 2>&1 &"
```
1. Wait with a single background command until `/root/vbench/write_$W.done` exists.
2. Add up the writer's recorded cost with the owner's HUD balance reading.
3. Continue with the next writer (`luna-codex`, then `sonnet-terminus`, then `luna-terminus`) **only if the cumulative spend plus that writer's smoke-test cost × 10 stays under $100.** Otherwise stop and report the spend so far.

Then count the collected verifiers and the leak findings:
```bash
ssh root@178.156.248.194 'find /root/vbench/verifiers -name test.sh -path "*/tests/*" | sort; python3 -c "
import json, glob
for f in glob.glob(\"/root/vbench/verifiers/*/collect_*.json\"):
    for r in json.load(open(f)):
        if r[\"leak\"] or not r[\"ok\"]: print(r[\"writer\"], r[\"task\"], r[\"trial\"], \"ok\" if r[\"ok\"] else \"no_verifier\", r[\"leak\"])"'
```
Expected: up to 40 verifiers (4 writers × 5 tasks × 2 attempts).
- An attempt that produced no folder is `no_verifier`, recorded in the collect JSON and reported, not rerun.
- A verifier with leak findings is still scored but excluded from its writer's numbers and listed in the report. **Read each finding before accepting it:** a verifier that merely mentions "terminal-bench" in a comment is a false positive. In that case narrow `FETCH` to fetch commands and rerun the scan, rather than excluding honest work.

- [ ] **Step 6: Evaluate** (compute only; 43 verifiers × about 22 entries, so roughly 8 to 12 hours at 4 regrades at a time; run it overnight)

```bash
ssh root@178.156.248.194 'cd /root/vbench/repo && nohup python3 -u -m vb.evaluate > /root/vbench/eval.out 2>&1 & echo started'
```
Wait until `/root/vbench/eval.out` ends with a run directory path or a traceback, using one background `until` loop that matches both. Then:
```bash
ssh root@178.156.248.194 'd=$(tail -1 /root/vbench/eval.out); ls $d; cat $d/summary.md'
```
Expected: no `BASELINE_FAILED.txt`, and `summary.md` shows each writer's TPR, TNR, BA and TNR by source per task.

- [ ] **Step 7: Write `README.md`, bring the results home and commit**

`README.md`:
```markdown
# Verifier Bench

How well do agents write verifiers, and how much of that is the model versus the harness around it?
An agent gets a task and its environment and writes the tests that decide whether an attempt succeeded.
We score those tests on real attempts whose pass/fail we already know from the task's own human-written
tests, for each model in its own harness and in a shared one.

- Design: `docs/design.md`
- M1 plan: `docs/plan_m1.md`
- M1 results: `runs/m1/summary.md`

Runs on the server with Harbor and Docker: `python3 -m vb.index`, `vb.select`, `vb.pool --task <t>`,
`vb.writer --make`, `vb.writer --run <luna|sonnet>`, `vb.evaluate`.
```

```bash
mkdir -p runs/m1 && ssh root@178.156.248.194 'tail -1 /root/vbench/eval.out' | xargs -I{} scp root@178.156.248.194:'{}/summary.md' root@178.156.248.194:'{}/scores.json' root@178.156.248.194:'{}/verdicts.jsonl' runs/m1/
scp -r root@178.156.248.194:/root/vbench/verifiers runs/m1/verifiers
git add README.md vb/evaluate.py vb/report.py tests/test_report.py runs/m1
git commit -m "M1: verifiers from 2 models x 2 harnesses scored on five Terminal-Bench 4.0 tasks against leaderboard attempts; baselines hold"
```

Then report to the owner:
- the model × harness table: the model effect in the terminus-2 column, the harness effect within each row;
- the writer table and the per-task table from `summary.md`;
- per-writer cost;
- any `no_verifier` attempts;
- dropped pool entries with their reasons;
- tasks excluded for community-reported test problems.
