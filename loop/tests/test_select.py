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


def test_categories_are_ordered_by_their_strongest_pool(tmp_path):
    make(tmp_path, "a_weak", category="Alpha")
    make(tmp_path, "z_strong", category="Zulu")
    counts = {"a_weak": {"pass": 11, "fail": 50, "error": 0}, "z_strong": {"pass": 60, "fail": 60, "error": 0}}
    assert select(tmp_path, counts, {}, n=1)[0] == ["z_strong"]
