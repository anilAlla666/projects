import tomllib
from pathlib import Path

import pytest

from vb.leakscan import scan
from vb.writer import WRITERS, dump_toml, make_writer_task


def test_leakscan_finds_copies_and_fetches_but_not_the_environment(tmp_path: Path):
    official = tmp_path / "official"
    official.mkdir()
    body = "def test_x():\n    assert compute() == 42, 'the answer must be forty-two, as the task says'\n"
    (official / "test_outputs.py").write_text(body)
    env = "harborframework/terminal-bench:demo-environment-1@sha256:abc"
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "Dockerfile").write_text(f"FROM {env}\nCOPY . /tests/\n")
    (mine / "test.sh").write_text("#!/bin/bash\npytest /tests/test_mine.py && echo 1 > /logs/verifier/reward.txt\n")
    (mine / "run.sh").write_text("# harbor-" + "canary GUID 1234 (copied from the environment)\n")
    assert scan(mine, official, (env,)) == []
    assert scan(mine, official) != []                                   # without the allowance, the FROM is flagged
    (mine / "copied.py").write_text(body)
    (mine / "Dockerfile").write_text("FROM harborframework/terminal-bench:demo-verifier-1@sha256:def\nCOPY . /tests/\n")
    (mine / "fetch.sh").write_text("curl -sL https://raw.githubusercontent.com/x/y/z.py\n")
    found = " | ".join(scan(mine, official, (env,)))
    assert "byte-identical" in found and "harborframework" in found and "githubusercontent" in found
    given = tmp_path / "environment"
    given.mkdir()
    (given / "input.py").write_text(body)                               # the copied file is one of the task's inputs
    assert "byte-identical" not in " | ".join(scan(mine, official, (env,), given))


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
    (src / "instruction.md").write_text("<!-- harbor-" + "canary GUID 1234 -->\nProduce /results/answer.csv with the totals.")
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
    assert "SECRET" not in every and "secret solution" not in every and "canary" not in every
    assert not (out / "solution").exists()


def test_v2_writer_task_uses_the_protocol_and_more_time(tmp_path: Path):
    from vb.writer import ALL_WRITERS, WRITERS_V2
    src = tmp_path / "demo"
    (src / "tests").mkdir(parents=True)
    (src / "environment").mkdir()
    (src / "task.toml").write_text(TOML)
    (src / "instruction.md").write_text("Produce /results/answer.csv with the totals.")
    (src / "environment" / "Dockerfile").write_text("FROM debian\n")
    out = make_writer_task(src, tmp_path / "v2", variant="v2")
    t = tomllib.loads((out / "task.toml").read_text())
    assert t["agent"]["timeout_sec"] == 5400.0
    text = (out / "instruction.md").read_text()
    assert "requirements.md" in text and "90 minutes" in text and "FROM x-environment@sha256:2" in text and "{" not in text
    assert set(WRITERS_V2) == {n + "-v2" for n in WRITERS} and all(n in ALL_WRITERS for n in WRITERS_V2)
    assert "max_budget_usd=8" in WRITERS_V2["sonnet-claude-code-v2"]["kwargs"]
    assert "max_turns=150" in WRITERS_V2["luna-terminus-v2"]["kwargs"]
    assert WRITERS_V2["luna-codex-v2"]["harness"] == "codex+protocol"
