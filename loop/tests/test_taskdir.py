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


def test_pin_verifier_image_replaces_the_official_pin():
    from vb.taskdir import official_verifier_image, pin_verifier_image
    out = tomllib.loads(pin_verifier_image(TOML, "vbv:abc123"))
    assert out["verifier"]["environment"]["docker_image"] == "vbv:abc123"
    assert out["environment"]["docker_image"].endswith("sha256:def")
    # a task that no longer pins an image (the October checkout) gets a [verifier.environment] table
    added = tomllib.loads(pin_verifier_image('[verifier]\nenvironment_mode = "separate"\n', "vbv:x"))
    assert added["verifier"]["environment"]["docker_image"] == "vbv:x"
    assert added["verifier"]["environment_mode"] == "separate"


def test_make_verifier_task_with_an_image(tmp_path):
    src = task(tmp_path)
    from vb.taskdir import official_verifier_image
    assert official_verifier_image(src).endswith("sha256:abc")
    out = make_verifier_task(src, BASELINES["always_fail"], tmp_path / "c", image="vbv:0123")
    t = tomllib.loads((out / "task.toml").read_text())
    assert t["verifier"]["environment"]["docker_image"] == "vbv:0123" and t["verifier"]["network_mode"] == "no-network"
