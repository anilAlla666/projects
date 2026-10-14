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


def test_tests_hash_follows_content_not_timestamps(tmp_path):
    import os
    from vb.regrade import tests_hash
    a, b = tmp_path / "a", tmp_path / "b"
    for d, body in ((a, "echo 1"), (b, "echo 0")):
        d.mkdir()
        (d / "test.sh").write_text(body)
        os.utime(d / "test.sh", (1_000_000, 1_000_000))              # same size, same timestamp
    assert tests_hash(a) != tests_hash(b)
    (b / "test.sh").write_text("echo 1")
    assert tests_hash(a) == tests_hash(b)



def test_infra_failures_only_flags_machinery(tmp_path):
    from vb.regrade import infra_failures
    job = tmp_path / "j"
    trial(job, "a", "src1", reward=1.0)
    trial(job, "b", "src2", exc={"exception_message": "... Error response from daemon: all predefined address pools have been fully subnetted"})
    trial(job, "c", "src3", exc={"exception_message": "Docker compose command failed ... container exited (ENTRYPOINT)"})
    assert infra_failures(job) == ["src2"]
