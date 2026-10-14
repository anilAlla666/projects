import importlib.util
import json
from pathlib import Path

from vb import hudtask

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grader", ROOT / "templates" / "hud" / "grader.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def test_verdict_only_reward_one_passes(tmp_path):
    assert grader.verdict_of(tmp_path) == "fail"                      # no file: reject
    (tmp_path / "reward.txt").write_text("1\n")
    assert grader.verdict_of(tmp_path) == "pass"
    for bad in ("0", "0.99", "", "yes", "1.5", "nan"):
        (tmp_path / "reward.txt").write_text(bad)
        assert grader.verdict_of(tmp_path) == "fail", bad
    (tmp_path / "reward.txt").unlink()
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 1.0}))
    assert grader.verdict_of(tmp_path) == "pass"


def test_score_floors_trivial_verifiers_and_moves_with_skill():
    labels = ["pass"] * 4 + ["fail"] * 6
    def rows(verdicts):
        return [{"label": l, "verdict": v} for l, v in zip(labels, verdicts)]
    assert grader.score(rows(["pass"] * 10))["reward"] == 0.0          # always pass
    assert grader.score(rows(["fail"] * 10))["reward"] == 0.0          # always fail
    perfect = grader.score(rows(labels))
    assert perfect["reward"] == 1.0 and perfect["balanced_accuracy"] == 1.0
    half = grader.score(rows(["pass", "pass", "fail", "fail"] + ["fail"] * 6))   # TPR .5, TNR 1
    assert half["reward"] == 0.5 and half["accepted_correct"] == "2/4" and half["rejected_wrong"] == "6/6"
    inverted = grader.score(rows(["fail"] * 4 + ["pass"] * 6))
    assert inverted["reward"] == 0.0


def test_bwrap_hides_grader_and_drops_caps(tmp_path):
    argv = grader.bwrap_argv(tmp_path, ["/app/out.txt"], {"PATH": "/bin"}, "/app", ["/bin/true"])
    joined = " ".join(argv)
    assert f"--tmpfs {grader.GRADER}" in joined and "--tmpfs /verifier_out" in joined
    assert "--cap-drop ALL" in joined and "--unshare-net" in joined and "--clearenv" in joined
    assert "/app/out.txt" not in argv                                   # artifact absent in stage: not bound
    (tmp_path / "art" / "app").mkdir(parents=True)
    (tmp_path / "art" / "app" / "out.txt").write_text("x")
    (tmp_path / "rw" / "app").mkdir(parents=True)
    argv = grader.bwrap_argv(tmp_path, ["/app/out.txt"], {}, "/", ["/bin/true"])
    assert "/app/out.txt" in argv
    i = argv.index(str(tmp_path / "rw" / "app"))
    assert argv[i - 1] == "--bind" and argv[i + 1] == "/app"
    assert argv.index("--ro-bind") < i < argv.index("/app/out.txt")      # copy of /app over the read-only image


def test_grading_dockerfile_drops_copy_of_official_tests(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "Dockerfile").write_text(
        "# harbor-" "canary GUID x\nFROM python:3.13-slim\nRUN pip install pytest\nCOPY . /tests/\nRUN mkdir -p /app\n")
    text = hudtask.grading_dockerfile(tmp_path)
    assert "canary" not in text and "COPY" not in text
    assert "FROM python:3.13-slim" in text and "RUN mkdir -p /app" in text
