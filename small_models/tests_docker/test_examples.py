import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_judge_data import _task_examples  # noqa: E402
from tests_docker.test_task import LIB, TESTS  # noqa: E402
from vm.sandbox.runner import AS_USER, DOCKER, IMAGE  # noqa: E402
from vm.tasks.mutate import pick_heldout  # noqa: E402
from vm.tasks.task import make_tasks  # noqa: E402


def test_examples_end_to_end(tmp_path):
    lib = tmp_path / "lib"
    src = lib / "src"
    (src / "tests").mkdir(parents=True)
    (src / "lib.py").write_text(LIB)
    (src / "tests" / "test_lib.py").write_text(TESTS)
    (src / "pyproject.toml").write_text('[project]\nname = "lib"\nversion = "0"\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n[tool.setuptools]\npy-modules = ["lib"]\n')
    subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "-v", f"{lib}:/work", "-w", "/work/src", IMAGE, "sh", "-c",
                    "python -m venv /work/venv && /work/venv/bin/pip install -q -e . pytest"], check=True)
    tests = [f"tests/test_lib.py::test_filler_{i}" for i in range(120)] + [f"tests/test_lib.py::test_{n}" for n in ("equal", "first", "second")]
    task = make_tasks({"name": "lib", "split": "test", "ok": True, "tests": tests, "visible": tests[::2]}, tmp_path)[0]
    exs = _task_examples(task, tmp_path, [], pick_heldout())
    by = {}
    for e in exs:
        by.setdefault(e["attempt_kind"], []).append(e)
    assert by["revert"][0]["label"] == 1 and by["revert"][0]["pointer"] == 0
    for kind in ("fake_output", "persuasion", "test_edit"):
        if kind in by:
            assert all(e["label"] == 0 for e in by[kind]), kind
    assert by["persuasion"][0]["untrusted"]
    for e in exs:
        assert e["evidence"][0]["text"].startswith("visible tests:")
        assert 0 <= e["pointer"] < len(e["evidence"])
