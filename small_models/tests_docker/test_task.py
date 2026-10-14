import subprocess
from pathlib import Path

from vm.sandbox.runner import AS_USER, DOCKER, IMAGE
from vm.tasks.task import make_tasks

LIB = 'def bigger(a, b):\n    """Return the larger of a and b."""\n    if a >= b:\n        return a\n    return b\n'
TESTS = "from lib import bigger\n\n" + "".join(f"def test_filler_{i}():\n    assert {i} == {i}\n\n" for i in range(120)) + (
    "def test_equal():\n    assert bigger(2, 2) == 2\n\ndef test_first():\n    assert bigger(3, 1) == 3\n\n"
    "def test_second():\n    assert bigger(1, 3) == 3\n")


def test_tasks_hide_everything_but_the_named_failing_test(tmp_path):
    lib = tmp_path / "lib"
    src = lib / "src"
    (src / "tests").mkdir(parents=True)
    (src / "lib.py").write_text(LIB)
    (src / "tests" / "test_lib.py").write_text(TESTS)
    (src / "pyproject.toml").write_text('[project]\nname = "lib"\nversion = "0"\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n[tool.setuptools]\npy-modules = ["lib"]\n')
    subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "-v", f"{lib}:/work", "-w", "/work/src", IMAGE, "sh", "-c",
                    "python -m venv /work/venv && /work/venv/bin/pip install -q -e . pytest"], check=True)
    tests = [f"tests/test_lib.py::test_filler_{i}" for i in range(120)] + [f"tests/test_lib.py::test_{n}" for n in ("equal", "first", "second")]
    rec = {"name": "lib", "split": "train", "ok": True, "tests": tests, "visible": tests[::2]}
    tasks = make_tasks(rec, tmp_path, max_tasks=12)
    assert tasks
    for t in tasks:
        assert t["named_test"] in t["failing"] and t["named_test"] not in t["oracle_only"]
        assert set(t["failing"]) - {t["named_test"]} <= set(t["oracle_only"])
        others = [f.split("::")[1] for f in t["failing"] if f != t["named_test"]]
        assert not any(o in t["task_text"] for o in others)
        assert t["function_lines"][t["bug_line"]] != t["original_source"].splitlines()[t["function_start"] - 1 + t["bug_line"]]
