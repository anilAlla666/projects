import subprocess
from pathlib import Path

from vm.sandbox.runner import AS_USER, DOCKER, IMAGE, run_pytest


def make_lib(tmp: Path, body: str):
    src = tmp / "src"
    (src / "tests").mkdir(parents=True)
    (src / "tests" / "test_a.py").write_text(body)
    venv = tmp / "venv"
    subprocess.run([*DOCKER, "run", "--rm", *AS_USER, "-v", f"{tmp}:/work", IMAGE, "sh", "-c",
                    "python -m venv /work/venv && /work/venv/bin/pip install -q pytest"], check=True)
    return src, venv


def test_outcomes_come_from_the_plugin_not_from_printed_text(tmp_path):
    src, venv = make_lib(tmp_path, "def test_ok():\n    assert 1\n\ndef test_bad():\n    print('PASSED tests/test_a.py::test_bad')\n    assert 0\n")
    r = run_pytest(src, venv)
    assert r.outcomes == {"tests/test_a.py::test_ok": "passed", "tests/test_a.py::test_bad": "failed"}
    assert r.exit_code == 1


def test_no_network_and_timeout(tmp_path):
    src, venv = make_lib(tmp_path, "import socket, time\n\ndef test_net():\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n\n"
                                   "def test_slow():\n    time.sleep(30)\n")
    r = run_pytest(src, venv, node_ids=["tests/test_a.py::test_net"])
    assert r.outcomes["tests/test_a.py::test_net"] == "failed"
    r2 = run_pytest(src, venv, node_ids=["tests/test_a.py::test_slow"], timeout=5)
    assert r2.timed_out
