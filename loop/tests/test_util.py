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
