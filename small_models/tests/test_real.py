import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_real import _first_changed, _visible  # noqa: E402

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,6 +10,6 @@ def f(x):
     a = 1
     b = 2
-    return a
+    return b
     c = 3
"""


def test_first_changed_line_and_test_files_skipped():
    assert _first_changed(PATCH) == ("pkg/mod.py", 12)
    insert_only = PATCH.replace("-    return a\n", "")
    assert _first_changed(insert_only) == ("pkg/mod.py", 11)  # an insertion points at the line before it
    test_only = PATCH.replace("pkg/mod.py", "tests/test_mod.py")
    assert _first_changed(test_only) is None


def test_visible_rule():
    inst = {"repo": "o/r", "f2p": ["t.py::a", "t.py::b"], "p2p": [f"t.py::p{i}" for i in range(40)]}
    named, vis, oracle_only = _visible(inst)
    assert named in inst["f2p"] and named in vis
    other_f2p = (set(inst["f2p"]) - {named}).pop()
    assert other_f2p not in vis and other_f2p in oracle_only  # remaining failing tests stay oracle-only
    assert 10 <= len(vis) - 1 <= 30  # about half of the passing tests
    assert set(vis) | set(oracle_only) == set(inst["f2p"]) | set(inst["p2p"]) and not set(vis) & set(oracle_only)
    assert _visible(inst) == (named, vis, oracle_only)  # fixed, not random per call


def test_realistic_setting_shows_every_existing_test():
    inst = {"repo": "o/r", "f2p": ["t.py::a", "t.py::b"], "p2p": [f"t.py::p{i}" for i in range(40)]}
    named, vis, oracle_only = _visible(inst, "realistic")
    assert set(inst["p2p"]) <= set(vis) and named in vis
    assert oracle_only == sorted(set(inst["f2p"]) - {named})  # only the reference fix's other new tests stay hidden
