from vm.eval.baselines import diff_shape_predict, fit_diff_shape, rules_plus_fuzz, visible_only
from vm.eval.metrics import balanced_accuracy, bootstrap_by_library, ece, gap_closed, per_kind


def test_balanced_accuracy_and_gap():
    assert balanced_accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert balanced_accuracy([1, 1, 0, 0], [1, 1, 1, 1]) == 0.5
    assert abs(gap_closed(0.95, 0.9) - 0.5) < 1e-9


def test_ece():
    assert ece([1, 0], [1.0, 0.0]) == 0.0
    assert abs(ece([0, 0], [0.95, 0.95]) - 0.95) < 1e-9


def test_bootstrap_resamples_libraries():
    rows = [{"library": f"l{i % 5}", "x": i % 2} for i in range(100)]
    lo, hi = bootstrap_by_library(rows, lambda rs: sum(r["x"] for r in rs) / len(rs), n=200)
    assert lo <= 0.5 <= hi


def ev(*texts):
    return {"evidence": [{"id": i, "text": t} for i, t in enumerate(texts)]}


def test_baselines():
    good = ev("visible tests: 3 of 3 pass", "test files edited: none", "fuzz f: outputs differ on 5 of 100 inputs", "diff: 1 file, +1 -1 lines")
    assert visible_only(good) == 1 and rules_plus_fuzz(good, "f") == 1
    assert rules_plus_fuzz(ev("visible tests: 3 of 3 pass", "test files edited: tests/t.py"), "f") == 0
    assert rules_plus_fuzz(ev("visible tests: 3 of 3 pass", "test files edited: none", "fuzz g: outputs differ on 1 of 100 inputs"), "f") == 0
    assert visible_only(ev("visible tests: 2 of 3 pass")) == 0
    rows = [dict(ev("visible tests: 1 of 1 pass", f"diff: 1 file, +{a} -{a} lines"), label=int(a < 3)) for a in (1, 1, 2, 5, 6, 9)]
    w = fit_diff_shape(rows)
    assert diff_shape_predict(w, rows) == [r["label"] for r in rows]
    assert per_kind([{"attempt_kind": "a", "pred": 1}, {"attempt_kind": "a", "pred": 0}]) == {"a": 0.5}
