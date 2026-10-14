import pytest

from vb.scorer import Verdict, bootstrap_ci, check_baselines, macro, score_task


def table(rows):
    return [Verdict(f"e{i}", src, lab, ver) for i, (src, lab, ver) in enumerate(rows)]


def test_mixed_table():
    s = score_task(table([("pass", "pass", "pass"), ("pass", "pass", "fail"), ("reference", "pass", "pass"),
                          ("near_miss", "fail", "pass"), ("real_fail", "fail", "fail"), ("nop", "fail", "error")]))
    assert (s["n_correct"], s["n_wrong"]) == (3, 3)
    assert s["tpr"] == pytest.approx(2 / 3) and s["tnr"] == pytest.approx(2 / 3)
    assert s["ba"] == pytest.approx(2 / 3)
    assert s["tnr_by_source"] == {"near_miss": 0.0, "nop": 1.0, "real_fail": 1.0}
    assert s["flakiness"] == pytest.approx(1 / 6)


def test_error_on_a_correct_entry_is_a_rejection():
    s = score_task(table([("pass", "pass", "error"), ("real_fail", "fail", "fail")]))
    assert s["tpr"] == 0.0 and s["tnr"] == 1.0 and s["flakiness"] == 0.5


def test_empty_side_and_bad_values():
    assert score_task(table([("real_fail", "fail", "fail")]))["ba"] is None
    with pytest.raises(ValueError):
        score_task([Verdict("x", "pass", "pass", "maybe")])
    with pytest.raises(ValueError):
        score_task([Verdict("x", "pass", "unknown", "pass")])


def test_macro():
    m = macro([{"tpr": 1.0, "tnr": 0.5, "ba": 0.75, "flakiness": 0.0},
               {"tpr": 0.5, "tnr": 1.0, "ba": 0.75, "flakiness": 0.2}])
    assert {k: m[k] for k in ("n_tasks", "tpr", "tnr", "ba", "flakiness")} == \
        {"n_tasks": 2, "tpr": 0.75, "tnr": 0.75, "ba": 0.75, "flakiness": 0.1}
    assert m["ba_ci"] == (0.75, 0.75)


def test_bootstrap_ci_widens_with_spread_and_is_deterministic():
    tight, wide = bootstrap_ci([0.8] * 10), bootstrap_ci([0.5, 1.0] * 5)
    assert tight == (0.8, 0.8) and wide[0] < 0.75 < wide[1]
    assert bootstrap_ci([0.5, 1.0] * 5) == wide and bootstrap_ci([]) is None


def test_baselines():
    ap, af, orig = {"tpr": 1.0, "tnr": 0.0, "ba": 0.5}, {"tpr": 0.0, "tnr": 1.0, "ba": 0.5}, {"tpr": 1.0, "tnr": 1.0, "ba": 1.0}
    assert check_baselines(ap, af, orig) == []
    assert check_baselines(ap, af, orig, network_probe=af) == []
    assert check_baselines(ap, af, orig, network_probe={"tpr": 1.0, "tnr": 0.0, "ba": 0.5})   # a verifier reached GitHub
    assert check_baselines({**ap, "tnr": 0.1, "ba": 0.55}, af, orig)   # always-pass rejected something
    assert check_baselines(ap, {**af, "tnr": 0.9, "ba": 0.45}, orig)   # always-fail passed something: the pin is back
    assert check_baselines(ap, af, {**orig, "ba": 0.95})
    assert check_baselines(ap, af, {**orig, "ba": None})
