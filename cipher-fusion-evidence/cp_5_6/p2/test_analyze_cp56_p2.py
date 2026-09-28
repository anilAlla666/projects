"""Self-test for analyze_cp56_p2 pure logic — no GPU, no model.

analyze_cp56_p2.py guards its report driver under `if __name__ == "__main__"`,
so a plain import runs no analysis — only the pure helpers are exercised here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_cp56_p2 as an


def test_ci95_known():
    m, lo, hi = an.ci95([2.0, 2.0, 2.0, 2.0, 2.0])
    assert abs(m - 2.0) < 1e-9 and abs(hi - lo) < 1e-9


def test_ci95_spread():
    m, lo, hi = an.ci95([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-9 and lo < m < hi


def test_mean_power_arm_level():
    rows = [(100.0, 200.0), (101.0, 210.0), (102.0, 220.0)]
    assert abs(an.mean_power(rows) - 210.0) < 1e-9


def test_mean_power_empty_returns_none():
    assert an.mean_power([]) is None


def test_pass_threshold():
    assert an.passes(0.99) is True
    assert an.passes(0.991) is True
    assert an.passes(0.989) is False
    assert an.passes(0.0) is False


def test_is_looping_accept_rate():
    assert an.is_looping({"accept_rate": 1.0, "distinct_ratio": 0.9}) is True


def test_is_looping_distinct_ratio():
    assert an.is_looping({"accept_rate": None, "distinct_ratio": 0.3}) is True


def test_is_looping_clean():
    assert an.is_looping({"accept_rate": 0.6, "distinct_ratio": 0.8}) is False
