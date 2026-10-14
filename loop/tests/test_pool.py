import json

from vb.pool import classify_fail, ctrf_counts, round_robin


def att(job, i, rank=1):
    return {"trial_id": f"{job}-{i}", "job": job, "rank": rank}


def test_round_robin_interleaves_runs_in_rank_order():
    atts = [att("b", 1, 2), att("b", 2, 2), att("a", 1, 1), att("a", 2, 1), att("a", 3, 1), att("c", 1, 3)]
    assert [x["trial_id"] for x in round_robin(atts)] == ["a-1", "b-1", "c-1", "a-2", "b-2", "a-3"]
    assert round_robin(list(reversed(atts))) == round_robin(atts)


def test_ctrf_counts(tmp_path):
    (tmp_path / "verifier").mkdir()
    assert ctrf_counts(tmp_path) == (0, 0)
    (tmp_path / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"summary": {"tests": 10, "passed": 9, "failed": 1}}}))
    assert ctrf_counts(tmp_path) == (9, 10)


def test_classify_fail():
    assert classify_fail(9, 10) == "near_miss"
    assert classify_fail(8, 10) == "near_miss"
    assert classify_fail(7, 10) == "real_fail"
    assert classify_fail(0, 0) == "real_fail"
