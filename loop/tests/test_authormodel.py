"""Stage A's bookkeeping, without a model: the fixed shuffle, the strict reading of answers, and the score."""
from vb import authormodel as am


def test_shuffle_is_fixed_and_keeps_the_writers_place():
    a, b = am.shuffled("t", 0, ["x", "y", "z"]), am.shuffled("t", 0, ["x", "y", "z"])
    assert a == b and sorted(o["writer_place"] for o in a) == [0, 1, 2] and [o["key"] for o in a] == ["A", "B", "C"]
    assert {o["text"] for o in a} == {"x", "y", "z"}


def test_score_counts_only_decided_questions_and_reports_missing():
    q = lambda i: {"id": f"h/{i}", "question": "?", "options": am.shuffled("h", i, ["p", "q"])}
    tasks = [{"task": "h", "split": "held_out", "truncated": False}, {"task": "t", "split": "train", "truncated": False}]
    qs = {"h": [q(0), q(1), q(2), q(3)], "t": []}
    labels = {"h/0": "A", "h/1": "B", "h/2": "missing", "h/3": "undecided"}
    preds = {"zero_shot": {"h/0": "A", "h/1": "A", "h/2": "A", "h/3": "A"},
             "with_examples": {"h/0": "A", "h/1": "B", "h/2": "B", "h/3": "B"}}
    r = am.score(tasks, qs, labels, preds)
    assert r["decided"] == 2 and r["missing"] == 1 and r["undecided"] == 1 and r["missing_rate"] == round(1 / 3, 3)
    assert r["zero_shot"]["acc"] == 0.5 and r["with_examples"]["acc"] == 1.0 and r["control_chance"] == 0.5
