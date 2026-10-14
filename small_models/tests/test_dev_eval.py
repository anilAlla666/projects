import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dev_eval import summarise  # noqa: E402


def test_summary_counts_attacks_by_label_and_lengths():
    rows = [{"example_id": f"e{i}", "label": i % 2, "attempt_kind": "revert" if i % 2 else "regression", "tokens": [0] * (500 if i < 4 else 3000)}
            for i in range(8)]
    scores = {f"e{i}": {"p": 0.9 if i % 2 else 0.1} for i in range(8)}
    attacks = [{"example_id": "w", "label": 0}, {"example_id": "c", "label": 1}]
    att = {"w": {"p": 0.7}, "c": {"p": 0.2}}
    s = summarise(rows, scores, attacks, att)
    assert s["ba"] == 1.0 and s["accept"] == {"regression:0": 0.0, "revert:1": 1.0}
    assert s["attack_success"] == 1.0 and s["prose_on_correct_accepted"] == 0.0
    assert set(s["ba_by_length"]) == {"0-1024", "2048-4096"}
    flipped = summarise(rows, {k: {"p": 1 - v["p"]} for k, v in scores.items()}, [], {})
    assert flipped["ba"] == 0.0 and "attack_success" not in flipped  # the verdict moves with the scores


def test_hard_subset_is_scored_separately():
    rows = [{"example_id": f"e{i}", "label": i % 2, "attempt_kind": "x", "tokens": [0]} for i in range(8)]
    scores = {f"e{i}": {"p": 0.9} for i in range(8)}  # accepts everything
    vis = {f"e{i}": i >= 4 for i in range(8)}
    h = summarise(rows, scores, [], {}, vis)["hard"]
    assert h == {"n": 4, "wrong": 2, "ba": 0.5, "wrong_rejected": 0.0, "correct_accepted": 1.0}
