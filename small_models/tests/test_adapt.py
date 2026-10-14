import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adapt import balanced_sample, split_library  # noqa: E402


def test_no_task_on_both_sides_and_balanced_samples():
    rows = [{"example_id": f"lib:{t}/kind/{i}", "label": i % 2} for t in range(6) for i in range(4)]
    pool, ev = split_library(rows)
    assert {r["example_id"].split("/")[0] for r in pool}.isdisjoint({r["example_id"].split("/")[0] for r in ev})
    assert len(pool) + len(ev) == len(rows) and pool and ev
    s = balanced_sample(pool, 10, 0)
    assert len(s) == 10 and sum(r["label"] for r in s) == 5
    assert balanced_sample(pool, 10, 0) == s and balanced_sample(pool, 10, 1) != s  # seeded, and the seed matters
