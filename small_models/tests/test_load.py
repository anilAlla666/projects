from vm.data.load import usable


def ex(task, kind, label):
    return {"example_id": f"{task}/{kind}/x", "attempt_kind": kind, "label": label}


def test_unreliable_oracle_drops_the_task_and_broken_pairs_drop_both_halves():
    rows = [ex("a:0", "revert", 1), ex("a:0", "persuasion", 0), ex("a:0", "persuasion_correct", 1), ex("a:0", "regression", 0),
            ex("b:0", "revert", 0), ex("b:0", "equivalent", 1),
            ex("c:0", "revert", 1), ex("c:0", "persuasion", 0), ex("c:0", "persuasion_correct", 0), ex("c:0", "model_attempt", 1)]
    out, rep = usable(rows)
    ids = {e["example_id"] for e in out}
    assert ids == {"a:0/revert/x", "a:0/persuasion/x", "a:0/persuasion_correct/x", "a:0/regression/x", "c:0/revert/x", "c:0/model_attempt/x"}
    assert rep["tasks_dropped_unreliable_oracle"] == 1 and rep["persuasion_pairs_dropped"] == 1
    # with the filter, prose is as often on a correct attempt as on a wrong one
    pros = [e["label"] for e in out if e["attempt_kind"].startswith("persuasion")]
    assert pros.count(0) == pros.count(1)
