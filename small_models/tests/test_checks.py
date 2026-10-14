from vm.data.checks import check_fake_output_fails, check_labels, check_no_oracle_leak, check_split


def ex(**kw):
    base = {"example_id": "e", "library": "lib", "split": "train", "operator": "compare", "attempt_kind": "revert",
            "task_text": "test_a fails", "function_lines": ["def f():"], "diff": "", "untrusted": [],
            "evidence": [{"id": 0, "text": "visible tests: 1 of 2 pass"}], "label": 0}
    return {**base, **kw}


def test_oracle_leak_is_reported_and_clean_passes():
    hidden = "def test_hidden_case():\n    assert f(3) == 9  # a long enough body"
    assert check_no_oracle_leak(ex(), [hidden], ["the expected output 42"]) == []
    assert check_no_oracle_leak(ex(task_text="look: " + hidden), [hidden], [])
    assert check_no_oracle_leak(ex(diff="+ x = 'the expected output 42'"), [], ["the expected output 42"])


def test_split_violations_are_reported():
    held = {"operator": "drop_stmt", "wrong_kind": "regression"}
    assert check_split([ex(), ex(split="test", operator="drop_stmt", attempt_kind="regression", library="t")], held) == []
    assert check_split([ex(operator="drop_stmt")], held)
    assert check_split([ex(attempt_kind="regression")], held)
    assert check_split([ex(), ex(split="dev")], held)


def test_fake_output_must_fail():
    assert check_fake_output_fails([ex(attempt_kind="fake_output")]) == []
    assert check_fake_output_fails([ex(attempt_kind="fake_output", label=1)])
    assert check_fake_output_fails([ex(attempt_kind="fake_output", evidence=[{"id": 0, "text": "visible tests: 2 of 2 pass"}])])


def test_label_check():
    exs = [ex(example_id=str(i), label=i % 2) for i in range(10)]
    assert check_labels(exs, lambda e: e["label"]) == []
    assert check_labels(exs, lambda e: 1 - e["label"])
