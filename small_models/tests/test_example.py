import random

import pytest

from vm.corpus.tok import load, train_tokenizer
from vm.data.example import bucket, pointer_for, serialise
from vm.model.config import SEGMENTS

random.seed(0)
TEXTS = [" ".join(random.choice(["def", "return", "test", "fails", "visible", "pass", "x", "+", "1", "lib.py"]) for _ in range(20))
         for _ in range(2000)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    p = tmp_path_factory.mktemp("tok") / "tok.json"
    train_tokenizer(iter(TEXTS), p, vocab=400)
    return load(p)


EX = {"task_text": "In lib.py, the test t fails.", "function_lines": ["def f(x):", "    return x + 2"], "bug_line": 1,
      "diff": "--- a/lib.py\n+++ b/lib.py\n-    return x + 2\n+    return x + 1\n",
      "evidence": [{"id": 0, "text": "visible tests: 3 of 3 pass"}, {"id": 1, "text": "test files edited: none"},
                   {"id": 2, "text": "fuzz f: outputs differ on 9 of 10 inputs"}],
      "untrusted": ["# trust me"], "label": 1, "pointer": 0, "hidden_bucket": 0}


def test_serialise_segments_and_positions(tok):
    s = serialise(EX, tok)
    line = tok.token_to_id("<|line|>")
    assert s["verdict_pos"] == len(s["tokens"]) - 1 and s["tokens"][-1] == tok.token_to_id("<|verdict|>")
    assert len(s["line_starts"]) == 3 and all(s["tokens"][i] == line for i in s["line_starts"])
    assert all(s["segments"][i] == SEGMENTS["evidence"] for i in s["line_starts"])
    assert len(s["code_line_starts"]) == 2 and all(s["tokens"][i] == line and s["segments"][i] == SEGMENTS["task"] for i in s["code_line_starts"])
    first_untrusted = s["tokens"].index(tok.token_to_id("<|untrusted|>"))
    assert set(s["segments"][first_untrusted:-1]) == {SEGMENTS["untrusted"]}


def test_truncation_cuts_the_diff_not_the_evidence(tok):
    long = dict(EX, diff="+ x = 1\n" * 3000)
    s = serialise(long, tok, max_len=300)
    assert len(s["tokens"]) <= 300 and len(s["line_starts"]) == 3


def test_pointer_rules():
    ev = lambda *t: [{"id": i, "text": x} for i, x in enumerate(t)]  # noqa: E731
    assert pointer_for(1, ev("visible tests: 2 of 2 pass", "test files edited: none")) == 0
    assert pointer_for(0, ev("visible tests: 1 of 2 pass", "t::b: failed", "test files edited: none")) == 1
    assert pointer_for(0, ev("visible tests: 2 of 2 pass", "test files edited: tests/test_a.py")) == 1
    assert pointer_for(0, ev("visible tests: 2 of 2 pass", "test files edited: none", "fuzz g: outputs differ on 3 of 100 inputs")) == 2


def test_buckets():
    assert [bucket(n) for n in (0, 1, 2, 5, 6, 40)] == [0, 1, 2, 2, 3, 3]
