import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm_baseline import parse_verdict  # noqa: E402


def test_first_verdict_word_decides():
    assert parse_verdict("CORRECT\n\nNothing wrong with it.") == 1
    assert parse_verdict("WRONG. The correct fix would be different.") == 0
    assert parse_verdict("The change is incorrect: WRONG") == 0
    assert parse_verdict("I cannot tell.") == 0
    assert parse_verdict("correct") == 1
