import ast

from vm.tasks.mutate import OPERATORS, mutants, pick_heldout

SRC = '''def clamp(x, lo, hi):
    """Clamp x into [lo, hi]."""
    if x < lo and lo <= hi:
        return lo
    total = x + 1
    helper(lo, hi)
    return min(x, hi)
'''


def test_every_operator_fires_once_where_it_applies():
    ms = mutants(SRC, "pkg/m.py")
    ops = {m.operator for m in ms}
    assert ops == set(OPERATORS)
    for m in ms:
        ast.parse(m.new_source)
        assert m.new_source != SRC
        assert m.function == "clamp" and m.function_start == 1 and m.function_end == 7


def test_compare_flip_is_exact():
    flips = [m for m in mutants(SRC, "pkg/m.py") if m.operator == "compare"]
    assert any("if x <= lo and lo <= hi:" in m.new_source for m in flips)
    assert any("if x < lo and lo < hi:" in m.new_source for m in flips)


def test_heldout_choice_is_fixed():
    a, b = pick_heldout(), pick_heldout()
    assert a == b and a["operator"] in OPERATORS
