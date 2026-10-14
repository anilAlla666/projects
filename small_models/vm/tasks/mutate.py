"""Single-site mutation operators over Python source, edited by exact character offsets so formatting and
comments survive. Ported from Verifier Bench's correct-by-construction pilot and widened to seven operators."""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass

OPERATORS = ["compare", "boolop", "arith", "const", "swap_args", "drop_stmt", "return_none"]
WRONG_KINDS = ["regression", "visible_only", "overfit"]
_FLIP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
         ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}
_ARITH = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "//"), ast.FloorDiv: ("//", "*")}


@dataclass(frozen=True)
class Mutant:
    path: str
    operator: str
    line: int
    function: str
    function_start: int
    function_end: int
    new_source: str


def pick_heldout() -> dict:
    r = random.Random("vm-heldout-2026-10-08")
    return {"operator": r.choice(OPERATORS), "wrong_kind": r.choice(WRONG_KINDS)}


def mutants(source: str, path: str) -> list[Mutant]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    raw = source.encode()

    def off(line, col):  # AST columns are UTF-8 byte offsets
        return len(raw[: len(source[: starts[line - 1]].encode()) + col].decode(errors="ignore"))

    funcs = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def owner(line):
        best = None
        for a, b, name in funcs:
            if a <= line <= b and (best is None or a > best[0]):
                best = (a, b, name)
        return best

    out = []

    def add(operator, node, a, b, old, new):
        seg = source[a:b]
        idx = [i for i in range(len(seg)) if seg.startswith(old, i)
               and not (old in "<>" and seg[i + 1 : i + 2] == "=")
               and not (old in ("<", ">", "=") and seg[i - 1 : i] in "<>=!")]
        if len(idx) != 1:
            return
        new_src = source[: a + idx[0]] + new + source[a + idx[0] + len(old):]
        _emit(operator, node.lineno, new_src)

    def _emit(operator, line, new_src):
        f = owner(line)
        if f is None:
            return
        try:
            ast.parse(new_src)
        except SyntaxError:
            return
        out.append(Mutant(path, operator, line, f[2], f[0], f[1], new_src))

    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            prev = n.left
            for op, right in zip(n.ops, n.comparators):
                if type(op) in _FLIP:
                    old, new = _FLIP[type(op)]
                    add("compare", n, off(prev.end_lineno, prev.end_col_offset), off(right.lineno, right.col_offset), old, new)
                prev = right
        elif isinstance(n, ast.BoolOp) and len(n.values) >= 2:
            word, other = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
            a, b = n.values[0], n.values[1]
            add("boolop", n, off(a.end_lineno, a.end_col_offset), off(b.lineno, b.col_offset), f" {word} ", f" {other} ")
        elif isinstance(n, ast.BinOp) and type(n.op) in _ARITH:
            old, new = _ARITH[type(n.op)]
            add("arith", n, off(n.left.end_lineno, n.left.end_col_offset), off(n.right.lineno, n.right.col_offset), old, new)
        elif isinstance(n, ast.Constant) and type(n.value) is int and n.end_lineno == n.lineno:
            a, b = off(n.lineno, n.col_offset), off(n.end_lineno, n.end_col_offset)
            if source[a:b].isdigit():
                _emit("const", n.lineno, source[:a] + str(n.value + 1) + source[b:])
        elif isinstance(n, ast.Call) and len(n.args) >= 2 and not n.keywords and all(
                x.lineno == x.end_lineno for x in n.args[:2]):
            x, y = n.args[0], n.args[1]
            ax, bx = off(x.lineno, x.col_offset), off(x.end_lineno, x.end_col_offset)
            ay, by = off(y.lineno, y.col_offset), off(y.end_lineno, y.end_col_offset)
            if source[ax:bx] != source[ay:by]:
                _emit("swap_args", n.lineno, source[:ax] + source[ay:by] + source[bx:ay] + source[ax:bx] + source[by:])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [s for s in n.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
            for s in body:
                if isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign)) and s.lineno == s.end_lineno and len(body) > 1:
                    a, b = off(s.lineno, s.col_offset), off(s.end_lineno, s.end_col_offset)
                    _emit("drop_stmt", s.lineno, source[:a] + "pass" + source[b:])
                if isinstance(s, ast.Return) and s.value is not None and s.lineno == s.end_lineno:
                    a, b = off(s.value.lineno, s.value.col_offset), off(s.value.end_lineno, s.value.end_col_offset)
                    _emit("return_none", s.lineno, source[:a] + "None" + source[b:])
    seen, unique = set(), []
    for m in sorted(out, key=lambda m: (m.line, m.operator, m.new_source)):
        if m.new_source not in seen:
            seen.add(m.new_source)
            unique.append(m)
    return unique
