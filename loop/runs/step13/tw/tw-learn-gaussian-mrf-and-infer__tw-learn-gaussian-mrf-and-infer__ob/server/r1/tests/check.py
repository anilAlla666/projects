#!/usr/bin/env python3
"""Verifier for the Gaussian MRF structure learning / inference / sampling task."""
import os, sys, traceback
import numpy as np
import pandas as pd

APP = os.environ.get("APP_DIR", "/app")
DATA = os.path.join(APP, "mrf_sample_10k.csv")
EDGES_F = os.path.join(APP, "learned_mrf.csv")
MARG_F = os.path.join(APP, "inferred_marginals.csv")
SAMP_F = os.path.join(APP, "final_mrf_sample.csv")
NODES = ['A', 'B', 'C', 'D', 'V', 'W', 'X', 'Y']
# Fallback (recomputed on the task data: top-8 |partial correlation|, ~0.20 vs <=0.021 for the rest)
FALLBACK_EDGES = {frozenset(e) for e in [('C', 'V'), ('W', 'Y'), ('V', 'W'), ('D', 'X'),
                                         ('V', 'X'), ('A', 'B'), ('A', 'V'), ('B', 'V')]}


def partial_corr(M):
    P = np.linalg.inv(np.cov(M.T))
    d = np.sqrt(np.diag(P))
    return -P / np.outer(d, d)


def true_edges():
    """Recompute the structure from the task data: the 8 pairs with largest |partial corr|."""
    try:
        d = pd.read_csv(DATA)[NODES].values.astype(float)
        pc = partial_corr(d)
        pairs = sorted(((abs(pc[i, j]), frozenset((NODES[i], NODES[j])))
                        for i in range(8) for j in range(i + 1, 8)), key=lambda t: -t[0])
        # require a clear gap so the top-8 is unambiguous
        if pairs[7][0] > 2.5 * pairs[8][0]:
            return {p for _, p in pairs[:8]}, d
        return FALLBACK_EDGES, d
    except Exception:
        return FALLBACK_EDGES, None


def read_edges():
    assert os.path.exists(EDGES_F), "learned_mrf.csv missing"
    lines = [l.strip() for l in open(EDGES_F).read().splitlines() if l.strip()]
    edges = []
    for l in lines:
        parts = [p.strip() for p in l.split(',')]
        assert len(parts) == 2, f"bad edge line {l!r}"
        a, b = parts
        assert a in NODES and b in NODES, f"invalid node in {l!r} (case-sensitive A,B,C,D,V,W,X,Y)"
        assert a != b, f"self loop {l!r}"
        edges.append(frozenset((a, b)))
    return edges


def check_edges(expected):
    edges = read_edges()
    assert len(edges) == 8, f"expected exactly 8 edges, got {len(edges)}"
    assert len(set(edges)) == 8, "duplicate edge"
    deg = {n: 0 for n in NODES}
    for e in edges:
        for n in e:
            deg[n] += 1
    assert all(deg['V'] > deg[n] for n in NODES if n != 'V'), f"V not strictly highest degree: {deg}"
    es = set(edges)
    assert es == expected, f"wrong structure: missing {[tuple(sorted(e)) for e in expected - es]}, extra {[tuple(sorted(e)) for e in es - expected]}"
    return es


def check_marginals():
    assert os.path.exists(MARG_F), "inferred_marginals.csv missing"
    m = pd.read_csv(MARG_F)
    assert [c.strip() for c in m.columns] == ['w_value', 'probability'], f"bad header {list(m.columns)}"
    m.columns = ['w_value', 'probability']
    assert len(m) >= 2, "marginal must be a discretized distribution over several W values"
    w = pd.to_numeric(m['w_value'], errors='coerce').values
    p = pd.to_numeric(m['probability'], errors='coerce').values
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(p)), "non-numeric / non-finite values"
    assert len(np.unique(w)) == len(w), "duplicate w_value"
    assert np.all(p >= 0), "negative probability"
    assert 0.95 <= p.sum() <= 1.05, f"probabilities sum to {p.sum()}"


def check_samples(edges, data):
    assert os.path.exists(SAMP_F), "final_mrf_sample.csv missing"
    s = pd.read_csv(SAMP_F)
    s.columns = [str(c).strip() for c in s.columns]
    for n in NODES:
        assert n in s.columns, f"missing column {n}"
    assert len(s) == 10000, f"expected 10000 rows, got {len(s)}"
    S = s[NODES].apply(pd.to_numeric, errors='coerce').values.astype(float)
    assert np.all(np.isfinite(S)), "NaN/Inf/non-numeric sample values"
    assert np.all(S.std(0) > 0.1), "degenerate samples (near-constant column)"
    # Shortcut: samples must be generated, not copied / resampled from the input data
    if data is not None:
        din = set(map(tuple, np.round(data, 2)))
        hits = sum(tuple(r) in din for r in np.round(S, 2))
        assert hits <= 100, f"{hits} sample rows are rows of the input data (copied/bootstrapped, not generated)"
    # Samples must follow the learned MRF: Markov property of the learned graph
    pc = partial_corr(S)
    e_pc, ne_pc = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            v = abs(pc[i, j])
            (e_pc if frozenset((NODES[i], NODES[j])) in edges else ne_pc).append(v)
    assert max(ne_pc) < 0.07, f"samples show dependence on a non-edge (|pcorr|={max(ne_pc):.3f})"
    assert np.mean(e_pc) >= 0.03, f"samples show no dependence along learned edges (mean |pcorr|={np.mean(e_pc):.3f})"


def main():
    expected, data = true_edges()
    ok = True
    checks = [("edges", lambda: check_edges(expected)), ("marginals", check_marginals),
              ("samples", lambda: check_samples(expected, data))]
    for name, fn in checks:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            ok = False
            print(f"FAIL {name}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
