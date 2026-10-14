#!/usr/bin/env python3
"""Verifier for the Gaussian MRF structure learning / inference / sampling task."""
import os
import numpy as np
import pandas as pd

NODES = ['A', 'B', 'C', 'D', 'V', 'W', 'X', 'Y']
EDGES_F = '/app/learned_mrf.csv'
MARG_F = '/app/inferred_marginals.csv'
SAMP_F = '/app/final_mrf_sample.csv'
DATA_F = '/app/mrf_sample_10k.csv'
REF_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_data.csv')


def _data():
    f = DATA_F if os.path.exists(DATA_F) else REF_F
    return pd.read_csv(f)[NODES]


def _pcor(values):
    P = np.linalg.inv(np.cov(np.asarray(values, float).T))
    s = np.sqrt(np.diag(P))
    return -P / np.outer(s, s)


def _edges():
    assert os.path.exists(EDGES_F), "learned_mrf.csv not found"
    with open(EDGES_F) as fh:
        lines = [l.strip() for l in fh.read().splitlines() if l.strip()]
    edges = []
    for l in lines:
        parts = [p.strip() for p in l.split(',')]
        assert len(parts) == 2, f"Bad edge line: {l!r}"
        edges.append(tuple(parts))
    return edges


def _marg():
    assert os.path.exists(MARG_F), "inferred_marginals.csv not found"
    m = pd.read_csv(MARG_F)
    m.columns = [c.strip() for c in m.columns]
    assert list(m.columns) == ['w_value', 'probability'], f"Bad header {list(m.columns)}"
    w = pd.to_numeric(m['w_value'], errors='raise').to_numpy(float)
    p = pd.to_numeric(m['probability'], errors='raise').to_numpy(float)
    return w, p


def _samples():
    assert os.path.exists(SAMP_F), "final_mrf_sample.csv not found"
    s = pd.read_csv(SAMP_F)
    s.columns = [str(c).strip() for c in s.columns]
    return s


def test_output_files_exist():
    for f in (EDGES_F, MARG_F, SAMP_F):
        assert os.path.exists(f), f"{f} not found"


def test_edge_format_and_count():
    edges = _edges()
    assert len(edges) == 8, f"Expected 8 edges, got {len(edges)}"
    seen = set()
    for a, b in edges:
        assert a in NODES and b in NODES, f"Invalid node in edge {a},{b}"
        assert a != b, f"Self loop {a},{b}"
        e = frozenset((a, b))
        assert e not in seen, f"Duplicate edge {a},{b}"
        seen.add(e)


def test_v_has_strictly_highest_degree():
    deg = {n: 0 for n in NODES}
    for a, b in _edges():
        deg[a] += 1
        deg[b] += 1
    others = max(v for k, v in deg.items() if k != 'V')
    assert deg['V'] > others, f"V degree {deg['V']} not strictly highest ({deg})"


def test_structure_matches_data():
    """The 8 strongest partial correlations of the data (partial correlation
    thresholding, which the task names) define the true graph. The gap between the
    8th and 9th is checked so this is only asserted when the structure is unambiguous."""
    d = _data()
    pc = np.abs(_pcor(d.values))
    pairs = sorted(((pc[i, j], frozenset((NODES[i], NODES[j])))
                    for i in range(8) for j in range(i)), key=lambda t: -t[0])
    assert pairs[7][0] > 3 * pairs[8][0], "reference data structure ambiguous"
    expected = {p for _, p in pairs[:8]}
    got = {frozenset(e) for e in _edges()}
    assert got == expected, f"Wrong edges. Missing {[tuple(sorted(e)) for e in expected-got]}, extra {[tuple(sorted(e)) for e in got-expected]}"


def test_marginals_valid_distribution():
    w, p = _marg()
    assert len(w) >= 2, "Need a discretized distribution"
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(p))
    assert np.all(p >= 0), "Negative probabilities"
    assert len(np.unique(w)) == len(w), "Duplicate w_value entries"
    assert 0.95 <= p.sum() <= 1.05, f"Probabilities sum to {p.sum()}"


def test_samples_format():
    s = _samples()
    assert len(s) == 10000, f"Expected 10000 samples, got {len(s)}"
    for n in NODES:
        assert n in s.columns, f"Missing column {n}"
        col = pd.to_numeric(s[n], errors='raise').to_numpy(float)
        assert np.all(np.isfinite(col)), f"Non-finite values in {n}"


def test_samples_not_copied_from_data():
    """Samples must be generated from the learned model, not the observed rows
    (copied, permuted or bootstrapped)."""
    s = _samples()[NODES].astype(float).round(6)
    srows = set(map(tuple, s.values.tolist()))
    for f in (DATA_F, REF_F):
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)[NODES].astype(float).round(6)
        drows = set(map(tuple, d.values.tolist()))
        frac = len(srows & drows) / max(len(srows), 1)
        assert frac < 0.01, f"{frac:.1%} of sample rows are rows of the observed data"


def test_samples_follow_learned_graph():
    """In samples from the learned Gaussian MRF, non-edges have ~zero partial correlation."""
    s = _samples()[NODES].astype(float)
    pc = _pcor(s.values)
    es = {frozenset(e) for e in _edges()}
    for i in range(8):
        for j in range(i):
            if frozenset((NODES[i], NODES[j])) not in es:
                assert abs(pc[i, j]) < 0.08, f"Non-edge {NODES[i]}-{NODES[j]} has partial corr {pc[i, j]:.3f} in samples"


def test_conditional_consistent_with_samples():
    """P(W | V=1, X=0.5) must come from the same learned Gaussian model the samples
    come from: its mean/sd must match the Gaussian conditional implied by the samples."""
    w, p = _marg()
    p = p / p.sum()
    mu = float((p * w).sum())
    sd = float(np.sqrt((p * (w - mu) ** 2).sum()))
    s = _samples()[NODES].astype(float)
    Xm = np.column_stack([np.ones(len(s)), s['V'], s['X']])
    b, *_ = np.linalg.lstsq(Xm, s['W'].to_numpy(), rcond=None)
    r = s['W'].to_numpy() - Xm @ b
    mu_s = float(b @ np.array([1.0, 1.0, 0.5]))
    sd_s = float(r.std())
    assert abs(mu - mu_s) < 0.06, f"Conditional mean {mu:.4f} inconsistent with samples ({mu_s:.4f})"
    assert abs(sd / sd_s - 1) < 0.2, f"Conditional sd {sd:.4f} inconsistent with samples ({sd_s:.4f})"
