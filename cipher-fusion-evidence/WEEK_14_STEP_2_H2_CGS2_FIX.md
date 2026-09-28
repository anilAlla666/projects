# W14 Step 2 H2 fix: rand_svd CGS2 (twice-MGS) loss-of-orthogonality correction

**Date:** 2026-05-23
**Adjudication:** 2026-05-23 (ζ-deep continuation, day 3)
**Commit:** cipher_rt_phase4 `3590f4d` tag `week-14-step-2-h2-cgs2-fix`
**Status:** Standalone substrate fix landed; does NOT close W14 Step 2 G — separate downstream bug remains
**Successor:** ζ-deep day 3 direct kernel diagnostic (Step 2 of Anil's 2026-05-23 adjudication sequence)

## 1. Background

`WEEK_14_STEP_2_B1_EDMD_VERIFY.md` (commit `4824b1e`) closed S2.B.1 by
verifying the EDMD pipeline operates correctly on synthetic snapshots. The
load-bearing S2.E gate (Koopman lane fires on real TinyLlama LM head with
output quality matching cuBLAS) failed catastrophically:

- top-1 token match between vanilla and CIPHER softmax: **0.00%** (gate ≥ 99%)
- KL divergence: **14.58** (gate ≤ 5.5e-5; originally specified, later
  retracted by Anil as ungrounded number)
- residual ratio: **0.84** (may13's documented gate < 0.10)

ζ-deep investigation (adjudication 2026-05-23) was scoped at 3 eng-days
to characterize the bug and either fix it or surface as architectural
finding. H2 is the highest-priority investigation per the adjudication
tree: "Why does cipher report energy_X = 1.0+ when Python ground-truth on
the same data reports energy_X < 1.0?"

## 2. Bug diagnosis

cipher_edmd_live reported energy_X values up to 1.99 for the LM head
shape (K=2048, N=32000). This is **mathematically impossible** —
energy_X = ||X_r||_F^2 / ||X||_F^2 must be ≤ 1.0 because the rank-r
approximation of any matrix has Frobenius norm bounded by the full matrix.

The diagnostic instrumentation (rolled back; see commit message of
`3590f4d`) dumped sigma spectrum + per-row diversity statistics at each
EDMD live FIT event. For LM head:

```
[CIPHER DIAG-H2] FIT K=2048 N=32000 m=2000 rank=64
  ||X||_F^2 = 1.4928e+07  top-64 sigma^2 = 2.9714e+07  energy = 1.990433
  sigma[0..4] = 4.4354e+03 2.6716e+03 1.3728e+03 6.3494e+02 3.9212e+02
```

**Sum of top-64 σ² = 2× ||X||_F²** — structurally impossible if σ are
true singular values of X (because Σ σ² = ||X||_F² for full SVD).

X_host was dumped to disk and ground-truth-SVD'd in Python
(`/tmp/step13_2_baseline/h2_verify.py`):

```
||X||_F^2 = 1.4929e+07              (matches cipher's xf2)
true σ_0 = 3137.47                  (cipher: 4435 — 41% larger; ratio √2)
true σ_63 = 0.0862                  (cipher: 0.0895; close)
true top-64 sum σ^2 = 1.4929e+07    (cipher: 2.97e7 — exactly 2× larger)
TRUE energy_X = 1.0000              (data is effectively rank-64)
```

**Cipher's rand_svd is inflating σ² by exactly 2.0 for this matrix.**

The bug was localized further via a Python clone of cipher_rs::
rand_svd_rank_r (`/tmp/step13_2_baseline/h2_randsvd_clone.py`):

```
After cipher's MGS orthonormalization of Y (K=2000 × l=74):
  Y^T·Y diagonal: all 1.0   (each column individually unit-norm)
  Y^T·Y off-diagonal abs mean: 1.48e-3
  Y^T·Y off-diagonal abs MAX:  9.90e-01
  ||Y^T·Y - I||_F = 2.45     (should be ≈ 1e-14 for orthonormal Y)
```

Modified Gram-Schmidt suffers from classical **loss of orthogonality**
when input columns are near-linearly-dependent. The l = r + 10 = 74
columns include 10 that lie in the same r=64-dim subspace as the
others; MGS produces a Y whose columns are individually normalized
but mutually non-orthogonal.

The downstream BBt = Y^T·X·X^T·Y computation reads Y as if it were
orthonormal. With non-orthonormal Y, BBt has inflated trace:

```
||Btil||_F^2 = ||Y^T·X||_F^2 = 2.87e7  (would be = ||X||_F^2 = 1.49e7
                                         if Y were truly orthonormal)
```

Eigenvalues of BBt = σ² are correspondingly inflated. The 2× scaling
matches the bug observation exactly.

## 3. Why this is specific to LM head

For other ported shapes (K=2048 N=256, K=2048 N=2048, K=5632 N=2048,
K=2048 N=5632), cipher reported energy_X in [0.96, 0.99] — close to 1
but never over. The LM head was the only shape exhibiting energy > 1.0.

Reading the diagnostic σ spectrum: LM head has σ_0 = 4435 (cipher) and
~3137 (true), while other shapes have σ_0 in [288, 552]. The bug
manifests when X has **strong rank concentration**:
- LM head input (post-RMSNorm pre-LM-head hidden state from
  autoregressive decode of WARMUP_PROMPT): effective rank 64
- Other shape inputs: higher effective rank, far from r

Mathematically the bug exists for all shapes but is hidden when true rank
> r because the "extra" eigenvalues from non-orthogonal Y fall into the
discarded top-r..top-l range and don't show up in top-r sum.

## 4. Fix: CGS2 (twice-MGS)

The fix is the standard "twice is enough" technique from Giraud, Langou,
and Rozloznik 2005 (*Doubly-Modified Gram-Schmidt*) and Hoffmann 1989
(*Iterative algorithms for Gram-Schmidt orthogonalization*).

```c
/* cipher_randsvd.h L182-190 — apply mgs_qr_thin twice at each of 3 sites */
for (int it = 0; it < power_iters; it++) {
    mgs_qr_thin(Y, K, l);
    mgs_qr_thin(Y, K, l);   /* CGS2 pass 2 */
    mm_tn(W_kn, Y, Z, K, N, l);
    mgs_qr_thin(Z, N, l);
    mgs_qr_thin(Z, N, l);   /* CGS2 pass 2 */
    mm_rm(W_kn, Z, Y, K, N, l);
}
mgs_qr_thin(Y, K, l);       /* final pass 1 */
mgs_qr_thin(Y, K, l);       /* final pass 2 (CGS2) */
```

Three sites × 2 calls = 6 mgs_qr_thin invocations per rand_svd call.
Each pass is O(K·l²) work; total cost overhead is 2× the original
single-pass cost. For l=74 and K=2000, this is ~22 million additional
multiply-adds — well under the existing SVD pipeline's CPU budget.

## 5. Verification (Python clone identical to cipher's algorithm)

`/tmp/step13_2_baseline/h2_randsvd_clone.py` re-runs the algorithm on
cipher's own dumped X_host with twice-MGS applied:

```
After twice-MGS final Y:
  ||Y^T·Y - I||_F = 6.6e-14       (machine precision; was 2.45)
  σ_0 ratio CIPHER-clone / TRUE = 1.0000   (was √2 = 1.414)
  sum σ² ratio cipher-clone / TRUE = 1.0000 (was 2.0)
```

Numpy's Householder QR on the same input produces identical sigmas to
twice-MGS (||Y^T·Y - I||_F = 8.3e-14, sigmas exactly match TRUE SVD).
Confirms twice-MGS restores stability to Householder-comparable quality.

Post-fix live cipher behavior (autoregressive warmup + s2e_measure.py):
- LM head FIT: `energy_X = 1.0000 sigma0 = 3.07e+03` (matches true σ_0 = 3137)
- All 5 shapes report energy_X ≤ 1.0 (no impossible values)

## 6. What this fix DOES NOT close

The H2 fix corrects the σ value computation but does NOT deliver the
S2.E top-1 ≥ 99% gate:

| Metric                | Pre-H2 fix | Post-H2 fix |
|-----------------------|------------|-------------|
| LM head energy_X      | 1.66       | 1.0000      |
| LM head σ_0           | 4435       | 3070        |
| top-1 token match     | 0.00%      | **0.00%**   |
| KL mean               | 14.58      | 14.78       |
| residual_ratio        | 0.84       | 1.36        |

**Top-1 match remains catastrophically at 0%** even with sigma values
now correct. The kernel arithmetic produces output that is
fundamentally inconsistent with the formula's mathematical specification.

Per Anil's 2026-05-23 adjudication, ζ-deep day 3 follows H2 with a direct
kernel diagnostic test: Python ctypes harness calling
cipher_koopman_fp16_launch_shape with hand-crafted V_T / K_op / W and
known input, comparing against Python formula evaluation. The result
will distinguish:
  D1 — kernel correct, bug is in cipher_edmd_live matrix preparation
  D2 — kernel layout assumption flipped, transpose fix
  D3 — kernel arithmetic fundamentally wrong, surface to Anil

## 7. Files

- `cipher_rt_phase4/include/may13/cipher_randsvd.h` (+13 LOC, -1)
- `/tmp/step13_2_baseline/h2_randsvd_clone.py` (Python verification)
- `/tmp/step13_2_baseline/h2_verify.py` (Python ground-truth check)
- `/tmp/step13_2_baseline/h2_lmhead_xhost.bin` (cipher's dumped X_host)
- `cipher-fusion-evidence/WEEK_14_STEP_2_H2_CGS2_FIX.md` (this doc)

## 8. Anchors

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` | `week-14-step-2-h2-cgs2-fix` (3590f4d) | **NEW** |
| Predecessor | `week-14-step-2-port-landmark` (ce4c1b8) | unchanged |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_kv.py` | `2b6cedab` | UNCHANGED |

Step 2 G NOT closed by this fix. Separate downstream investigation
follows; if successful → Step 2 G closes on top of `3590f4d`; if
unsuccessful → Step 2 G remains open and architectural finding surfaces
to Anil for re-adjudication per the 2026-05-23 day-3 close decision tree.
