# R.A+R.B GOODPUT+RECOVERY — RULE-4 PRE-REGISTRATION (written BEFORE measurement)

Timestamp intent: this file is committed before any rc_ab run produces a number. Measured results go in the
report and must be compared back to these predictions; surprises are reported, not hidden.

## Fault model + definitions
- Greedy decode (temperature=0) is autoregressive ⇒ once one emitted token's argmax flips, the entire suffix
  diverges from the clean reference (different context → different tokens = cascade).
- `T` = decode tokens/seq. `S` = fault onset step. `N` = detector check period. Injection = persistent bit-14 flip
  of one element of layer-0 down_proj output (R.A's `inject_gemm=3`), active every step from `S`.
- `f` = first step whose **emitted token** differs from the clean reference (argmax flip), `f ≥ S`. (May be `f = ∞`
  if the perturbation is fully absorbed and no token ever flips — the "benign corruption" regime.)
- `S'` = first detector check step `≥ S` = persistent-fault detection step. Detection latency `= S' − S ∈ [0, N−1]`.

## How per-token corruption is determined (honest denominator)
A token is **useful** iff it (a) matches the clean reference token-id AND (b) is actually served as valid (not
quarantined). A token that differs from clean and is served = **0 goodput AND silently-wrong (garbage)**. A token that
is quarantined = 0 goodput but **safe** (not served as garbage). Corruption ground-truth = token-id ≠ clean-reference
token-id (measured by a clean no-fault reference run). Detection is **periodic** (every N steps), so CIPHER's knowledge
is per-check-window, not per-token; useful-goodput for WITH-CIPHER is therefore counted over served (pre-quarantine)
windows. Detector overhead is counted in the time denominator (NOT free).

## Goodput arithmetic (on paper)
| | served tokens | useful (match clean & served) | garbage served (wrong & served) | time |
|---|---|---|---|---|
| **WITHOUT** (fault, no detector, all served) | `[0,T)` | `f` | `T − f` | baseline |
| **WITH** (fault + detector + quarantine from S') | `[0,S')` | `min(f,S')` | `max(0, S'−f)` (≤ N) | baseline × (1+overhead(N)) |

Three regimes (which obtains is **measured**, not assumed):
1. **Detection beats divergence (`S' ≤ f`)**: WITH serves **0 garbage**, useful_WITH = `S'` ≤ useful_WITHOUT = `f`.
   CIPHER *discards* the still-good tokens `[S', f)` + pays overhead ⇒ **useful-goodput LOWER**, garbage 0 vs `T−f`.
2. **Divergence beats detection (`f < S'`)**: useful_WITH = `f` = useful_WITHOUT; garbage_WITH = `S'−f` (≤N) ≪ `T−f`.
   ⇒ useful-goodput ~equal (minus overhead), garbage nearly eliminated.
3. **Benign corruption (`f = ∞`, token never flips)**: useful_WITHOUT = `T` (all correct, no garbage); useful_WITH =
   `S'` (CIPHER quarantines good output it flagged at the GEMM level) ⇒ **CIPHER is pure goodput loss, 0 garbage avoided.**

## PRE-REGISTERED EXPECTATIONS (commit before measuring)
1. **Useful-goodput: WITH ≤ WITHOUT in every regime.** CIPHER does NOT increase useful tokens/s; it costs overhead and
   (regimes 1,3) conservatively discards post-detection tokens. **I expect to report a useful-goodput LOSS for
   WITH-CIPHER**, larger at small N (more overhead) and in the benign/late-divergence regime. This is the honest trade.
2. **The value of CIPHER is garbage-elimination, not goodput:** garbage_WITHOUT = `T − f` (silently served wrong
   tokens) vs garbage_WITH ≤ `N` (or 0 if S'≤f). CIPHER converts unknown-wrong into known-bad/quarantined.
3. **tok/joule: WITH ≤ WITHOUT** (overhead inflates energy AND time; fewer useful tokens). Measured via NVML
   power.draw integrated (trapezoid) over the run.
4. **Quarantine (persistent):** corrupted response flagged/quarantined within ≤ N steps of onset (at S'), with
   **0 false quarantine on clean tokens** (T=0 ⇒ no clean check ever fires). Pre-onset windows served normally.
5. **Quarantine boundary — substrate question:** the shim can mark/flag at the cuBLAS boundary and the (scratch) driver
   can refuse to serve the flagged response. Preventing a *streamed* token mid-flight may need vLLM cooperation →
   if so, record WALL-WITH-MECHANISM (CIPHER flags the response corrupt for the operator; preventing incremental
   serve of an already-emitted token needs vLLM).
6. **Transient single-step fault:** I expect the periodic GEMM-recompute detector to **MISS** a transient that lands
   *between* checks (at the next check the down_proj GEMM is clean again → recompute matches → not detected), and to
   detect it only if it lands *on* a check step (prob ~1/N), and even then *after* that token was emitted. So the honest
   bound is stronger than "detected-after-serve": a between-check transient is **not detected at all** by this detector;
   its propagated KV-cache effect is invisible to GEMM-recompute. (This is the fork-2 periodic-detection wall.) MEASURE
   both placements (on-check vs between-check).

## Breakeven / decision rule
CIPHER is a net win only if the operator values avoiding `T−f` silently-wrong tokens more than losing `(f − useful_WITH)`
good tokens + `overhead(N)` throughput. For wrong-answers-are-worse-than-no-answer workloads (most), garbage-elimination
dominates and CIPHER wins on *safety* despite losing on *useful-goodput*. The report states this as a trade, with the
measured numbers, and does not claim a useful-goodput gain unless the measurement shows one.
