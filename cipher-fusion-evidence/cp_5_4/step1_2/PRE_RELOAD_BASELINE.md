# CP 5.4 Step 1.2 — Phase 0: pre-reload regression baseline

**Date:** 2026-05-18. Captured on the **current kmod `e2f50452`** +
`libcipher_rt a7ac8e97`, before any kmod change. These are the baselines the
new kmod `90103f82` must match (±3%) in Phase 3.

---

## W1 — Phase B N=8 TinyLlama cross-tenant batched (3 reps)

`run_arm3.sh`, N=8, TinyLlama-1.1B, 128 tok, gen-based batched executor.

| rep | agg tok/s | mean power (W) | tok/W | total tok | KL_max |
|---|---|---|---|---|---|
| 1 | 496.11 | 138.2 | 3.590 | 4224 | 3.1e-5 |
| 2 | 502.24 | 139.9 | 3.590 | 4224 | 3.1e-5 |
| 3 | 476.99 | 138.9 | 3.434 | 4224 | 3.1e-5 |
| **mean** | **491.78** | **139.0** | **3.538** | — | ~3.1e-5 |

Correctness: teacher-forced `TFGATE` KL_max ≈ 3.1e-5 every rep (PASS).

**Variance note for the Phase-3 comparison:** the natural 3-rep spread is
already ~±3% on tok/s (477–502) — rep 3 sits ~3% below the mean on its own.
The Phase-3 regression gate must therefore be applied **mean-vs-mean** (3-rep
mean new kmod vs 3-rep mean baseline), not per-rep — a single per-rep
comparison would flag run-to-run noise as a regression. Baseline mean to beat:
**agg tok/s 491.8, tok/W 3.54.**

## W2 — CP 5.3 STEP 2B Arm-c (F1-fixed spec-decode)

`step2b_arm.py` arm c, Llama-3.1-8B + Llama-3.2-1B draft, full substrate,
64-tok, 1 rep.

| metric | baseline |
|---|---|
| accept_rate | **0.3796** (41/108) |
| rounds | 22 |
| tok/round | 2.909 |
| wall_s | 4.56 |
| decode text | coherent ("…63 million square miles. It is bordered by Asia and Australia…") |

Regression metric: **accept_rate 0.380** (±3% → [0.368, 0.391]) and coherent
text. Reproduces the STEP 2B Step 1 measurement exactly.

## W3 — Track 2 SC2 VMM weight arena

`sc2_verify.py`, TinyLlama-1.1B, weights rebound into the CIPHER VMM weight
arena, single tenant.

| gate | baseline |
|---|---|
| gate_1 VMM-backed / gate_2 page_info / gate_3 memcmp / gate_4 KL / gate_5 export | all **PASS** |
| teacher-forced KL_max | **0.0** |
| SC2_PASS | **true** |

Regression metric: **all gates PASS, KL_max 0.0** (binary — any gate flip or
KL > 0.1 is a regression).

## dmesg baseline signature

`dmesg | grep -iE 'cipher|ARB'` before and after the Phase 0 suite: **identical
— no new kmod log lines emitted during any workload.** The only cipher dmesg
lines are pre-existing benign `cipher_clock: set N MHz` entries (timestamps
~416474–416557, from earlier-session DVFS use). Kmod `pr_*` output is silent
during normal workloads.

**Expected Phase-3 dmesg diff** (acceptable, not a regression): the CP 5.4
kmod's load message (`CP 5.4 arbitration ledger — 16 × 8-SM groups…`) at
`insmod`, and — *if* a workload's ARB-poll thread fires — the userspace
`[cipher_v2] ARB-poll: … rc=… errno=…` lines go to **process stderr, not
dmesg**, so dmesg should stay as quiet as the baseline. **Any new kmod `pr_err`
/ `pr_warn` line during a Phase-3 workload is a regression signal.**

## Phase 0 status

Baseline captured, all three workloads ran clean on the current kmod. Phase 0
is the safe prerequisite — no kmod change, no anchor touched. Fallback
`e2f50452` preserved at `/home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_cp5_4`.
Next: Phase 1 (reload) → Phase 2 (isolation) → Phase 3 (regression vs the
above).
