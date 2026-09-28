# W.4b — Form B′ cross-tenant POOL transport — CLOSE REPORT

**Date:** 2026-05-28. **Anchors (UNCHANGED — userspace harness + an existing env
flag only, no substrate edit):** cipher_rt_phase4 `8b5e928` (libcipher_rt.so md5
`9fe23143`); cipher_kmod `02fc2d1` (0.6.6, NR 30/31); cipher_kv_bridge `5a3db034`
(Track-2 SC3 arena). Tag chain `w4b-2-prototype` → `w4b-3-blockloop` → `w4b-4-n4`
→ `w4b-close` → **`w4b-6-weightshare` (Step 1) → `w4b-6-amortize` (Steps 2–3, gate
CLEARS)**. Models: TinyLlama-1.1B, Llama-3.2-1B (negative control), Mistral-7B-v0.1
(engagement regime). **W.4b.6 (this update): gate CLEARS at 3.239× — §E/§G +
W4B_6_FINDINGS.md.**

## A — Transport architecture (Form B′, executor-owned-KV)

Each tenant is its own injected GPU process running the model (real W.6 fp,
NR 30/31 registration). Steady-state a tenant embeds the previous token, ships the
`[1,hidden]` activation to a **vanilla** executor over cuIPC, blocks on a per-token
barrier, receives its `[1,vocab]` logit row, argmaxes, ships the next embed. The
executor gathers `[N,hidden]`, runs HF's B=N forward with **executor-owned batched
KV** (`formb_executor.py`), and scatters the N logit rows back.

- cuIPC transport: `formb_ipc.py` (cipher_kv_bridge VMM arenas, `cuMemExport`
  POSIX-FD over `socket.send_fds`/SCM_RIGHTS, persistent per-token reuse).
- executor stays VANILLA (no `CUDA_INJECTION64_PATH` → zero cuBLAS-shim tax,
  the W.4b.1 finding); only tenants inject `libcipher_rt.so`.
- per-token barrier + gather/scatter are substrate-line (cuIPC + Unix socket); no
  vLLM source, no scheduler hook (Memory #24).
- the memo-literal tenant-owned cuIPC-mapped peer-KV attention is **deferred to
  W.4b.2b (R-4)** — it cannot use HF's batched forward (HF reallocates KV/step).

## B — Gate + correctness backstop + block/disable loop

- **Eligibility gate** (`formb_cohort.py`): the vanilla executor reads the live
  NR 31 cohort via the raw QUERY ioctl with `caller_fingerprint=0` (query-only, no
  self-register) and runs the **audited W.4a guard** `cipher_rt_pool_partition`
  via ctypes (reused verbatim). Admits only co-resident tenants whose fp == the
  executor's deployed-model fp (materialized by a one-time injected warmup,
  memo §2 #2a).
- **First-coalesce correctness backstop** (W.4b.3): each tenant's batched prefill
  row vs its solo B=1 reference, fed to the audited
  `cipher_rt_pool_correctness_check` on a **top-K-by-solo-index projection** (raw-
  logit element-wise rel-diff is dominated by near-zero entries — legit max_rel
  ~122 at max_abs ~0.02; the top-64 slice gives legit ~2.5e-3 vs tol 0.15, and a
  wrong-model / fault row >> tol). Fail → `cipher_rt_pool_mark_blocked`.
- **Block → disable** (W.4b.3): a blocked group falls to solo; ratio policy (1
  blocked / 1 offered = 100% > 50% → session disable) is executor-side W.4b layer
  on the audited primitives. **Forced-fault test (3.3):** synthetic divergent row
  → top-K max_rel 20.5 >> tol → `fails=1` → `mark_blocked` → both tenants fall solo
  with CORRECT output, zero corruption propagated (`proto_fault/`).

## C — Memory #11 negative control (zero cross-fp coalesce), every N

| N | composition | result |
|---|---|---|
| 2 | TinyLlama + Llama-3.2-1B | distinct REJECTED, `eligible=0`, both solo, no corruption |
| 4 | 3× TinyLlama + Llama-3.2-1B | 3 coalesce, Llama REJECTED → solo, no corruption |
| 8 | 7× TinyLlama + Llama-3.2-1B | 7 coalesce, Llama REJECTED → solo, no corruption |
| 8 | 7× **Mistral-7B** + Llama-3.2-1B (W.4b.6) | 7 coalesce 64/64 exact, Llama `distinct_rejected=1` → solo, no corruption |

Every distinct-fp process was refused coalescing by the audited guard at every N.
Zero cross-fingerprint coalesce events across the campaign. This is the Form-B
value proof a one-model Form-A executor cannot pose. (The negative control is
**model-agnostic by construction** — the guard compares W.6 fingerprints, not
model identities — so it was verified on TinyLlama at N=2/4/8 and not re-run for
Mistral; the fingerprint-comparison mechanism is identical.)

**Token correctness (exact greedy match vs solo).** TinyLlama N=8 positive: all
8 tenants 64/64 exact (KL ≤ 5.3e-5). **Mistral-7B N=8 positive (W.4b.6): 7/8 tenants
64/64 exact; row-3 = 20/64** — adjudicated benign via the decisive teacher-forced
KL test (`formb6_tf_kl.py`): the FIRST causal divergence is a near-tie (solo
top1−top2 gap 0.00781 at step 20, the W.4b.5 signature) that greedy-cascades; under
identical context the batched-B=8 logits match solo to **worst KL 7.04e-5** (row-3
argmax mismatch 1/64). The substrate is faithful; the cascade is FP-reduction-order
at a coin-flip, NOT corruption. Mistral-7B N=4 positive: 255/256 exact
(3×64/64 + 1×63/64). The single flip (tenant3 step 20) was INVESTIGATED per
Memory #11: the solo model's top1−top2 logit gap there was **0.0078** (a near-tie)
and step-KL = 5.8e-6 — i.e. a benign batched(B=4)-vs-unbatched(B=1)
FP-reduction-order greedy coin-flip, NOT corruption (corruption ⇒ large gap +
KL≫1). NOT a HARD STOP. The forced-fault test (§B) confirms the guard DOES fire
on real divergence (top-K max_rel 20.5 ≫ tol → BLOCK).

## D — Transition handling

(tenant join/leave/fp-change — see §F debt; the prototype runs a fixed cohort.
The gate re-queries the live NR 31 cohort each batch-formation, so a join/leave
changes the admitted set on the next formation; explicit churn soak deferred.)

## E — Engagement table (cross-process vs in-process ceiling vs naive)

Decode-window tok/W (power sampled over DECODE_START→END). Substrate-attributable
= cross-process batched ÷ naive N-concurrent.

**TinyLlama-1.1B (NOT GPU-bound — see §E note):**

| N | cross-proc tok/W | in-proc ceiling tok/W | naive tok/W | in-proc lever | substrate-attrib |
|---|---|---|---|---|---|
| 2 | 0.182 (pre-amort) | 0.528 | 0.473 | 1.115× | 0.385× |
| 4 | 0.329 | 0.731 (0.970*) | 0.627 (0.916*) | 1.17× | 0.57× |
| 8 | 0.685 | 1.436 | 1.031 | **1.393×** | 0.665× |

*N=4 in-proc/naive raw tok/W from the un-windowed CSV; windowed values in the
lever column. Cross-process tok/s scales ~linearly (18→36→70 for N=2→4→8) — the
per-token rendezvous amortizes over N.

**Mistral-7B-v0.1 N=8 — THE GATE REGIME, now reachable via Track-2 weight-sharing
(W.4b.6 — GATE CLEARS).** Weight-share (lossless, KL=0) fits executor + 8 tenants
in **13.49 GiB** (vs 9 × 14 = 126 GiB OOM); the executor runs the whole B=N decode
(per-token rendezvous *banked*). Vanilla executor restored (see CUPTI note below).

| | tok/s | W | tok/W |
|---|---|---|---|
| cross-process full-handoff | 157.7 | 151.5 | **1.041** |
| in-proc B=8 ceiling | 156.7 | 148.0 | 1.059 |
| naive 8-concurrent (weight-shared) | 93.0 | 289.5 | 0.321 |

- **substrate-attributable = cross / naive = 3.239×** → **CLEARS** the >1.0×
  break-even and approaches the 3.69× CP-5.6 target (located at exactly this regime).
- in-proc lever = **3.296×** (vs only 1.48× at the testable N=4 in W.4b.5 — the
  density lever's true magnitude appears only at N=8 GPU-bound, now reachable).
- cross / inproc = **0.983×** — full-handoff realizes 98.3 % of the single-process
  ceiling; transport is genuinely banked, not merely offset.
- **Caveat (honest):** the 3.24× has a throughput half and a power half. Throughput
  alone = cross 157.7 ÷ naive 93.0 = **1.70× — clears break-even regardless of
  power**. The power half (naive 289 W vs cross/in-proc ~150 W) is measured in
  default compute mode; MPS on the naive 8-concurrent case could narrow it. Cross
  sits at 98 % of this harness's in-proc ceiling (3.296×); the residual gap to CP
  5.6's 3.69× is a ceiling/methodology difference, not a cross-process loss. The
  naive denominator is **vanilla (uninjected)** — the ratio is not inflated by
  taxing the baseline.

**The cross-process tax was a self-inflicted CUPTI artifact (W.4b.6 root-cause).**
First pass gave cross 0.504 tok/W → substrate-attrib 1.563× (cross/inproc 0.475×).
A 5-control elimination (idle bare ctx 154.7 / warmed-vanilla 155.7 / warmed-injected
157.8 / arena-weights 408 / backstop 421 tok/s — all full speed; pmon: batched
tenants sm=0, executor starved at 13 % sm) proved the 2.6× was NOT co-residence,
NOT transport. Root cause: the executor ctypes-loads `libcipher_rt.so` for the
audited gate guard, and its constructor auto-subscribes CUPTI kernel-launch
callbacks (`cipher_inject.c:159` → `cipher_cupti.c`) — re-arming the W.4b.1 shim on
the "vanilla" executor. Fix: `CIPHER_RT_DISABLE_AUTO_INIT=1` (existing escape hatch,
guard is a pure CPU fn) → truly-vanilla decode → the cleared numbers above.
Substrate/kmod/bridge UNCHANGED. (W4B_6_FINDINGS.md has the full table.)

**Historical — Mistral-7B N=4 (W.4b.5, per-token-barrier architecture):** in-proc
B=4 0.527 / naive 0.355 / cross-proc-per-token 0.273 tok/W; in-proc lever 1.483×,
cross substrate-attrib 0.768× — the per-token barrier (~79 ms/step) offset the
lever. W.4b.6 supersedes this by (a) reaching N=8 via weight-share and (b) banking
the rendezvous via full-handoff.

**Note — why TinyLlama shows a weak/no tok/s lever.** On a 1.1B model the GPU is
never saturated even by 8 concurrent B=1 streams (decode at 1980 MHz throughout),
so naive N-concurrent scales ~linearly in tok/s (in-proc B=8 172.7 ≈ naive 164.6
tok/s) and the batching lever is **power-driven only** (naive burns 160 W vs
in-proc 120 W → 1.39× tok/W at N=8). CP 5.6's contention-driven lever (3.69× at
N=8) lives in a GPU-bound regime; Mistral-7B (memory-bandwidth-bound at B=1)
reproduces it (in-proc lever 1.48× at N=4 here).

## F — Debt forecast

- **R-4 cuIPC-mapped peer-KV attention → W.4b.2b** (branch B: per-token
  cross-process coalescing / heterogeneous-compute rebalance; net-new; v1-buildable
  vs v2-research adjudicated there). W.4b.6 clears the gate for the *homogeneous
  same-model density* lever (branch A) only — branch B is unexercised by full-handoff.
- **Rendezvous floor — RESOLVED for branch A** by full-handoff (executor runs the
  whole B=N decode; cross/inproc = 0.983×). The per-token barrier remains the cost
  *if* branch B (per-token coalescing) is pursued.
- Self host-tgid resolution (bare-metal == no namespace; W.4a debt #2/#3).
- 100-tenant scale (V.1); heterogeneous-model density (CP 5.5 / Weeks 13-14).
- **Productionize the gate-guard call without the CUPTI auto-init** (W.4b.6 used the
  `CIPHER_RT_DISABLE_AUTO_INIT` env hatch; a dedicated pure-guard entry point or a
  separate gate subprocess is the clean form).

## G — Anchors + Goal 1 status + next

substrate `8b5e928` + kmod `02fc2d1` + cipher_kv_bridge `5a3db034` UNCHANGED across
W.4b.2→.6 (userspace harness + an existing env-var flag only — no substrate edit, so
no backfill/9-cell regression required per discipline (a)).

**Goal 1 status — DENSITY DELIVERED at N=8 Mistral-7B; the engagement gate CLEARS
(W.4b.6).** The W.4b.5 "regime-unreachable" limit is closed by Track-2
weight-sharing, and the engagement number clears break-even decisively:

- **Gate result: substrate-attributable = 3.239×** (cross-process full-handoff
  1.041 ÷ naive-8-concurrent 0.321 tok/W), approaching CP 5.6's **3.69×** located at
  this exact N=8 Mistral regime. In-proc ceiling lever = 3.296×; cross/inproc =
  0.983× (transport banked).
- **Regime now reachable:** lossless weight-sharing (KL = 0, bit-identical
  TinyLlama + Mistral) fits executor + 8 Mistral tenants in **13.49 GiB** (was
  9 × 14 = 126 GiB → OOM). Step 1.2/1.3 PASS.
- **The earlier 0.475× cross/inproc tax was a self-inflicted CUPTI artifact**
  (gate-guard ctypes load auto-subscribed CUPTI launch callbacks), root-caused by a
  5-control elimination + pmon and fixed with `CIPHER_RT_DISABLE_AUTO_INIT=1`. NOT a
  fundamental cross-process or rendezvous ceiling.
- **Safety intact at the gate regime:** audited guard rejects distinct-fp at N=8
  Mistral (`distinct_rejected=1`, zero cross-fp coalesce); correctness exact-or-
  benign-near-tie (teacher-forced KL ≤ 7.04e-5; row-3 cascade proven a coin-flip,
  not corruption); FIRSTCOALESCE backstop + block/disable loop intact.

**Scope boundary (the A/B fork — for Anil).** The gate clears for **branch A**:
*homogeneous same-model density* — agents submit a request and await a batched
completion (how real multiplexed serving works), the executor batches B=N on shared
weights. This IS the CP-5.6 / Goal-1-v1 density lever, now realized cross-process
with real per-process admission. It does NOT exercise **branch B** — *per-token
cross-process coalescing* / heterogeneous-compute rebalance (memo-literal peer-KV
attention), which W.4b.5 showed is rendezvous-bound and remains **R-4 / W.4b.2b**
future scope. For the v1 same-model density goal, branch A is the faithful and
sufficient realization.

**Disposition:** R-2 / engagement gate **MET** for v1 same-model density. W.4b
closes clean on this basis. (Full detail: W4B_6_FINDINGS.md.) **Next:** per W.4a §G,
W.7 NCCL is unblocked; branch-B heterogeneous density is R-4/CP-5.5 future scope.

---

## H — W.4b FINAL CLOSE on branch A (2026-05-29, Anil-adjudicated)

After W.4b.6 cleared the gate, Anil elected to **harden via option (1)** before
closing. **W.4b.7** (`W4B_7_FINDINGS.md`, commits `479c728`+`61dcc14`) decomposed the
per-token rendezvous and **falsified the "rendezvous IS the gap" premise**: the
rendezvous is **~8 %** of the step (4.4 ms); the **B=4 Mistral decode forward (50.9 ms,
92 %) is kernel-launch-latency bound** (~52 ms at 1980 MHz == ~51 ms at 1005 MHz). The
per-token cross-process path is **already at 0.98× of the in-proc throughput ceiling at
baseline**, and its apparent tok/W edge is a clock-droop by-product of the rendezvous
idle gaps (so amortizing — memo items 2–6 — would be *counterproductive*; Anil retired
2–6). §4 GREENLIGHT met at baseline. **Correctness (Memory #11) PASS** throughout
(tf 255–256/256; benign near-ties, gaps 0.0 / 0.0078; FIRSTCOALESCE fails=0).

**W.4b is CLOSED on branch A** (homogeneous same-model density; gate MET 3.239×,
W.4b.6). Branch B (R-4 per-token peer-KV coalescing) is **deferred** — W.4b.7 showed
its real costs (peer-KV transport that grows with seq len; sitting on the launch-bound
forward) are **not** de-risked by the cheap fixed-embed rendezvous.

**Next substep (Anil-directed 2026-05-29): NOT W.7 NCCL yet.** The launch-bound forward
that W.4b.7 surfaced as the true ceiling-limiter is the next target — a **CUDA-graph /
batch-size probe** on the ~52 ms B=N decode forward (orthogonal to branch A/B; lifts
branch A's absolute speed and bounds any future R-4). Design memo to follow (design-memo
→ approve → build). W.7 NCCL deferred until after that probe.

**Anchors at final close UNCHANGED across all of W.4b.2→.7:** cipher_rt_phase4
`8b5e928` (libcipher_rt.so md5 `9fe23143`) / cipher_kmod `02fc2d1` (0.6.6) /
cipher_kv_bridge `5a3db034`. `w4b-close` tag moved to the W.4b.7-close commit.

Next (after Anil adjudication): per (3)→(1)/(2), then W.7 NCCL (W.4a §G).
