# CP 5.4 — Step 1.6 (mixed-deployment verification) — DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, STOP for
adjudication. No GPU run, no source modified, no measurement. Anchors
unchanged: kmod `8d777dfb`, libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`,
cipher_kv_bridge `fca6843d`.

---

## §0 — Where this sits

Step 1.5 closed (green-ctx churn ~free; Marlin cubin one-shot). CP 5.4 is
**6/8** sub-steps done. Step 1.6 = **mixed-deployment verification**: N
PARTITION tenants (isolated, each on its own green-context SM slice, running on
`libcipher_rt`) running concurrently with the POOL batched executor (on
`libcipher_v2` + `cp54_pool`), all arbitrated by the one kmod CP 5.4 group
ledger.

**Step 1.6 is substantively different from 1.4/1.5.** Those swept one subsystem
in isolation (the confined pool; green-ctx churn). Step 1.6 is the first time
the two tenant *classes* run real workloads *concurrently* — it tests the
integration, not a parameter. Step 1.3b' Test B proved POOL + 1 idle PARTITION
are disjoint *by construction*; Step 1.6 proves N PARTITION + POOL are correct,
isolated and disjoint *under concurrent load*.

## §1 — Source-grounded starting point

- **PARTITION tenant runner.** `cp54_s15_partition_driver.py` (Step 1.5) only
  `ALLOCATE`s a partition and holds it — it runs **no workload**.
  `cp54_tenant.py` (Step 1.3 Phase D) runs a real TinyLlama forward + 24-tok
  decode under `libcipher_rt` with `CIPHER_QOS_CLASS` / `CIPHER_SM_COUNT` via
  env — but it is a one-shot (load + forward + exit), not a sustained
  concurrent decoder. **Step 1.6 needs a sustained-decode PARTITION tenant** =
  `cp54_tenant.py`'s model-running path + a multi-round decode loop +
  per-round instrumentation. New throwaway harness, no anchor touch.
- **POOL executor.** `cipher_batch_executor_gen.py` + `cp54_pool.py` — already
  a CP 5.4 POOL tenant (Step 1.3b'); `run_arm3.sh` orchestrates it (executor +
  N socket clients + power sampler). Step 1.6's orchestrator extends
  `run_arm3.sh`: launch the POOL executor **and** N PARTITION tenant processes
  concurrently, sharing `/dev/cipher`.
- **Common authority.** PARTITION tenants run `libcipher_rt`; the POOL executor
  runs `libcipher_v2`. They never coordinate directly — the kmod CP 5.4 ledger
  is the sole cross-process authority (disjoint `grp_mask`s by construction).
  That architecture is what Step 1.6 verifies under load.

## §2 — Q1: workload choice for PARTITION tenants

| question | recommendation |
|---|---|
| model | **TinyLlama-1.1B for both PARTITION and POOL tenants** (v1). Same model isolates the *arbitration* variable; an asymmetric-model matrix adds a model-size confound. Asymmetric models = deferred realism extension (§7). |
| prompts | **the WL01 5-prompt / 128-tok set**, same as Phase B, for both classes — so each tenant's output is gradeable against the existing `gold.json` / `gold_logits.pt`. |
| "concurrent" | **steady-state simultaneous decode** — all tenants in their decode window at once, the contention regime. Staggered arrival is the *churn* case, already priced in Step 1.5 (~1.7 ms/resize); 1.6 measures steady-state contention, not arrival transients. |

The Nemotron N=5 pitch (§4) is a *configuration* (the SM split), not the
Nemotron *model* — per Q6, the Nemotron benchmark itself is post-CP-5.4. Step
1.6 runs TinyLlama at the N=5 SM configuration.

## §3 — Q2: methodology focus — and a metric the campaign has already ruled out

**Aggregate tok/W "lift" is NOT a defensible Step 1.6 headline.** Per
[[cipher-lift-framing]] — "aggregate-TPW lift on WL01–WL24 is mechanically
unsupported for SM partitioning; ship on MFU gates not TPW lift." Q2 listed
"aggregate substrate-attributable lift" as a candidate headline; this memo
recommends **rejecting it as the headline** — it is exactly the class of claim
that memory says cannot be mechanically supported for SM partitioning.

**The load-bearing headline is per-tenant latency variance / tail-tightness.**
Per [[cipher-t424e-isolation-confirmed]] — "isolation is variance reduction,
not p99 reduction." The substrate's value proposition for a PARTITION tenant is
*isolation*: a tenant on its own SM slice should see **tighter per-tenant
decode-latency variance** than the same tenant contending naively for the whole
GPU. So:

- **Headline finding:** at the claimed operating points, PARTITION tenants
  under CIPHER show **tighter per-tenant latency variance** (max/mean, or
  p99/p50) than under naive multi-process contention — **at correctness**.
- **Secondary, gating:** per-tenant correctness under concurrency — every
  PARTITION and POOL tenant passes its TFGATE / logit-KL gate (≤ 0.1; campaign
  norm ~1e-4).
- **Descriptive, NOT a lift claim:** aggregate throughput (Σ tok/s across all
  tenants) is reported as characterization of the operating point — *not*
  framed as a substrate "lift." Tagged explicitly as such.

This makes the substrate's value falsifiable on the axis it actually delivers
(variance), and does not re-litigate the retracted TPW-lift framing.

## §4 — Q3: operating points

Constraint: 15 8-SM groups total (kmod ledger, [[cipher-cp54-15groups]]);
`partition groups + pool groups = 15`; the 12-SM remainder is unallocatable.

**Configs as adjudicated 2026-05-19 — pinned in §11; the rows below carry the
adjudicated values (OP-2 and OP-asym corrected from the pre-adjudication draft).**

| point | config | groups (part + pool = 15) | role |
|---|---|---|---|
| **OP-5** (headline) | 5 PARTITION × 16 SM + POOL 40 SM | (5×2) + 5 = 15 | the Nemotron N=5 architectural pitch point |
| OP-2 | 2 PARTITION × 16 SM + POOL 88 SM | (2×2) + 11 = 15 | few partitions, 16-SM each (consistent with OP-5) — characterization. **See §11 — the adjudication's "13 groups/104 SM" pool does not reconcile with 2×16-SM partitions; flagged for confirm.** |
| OP-3 | 3 PARTITION × 24 SM + POOL 48 SM | (3×3) + 6 = 15 | mid characterization (optional, if scope allows) |
| OP-asym | 4 PARTITIONs 8 + 16 + 16 + 24 SM + POOL 56 SM | (1+2+2+3) + 7 = 15 | **asymmetric** — the only point in the whole CP 5.4 campaign that exercises a non-uniform ledger allocation end-to-end; proves the variable-size ledger + slot→SM bridge handle mixed `sm_count`s |

**Recommendation:** OP-5 headline; OP-2 and OP-asym are the load-bearing
characterization companions (OP-3 optional). One asymmetric point is enough —
asymmetric is a ledger/bridge correctness check, not a sweep.

## §5 — Q4: comparison baseline — a deliberate deviation from the Phase B 3-arm

Phase B / CP 5.6 used a 3-arm methodology: Arm 1 naive multi-process HF, Arm 2
vLLM-per-tenant, Arm 3 CIPHER. Q4 asks Step 1.6 to align or **document the
deviation**.

**Recommended (surfaced for adjudication): a 2-arm headline.**
- **Arm A — naive multi-process.** All tenants (partition-class + pool-class)
  as independent HF processes on the full GPU, **no green contexts, no kmod
  arbitration** — pure time-sliced contention. This is the honest "naive
  multi-tenant equivalent of substrate-mediated arbitration."
- **Arm B — CIPHER mixed deployment.** Same tenants; PARTITION tenants get
  isolated kmod-arbitrated green-ctx SM slices, POOL tenants get the batched
  executor.

This is the **same 2-arm shape as the T4.2.4e isolation measurement**
([[cipher-t424e-isolation-confirmed]] — partition vs none), extended from a
single-process A/B to multi-tenant concurrent — i.e. the deviation reuses an
existing campaign methodology, it is not novel.

**Why deviate from 3-arm:** vLLM-per-tenant (Arm 2) is the natural comparator
for the **POOL** (batched-throughput) workload; it has **no clean analogue for
the PARTITION workload** — vLLM exposes no SM-partitioning primitive, so a
"vLLM mixed deployment" does not exist to compare against. Forcing vLLM into the
mixed-deployment matrix would compare different things. **Optional
characterization add-on:** a vLLM-per-tenant run of the POOL-class tenants only,
as a throughput reference point — explicitly tagged "comparison-class deviation
from the Phase B 3-arm, by design."

**Adjudication item:** accept the 2-arm headline (+ optional vLLM POOL-only
characterization), or require the full 3-arm.

## §6 — Q5: stop conditions under concurrency

| gate | spec |
|---|---|
| **per-tenant correctness** | every PARTITION and POOL tenant passes its logit-KL / TFGATE gate (≤ 0.1). **Any** tenant fails → STOP, surface. |
| **runtime SM-disjointness** (load-bearing) | a continuous per-round probe — see below. |
| aggregate throughput | descriptive — *not* a hard gate (§3). A collapse vs the sum of isolated single-tenant runs is flagged for adjudication, not an auto-stop. |

**The runtime-disjointness probe — concretely specified.** A one-shot
self-verify at green-ctx build (as `cp54_pool.py` does today) cannot catch a
partition that *stops* enforcing mid-run (the finding-F1 class — a green ctx
silently ignored). Step 1.6 adds a **continuous probe**:

- **Where:** the `%smid` probe kernel runs **on each tenant's own
  green-context stream** (not the default stream — only a launch on the
  tenant's own stream proves where *that tenant's* kernels land).
- **Cadence:** once per decode round (~1.86 s/round; a `%smid` probe is
  sub-ms — negligible).
- **Pass condition (two clauses):**
  1. tenant *t*'s observed SM set ⊆ the SMs of *t*'s allocated `grp_mask`; **and**
  2. tenant *t*'s observed SM set ∩ every *other* tenant's allocated SM set = ∅.
- **Why clause 2 is not redundant:** clause 1 alone ("I am inside my own
  partition") cannot detect *another* tenant also running inside *t*'s
  partition — a green-ctx-not-enforcing failure on the *other* tenant. The
  pairwise-intersection clause is what catches a silent disjointness break.
- **Where the check runs:** each tenant probes only its *own* observed SM set
  on its own green-ctx stream and reports it back; the **orchestrator** runs
  the load-bearing computation centrally — it collects every tenant's per-round
  observed-SM-set, queries the kmod ledger (`CIPHER_CP54_QUERY`) for every
  tenant's allocated `grp_mask`, and does the clause-1 and pairwise clause-2
  intersection checks. No tenant needs to know another tenant's allocation.
- **Failure mode caught:** without this probe a silent disjointness break
  surfaces only as a throughput/contention anomaly — indistinguishable from
  noise. The probe makes it a hard, attributable STOP.

**Stop conditions (explicit):** any per-tenant correctness gate fails; the
disjointness probe fails either clause; a tenant process crashes (crash
*handling* is Step 1.7, but an unexpected crash during 1.6 still stops the run
and is surfaced).

## §7 — Q6: what Step 1.6 is NOT

- **Not** extending the lift curve below 40 SMs — that is a Step 1.4 addendum
  (the K=5 noise flag), explicitly deferred there at Step 1.5 §3.
- **Not** failure injection / crash-path / reload-rollback / resize-race
  coverage — that is **Step 1.7**.
- **Not** the Nemotron N=5 benchmark itself — post-CP-5.4-closure. Step 1.6
  uses TinyLlama at the N=5 *configuration*.
- **Not** a scheduler / dynamic per-request arbitration — v1 is static
  per-session `qos_class` (CP 5.4 scope §1); dynamic arbitration is v2.
- **Not** an asymmetric-*model* matrix — same model both classes in v1;
  asymmetric models are a deferred realism extension.

## §8 — Proposed Step 1.6 plan (pending adjudication)

1. **1.6A** (this memo) — adjudicate Q1–Q5, the §5 2-arm deviation, the §3
   metric reframing.
2. **1.6B — harnesses.** (i) a sustained-decode PARTITION tenant
   (`cp54_tenant.py` model path + multi-round decode loop + per-round `%smid`
   probe on the green-ctx stream + per-tenant KL); (ii) a mixed-deployment
   orchestrator extending `run_arm3.sh` — launches the POOL executor + N
   PARTITION tenants concurrently, collects per-tenant correctness + latency +
   the central disjointness check; (iii) the naive multi-process Arm A harness.
   Sanity-check.
3. **1.6C — runs.** Arm A + Arm B at OP-5, OP-2, OP-asym (OP-3 if scope), 3
   reps each. Per-tenant correctness gated; disjointness probed every round.
4. **1.6D — analysis / report.** Per-tenant variance (Arm B vs Arm A) =
   headline; per-tenant correctness = gate; aggregate throughput =
   characterization; disjointness = PASS/FAIL. Evidence
   `cp_5_4_step1_6_evidence.tar.gz` (+ `.md5`).

Anchors: no rotation expected — Step 1.6 is verification; all new files are
throwaway harnesses. If a defect requires a substrate fix, that rotates an
anchor and re-enters design-memo discipline.

## §9 — Anchors

Unchanged by this memo and expected unchanged by all of Step 1.6 (verification
step): kmod `8d777dfb`, libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`,
cipher_kv_bridge `fca6843d`.

## §10 — Adjudication ask

**STOPPING HERE for adjudication.** No run, no source modified, no measurement.
Decisions:

1. **Q1 / §2** — TinyLlama both classes / WL01 prompts / steady-state-concurrent.
2. **Q2 / §3** — accept that aggregate TPW-lift is *not* the headline
   ([[cipher-lift-framing]]); headline = per-tenant latency variance
   ([[cipher-t424e-isolation-confirmed]]); aggregate throughput = descriptive.
3. **Q3 / §4** — operating points: OP-5 headline + OP-2 + OP-asym (+ OP-3
   optional).
4. **Q4 / §5** — accept the 2-arm headline (naive vs CIPHER) + optional vLLM
   POOL-only characterization, *or* require the full 3-arm.
5. **Q5 / §6** — accept the continuous per-round, two-clause,
   orchestrator-centralised disjointness probe as the load-bearing runtime gate.

No code until this memo is adjudicated.

---

## §11 — Adjudication outcome (2026-05-19) & build-pinned parameters

All five decisions **ADOPTED**. Step 1.6 build + measurement (Phase 1.6B and
beyond) **AUTHORIZED**. This section pins the parameters the build needs and
records the additional constraints; it supersedes any pre-adjudication value.

### §11.1 — Operating points (pinned)

| point | partitions | partition groups | POOL | total | reps |
|---|---|---|---|---|---|
| **OP-5** (headline) | 5 × 16 SM | 5 × 2 = 10 | 5 grp / 40 SM | 15 ✓ | 3 |
| OP-2 | 2 × 16 SM | 2 × 2 = 4 | 11 grp / 88 SM | 15 ✓ | 3 |
| OP-3 (optional) | 3 × 24 SM | 3 × 3 = 9 | 6 grp / 48 SM | 15 ✓ | 3 |
| OP-asym | 8 + 16 + 16 + 24 SM | 1+2+2+3 = 8 | 7 grp / 56 SM | 15 ✓ | 3 |

**OP-2 arithmetic flag (must confirm before Phase 1.6B-4).** The adjudication
text said *"2×16 SM … with pool = 13 groups = 104 SMs."* That does not
reconcile: 2 partitions × 16 SM = 2 × 2 groups = **4** partition groups, leaving
**11** pool groups = **88 SM** — not 13/104. A 13-group/104-SM pool implies
**2 partition groups total**, i.e. 2 × **8** SM partitions. The adjudication
also explicitly motivated 2×16 SM as "clean per-partition consistency" with
OP-5's 16-SM partitions. Pinned reading, provisionally: **OP-2 = 2 × 16 SM +
88-SM/11-group pool** (keeps the 16-SM partition consistent with OP-5). This
does **not** block Phase 1.6B-1/-2/-3; it is needed only at **1.6B-4**. Flagged
for the user to confirm before the sweep.

**OP-asym (pinned).** Four partitions at **8, 16, 16, 24 SM** (1+2+2+3 = 8
groups) + a **56-SM / 7-group** POOL. Per additional-constraint 4, OP-asym
verification must check: (a) the kmod ledger grants the four non-uniform
`grp_mask`s correctly and disjointly; (b) each tenant's green context's
verified SM count matches its own `sm_count`; (c) the per-round disjointness
probe (§6) passes across the heterogeneous set.

### §11.2 — Per-tenant latency variance — measurement design (constraint 2)

- **Metric:** **per-decode-step wall latency** — each single-token decode step
  timed `synchronize → t0 → model step → synchronize → t1`. The **prefill step
  (step 0, whole-prompt forward) is recorded separately** and excluded from the
  decode-variance statistic — it is a different-sized op. Decode steps (1…gen-1)
  are the variance population.
- **Per-tenant statistic:** p50 / p95 / p99 of the decode-step latency, and the
  **coefficient of variation CV = std/mean** as the headline tail-tightness
  number (CV is scale-free → comparable Arm A vs Arm B and across OPs).
- **Naive baseline fairness (Arm A):** N independent HF processes, **standard
  CUDA time-slicing — no MPS, no green contexts, no `qos_class`, no
  partitioning**; every process sees the full 132 SM; identical model, prompts,
  gen length, and the *identical* per-step timing code path as Arm B. The only
  variable is substrate-mediated SM partitioning vs none.
- **Confidence:** 3 reps per OP; the decode-step population per tenant is large
  (≈ rounds × ~123 decode steps). Report the CV with a **bootstrap 95 % CI**,
  Arm B vs Arm A, per tenant. The headline claim — "CIPHER PARTITION tenants
  show lower CV than naive" — holds only if the CIs separate; if they overlap,
  that is reported honestly as inconclusive (a real finding, per the stop
  conditions).

### §11.3 — Aggregate throughput reporting (constraint 3)

Reported as **descriptive characterization, never as "substrate lift."** In
addition, run **isolated single-tenant baselines at each partition size** (one
16-SM / 24-SM / 8-SM PARTITION tenant alone) so the report can show "each
partition individually runs as expected" — aggregate ≈ Σ isolated is honest
verification that partitions compose, *not* a lift claim.

### §11.4 — Report methodology section (constraint 1)

The Step 1.6 **report** must carry an explicit methodology section explaining
the **2-arm deviation** from the Phase B 3-arm, with the [[cipher-lift-framing]]
rationale spelled out for diligence reviewers: aggregate tok/W across N
partitions is *not* substrate-attributable the way Phase B's cross-tenant
batching was; per-tenant tail latency under contention *is* (CIPHER isolates;
naive multi-process time-slices). The campaign now has two honest methodology
lineages — Phase B 3-arm (concurrent-serving comparison) and Step 1.6 2-arm
(SM-partitioning-vs-time-sliced comparison) — and the report names both.

### §11.6 — Loader invariant for libcipher_rt-loaded tenants (Finding 1, adjudicated)

**A PARTITION (libcipher_rt) tenant MUST be launched via
`CUDA_INJECTION64_PATH=<libcipher_rt.so>`, not `LD_PRELOAD`.** libcipher_rt's
green-context init (`cipher_rt_green_ctx_cp54_init` → `CIPHER_CP54_ALLOCATE`)
runs inside `InitializeInjection` / `InitializeInjection2` (`cipher_inject.c`),
which the CUDA driver invokes **only** for the `CUDA_INJECTION64_PATH` library
at `cuInit`. CP 2.5 made the substrate LD_PRELOAD-free (GOT-patching driven
from `InitializeInjection2`). A tenant launched with plain `LD_PRELOAD` runs
libcipher_rt's constructor but **never** `InitializeInjection` → no green
context → no partition (verified in Step 1.6B-1: `%smid` showed all 132 SMs).
The Phase 1.6B-2 orchestrator launches every PARTITION tenant with
`CUDA_INJECTION64_PATH`. Allocation is read from the kmod ledger
(`CIPHER_CP54_QUERY`), not `cipher_rt_green_ctx_sm_count()` (Finding 2).

### §11.5 — Phase plan (authorized) & checkpoint discipline

Phase 1.6B-1 partition-tenant runner → 1.6B-2 orchestrator → 1.6B-3 naive
baseline → 1.6B-4 OP sweep (OP-5/OP-2/OP-asym, 3 reps; OP-3 if scope). Est.
~7–10 h — **multi-session**; checkpoint (harness + partial JSON + a sub-phase
note) at every 1.6B-* boundary so progress survives across sessions; status
report at each boundary. New harness code is throwaway, `cp54_s16_*.py`
naming; no substrate source modification expected, anchors held. Stop
conditions: per-tenant KL > 0.1; disjointness probe clause-1/2 failure (a
substrate correctness issue); aggregate catastrophically off Σ-isolated; an
inconclusive variance claim is documented honestly, not suppressed.
