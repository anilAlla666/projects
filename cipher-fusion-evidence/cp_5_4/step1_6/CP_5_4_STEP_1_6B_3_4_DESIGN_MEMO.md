# CP 5.4 — Step 1.6B-3 + 1.6B-4 — COMBINED DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source,
no GPU, no measurement. STOP for adjudication before the 1.6B-3 build.

**Combined memo rationale.** 1.6B-3 (naive Arm-A baseline) and 1.6B-4 (the
CIPHER operating-point sweep, Arm B) are **paired measurement work** — one
measurement framework, CIPHER off vs on. Comparison validity depends on a
*single* methodology (identical model, prompts, N, per-step timing code path,
variance statistic). Designing them apart invites a subtle Arm-A/Arm-B
mismatch. Track 2 / Track 3 used separate memos for *independent build*
phases; 1.6B-3/4 is not that — it is one framework measured twice.

**This memo does NOT re-litigate Step 1.6 §11.** The Step 1.6 design memo
(`CP_5_4_STEP_1_6_DESIGN_MEMO.md`) was adjudicated 2026-05-19 — §11 pinned the
operating points, the 2-arm methodology, the per-decode-step CV metric, the
disjointness probe, the loader invariant. Those are **settled**; this memo
cites them and builds the 1.6B-3/4 execution on top. It addresses only (a) the
items genuinely *new since §11* — chiefly Track 2 having since closed — and
(b) the §11.1 OP-2 arithmetic flag, which §11 itself deferred "to confirm
before 1.6B-4."

---

## §0 — Anchor reality: the substrate under 1.6B-1/1.6B-2 has been rotated out

**Load-bearing — read first.** The Step 1.6 design memo §9 records the
substrate as kmod `8d777dfb`, libcipher_rt `ebc0baaa`, libcipher_v2
`86618c30`, cipher_kv_bridge `fca6843d`. The 1.6B-1 partition-tenant runner
and 1.6B-2 orchestrator were built and sanity-passed against **that**
substrate.

Since then — same calendar day, but after 1.6B-2A:

- **Track 3 (DSM)** rotated **libcipher_rt** `ebc0baaa → 83afd1ca` (DSM was
  additive, but it *modified the libcipher_rt source* that PARTITION tenants
  load) and the kmod through the Track 3 line.
- **Track 2** rotated the **kmod** to `285d102e` (Track 3 close) then
  `285d102e → 008b3c66` (SC5 weight-arena) and **cipher_kv_bridge**
  `fca6843d → c04b0c39` (SC3).

**Current substrate: kmod `008b3c66`, libcipher_rt `83afd1ca`, cipher_kv_bridge
`c04b0c39`, libcipher_v2 `cc0479b8`** (verified by `md5sum` at this memo's
pre-baseline orientation; `/proc/cipher/{arenas,migrations}` both live; kmod
loaded).

So the substrate that 1.6B-1/1.6B-2 were validated against **no longer
exists.** The PARTITION-tenant path (libcipher_rt green-ctx init →
`CIPHER_CP54_ALLOCATE`) runs on a libcipher_rt that Track 3 rebuilt and a kmod
Track 2 rebuilt twice. The CP54 ALLOCATE path *should* be untouched (Track 3
DSM and Track 2 weight-arena were both additive), but **"should be" is not
"verified."**

⇒ **The 1.6B-3 build does not start until 1.6B-2A is re-run and re-passes on
the current anchors** — see §6, this is a load-bearing gate, not a generic
regression line.

---

## §1 — What is settled (Step 1.6 §11) — cited, not re-opened

| dimension | pinned by §11 | used here as |
|---|---|---|
| model | TinyLlama-1.1B, **both** tenant classes (§2 — isolates the arbitration variable from a model-size confound) | fixed — see §3 item (a) for the user's re-open |
| prompts | WL01 5-prompt / 128-tok set, gradeable vs `gold.json` | fixed |
| concurrency | steady-state simultaneous decode (contention regime) | fixed |
| headline metric | per-decode-step wall latency; **CV = std/mean** per tenant; prefill step excluded; bootstrap 95 % CI, Arm B vs Arm A | fixed (§11.2) |
| operating points | OP-5 (headline), OP-2, OP-asym; OP-3 optional; 3 reps | fixed — OP-2 arithmetic flag resolved in §4 |
| arms | 2-arm: Arm A naive multi-process / Arm B CIPHER mixed | fixed (§5) |
| disjointness gate | continuous per-round, two-clause, orchestrator-centralised `%smid` probe | fixed (§6 of the Step 1.6 memo) |
| loader invariant | PARTITION tenants launched via `CUDA_INJECTION64_PATH`, never `LD_PRELOAD` (§11.6, Finding 1) | fixed |
| anchors | no rotation expected — 1.6B-3/4 is verification, all new code throwaway | see §7 |

---

## §2 — 1.6B-3: the naive Arm-A baseline — precise definition

Arm A is the honest "what multi-tenant looks like with no substrate." Per
§11.2's fairness clause, pinned precisely:

- **Processes.** At each operating point, Arm A = **one independent HF process
  per logical tenant** — no POOL executor, no batching (naive multi-tenant
  does not batch across tenants), no green contexts, no kmod arbitration, no
  `LD_PRELOAD`, no `CUDA_INJECTION64_PATH`, **no MPS, no MIG**. Pure CUDA
  time-slicing on the full 132 SMs.
  - OP-5 → **10 processes** (5 partition-class + 5 pool-class logical tenants,
    all just HF processes in Arm A — the class distinction is an Arm-B-only
    concept).
  - OP-2 → 2 + N_pool processes; OP-asym → 4 + N_pool processes. The pool-class
    logical-tenant count per OP is pinned in §4.
- **Per-process workload.** Identical to Arm B's PARTITION tenant: TinyLlama-1.1B,
  the WL01 5-prompt set, the same generation length, **the same per-decode-step
  timing code path** (`synchronize → t0 → step → synchronize → t1`). The *only*
  variable between Arm A and Arm B is substrate-mediated SM partitioning vs
  none.
- **Correctness.** Each Arm-A process is logit-KL gated (≤ 0.1) exactly as
  Arm B — a naive baseline that is silently wrong is not a baseline.
- **Measurement points.** idle / loaded / per-decode-step latency / aggregate
  tok/s, plus the 3-point framebuffer convention from SC6 (idle / loaded /
  exited) so Arm A's memory cost is on record too.

1.6B-3 deliverable: `cp54_s16_naive.py` (the Arm-A tenant — throwaway,
`cp54_s16_*` naming) + the Arm-A path in the orchestrator; a sanity run at
one OP before the 1.6B-4 sweep.

## §3 — Items genuinely new since §11 — the memo's real content

### (a) Model selection — recommend HOLDING §2's TinyLlama pin

The user's authorization re-opens "TinyLlama, Mistral-7B, or both?" — Track 2
having since closed on Mistral-7B. **Recommendation: hold TinyLlama for both
classes, per Step 1.6 §2.** The §2 rationale is unchanged by Track 2's
closure: Step 1.6 measures **SM-partition-vs-time-sliced contention** — model
size is a *confound* on that axis, not a variable of interest. Track 2's
Mistral-7B headline was a *memory* measurement, where model size is the point;
Step 1.6 is a *latency-variance* measurement, where it is noise. A Mistral-7B
run is a legitimate **add-on characterization** (like Track 2's two-model
table), explicitly tagged — *not* the 1.6 headline. **Adjudication item (a):**
hold TinyLlama-only headline (recommended), or add a tagged Mistral-7B OP-5
characterization run.

### (b) Tenant count — already a sweep; reconcile, do not re-open

The user's authorization frames an open "N=4 vs N=8 vs sweep." **This does not
map onto the pinned structure** — OP-2 / OP-5 / OP-asym (§11.1) **are** the
tenant-count sweep: OP-5 = 5 PARTITION tenants (the headline, the Nemotron N=5
configuration), OP-2 = 2, OP-asym = 4 (non-uniform). There is no separate
"N=4/N=8" decision to make — the headline N is **5** (OP-5) and the sweep is
{2, 5, 4-asym}. Recorded as reconciled, not as an adjudication item. (If the
user specifically wants an N=8 point, that would need a 4th OP and a 15-group
arithmetic — 8×... does not divide the 15 8-SM groups cleanly at 16 SM each;
flag separately if desired, but it is not in the §11 pin.)

### (c) Track 2 weight sharing in 1.6B-4 — recommend a SEPARATE tagged scenario

Track 2 closed *after* the Step 1.6 memo, so §11 could not address it. The
question: does 1.6B-4 enable cross-tenant weight sharing (all same-model
tenants share one arena)?

**Recommendation: the 1.6B-4 headline arms (A naive, B CIPHER) run WITHOUT
weight sharing; add an optional third scenario "Arm B+WS" — CIPHER mixed
deployment with Track 2 weight sharing on — as a tagged characterization
point.** Reasoning:

- Step 1.6's headline is **latency variance**. Weight sharing is a **memory**
  lever (Track 2's axis), already measured and closed (`TRACK_2_SC6_MEMORY.md`).
  Folding it into the headline arms confounds the variance measurement with a
  memory-layout change.
- But weight sharing *could* perturb latency (shared read-only weights →
  different L2 behaviour under contention). Measuring **Arm B+WS vs Arm B** as
  an isolated, tagged scenario answers "do the two substrate primitives
  compose without a latency regression?" — which is the genuine CP-5.5-relevant
  question — *without* contaminating the A-vs-B headline.
- This mirrors the Track 2 discipline: keep the measured axis clean, put the
  second lever in its own labelled scenario.

**Adjudication item (c):** accept "Arm B+WS as an optional tagged scenario,
headline arms weight-sharing-off" — or direct weight sharing into the headline.

### (d) NVML power sampling cadence

Step 1.6 §3 ruled aggregate tok/W **out as a headline** ([[cipher-lift-framing]])
— power is GPU-global and cannot be attributed per tenant on a shared device.
It is still worth sampling **descriptively**. **Recommendation: NVML power
sampled at a fixed 100 ms cadence**, GPU-aggregate only, reported as
characterization, **never as a substrate "lift"** (a decode round is ~1.86 s →
~18 samples/round, enough to see decode structure; 100 ms is far too coarse to
perturb the workload). tok/W, if reported, is aggregate-GPU and explicitly
tagged descriptive. **Adjudication item (d):** confirm 100 ms aggregate-only
descriptive, or specify otherwise.

### (e) Cold-start vs warm steady-state

§11.2 already excludes the prefill step (step 0) from the decode-variance
population. Cold-start effects beyond prefill — first-round cubin compile,
caching-allocator warmup, clock spin-up — still contaminate the *first decode
round*. **Recommendation: warm steady-state convention — discard the first
decode round per tenant per rep as warmup; the CV statistic is computed over
warm rounds only.** This matches §2's "steady-state contention" framing. The
discarded warmup round is still *recorded* (so a cold-start anomaly is visible),
just excluded from the headline statistic. **Adjudication item (e):** confirm
"first decode round discarded as warmup, recorded but excluded."

### (f) Step 1.6P — recommend a SEPARATE design memo

**Recommendation: 1.6P (POOL-scaling at B=20/50/100) is NOT in this memo.** The
combined-memo rationale is "one measurement framework." 1.6P is a *different*
framework: POOL-only, the batched executor at high batch size, the metric is
POOL throughput scaling, and it is the step where Track 2 weight sharing is
load-bearing (100 POOL clients cannot co-reside without it). 1.6B-3/4 is
mixed-deployment latency variance. Folding 1.6P in would dilute both and break
the "identical methodology" rationale that justifies combining 1.6B-3 with
1.6B-4 in the first place. **1.6P gets its own design memo after 1.6B-4
closes.** Adjudication item (f): confirm 1.6P separate.

## §4 — The OP-2 arithmetic flag (§11.1) — resolved

Step 1.6 §11.1 flagged: the adjudication text said *"OP-2 = 2×16 SM … pool =
13 groups = 104 SM,"* which does not reconcile — 2 partitions × 16 SM = 2 × 2
groups = **4** partition groups, leaving **11** pool groups = **88 SM**. A
13-group/104-SM pool would require only **2** partition groups total = 2 × **8**
SM partitions.

**Recommendation (pinned, pending confirm): OP-2 = 2 × 16-SM PARTITION + an
88-SM / 11-group POOL.** This keeps the 16-SM partition size **consistent with
OP-5** (the adjudication explicitly motivated 2×16 SM for "clean per-partition
consistency"), and `2×2 + 11 = 15` groups checks out. The "13 groups / 104 SM"
phrasing is read as an arithmetic slip in the adjudication text. **Adjudication
item (the §11.1 confirm): accept OP-2 = 2×16 SM + 88-SM/11-group POOL.** This
must be settled before 1.6B-4 (it does not block the 1.6B-3 build).

**Pool-class logical-tenant count per OP** (needed so Arm A knows its process
count — pinned here for confirmation): POOL serves a fixed **5 pool-class
logical tenants** at every OP (the POOL executor batches them; the partition
count is what varies). So Arm-A process counts: OP-5 → 10, OP-2 → 7, OP-asym
→ 9. Adjudication may revise the pool-tenant count.

## §4.5 — Deferral-audit items D1 & D4 — they gate this step

The CP 5.4 deferral audit (`CP_5_4_DEFERRAL_AUDIT.md`, 2026-05-19) bucketed
**D1** and **D4** as "gate Step 1.6 itself — adjudicate before 1.6B-3/4." This
memo must close them.

### D1 — multi-partition free-order POOL fragmentation — resolved by Track 3

D1, "the scariest": the POOL green context uses count-only allocation
(`torch.cuda.GreenContext.create(N)`); disjointness holds only because torch
picks the low SM prefix. ≥2 partitions freeing **out of LIFO order** fragment
the POOL off that low prefix → disjointness breaks. `b2_op5_b2a` passed only
because its 2 partitions were FIFO/lockstep.

**Resolution path:** D1 was adjudicated → became Step 1.6X-1 → the user chose
**Option A (live SM migration)** → that shipped and **closed as Track 3 (DSM)**
([[cipher-track3-dsm]]). The production mechanism for POOL defragmentation now
exists. **For the 1.6B-4 sweep specifically**, D1 is additionally handled by
**measurement design**: each OP rep allocates all PARTITION tenants up front
against a **fully-drained, clean ledger** (contiguous), measures steady-state
decode, and frees all at teardown. **The orchestrator verifies the clean
ledger between reps** — after each rep it queries `CIPHER_CP54_QUERY` and
asserts 0 partition groups held before the next rep allocates; a
non-empty/fragmented ledger **aborts the run** rather than measuring on a
polluted state. (Relying on Python process-teardown order to free in LIFO is
*not* safe — teardown order is not deterministic; the explicit between-rep
`CIPHER_CP54_QUERY` check is what guarantees the clean-start invariant.) No
mid-run free-order churn is exercised inside the measured window — mid-run
churn is Step 1.7 / 1.6P territory, not 1.6B-4's steady-state contention
regime. **Recommendation:** 1.6B-4 runs OP-5 / OP-asym with the
clean-allocate-per-rep discipline above; DSM is the documented production
defrag mechanism but 1.6B-4 does not need to invoke it. **Adjudication item
(D1):** accept "clean-allocate/drain per rep; no mid-run free-order churn in
1.6B-4; DSM is the production answer, exercised under churn in 1.7/1.6P."

### D4 — Marlin × partition grid-sizing — trace it, do not assume

D4: Marlin × partition is **unexercised, not broken**. PARTITION tenants are
B=1 decode (Marlin's regime is B≥8); the POOL runs `libcipher_v2` (no Marlin).
So Marlin **likely never fires** in any Step 1.6 OP — but "likely" is not
"shown." **Recommendation:** 1.6B-3/4 **CUPTI-traces one rep per OP** for
Marlin cubin launches, so the report can state definitively whether Marlin is
in or out of the measured path (the audit's recommendation, adopted). This is
a cheap addition to the §5 measurement set. **Adjudication item (D4):** accept
the per-OP CUPTI Marlin trace as part of the 1.6B-3/4 measurement.

### O2 — Test D −2.3% — considered closed

The deferral audit's open item O2 (the Test D −2.3% regression magnitude) is
treated as **closed**: Step 1.4's 120-SM anchor already *is* the
recalibration ([[cipher-cp54-step1-4]]). No 1.6B-3/4 action; noted for
completeness.

## §5 — 1.6B-4: the operating-point sweep — execution

- **Arms per OP:** Arm A (naive, §2) and Arm B (CIPHER mixed, §11). Optional
  tagged **Arm B+WS** (§3c).
- **OPs:** OP-5 (headline), OP-2, OP-asym; OP-3 if scope. **3 reps each.**
- **Per rep:** all tenants in steady-state simultaneous decode; per-decode-step
  latency captured (warm rounds only, §3e); the per-round two-clause
  disjointness probe on Arm B / Arm B+WS (Step 1.6 §6); per-tenant logit-KL
  gate; NVML power at 100 ms (§3d); 3-point framebuffer. **Clean ledger
  allocate/drain per rep** (§4.5 D1) — no fragmentation carried across reps.
- **One rep per OP CUPTI-traced** for Marlin cubin launches (§4.5 D4) — the
  report states Marlin in/out of the measured path definitively.
- **Isolated single-tenant baselines** at each partition size (§11.3) — one
  16 / 24 / 8-SM PARTITION tenant alone — so "aggregate ≈ Σ isolated" is an
  honest composition check, not a lift claim.
- **Headline analysis:** per-tenant decode-step **CV**, Arm B vs Arm A,
  bootstrap 95 % CI per tenant. Claim "CIPHER PARTITION tenants show lower CV
  than naive" holds **only if the CIs separate**; overlap → reported honestly
  as inconclusive (a real finding, per the stop conditions).
- **Stop conditions** (Step 1.6 §6): any per-tenant KL > 0.1; disjointness
  probe clause-1 or clause-2 failure; an unexpected tenant crash. Aggregate
  throughput is descriptive — a collapse vs Σ-isolated is *flagged for
  adjudication*, not an auto-stop.

## §6 — Pre-measurement baseline — the load-bearing gate

Before **any** 1.6B-3 code is written, on the **current** anchors
(`008b3c66 / 83afd1ca / c04b0c39 / cc0479b8`):

1. **`md5sum` all four anchors** — confirm the current set; record it.
2. **`/proc/cipher/{arenas,migrations}` both operational** — done at this
   memo's orientation; re-confirm at build time.
3. **★ Re-run the 1.6B-2A sanity** — the 2×16-SM PARTITION + POOL co-resident
   lockstep run that 1.6B-2A passed (disjointness clause-1/2 = 0, co-resident)
   — **on the current substrate.** This is the **load-bearing gate** (§0): it
   directly exercises the libcipher_rt green-ctx-init → `CIPHER_CP54_ALLOCATE`
   path that Track 3 rebuilt and the kmod Track 2 rebuilt. Generic regression
   gates (W3, isolation 15/15, SC6) do **not** touch this path.
   - **1.6B-2A re-passes** → the PARTITION path is intact on `008b3c66 /
     83afd1ca`; 1.6B-3 build proceeds.
   - **1.6B-2A fails** → a Track-2/Track-3 change silently touched the CP54
     ALLOCATE / green-ctx path. **STOP, surface** — this is a finding, not a
     gap; 1.6B-3 is not built until it is root-caused and adjudicated.
4. Supporting regression (substrate-health, not the gate): isolation 15/15,
   W3 `SC2_PASS`, dmesg clean.

## §7 — Anchors

No rotation expected — 1.6B-3/4 is verification; all new code is throwaway
(`cp54_s16_*` naming). Expected unchanged through 1.6B-3/4: kmod `008b3c66`,
libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`, libcipher_v2 `cc0479b8`.
**If a defect requires a substrate fix → STOP, surface; that rotates an anchor
and re-enters design-memo discipline** (a measurement that needs a substrate
change is itself a finding).

## §8 — Plan & checkpoint discipline

`1.6B-3` (naive Arm-A harness + sanity) → `1.6B-4` (Arm A + Arm B sweep at
OP-5/OP-2/OP-asym, 3 reps; OP-3 if scope) → `1.6D` analysis/report. Est.
~6–9 h, **multi-session** — checkpoint (harness + partial JSON + sub-phase
note) at every 1.6B-* boundary. Evidence under
`cipher-fusion-evidence/cp_5_4/step1_6/`.

## §9 — Adjudication ask

**STOPPING — no source, no GPU, no measurement.** Decisions:

1. **(a) Model** — hold TinyLlama-only headline (recommended), or add a tagged
   Mistral-7B OP-5 characterization run.
2. **(b) Tenant count** — *reconciled, not an ask*: OP-5 (N=5) is the headline,
   {OP-2, OP-5, OP-asym} is the sweep. Confirm no separate N=8 point is wanted.
3. **(c) Weight sharing** — accept "Arm B+WS as an optional tagged scenario,
   headline arms weight-sharing-off" (recommended), or fold WS into the
   headline.
4. **(d) Power** — confirm NVML 100 ms, GPU-aggregate, descriptive-only.
5. **(e) Cold/warm** — confirm first decode round discarded as warmup
   (recorded, excluded from the CV statistic).
6. **(f) 1.6P** — confirm 1.6P (POOL-scaling) gets its own later design memo,
   not folded here.
7. **(§11.1 confirm) OP-2** — accept OP-2 = 2 × 16-SM PARTITION + 88-SM /
   11-group POOL; and the per-OP pool-class tenant count (pinned provisionally
   at 5).
8. **(§6 gate)** — accept 1.6B-2A re-validation on the current anchors as the
   load-bearing pre-build gate; a 1.6B-2A failure STOPs the 1.6B-3 build.
9. **(D1, §4.5)** — accept "clean-allocate/drain per rep; no mid-run
   free-order churn in 1.6B-4; Track 3 DSM is the production defrag answer,
   exercised under churn in 1.7/1.6P, not here."
10. **(D4, §4.5)** — accept the per-OP CUPTI Marlin trace as part of the
    1.6B-3/4 measurement set.

On adjudication: proceed to **1.6B-3** — pre-baseline §6 first, then the
naive Arm-A harness.
