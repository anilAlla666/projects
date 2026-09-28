# CP 5.4 — DEFERRAL AUDIT (Steps 1.1 → 1.6B-2A)

**Date:** 2026-05-19. **Type:** paperwork — no GPU, no source modified, no
measurement. Anchors unchanged: kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`.

**Purpose.** Before Step 1.6B-3 (naive baseline) and 1.6B-4 (the headline OP
sweep) run, audit the entire CP 5.4 build to date for anything deferred,
skipped, marked optional/future/sidebar, or implicitly assumed — that would
matter at **production scale**. The 1.6B-4 substrate-value measurement must
measure the right thing with full deferral context.

**STOP after this audit** — per-finding adjudication required before 1.6B-3/4.

---

## §0 — What "production scale" means here, and why it changes the lens

Production scale = **100-tenant Nemotron Nano on one H100** (the CP 5.5 target).
This is **not** a uniform scale-up of the Step 1.6 sanity (2-tenant TinyLlama).
The structural facts that govern every finding below:

1. **The kmod ledger has 15 8-SM groups** ([[cipher-cp54-15groups]]). You
   **cannot** give 100 tenants each a PARTITION. A 100-tenant deployment is
   therefore **POOL-dominated**: a small number of PARTITION tenants (isolated,
   latency-sensitive) + the POOL batched executor serving the *bulk* of the 100
   as `B≈N` socket clients of one model process. The CP 5.4 POOL executor
   (`cipher_batch_executor_gen.py`) already *is* that N-client model.
2. So at N=100 the load lands as: **few PARTITION tenants** (each its own
   green-ctx SM slice) **+ one POOL process at `B`≈(100−few)** running a
   confined green context on the residual groups. Per-finding impact is judged
   against *that* shape, not against "100 partitions."
3. CP 5.4 v1 is **static per-session `qos_class`** — no dynamic/per-request
   arbitration (scope memo §1). The N=100 mix is fixed at deploy time.
4. **The Step 1.6 sanity that just passed (`b2_op5_b2a`) ran 2 PARTITION tenants
   that allocated and freed in lockstep.** OP-5 runs **5** PARTITION tenants,
   OP-asym runs **4** with non-uniform sizes. Several findings below are invisible
   at N=2 and first bite at OP-5/OP-asym — i.e. inside Step 1.6 itself.

Findings are **D**# (deferred), **N**# (NOT deferred — built/verified, with
evidence), **O**# (open question, not a deferral). §16 buckets the D-items.

---

# PART I — DEFERRALS (D1–D15)

## D1 — Multi-partition free-order POOL fragmentation (count-only torch API)

**a) What.** The POOL executor builds its green context with
`torch.cuda.GreenContext.create(num_sms)` — a **count-only** API (Step 1.3b'
Phase A, the load-bearing FINDING). It has no SM-set parameter. Disjointness
holds *only* because torch's `create(8K)` deterministically picks the **low
prefix** groups 0..K−1 (Appendix-A probe, 4/4 runs) and the kmod places the
POOL on that low prefix while PARTITIONs take the high end. If ≥2 PARTITIONs
**free out of LIFO order**, the kmod POOL becomes a **non-contiguous, non-low-
prefix** group set — which a count-only `create(N)` can no longer match.

**b) Where.** `CP_5_4_STEP_1_3B_DESIGN_MEMO.md` Appendix A cost 3
("Multi-partition free-order caveat … Out of Step 1.3b' scope; flag for Step 1.6
— needs kmod compaction or an accepted-fragmentation policy");
`CP_5_4_STEP_1_3B_REPORT.md` "Scope deferred" — explicitly punted **to Step 1.6**.

**c) Production-scale impact.** This is the **scariest in-scope finding**: it
breaks **inside Step 1.6**, not just CP 5.5. OP-5 = 5 PARTITION tenants, OP-asym
= 4. The `b2_op5_b2a` sanity passed only because its 2 partitions allocated and
freed in lockstep (FIFO). The moment OP-5's tenants finish at different times
(realistic — different prompt lengths) and free out of order, the POOL's groups
fragment off the low prefix; torch's `create(N)` picks the low N/8 groups, which
now **overlaps a still-live PARTITION**. The kmod ledger says "disjoint";
physical SMs are not. The POOL's `%smid` self-verify *raises* (good — it fails
loud, not silent) — but a raise is a **STOP, not a recovery**: the run aborts.
At N=100 with continuous tenant turnover this is not an edge case, it is the
steady state.

**d) Closing it requires.** One of: (i) **kmod group compaction** — on FREE,
the kmod migrates the POOL back onto a contiguous low prefix (medium effort,
reopens the kmod → Step 1.1/1.2 cycle, anchor rotation); (ii) **accepted-
fragmentation policy** — POOL keeps a non-contiguous mask and the executor gets
a grp_mask-precise green context (needs libcipher_rt-class driver-API green-ctx,
i.e. Option 3 from the 1.3b' memo — rejected once for Phase B methodology
continuity); (iii) **constrain Step 1.6 to FIFO-free OPs and document the gap**
(zero effort, but OP-5/OP-asym realism is then artificial). Effort: (i)
medium-high; (ii) high; (iii) trivial but narrows the claim.

**e) Recommendation.** **Close in Step 1.6** (it gates OP-5/OP-asym validity).
Recommend (iii) for the 1.6B-4 sweep itself — run the OPs FIFO-free and **state
the constraint explicitly in the report** — plus open a tracked pre-CP-5.5
workstream for (i) kmod compaction. Rationale: 1.6B-4 must not silently measure
a FIFO-only regime and present it as general; and N=100 turnover *needs* (i).

## D2 — `CIPHER_CP54_MAX_ALLOCS = 64` metadata table

**a) What.** The kmod CP 5.4 per-PID metadata table is a fixed array of **64**
entries (`cipher_cp54_allocs`).

**b) Where.** `CP_5_4_STEP_1_1_BUILD_LOG.md` "Design notes" item 2 — verbatim:
*"Fine for CP 5.4 Step 1 verification (2 partition + a handful of pool tenants);
**too small for CP 5.5's 100-tenant soak** — flagged to revisit (hashtable or
larger array) before CP 5.5."* An explicit, named deferral.

**c) Production-scale impact.** Under the §0 POOL-dominated shape, the *ledger*
entries = (few PARTITIONs + 1 POOL) — well under 64, so the table is **not**
the binding limit if the 100 tenants are POOL *socket clients* (they are not
ledger entries; only the one POOL process is). The risk is the opposite mistake:
if a future deployment registers many PARTITIONs, or a leak inflates the table,
the 65th ALLOCATE fails. The honest production statement: 64 is adequate for
"≤63 PARTITION + 1 POOL," which the 15-group ledger already caps far below 64
anyway — so the table is **over-provisioned relative to the 15-group ceiling**.
The flag in the build log slightly overstated the risk; the real CP 5.5 check
is whether anything *other* than ledger entries scales with N.

**d) Closing it requires.** Either confirm 64 ≥ max concurrent ledger entries
(trivial — bounded by 15 groups → ≤15 PARTITIONs + 1 POOL = 16; **64 is safe**)
and downgrade the flag; or, if a future ABI lets one PID hold multiple
allocations, resize. Effort: trivial (a confirmation note).

**e) Recommendation.** **Separate workstream pre-CP-5.5**, but as a
**5-minute confirmation, not a rebuild**: document that the 15-group ledger
ceiling (≤16 entries) makes `MAX_ALLOCS=64` permanently safe, and close the
Step 1.1 flag. Rationale: the Step 1.1 flag should not be left dangling into
CP 5.5 as if it were unresolved scaling work — the arithmetic resolves it.

## D3 — Sub-40-SM lift-curve extension

**a) What.** The Step 1.4 confined-pool lift curve was measured K=15..5
(120..40 SMs). Below 40 SMs is **unmeasured**. Step 1.4 flagged the K=5 reps as
noisy (~5% spread) — "a soft signal of *approaching* but not crossing the §8
knee."

**b) Where.** `CP_5_4_STEP_1_4_REPORT.md` close ("Step 1.5 may want to probe
past 40 SMs"); `CP_5_4_STEP_1_5_DESIGN_MEMO.md` §3 — *"defer it explicitly … a
Step 1.4 addendum or folded into Step 1.6"*; `CP_5_4_STEP_1_6_DESIGN_MEMO.md`
§7 — *"Not extending the lift curve below 40 SMs — that is a Step 1.4 addendum
… explicitly deferred."* Deferred twice, never scheduled.

**c) Production-scale impact.** OP-5 (the headline) puts the POOL at **40 SMs /
5 groups — exactly the lowest measured point**, the one with 5% noise and a
suspected nearby knee. At N=100 the POOL carries `B`≈95 clients on those 40
SMs. If the real knee is at, say, 32 SMs, OP-5's POOL is fine; if the K=5
noise *is* the knee onset, the POOL's batched throughput at 40 SMs is already
degrading and 1.6B-4's aggregate-throughput characterization sits on an
unstable point. The curve's behavior **at and just below** the headline
operating point is unknown.

**d) Closing it requires.** A 3–6-run sweep at K=4,3 (32/24 SMs), reusing the
Step 1.4 harness (`cp54_pool.py` `CIPHER_POOL_MAX_GROUPS` knob) unchanged.
Effort: low (~0.5 d, no new code).

**e) Recommendation.** **Separate workstream pre-CP-5.5** (a Step 1.4
addendum), OR fold a single K=4 confirmation point into 1.6B-4. Rationale: the
headline OP-5 measurement should be able to say "40 SMs is *on* the flat
region" with a measured point below it, not infer it from a noisy edge.

## D4 — Marlin partition-aware grid-sizing on variable / asymmetric SM counts

**a) What.** Marlin reads `cipher_rt_green_ctx_sm_count()` to size its GEMM
grid (CP 5.3 STEP 2A). CP 5.4 makes the green context **variable-size** (16,
24, and the OP-asym set 8/16/16/24). Whether Marlin's grid-sizing is correct
*and* performant at these specific non-uniform counts is **not exercised** by
any CP 5.4 step.

**b) Where.** `CP_5_4_STEP_1_3_DESIGN_MEMO.md` "Marlin already consumes
green-ctx size" — *"a wrong CP 5.4 SM count silently mis-sizes Marlin.
Composition noted; not a blocker."* No CP 5.4 step measures Marlin under a
CP 5.4 partition; OP-asym (the only non-uniform point) is in the 1.6B-4 sweep,
not yet run.

**c) Production-scale impact.** This is **"unexercised," not "broken"** — and
the distinction matters. The PARTITION tenants in Step 1.6
(`cp54_s16_partition_tenant.py`) run **sustained `B=1` decode**. Marlin's
designed regime is **`B≥8`** ([[cipher-t45-substrate-marlin]]) — at `B=1`
decode Marlin's GEMM path is very likely **not on the hot path at all**. The
POOL executor runs `libcipher_v2.so`, which has **no Marlin** (Step 1.3b').
So across the entire Step 1.6 OP matrix, Marlin × CP 5.4 partitioning may
**never fire**. The genuine production gap: a future PARTITION tenant doing
**batched prefill** (`B≥8`) on a 16- or 24-SM green context would be the first
real Marlin×partition composition — and grid-sizing on a 16-SM (2-group) or
asymmetric slice is verified nowhere. [[cipher-marlin-primary-ctx-pin]] /
finding F1 already showed Marlin is structurally full-GPU and needed a
primary-ctx pin; partition-confined Marlin is explicitly Phase 5 work.

**d) Closing it requires.** First a **cheap characterization**: confirm whether
Marlin fires at all in the Step 1.6 OP configs (CUPTI `cuModuleLoad`/launch
trace on a partition tenant — ~1 h). If it does not fire (expected), the
production gap is "batched-prefill PARTITION tenant" — a separate measurement
needing a `B≥8` partition workload at 16/24-SM. Effort: characterization low;
full closure medium (new workload + grid-sizing audit).

**e) Recommendation.** **Close the characterization in Step 1.6** (add the
CUPTI-trace check to 1.6B-3/4 so the report can state plainly "Marlin does/does
not participate at these OPs"); **defer the `B≥8`-partition Marlin grid-sizing
audit to a separate workstream** (it is Phase 5 / Marlin×partitioning, per
memory). Rationale: 1.6B-4's report must not leave a reader guessing whether
Marlin is in or out of the measured path.

## D5 — Green-context resize churn *rate* at sustained N=100 churn

**a) What.** Step 1.5 measured the cost of a **single** pool-resize (~1.7 ms)
and concluded the arbitration policy needs **no resize rate-limiting**. The
measurement was 10 resize cycles, one at a time. The **rate** of resizes under
sustained multi-tenant churn was not measured.

**b) Where.** `CP_5_4_STEP_1_5_REPORT.md` §1.5D — *"green-context churn is
operationally free; the CP 5.4 arbitration policy needs no resize rate-limiting
or amortization"* — derived from one resize = 0.09% of a round, and "even a
pathological tenant arriving and leaving every round."

**c) Production-scale impact.** The "free" conclusion was validated at low N
and a hand-picked pathological rate. At N=100 with tenants continuously
arriving/departing, **every** PARTITION arrival or departure forces the POOL
to resize its green context — a **synchronous ~1.7 ms stall** on the POOL
process. If the PARTITION churn rate is, say, 20 events/s, that is ~34 ms/s of
POOL stall (~3.4%) — right at the campaign's ±3% noise gate. The Step 1.5
threshold math ("a resize would have to cost >55 ms to breach ±3%") priced
*one* resize, not a *rate*. Sustained churn is a rate × cost product, and the
rate was never bounded.

**d) Closing it requires.** A churn-rate microbench: drive the
`cp54_s15_resize.py` harness with N partitions arriving/departing at varied
rates, measure cumulative POOL stall as a fraction of wall time, find the rate
at which it breaches ±3%. Effort: low (~0.5 d, harness exists).

**e) Recommendation.** **Fold into Step 1.7** (failure/robustness modes is the
natural home for "behavior under sustained churn"). Rationale: Step 1.5's
"free" verdict is correct *per event* but should not be cited at CP 5.5 as
"free at any rate" — 1.7 should bound the rate.

## D6 — Marlin cubin concurrent cold-compile at N-process cold start

**a) What.** Marlin's GEMM cubin is NVRTC-compiled once per process: **~11.9 s
cold**, **~0.1 s warm** (NVRTC on-disk cache). Step 1.5B-4 verified one process;
the warm path relies on the **shared NVRTC disk cache**.

**b) Where.** `CP_5_4_STEP_1_5_REPORT.md` §1.5B-4 — *"~11.9 s cold NVRTC
compile (first run), 111.6 ms warm (NVRTC disk cache)"*; §1.5D — *"a one-time
per-process cost, paid before steady state."*

**c) Production-scale impact.** Under §0, only the **few PARTITION tenants** run
`libcipher_rt` (which carries Marlin); the POOL runs `libcipher_v2` (no
Marlin). So at N=100 the cubin cost is paid by the *few* partitions, not 100×.
**But:** if those PARTITION processes start **concurrently from a cold disk
cache** (fresh pod, post-reboot, cache evicted), they each see an empty cache
and **all compile at once** — racing for NVRTC + GPU compile resources, and the
disk cache offers no benefit because none has finished writing it yet. The
cold-start window for the partition set is then ~12 s × (compile contention
factor), not 12 s amortized. Warm restart is fine; **cold concurrent start is
the brutal case** and is unmeasured. (If a future deployment puts Marlin in the
POOL path too — e.g. batched-prefill POOL — this becomes 100× and severe.)

**d) Closing it requires.** Measure: launch K partition processes concurrently
with the NVRTC cache cleared, time to all-warm. Mitigation if bad: a
**pre-warmed persistent cubin cache** seeded at pod provision time (so cold
start never happens in production). Effort: measurement low; persistent-cache
mitigation medium.

**e) Recommendation.** **Separate workstream pre-CP-5.5.** Rationale: not on
the Step 1.6 critical path (Step 1.6 runs warm), but a real production
cold-start cost that CP 5.5's 100-tenant soak should either measure or design
out with a seeded cache.

## D7 — Per-tenant power accounting / metering

**a) What.** The substrate reports power via **NVML — whole-GPU only**. There
is no per-tenant (per-PARTITION, and especially per-POOL-client) power
attribution. CUPTI power-by-context is not implemented.

**b) Where.** No CP 5.4 step builds it. `CP_5_4_STEP_1_6_DESIGN_MEMO.md` §3
explicitly makes aggregate tok/W **descriptive, never a per-tenant lift claim**;
[[cipher-lift-framing]] — "aggregate-TPW lift … is mechanically unsupported for
SM partitioning." The campaign measures *whole-GPU* watts throughout (T4.3
VOLT, every tok/W number).

**c) Production-scale impact.** A 100-tenant production deployment **needs
per-tenant power for billing**. Two layers of the gap: (i) NVML cannot split
power between concurrently-running PARTITION tenants on disjoint SM slices;
(ii) worse, under §0 the **POOL serves ~95 tenants as `B`-batched clients of
one process** — NVML and even CUPTI context-power **cannot attribute power
*between clients of the same POOL process at all***, because they share one
context, one set of kernels. Per-client billing for POOL tenants would need
token-count-proportional allocation of the POOL's measured power — a modeling
choice, not a measurement. This is a genuine, unaddressed production-billing
gap.

**d) Closing it requires.** A power-attribution design: per-PARTITION via
CUPTI context-power or SM-occupancy-weighted NVML; per-POOL-client via a
token-proportional split of the POOL process's power. This is **design +
build**, a real workstream — not a measurement tweak. Effort: medium-high.

**e) Recommendation.** **Separate workstream pre-CP-5.5** (or explicitly CP 5.5
scope). Rationale: not a substrate-correctness gap and not on the Step 1.6
critical path, but production billing cannot ship without it — it must be on
the books with an owner before CP 5.5 claims a 100-tenant deployment.

## D8 — KV cache & GPU-memory behavior at N=100 (POOL `B`≈95)

**a) What.** CP 5.4 does not touch KV cache or the GPU memory allocator. The
Step 1.6 workload is TinyLlama-1.1B; the POOL batches `B=2..8` in sanity. At
production the POOL runs Nemotron Nano at `B`≈95 and KV slab allocation
patterns, page-cache behavior, and HBM headroom are all different from the
N=8 regime CP 5.4 exercised.

**b) Where.** Not in any CP 5.4 step (CP 5.4 = arbitration/partitioning only).
KV is CP 5.1 (vLLM KV integration, [[cipher-cp51-closed]]) / CP 5.2 (offload,
[[cipher-cp52-closed]]) — both certified on TinyLlama + Mistral-7B at small
batch, **not** at `B`=95. [[cipher-phase-a-multitenant]] already found "weight-
cache = capacity not bandwidth" and "B=1 decode is overhead-bound."

**c) Production-scale impact.** At `B`≈95 the POOL's KV cache is the binding
HBM consumer: 95 concurrent Nemotron-Nano sequences × context length × KV
bytes. Whether it fits on one H100 alongside model weights, whether vLLM's
paged-KV (CP 5.1) holds at that batch, and whether KV slab allocation
fragments under 95-way churn — none is verified. This is arguably **the**
dominant CP 5.5 risk, and CP 5.4 neither addresses nor measures it (correctly —
it is out of CP 5.4 scope), but 1.6B-4 measuring TinyLlama at small `B` must
not be read as evidence the memory side scales.

**d) Closing it requires.** A CP 5.5-scope KV/memory capacity study at
`B`≈95 with the real Nemotron-Nano model: HBM budget, paged-KV behavior,
fragmentation under churn. Effort: high — this is core CP 5.5 work.

**e) Recommendation.** **Separate workstream — explicitly CP 5.5 scope.**
Rationale: out of CP 5.4 scope by design; flagged here so the 1.6B-4 report
states plainly that it measures *arbitration/partitioning*, not memory
scaling, and the CP 5.5 plan must carry the KV capacity study as a first-class
gate.

## D9 — Failure recovery beyond the single-process `do_exit` reaper

**a) What.** The verified failure path is the `do_exit` reaper reclaiming **one
exiting process's** groups (Step 1.2 Test 4, Step 1.3b' Test C). **Not** yet
covered: kmod `insmod` failure rollback under load, the §5 pool-resize
under-fill race, and **concurrent** crashes (≥2 partitions crashing
simultaneously, or a crash *mid-ALLOCATE*).

**b) Where.** `CP_5_4_SCOPE_V1.md` §7 (failure-recovery table — the §5 resize
race is "best-effort + retry — *acceptable for v1.0*") and §5 (the resize race
is "not a single transaction"); the whole of **Step 1.7** ("failure-mode
coverage — crash paths, reload rollback, resize race") is **planned, not
done** — CP 5.4 is 6/8 steps, 1.7 and 1.8 remain.

**c) Production-scale impact.** At N=100 with continuous turnover, crashes are
routine, not exceptional. Single-process exit is handled. The unverified cases:
(i) **concurrent crashes** — two PARTITION reapers (lock-free, atomic context)
running while a third tenant's ALLOCATE holds `cipher_cp54_lock`; Step 1.1
fixed the `pool_pid` clobber race and the module-exit teardown race, but
"N reapers concurrent with a live ALLOCATE" is reasoned-about, not tested.
(ii) **crash mid-ALLOCATE** — a process killed inside the ioctl: kernel
syscalls run to completion so the mutex is not orphaned by a userspace SIGKILL,
but this rests on reasoning, not a test. (iii) the **§5 resize race** under
real concurrency — accepted as best-effort, never exercised under load.

**d) Closing it requires.** This **is Step 1.7's defined scope** — crash-path
injection, reload-rollback-under-load, resize-race stress. Effort: the
~0.5–1 d already budgeted for Step 1.7 (`CP_5_4_SCOPE_V1.md` §8/§10).

**e) Recommendation.** **Fold into Step 1.7** — it is already 1.7's scope; this
audit's contribution is to make the **concurrent-crash and crash-mid-ALLOCATE**
cases *explicit* 1.7 test items (the scope memo lists "crash paths" generically;
N=100 makes the *concurrent* variant the one that matters). Rationale: no new
workstream needed; just ensure 1.7's plan names the concurrent cases.

## D10 — Dynamic / per-request arbitration (v1 → v2)

**a) What.** CP 5.4 v1 is **static per-session declared `qos_class`** — a
tenant's PARTITION/SHARED/POOL class and SM count are fixed at registration.
Dynamic, per-request arbitration (re-allocating SMs as load shifts) is **v2**.

**b) Where.** `CP_5_4_SCOPE_V1.md` §0 — *"v1 functional scope … static
per-session declared `qos_class` … dynamic/per-request arbitration deferred to
v2"*; `CP_5_4_STEP_1_6_DESIGN_MEMO.md` §7 — *"Not a scheduler / dynamic
per-request arbitration — v1 is static … dynamic arbitration is v2."*

**c) Production-scale impact.** At N=100 with bursty, heterogeneous load, a
static SM split has **no elasticity**: a PARTITION tenant idle between requests
still holds its groups; a hot POOL cannot borrow SMs from quiet partitions.
The deployment is provisioned for worst-case concurrent load and wastes SMs the
rest of the time. This is a real production efficiency ceiling — but it is a
**known, deliberate v1 boundary**, not an oversight.

**d) Closing it requires.** CP 5.4 v2 — a scheduler/arbitration policy on top
of the v1 ledger (the ledger's ALLOCATE/FREE/QUERY ioctls were designed to
support it; `QUERY` exists "for the scheduler client"). Effort: high — a
distinct CP.

**e) Recommendation.** **Separate workstream — v2 / post-CP-5.5.** Rationale:
explicit, adjudicated v1 scope boundary. Flagged here only so the 1.6B-4 report
and the CP 5.5 plan state plainly that the 100-tenant claim is for a *static*
allocation, and an idle-tenant efficiency loss is expected and unmeasured.

## D11 — Hot-swap / live-patch kmod reload

**a) What.** Updating the kmod requires a **drain**: stop all CIPHER tenants,
`rmmod`, `insmod`, tenants re-register. No hot-swap / live-patch path.

**b) Where.** `CP_5_4_SCOPE_V1.md` §6 — *"A hot-swap/live-patch path is out of
scope for v1.0"* — and the drain is justified as acceptable because this is a
**dev pod with no production tenants**.

**c) Production-scale impact.** The "dev pod, drain is fine" justification
**does not hold at production scale.** A 100-tenant production deployment
cannot drain 100 live tenants to ship a kmod bugfix or ABI addition — that is a
full-service outage. Every CP 5.4 reload so far (Step 1.2) was on a quiescent
pod; production needs either a maintenance-window policy or a live-update path.

**d) Closing it requires.** Either a documented **maintenance-window operational
policy** (low effort — paperwork: "kmod updates require a drain window") or an
actual **live-patch / hot-swap mechanism** (high effort — kernel livepatch or a
versioned side-by-side device, which the scope memo notes is "not possible"
with the same major/name). Effort: policy low; mechanism high.

**e) Recommendation.** **Separate workstream pre-CP-5.5** — at minimum the
operational policy must be written before a 100-tenant deployment. Rationale:
the v1.0 "drain is fine" rests entirely on "dev pod"; CP 5.5's production
framing invalidates that premise and the gap must be acknowledged, even if the
resolution is just a documented maintenance window.

## D12 — Asymmetric-model matrix

**a) What.** Step 1.6 runs **TinyLlama-1.1B for both** PARTITION and POOL
tenant classes. A mixed/asymmetric-*model* deployment (different model sizes
per tenant) is not exercised. (OP-asym varies SM *count*, not model.)

**b) Where.** `CP_5_4_STEP_1_6_DESIGN_MEMO.md` §2 — *"Same model isolates the
arbitration variable; an asymmetric-model matrix adds a model-size confound.
Asymmetric models = deferred realism extension (§7)"*; §7 — *"Not an
asymmetric-model matrix — same model both classes in v1."*

**c) Production-scale impact.** A real 100-tenant Nemotron-Nano deployment is
plausibly **heterogeneous** — different tenants running different model
variants / context lengths / quantizations. The CP 5.4 substrate is
model-agnostic (it arbitrates SMs, not models), so there is no *correctness*
risk; the gap is that the *measured* isolation/variance numbers (1.6B-4's
headline) are for the homogeneous case. Whether per-tenant variance isolation
holds when a heavy tenant and a light tenant share the ledger is unmeasured.

**d) Closing it requires.** A mixed-model OP run (e.g. one PARTITION on a 7B
model, others on 1.1B) — measurement only, no substrate change. Effort: low–
medium (model setup).

**e) Recommendation.** **Separate workstream — post-CP-5.4 realism extension**
(could be a CP 5.5 sub-item, since CP 5.5's Nemotron-Nano deployment is the
natural place). Rationale: deliberate v1 scoping; flagged so 1.6B-4's
variance-isolation headline is correctly stated as "homogeneous-model."

## D13 — DVFS / VOLT regime composition under mixed deployment

**a) What.** The VOLT clock-locking actuator (T4.3, [[cipher-t431-volt-shipped]]
/ [[cipher-t432-kmod-volt-ioctl]]) is not touched, composed, or measured by any
CP 5.4 step. CP 5.4 mixed deployment runs at whatever the default clock regime
is.

**b) Where.** Absent from all of CP 5.4 (Steps 1.1–1.6). VOLT's envelope
([[cipher-t43-envelope]]) is "+55% on memory-bandwidth-bound decode only;
7B+ regime needs recalibration."

**c) Production-scale impact.** H100 DVFS sets **one clock for the whole GPU** —
there is no per-SM-partition clock. At N=100 mixed deployment, a clock locked
for the POOL's memory-bound `B`-batched decode may be wrong for a PARTITION
tenant doing compute-bound prefill, and vice-versa. VOLT's gains were measured
on isolated single workloads; whether a locked clock helps or hurts the
*aggregate* of a heterogeneous 100-tenant mix is unknown. CP 5.4 correctly
does not claim any tok/W lift (§3, [[cipher-lift-framing]]), so this is not a
retracted-claim risk — it is an unmeasured composition.

**d) Closing it requires.** A VOLT × mixed-deployment measurement — does a
locked clock chosen for the POOL degrade PARTITION tenants? Effort: medium
(measurement; VOLT actuator exists).

**e) Recommendation.** **Separate workstream — post-CP-5.5** (or fold into the
CP 5.5 Nemotron benchmark, where tok/W is measured anyway). Rationale: not a
CP 5.4 gap (CP 5.4 makes no power claim); flagged so CP 5.5's power numbers
account for the whole-GPU-clock constraint under a heterogeneous mix.

## D14 — 12-SM remainder — unallocatable capacity

**a) What.** The H100 has 132 SMs; the kmod ledger is 15 × 8-SM groups = **120
SMs**. The **12-SM remainder is unallocatable** under 8-SM granularity.

**b) Where.** `CP_5_4_SCOPE_V1.md` §1 — *"The 4-SM remainder … is unallocatable
… inherited from the existing green-ctx design's accepted 3% loss"* (the memo
said 4 SMs / 16 groups; Step 1.3a corrected this to **15 groups / 12-SM
remainder**, [[cipher-cp54-15groups]]); `CP_5_4_STEP_1_6_DESIGN_MEMO.md` §4 —
"the 12-SM remainder is unallocatable."

**c) Production-scale impact.** 12/132 = **~9% of the GPU is permanently idle**
under CP 5.4 arbitration — larger than the "accepted 3%" the scope memo
inherited (because 15 not 16 groups). At N=100 that is ~9% of capacity the
deployment paid for and cannot use. It is a **known, accepted, fixed** loss —
not a bug — but the magnitude (9%, not 3%) should be stated explicitly in any
production capacity claim.

**d) Closing it requires.** Either accept it (zero effort — it is a hardware ×
green-ctx-granularity constraint) or investigate sub-8-SM green contexts (the
green-ctx `MIN_SM=8` is a CUDA constraint — likely not closable). Effort:
accept = zero.

**e) Recommendation.** **Accept; no action — but document the 9% in the CP 5.5
capacity statement.** Rationale: genuinely unclosable at the green-ctx layer;
the only deferral-audit action is to make sure the production capacity number
says "120 of 132 usable," not "132."

## D15 — Track 2 cross-process weight sharing (SC4 identity, SC5 arena lifetime)

**a) What.** Track 2 = the cross-process **weight-sharing** workstream
([[cipher-phase-c]] — VMM bridge interception, `cuMemExport` POSIX-FD). The
user's audit list names **SC4** (cross-process weight-sharing *identity
verification* — the attack surface: does process B get *exactly* process A's
weights, unmodified?) and **SC5** (kmod-owned arena lifetime — who owns the
shared weight arena, and what happens on owner exit?).

**b) Where.** **Not part of CP 5.4 at all.** CP 5.4 is SM arbitration /
partitioning; Track 2 is a separate Phase C workstream. Phase C status
([[cipher-phase-c]]): Track 2 design memo done, 4 methodology citations
verified, vLLM 0.21.0 installed isolated. SC4/SC5 are Track 2 success criteria,
their build state is **not advanced by any CP 5.4 step**.

**c) Production-scale impact.** Decisive for N=100 **if** the deployment shares
weights: 100 separate Nemotron-Nano weight copies do not fit one H100, so
either (i) the POOL-batched model (one weight copy, `B`-batched — the CP 5.4
path, no Track 2 needed) or (ii) Track 2 cross-process weight sharing for the
PARTITION tenants. SC4 (identity verification) is a **correctness/security
gate** — a partition tenant must not get subtly-wrong shared weights. SC5
(arena lifetime) is a **resource-safety gate** — the shared arena must outlive
every consumer. Under the §0 POOL-dominated shape, the bulk of N=100 goes
through the POOL (no Track 2 needed); Track 2 matters only for *PARTITION*
tenants wanting to share weights. So Track 2 is **not on the CP 5.4 / CP 5.5
critical path** *if* CP 5.5 is POOL-served — but if PARTITION tenants are
expected to run distinct large models, Track 2 SC4/SC5 become required.

**d) Closing it requires.** Track 2 execution — a separate Phase C workstream
with its own design memos and gates. Effort: high (weeks — per the Phase C
scope).

**e) Recommendation.** **Separate workstream — Track 2 / Phase C, not CP 5.4
or CP 5.5-blocking under the POOL-served model.** Rationale: out of CP 5.4
scope entirely; included here because the user listed it — the audit's finding
is that CP 5.4 neither advances nor depends on it, and whether CP 5.5 needs it
hinges on the unstated question "are the 100 tenants POOL-served (no Track 2)
or PARTITION-served-with-distinct-models (Track 2 required)?" — **that question
should be adjudicated.**

---

# PART II — NOT DEFERRED (built / verified — evidence trail)

## N1 — Slot→SM mapping bridge (the deferred T4.2.4c work) — BUILT

**Status: fully built and verified.** Scope memo §2 identified the slot→SM
bridge as "the deferred T4.2.4c work … never built … CP 5.4's load-bearing
work." It **was** built: Step 1.3 Phase C — `cipher_rt_green_ctx_cp54_init()`
issues `CIPHER_CP54_ALLOCATE`, caches the `grp_mask`, and `ensure()` selects
the set-bit groups from `cuDevSmResourceSplitByCount` → `cuGreenCtxCreate`
(variable-size). Verified: Step 1.3a probe proved **group g ↔ the same physical
SMs in every process** (`PHASE_1_3A_PROBE.md`, re-validated Step 1.3 Phase B);
Step 1.3 Test B confirmed a 16-SM partition gets exactly `grp_mask=0x0003`.
**Caveat (minor, for the record):** the group count is a hardware-hardcoded
constant `CIPHER_CP54_NUM_GROUPS = 15` (corrected from 16 in Step 1.3 Phase A);
`grp_mask` is a `u32` with `static_assert(NUM_GROUPS <= 32)`. Correct for this
H100; **not parameterized** for a different GPU (H200/B100 would need the
constant re-derived). Not a deferral — a documented hardware binding.

## N2 — Single-process crash recovery (`do_exit` reaper) — VERIFIED

**Status: built and empirically verified.** The kmod `do_exit` reaper
reclaims an exiting PID's groups. Verified: Step 1.2 Test 4 (a forked child
ALLOCATEs 24 SMs, `_exit()`s without FREE and without REGISTER_TENANT — all 3
groups reclaimed; this is the re-attempt that caught and fixed the reaper-guard
leak), Step 1.3 Test C, Step 1.3b' Test C (`kill -9` the POOL executor → all 15
groups reclaimed, no leak). The **multi-process / concurrent** crash case is
the *un*verified part — see D9.

## N3 — Single green-context resize cost — MEASURED

**Status: measured.** Step 1.5B-3 — the full production resize path costs
~1.7 ms (shrink) / ~1.6 ms (grow), ~0.6 ms over the bare-API floor, fully
accounted. What is *not* covered is the **sustained rate** of resizes — see D5.

## N4 — Marlin cubin one-shot-per-process — VERIFIED

**Status: verified.** Step 1.5B-4 — the Marlin cubin is `cuModuleLoadData`-
loaded **exactly once per process**, at warmup, and is **not** recompiled or
reloaded on green-context churn (4-assert PASS, CUPTI-traced). The *cold-compile
cost under concurrent N-process cold start* is the unmeasured part — see D6.

---

# PART III — OPEN QUESTIONS (not deferrals — must resolve before 1.6B-4)

## O1 — OP-2 arithmetic (88-SM vs 104-SM pool)

`CP_5_4_STEP_1_6_DESIGN_MEMO.md` §11.1 and `CP_5_4_STEP_1_6B_1_CHECKPOINT.md`
both flag it: the adjudication text said "OP-2 = 2×16 SM + 13-group/104-SM
pool," which does not reconcile — 2×16 SM = 4 partition groups → **11**-group/
**88**-SM pool. Provisionally pinned to **2×16 SM + 88-SM pool**. **Needs user
confirm before 1.6B-4.** Not a deferral — an unresolved spec arithmetic item.

## O2 — Step 1.3b' Test D −2.3% regression magnitude

`CP_5_4_STEP_1_3B_REPORT.md` Test D: the 120-SM green-ctx confinement costs
~2.3% mean throughput vs the 132-SM baseline; the mean-vs-mean ±3% gate passes
but two of three reps individually fall below the −3% floor. Flagged for
closure adjudication — "accept as-is, or re-measure with more reps / a
recalibrated 120-SM baseline." Not a deferral — an open closure-adjudication
item. (Note: Step 1.4 already adopted the 120-SM anchor as the curve's top
point, which is the recalibration the flag suggested — so O2 is arguably
already addressed and just needs formal closure.)

---

# §16 — Adjudication summary — recommended buckets

Per the user's framework: which deferrals close in **Step 1.6**, which fold into
**Step 1.7**, which become **separate workstreams pre-CP-5.5**, which are v2 /
accepted. Recommendations only — adjudicate per-item.

| # | Finding | Recommended bucket | Effort |
|---|---|---|---|
| **D1** | Multi-partition free-order POOL fragmentation | **Close in Step 1.6** (constrain 1.6B-4 to FIFO-free + document) **+ kmod-compaction workstream pre-CP-5.5** | trivial / med-high |
| **D4** | Marlin × partition grid-sizing | **Characterize in Step 1.6** (CUPTI trace: does Marlin fire?) + defer `B≥8`-partition audit | low / med |
| **D5** | Resize churn *rate* at N=100 | **Fold into Step 1.7** | low |
| **D9** | Concurrent-crash / crash-mid-ALLOCATE recovery | **Fold into Step 1.7** (already 1.7 scope — name the concurrent cases) | budgeted |
| **D2** | `MAX_ALLOCS=64` | Pre-CP-5.5 — **5-min confirmation** (15-group ceiling → safe), close the flag | trivial |
| **D3** | Sub-40-SM lift curve | Pre-CP-5.5 (Step 1.4 addendum) or one K=4 point in 1.6B-4 | low |
| **D6** | Marlin cubin concurrent cold-compile | Separate workstream pre-CP-5.5 | low / med |
| **D7** | Per-tenant power accounting | Separate workstream pre-CP-5.5 (or CP 5.5 scope) | med-high |
| **D8** | KV cache / memory at `B`≈95 | Separate workstream — **explicitly CP 5.5 scope** | high |
| **D11** | Hot-swap kmod reload | Pre-CP-5.5 — at minimum a written maintenance-window policy | low / high |
| **D10** | Dynamic per-request arbitration | v2 / post-CP-5.5 (deliberate v1 boundary) | high |
| **D12** | Asymmetric-model matrix | Post-CP-5.4 realism extension (CP 5.5 sub-item) | low-med |
| **D13** | DVFS/VOLT × mixed deployment | Post-CP-5.5 (or fold into CP 5.5 Nemotron benchmark) | med |
| **D15** | Track 2 SC4/SC5 weight sharing | Separate workstream (Track 2 / Phase C) — **adjudicate whether CP 5.5 needs it** | high |
| **D14** | 12-SM (~9%) remainder | Accept; no action — document "120 of 132 usable" in CP 5.5 capacity claim | zero |

**The two findings that gate Step 1.6 itself** (must adjudicate before
1.6B-3/4): **D1** (fragmentation can break OP-5/OP-asym) and **D4** (the report
must state whether Marlin is in the measured path). Everything else can run in
parallel with 1.6B-3/4 or after.

**Open items to resolve before 1.6B-4 (not deferrals):** O1 (OP-2 arithmetic —
needs user confirm), O2 (Test D −2.3% — formal closure).

**Verified / not deferred:** N1 slot→SM bridge (built), N2 single-process
reaper (verified), N3 single resize cost (measured), N4 Marlin cubin one-shot
(verified).

---

**STOP. Surfacing for per-finding adjudication.** No 1.6B-3 (naive baseline) or
1.6B-4 (OP sweep) until each finding above is adjudicated into a bucket. The
headline substrate-value measurement (1.6B-4) must measure the right thing with
this deferral context settled — in particular D1 (FIFO-free constraint) and D4
(Marlin in/out of path) change *what configuration 1.6B-4 measures* and *how
the report frames it*.
