# CP 5.4 — Step 1.4 (confined-pool batch-lift curve) — PHASE 1.4A DESIGN MEMO

**Date:** 2026-05-19. **Status: measurement design — two items differ from V1
§8; STOP for adjudication before the sweep** (per the Step 1.4 Phase 1.4A
discipline). No measurement run yet. Anchors unchanged.

---

## V1 §8 — the Step 1.4 spec as written

> Measure delivered substrate-attributable lift at pool sizes **128 / 112 / 96
> / 64 SMs** (16 / 14 / 12 / 8 groups), 3 reps each, teacher-forced KL ≤ 0.1
> per tenant. … the curve's top point is the **128-SM (16-group) pool**,
> reported as **delta from that point** … **Stop-and-adjudicate** if the curve
> collapses sub-linearly past the 64-SM point.

**Confirmed matching the Step 1.4 authorization** — 3 reps per size; TFGATE
KL ≤ 0.1; curve reported as delta from the top pool point (not Phase B's
3.69×/132-SM number); stop-and-adjudicate on sub-linear collapse; workload =
Phase B N=8 TinyLlama executor.

## Discrepancy 1 — pool sizes: §8's 16-group basis is invalid

§8 specifies **128/112/96/64 SMs = 16/14/12/8 groups**. This rests on the
**16-group** assumption that CP 5.4 Step 1.3a **disproved**:
`cuDevSmResourceSplitByCount(minCount=8)` on this H100 yields **15** 8-SM
groups + a 12-SM remainder (`PHASE_1_3A_PROBE.md`); the kmod was corrected to
`CIPHER_CP54_NUM_GROUPS = 15` (anchor `8d777dfb`). §8's literal sizes
**128 SMs (16 groups) and 96 SMs (12 groups, an even count not on a 15-down
sweep)** are not realizable.

The Step 1.4 authorization's set — **120/104/88/72/56/40 SMs = 15/13/11/9/7/5
groups** — is the corrected sweep: top point 120 (the real 15-group max,
already adopted as the anchor in the Step 1.3b' closure), every-other-group
down to 5. It is finer (6 points vs §8's 4) and deeper (40 SMs / 5 groups vs
§8's floor of 64 SMs / 8 groups) — and going past the old 64-SM point is
exactly what §8's sub-linear-collapse stop-condition was written to watch.

**Recommendation:** adopt the authorization's set {15,13,11,9,7,5 groups =
120,104,88,72,56,40 SMs}; §8's 16-group sizes are superseded by the measured
hardware. (Confirm.)

## Discrepancy 2 — how to set the pool size: an ordering complication

The Step 1.4 authorization says *"shrink pool via PARTITION tenants holding the
appropriate high groups."* For a pool size of K groups, the executor-POOL must
own the **low** groups [0..K-1] — that is the load-bearing requirement, because
`torch.cuda.GreenContext.create(K·8)` picks the **low-prefix** groups and the
executor's green-ctx self-verify (Step 1.3b') checks exactly that. So the
(15-K) partition-held groups must be the **high** ones.

**The complication:** the kmod's `cp54_claim` is **low-first**. A PARTITION
allocated into a ledger with no POOL present claims the **low** groups [0..N-1]
— the executor-POOL would then be left the **high** groups, and torch's
low-prefix green context would not match the pool's actual groups →
**self-verify FAIL**. A PARTITION lands on **high** groups only via
`cp54_pool_shrink`, which requires a POOL to already own them. So the
partition-holder method needs, per pool size, a **primer sequence**: a primer
POOL claims all 15 → the partition allocates (`pool_shrink` releases the high
(15-K), the partition takes them) → the primer FREEs (its low K groups go
free) → only then the executor starts and POOL-allocates the free low K. A
2-helper dance with explicit sequencing/ledger checks each step.

**Simpler alternative — a pool-size cap knob.** Add `CIPHER_POOL_MAX_GROUPS=K`
to `cp54_pool.py`: the executor still `ALLOCATE(POOL)`s, but builds its green
context at `min(grp_count, K)` groups. `create(K·8)` → torch's low-prefix pick
→ groups [0..K-1] → self-verify PASS; the executor runs confined to K SMs. This
measures the **identical** quantity the lift curve wants — executor throughput
vs pool SM count — **race-free, no partition holders, no primer**. It is
faithful: the lift curve prices the *pool's* throughput at each size; partition
*contention* is Step 1.6 (mixed deployment), not 1.4, and idle partition
holders would not contend anyway. The only nuance — the kmod ledger nominally
shows the POOL owning 15 while the green context uses K — is cosmetic for a
throughput measurement and will be documented.

**Recommendation:** use the **cap knob**. It is simpler, race-free, and
measures exactly the curve's quantity. If the user prefers real partition
occupancy for realism, the primer+partition dance is viable but slower and
racier — surfaced for the call.

## Proposed Step 1.4 plan (pending adjudication)

1. **1.4A** (this memo) — adjudicate Discrepancies 1 & 2.
2. **1.4B sweep** — for K ∈ {15,13,11,9,7,5} groups (120/104/88/72/56/40 SMs):
   set the pool to K (cap knob, or primer+partition per adjudication); run the
   Phase B N=8 TinyLlama executor 3 reps; capture mean tok/s, mean tok/W,
   TFGATE KL/rep, dmesg; the green-ctx self-verify confirms groups [0..K-1]
   each run. 6 sizes × 3 reps = 18 runs.
3. **1.4C analysis** — tok/s and tok/W vs pool size; delta from the 120-SM top
   point; locate the knee; flag if sub-linear past ~64 SMs (§8 stop-condition).

Anchors: no rotation — measurement step. The only source touch is the
`CIPHER_POOL_MAX_GROUPS` knob in `cp54_pool.py` (a Step 1.3b' test file, not an
anchor) if the cap-knob methodology is adjudicated.

**STOPPING HERE for adjudication of Discrepancies 1 & 2 — no sweep run, no
source modified. Anchors unchanged: kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`.**
