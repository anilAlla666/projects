# R-I1 SHIELD via SM-PARTITION — design memo (STOP for Anil approval; no build code)

**Date:** 2026-05-29. design-memo → **approve** → build → STOP at gate. R-I1 ONLY. The D.8
throttle-band HARD STOP (`D8_CLOSE_REPORT.md` 6a1eeae §3.5: a CPU-submission sleep is a
contention-*frequency* reducer — median recovers, but p99 is **collision-bound, stuck ~24×
baseline at any throttle, and only relieved by near-suppressing the neighbor** — because it cannot
preempt an in-flight kernel) re-scopes R-I1 to the architecturally correct lever: **spatial SM
partition** (green-context CU masking), which bounds the tail by giving the latency tenant SMs the
aggressor physically cannot touch — *without* suppressing the aggressor.

**R-D5 burst-fairness stays CLOSED-as-moot** (D.8 §4.4: decode throughput is overhead-bound, HBM
4–18%; freeing the GPU does not raise an overhead-bound victim's throughput). NOT rebuilt here.

## §0 Provenance / anchors (verified on disk)

| component | anchor | note |
|---|---|---|
| cipher_rt_phase4 | `ed130e7`, tag `d7-rh1-close`, anchor `.so 01d4effb` | UNCHANGED |
| cipher_kmod | **canonical `02fc2d1` (0.6.6)**; **LOADED `0.7.0`** (srcversion `E3887DF…`, inert additive D.8 ledger) | R-I1 uses the **CP54 path (ioctl NR 13)**, present in BOTH 0.6.6 and 0.7.0 — **SEPARATE from the D.8 ledger (NR 32)**. We run against the loaded 0.7.0 (CP54 + inert D.8). |
| cipher_kv_bridge.so | `5a3db034` | UNCHANGED |
| cipher-fusion-evidence | HEAD `6a1eeae` | — |

## §1 Built-vs-net-new (read on disk)

The SM-partition substrate is **almost entirely built** (CP 5.4 / Track 3). R-I1 is mostly
configuration + a Gate-B harness + verification, not new substrate.

**BUILT:**
- **CP54 ALLOCATE ledger** (`cipher_cp54_sched.c`, kmod ioctl **NR 13**) — the cross-process SM
  authority: partitions the H100 as **15 × 8-SM groups** (120 of 132 SMs; 12-SM remainder
  unallocatable) and hands a requesting PID a `grp_mask` of free groups. **Different PIDs get
  DISJOINT groups** (per-PID in_use marking under the ALLOCATE mutex) — the disjointness §2 needs
  is a ledger invariant, not new code.
- **Per-process green ctx bound to its CP54 groups** (`cipher_rt_green_ctx.c`,
  `cipher_rt_green_ctx_cp54_init`): reads env **`CIPHER_QOS_CLASS` / `CIPHER_SM_COUNT`**, issues
  CP54 ALLOCATE, caches `grp_mask`, builds the green ctx over exactly those 8-SM groups via
  `cuDevSmResourceSplitByCount(MIN_SM=8) → cuDevResourceGenerateDesc → cuGreenCtxCreate →
  cuCtxFromGreenCtx → cuGreenCtxStreamCreate`. PARTITION class = non-zero mask; SHARED = no groups
  (default-OFF). Graceful fallback if `/dev/cipher`/ALLOCATE absent.
- **T4.2.4d per-launch enforcement** (`cipher_cupti.c`) — the **load-bearing** routing: at every
  kernel-launch CUPTI callback the calling thread is made-current on its green CUcontext, so
  PyTorch's NULL-stream **and** cuBLAS launches resolve to the green partition's SMs (not the
  primary full-SM context). This is exactly the fix that turned the `cipher-t424c`
  "partition-not-enforced" null result into the `cipher-t424d` 11× confinement. Marlin-quant has a
  documented carve-out (not relevant to a plain-matmul aggressor).
- **Priority-band table** (`cipher_sm_set_priority`, `cipher_green_ctx.cu`) — built, unused by policy.

**NET-NEW (the R-I1 work):**
1. Route **BOTH** Gate-B tenants through the CP54 PARTITION path on **disjoint** group sets
   (aggressor band 0 → set A; latency tenant band ≥1 → set B; A∩B=∅) — primarily configuration via
   the existing `CIPHER_QOS_CLASS`/`CIPHER_SM_COUNT` env (the ledger guarantees disjointness across
   PIDs). Optional thin wiring to derive (class, size) from the D.8 SHIELD band instead of raw env.
2. The Gate-B harness on the clean diag2 probe (§4).
3. Verification instrumentation: confirm A∩B=∅ (grp_masks), confirm the aggressor is actually
   confined (its launches greened — `g_ctx_swaps_to_green>0` and its SMs ⊆ A), per-tenant KL=0.

## §2 DISJOINT-PARTITION requirement (load-bearing — the `cipher-t424c` lesson)

A one-sided victim reservation does **NOT** isolate: if the aggressor stays on the primary
full-SM context it floods all 132 SMs including the victim's, and the lever is inert (this is
exactly the `cipher-t424c` "passes API smoke but bomb-throughput A=B" failure). **Both** tenants'
launches must route through their own green-ctx partitions, with **A∩B=∅**:
- aggressor (band 0) confined to CU set A; latency tenant (band ≥1) confined to CU set B.
- The CP54 ledger gives disjoint groups per PID (§1) — but the build MUST verify it empirically:
  the aggressor's kernels must land on A only (not spill to B). Verification = (i) grp_masks
  disjoint, (ii) `g_ctx_swaps_to_green>0` for the aggressor, (iii) the contention-relief itself
  (if the victim's p99 recovers while the aggressor keeps full work, confinement is proven; if not,
  outcome (3)).
- **Caution carried (`cipher-t424e`):** prior CP54 A/B testing found partition delivers tail-
  *tightness* but at an **absolute-latency cost** — a tenant confined to 8 SMs runs its own work on
  fewer SMs. So the honest Gate-B reference is the victim **on its own partition B alone** (controls
  for the SM-count cost); we also report victim-on-full-GPU-solo for context. The diag2 victim is
  tiny (512³–1024³ GEMM, fits 8 SMs), so the SM-count cost should be small — but it is a real
  PARTIAL-outcome risk (§5).

## §3 Transparent rebind at the intercept layer (no app change)

CIPHER retargets a tenant's launches onto its green-ctx partition via the **CUPTI launch
callback** (`cuCtxSetCurrent`/make-current at ENTER) — **no torch/vLLM/NCCL source change** (Mem
#24). Semantics preserved:
- **Address space:** a green context shares the parent context's address space → weights, KV
  cache, and all device pointers remain valid (no copy, no remap). KV intact.
- **Stream/event:** the NULL stream resolves to the green ctx's default stream (on the partition);
  stream/event ordering within the tenant is unchanged (same program order, same dependencies).
- **KL=0 (Mem #11):** the partition changes only *which SMs* a kernel runs on — identical kernel,
  identical args, identical context address space, identical launch order ⇒ per-tenant output is
  bit-identical. Verified at the gate (KL=0 vs standalone); any divergence = HARD STOP.

## §4 Gate B re-run — CLEAN diag2 harness (NOT HF-generate)

Use `d8_diag2.py`'s probe (wall-clock submit→sync, GPU-bound victim, **truly saturating
aggressor**) — the harness that produced the trustworthy §3.5 numbers — extended so BOTH tenants
take CP54 PARTITION green ctxs (disjoint). Three numbers, plus the anti-spin control:
- **victim p99 SOLO** (victim alone — report both full-GPU-solo and partition-B-solo per §2),
- **victim p99 aggressor-OFF**… i.e. the un-partitioned contention baseline = the §3.5 **40×** number,
- **victim p99 aggressor-ON-with-PARTITION** (both confined, disjoint).
- **aggressor work held:** the aggressor must keep **~full matmul throughput** under partition
  (it owns set A) — the explicit contrast to the throttle's kill-switch (which dropped it to ~1%).

**The bar (outcome 1 / PASS):** victim p99 bounded near its solo floor **while the aggressor keeps
~full work**. Report the p99 **tail** with aggressor-work held — do **not** over-correct to PASS off
a recovered p50 (the D.8 trap); the gate is the tail.

## §5 Pre-registered outcomes (one run, three legitimate verdicts — anti-spin)

1. **PASS** — victim p99 bounded near solo under the aggressor, aggressor keeps ~full work →
   R-I1 closed; rotate anchor + tag.
2. **PARTIAL** — tail tightens but stays elevated (e.g. the `cipher-t424e` absolute-SM-cost), OR
   isolation only holds when the aggressor is degraded → report measured victim p99 + aggressor
   work; R-I1 partial on 1 H100.
3. **NOT-ENFORCED** — green-ctx cannot transparently confine the aggressor (the `cipher-t424c`
   not-enforced failure recurs: aggressor spills to the victim's SMs) → SM-partition also
   insufficient on this stack → surface as a **1-H100 isolation ceiling** (escalate: MPS / MIG /
   multi-GPU), NOT faked.

Whichever manifests is reported. No tuning to pass.

## §6 Discipline + close gate

- **Scope: R-I1 ONLY.** R-D5 stays closed-as-moot; not rebuilt.
- **Mem #24 substrate-line:** CP54 ALLOCATE (NR 13) + the CUPTI launch intercept ONLY. Any
  required torch/vLLM/NCCL source patch → HARD STOP, surface.
- **Mem #11:** rebind must not alter output — per-tenant KL=0 vs standalone; divergence = HARD STOP.
- **Mem #16 (HARD GATE):** fresh substrate; **default-OFF** (no `CIPHER_QOS_CLASS` ⇒ SHARED ⇒ no
  partition ⇒ byte-identical to current). Close ONLY after: **additivity-KL byte-identical OFF**
  + **30-min N=128 resolver soak** + **Gate B PASS (outcome 1)**.
- **Anchor NOT rotated until Gate B PASS.** kmod state pinned (§0): R-I1 uses CP54 NR 13 on the
  loaded 0.7.0; canonical anchor stays 0.6.6 unless a kmod change is needed (none expected — CP54 is
  built). bf16/INT4 throughout.

**STOP — awaiting Anil's approval before any build code.** On approval: wire both tenants through
disjoint CP54 partitions + the diag2 Gate-B harness + disjointness/confinement/KL verification, run
Gate B + the §6 regression, STOP at the per-outcome verdict.
