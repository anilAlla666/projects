# W.4a — cross-tenant POOL eligibility + correctness substrate — CLOSE REPORT

**Date:** 2026-05-28
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `9fe23143b12e67355e7acc0b606d0c25` (CUDA-13 in-container)
**Side anchor:** `build_cuda13/libcipher_rt.so.w4a` (identical bytes)

## Verdict
**W.4a PASS.** The unbreakable safety layer for Goal 1 heterogeneous
multiplexing is built and verified: co-resident tenants (W.6sC cohort) are
partitioned into same-model coalesce-groups (W.6sB fingerprint), with a
per-tenant correctness backstop. W.4a is **DECISION-ONLY** — it performs NO
operand transport and NO coalescing; every GEMM still passes through the
single-tenant path unchanged. The Memory #11 false-negative guard (never group
distinct/unwarmed fingerprints) is verified deterministically (19/19) AND live
(M2/M3 zero cross-fp grouping). Transport is the sequenced W.4b CP 5.6 re-port,
gated by this layer.

## A — Coalesce-group partitioning + cohort read
- **`cipher_rt_pool_partition()`** (`cipher_rt_pool.c`) — PURE decision function:
  adds every cohort peer whose `model_fingerprint == my_fingerprint` (both
  nonzero) to the group; counts the rest as `distinct_rejected`. `group_id ==
  fingerprint` (every same-model tenant computes the same id). Eligible iff
  `n_in_group >= 2`. `my_fingerprint == 0` (unwarmed) ⇒ solo.
- **Cohort read** — `cipher_rt_pool_update()` consumes the W.6sC snapshot from
  `cipher_rt_coresidence_update(model_fp, peers, ...)` in `classify_internal`
  (`src/cipher_workload_detect.cpp`), ~1/sec cadence (no extra ioctl). Publishes
  the group via a double-buffer (classifier writes, dispatch reads lock-free).
- **Dispatch hook** — `cipher_rt_pool_observe_gemm(k, n, Atype)` at the cuBLAS
  boundary (`cipher_rt_cublas_shim.c`, after `observe_gemm`) records the
  shared-weight `(K=k, N=n, dtype)` the group would coalesce at (for W.4b) and
  **always returns 0 = passthrough**. Reads the published atomic snapshot; no ioctl.

**PID-namespace finding (caught by the gate):** the kmod records `current->tgid`
(host PID namespace); the substrate's `getpid()` inside a container is the
*container* PID. An early version matched self by `getpid()`, never fired, and
injected a phantom self ⇒ single-tenant `group_size=2`. **Fixed**: identity is
by **fingerprint**, namespace-agnostic and exactly what coalescing needs (same
weights); self is already in the cohort snapshot (kmod heartbeat), so
`n_in_group` is simply the same-fingerprint count. Post-fix single-tenant
`group_size=1`. W.4b will need self's *host* tgid for coordination (debt §F).

## B — Per-tenant correctness backstop
- **`cipher_rt_pool_correctness_check(n, outputs, refs, len, tol, pass[])`** —
  max element-wise relative diff per tenant; passes iff `<= tol` (0.01). Returns
  the count of failing tenants. This is the check W.4b runs first-coalesce per
  group to catch any fingerprint-collision edge before scattering.
- **Deterministic validation** (`test_pool_eligibility.c`): same-fp coalesced
  outputs PASS (fails=0); a forced-divergent tenant is FLAGGED (fails=1,
  per-tenant pass[1]=0). The backstop works before any real transport exists.

**Enforcement scope (honest):** the blocking *table* (`cipher_rt_pool_mark_blocked`
/ `cipher_rt_pool_is_blocked`) is present, unit-testable, and wired into
`observe_gemm` (a blocked group won't record a coalesce shape). But the
*invocation site* — calling `mark_blocked` when the correctness check fails on
real coalesced+scattered outputs — lands in **W.4b** (W.4a has no transport, so
there are no real outputs to check; `mark_blocked` is never called here). The
spec's ratio-based auto-disable (>25% per-tenant / >50% session-wide) is **NOT
implemented in W.4a** — it needs real block events to count, so it is deferred
to W.4b (§F debt #1). W.4a delivers the per-group block primitive + the check;
W.4b wires the failure→block→disable loop.

## C — Memory #11 false-negative structural guard
The catastrophic case is coalescing different-model tenants → silent corruption.
W.4a prevents it structurally (partition only groups bit-identical nonzero
fingerprints) and is verified two ways:
- **Deterministic** (`test_pool_eligibility` 19/19): M2 distinct-fp → never
  grouped, symmetric; M3 mixed → only same-fp grouped; unwarmed self/peer →
  never grouped; backstop flags forced divergence.
- **Live** (real concurrent vLLM, `run_pool_multitenant_w4a.sh`): **M2** —
  TinyLlama-fp16 + TinyLlama-AWQ (same arch/dims, different quantization ⇒
  distinct fp): `eligible_groups=0`, `group_size=1` each, `distinct_fp_rejected`
  grows to 7528/28022 (continuous rejection). **Zero cross-fp grouping.**

## D — Transition handling
Eligibility is recomputed each classify (~1/sec) from the live cohort, so all
transitions are handled statelessly:
- **Join** — a new same-fp co-resident tenant raises the cohort count → next
  partition includes it (M1: group grew to 4 as tenants warmed).
- **Leave** — a tenant pruned from the cohort (W.6sC 30 s liveness) drops from
  the next partition.
- **Fingerprint change** — `cipher_rt_pool_partition` keys on the current fp, so
  a tenant that swaps models moves coalesce-group; the old group is unaffected
  (group_id == fingerprint). The transient warmup fp-shift (W.6sB:
  `model_fp` not stable until gemm_total≥64) manifests as a small, plateauing
  `distinct_fp_rejected` (single-tenant ≤2; vs M2's tens-of-thousands for a real
  distinct model) — the conservative warmup-gate (unwarmed ⇒ solo) holds.

## E — Eligibility engagement table

| Cell | co_resident | eligible_groups | group_size | distinct_fp_rejected | Expected | Result |
|------|-------------|-----------------|------------|----------------------|----------|--------|
| P1/P2/P3 single-tenant (gate) | 1 | 0 | 1 | ≤2 (transient) | solo | **PASS** |
| M1 4× TinyLlama same fp | 4 | thousands | 4 (3/4 tenants; tiny1 max 3) | 5-7 transient | group of 4 | **PASS** |
| M2 fp16 + AWQ distinct fp | 2 | **0** | **1** each | 7528 / 28022 (grows) | NO coalesce | **PASS (guard)** |
| M3 2×fp16 + 1×AWQ | 3 | thousands(fp16)/0(awq) | 2 (fp16) / 1 (awq) | large | fp16 pair + awq solo | **PASS** |
| N2 single Llama-3-8B B=8 | 1 | 0 | 1 | ≤2 | solo (=W.6sC P1) | **PASS** |

Counters in the CLASSIFY-POOL log: `eligible_groups`, `solo`,
`distinct_fp_rejected` (the guard-firing count), `group_size`, `blocked`.
9-cell close gate **9/9 PASS, NO SEGFAULTS**, all prior actuators preserved
(P1/P2 A4, P3 A3 int4, T1 A3, C1 A4). Single-tenant passthrough byte-identical
to W.6sC (decision-only, no transport).

*M1 tiny1 max_group_size=3 (not 4)*: tiny1 produced only ~16 CLASSIFY-POOL
samples (a short observation window — it ran far fewer classify passes than its
peers; tiny2 logged 6789), and within that window its max simultaneous-warm
overlap was 3. The 4-tenant eligible group demonstrably forms — tiny2/3/4
(thousands of samples each) observed `group_size=4`. Per-tenant max varies with
launch stagger + warmup; the partition decision is correct.

**Counter-semantics note:** `solo` is incremented whenever a published group is
`eligible==0`, which lumps two cases: (a) "queried, alone" (n_in_group=1) and
(b) the startup ticks before the first cohort query returns (empty snapshot,
n_in_group=0 — the 24 `group_size=0` ticks per single-tenant cell). Both are
correctly non-eligible; the distinction (haven't-queried vs alone) is
observability-only and does not affect the guard.

## F — Engineering debt forecast
1. **W.4b transport** (the next substep): re-port the CP 5.6 cross-process
   static coordinator (socket-IPC + decode-step barrier + cuIpc operand sharing),
   GATED by this eligibility layer + correctness backstop. W.4b also wires the
   **failure→block→disable loop**: call `cipher_rt_pool_mark_blocked` on a
   first-coalesce correctness failure, and implement the ratio-based auto-disable
   (>25% groups blocked ⇒ per-tenant disable; >50% ⇒ session-wide) — these need
   real block events to count and so are deferred from W.4a. Per discipline (j),
   verify at W.4b entry that the decode-step barrier stays substrate-side
   (cross-process libcipher_rt IPC + dispatch-boundary barrier), NOT a vLLM
   scheduler hook.
2. **Self host-tgid for W.4b coordination** — getpid() is the container PID; W.4b
   needs self's host tgid to know which snapshot entry is self (e.g., kmod QUERY
   returns caller tgid, or read /proc/self/status NSpid). v1.x / W.4b-entry.
3. **Same-container multi-process** — main + EngineCore are 2 host tgids; here
   main does no GEMMs so it never registers (1 registrant/container = 1/tenant,
   verified). If a future deployment has 2 GEMM-active processes per container,
   they'd co-group (harmless — same model) but inflate density counts; W.4b may
   want tenant (cgroup) identity vs process identity.
4. **Dense-8B co-residence** — 2× 8B OOMs vLLM's KV profiling at util=0.3
   (per-tenant memory partitioning is W.4b territory); M1/M2/M3 use light
   TinyLlama tenants (cohort/eligibility is model-agnostic).
5. **100-tenant scale** — partition is O(peers); CIPHER_POOL_MAX=128. Load-check
   at the Goal-1 target under V.1 / CP 5.5.
6. **distinct_fp_rejected conflation** — counts both transient warmup fp-lag
   (self's stale snapshot entry) and real distinct-model peers; the stable-state
   value (plateau vs continuous growth) distinguishes them. Could refine by
   identifying self once host-tgid is available (debt #2).

## G — Anchors + Goal 1 status

| Component | Pre-W.4a | Post-W.4a |
|-----------|----------|-----------|
| `cipher_rt_phase4` HEAD | 8f32441 (w6-subC-coresidence) | (commit) tag `w4a-pool-eligibility` |
| `cipher_rt_phase4` libcipher_rt.so md5 | 73fac17f | **9fe23143b12e67355e7acc0b606d0c25** |
| `cipher_kmod` HEAD | 02fc2d1 (0.6.6) | 02fc2d1 (**UNCHANGED** — reads NR 31) |
| `cipher-fusion-evidence` HEAD | ac61365 (w6-subC-soak) | (commit) tag `w4a-pool-close` |
| `cipher-platform` rev8 `/usr/lib/cipher` | 1f305ce6 | 1f305ce6 (**UNCHANGED**) |

**Goal 1 status:** the SAFE heterogeneous partition is delivered (batch only
within same-model groups, never across — verified live, M2/M3). This is the
unbreakable load-bearing wall. The DENSITY lever (3-6× tok/W by tenant density)
is the sequenced **W.4b** transport (CP 5.6 re-port gated by this layer). Goals
2/3-actuators/4/5 stand from W.1/W.2/W.3/W.5/W.6/rev8. **Next: W.4b**, then W.7
NCCL (Memory #23).
