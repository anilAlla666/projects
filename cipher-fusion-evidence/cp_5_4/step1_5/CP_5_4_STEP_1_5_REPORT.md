# CP 5.4 — Step 1.5 (green-context churn cost) — REPORT

**Date:** 2026-05-19. **Status: COMPLETE — all of Phase 1.5B (1.5B-1…1.5B-4)
executed; no stop-condition triggered.** Green-context churn is **cheap**
(per-event pool-resize stall ~1.7 ms) and the Marlin cubin is **not** recompiled
or reloaded on green-context churn (4-assert confirmation PASS). Anchors
unchanged. Awaiting Step 1.5 close adjudication.

---

## Adjudication inputs (recap)

Step 1.4 closed (user "continue"). Step 1.5 §7 adjudicated: two-tier microbench
ADOPTED; sub-40-SM floor probe DEFERRED to a Step 1.4 addendum; Marlin question
= a 4-assert confirmation test. Phase 1.5B authorized with sub-phases 1.5B-1…4.

## Phase 1.5B-1 — two-tier microbench setup

Three throwaway harnesses built and sanity-checked (all in
`cipher-fusion-evidence/cp_5_4/step1_5/`):

- `cp54_s15_bare.py` — Tier (i) bare-API floor: `torch.cuda.GreenContext.create(
  K·8)` create/teardown in a loop, no kmod ioctl, no self-verify.
- `cp54_s15_resize.py` + `cp54_s15_partition_driver.py` — Tier (ii) real
  pool-resize path: a POOL (`cp54_pool.PoolBinding`) plus a PARTITION
  subprocess arriving/departing, driving the production
  `check_resize() → build_green_ctx()` rebuild.
- `cp54_s15_marlin.py` + `cumod_count.c`/`libcumod_count.so` — the Marlin
  4-assert confirmation (1.5B-4).

Sanity: bare harness ran (K=1,15); resize harness ran one cycle
(15→13→15 groups, self-verify PASS each).

## Phase 1.5B-2 — bare-API floor

`cp54_s15_bare.py --iters 100 --groups 1,3,5,7,9,11,13,15` →
`bare_floor.json`. A primary context is established before timing so the
first create is not charged with primary-context bring-up.

| K (groups) | SMs | mean ms | p50 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| 1 | 8 | 1.065 | 1.039 | 1.570 | 1.590 |
| 3 | 24 | 1.074 | 1.040 | 1.563 | 1.583 |
| 5 | 40 | 1.054 | 1.021 | 1.570 | 1.704 |
| 7 | 56 | 1.051 | 1.020 | 1.542 | 1.551 |
| 9 | 72 | 1.042 | 1.020 | 1.534 | 1.535 |
| 11 | 88 | 1.054 | 1.023 | 1.543 | 1.549 |
| 13 | 104 | 1.060 | 1.027 | 1.533 | 1.696 |
| 15 | 120 | 1.057 | 1.025 | 1.573 | 1.574 |

**The bare-API create/teardown floor is ~1.05 ms mean (p99 ~1.55 ms), flat
across the entire group-count range** — green-context size does not affect
create cost. This is the lower bound the real path is measured against.

## Phase 1.5B-3 — real pool-resize path

`cp54_s15_resize.py --cycles 10 --part-sm 16` → `resize_path.json`. Each cycle:
a PARTITION subprocess `ALLOCATE`s 16 SM (2 groups) → kmod shrinks the POOL
15→13 groups → the POOL's `check_resize()` rebuilds the green context (timed);
the PARTITION `FREE`s → POOL grows 13→15 → `check_resize()` rebuilds again
(timed). The timed window includes a `torch.cuda.synchronize()` before the
`GreenContext` destroy (memo §5 — drains pending green-stream work so the
teardown is not under-attributed; the Step 1.4C cross-stream hazard class).

The timed `check_resize()` is the **full production path**: `ALLOCATE` ioctl +
`GreenContext` destroy + `GreenContext` create + green-ctx `Stream()` +
`set_context()` + the `%smid` self-verify kernel + sync.

| transition | n | mean ms | p50 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| shrink (partition in, 15→13) | 10 | 1.694 | 1.630 | 2.161 | 2.161 |
| grow (partition out, 13→15) | 10 | 1.562 | 1.553 | 1.610 | 1.610 |

All 20 resizes correct — `all_resized=True`, self-verify PASS at every size.

**The full pool-resize path costs ~1.7 ms** (shrink) / **~1.6 ms** (grow) — only
~0.6 ms over the bare-API floor. That ~0.6 ms is the ioctl + the `%smid`
self-verify kernel launch/sync + the stream creation — exactly the expected
extra work; **no unexpected gap** (the 1.5B stop-condition "real-resize ≫ floor
by an unexpected magnitude" is not triggered — the ratio is 1.6×, fully
accounted).

**Forward-progress stall.** `check_resize()` is called **synchronously** at each
round top in `cipher_batch_executor_gen.py` (Step 1.3b'); it is not an async
path. So the per-event executor stall **is** the measured ~1.7 ms — there is no
separate hidden stall to measure. A partition arrival or departure costs the
batch pool a single ~1.7 ms synchronous stall.

## Phase 1.5B-4 — Marlin cubin / green-ctx-churn 4-assert confirmation

`LD_PRELOAD=./libcumod_count.so python3 cp54_s15_marlin.py --swaps 20` →
`marlin_confirm.json`.

**Instrumentation note (methodology).** A plain `LD_PRELOAD` `cuModuleLoadData`
interposer would **false-negative** here: the Marlin engine resolves the symbol
via `dlsym(libcuda_handle, "cuModuleLoadData")` (`cipher_rt_marlin_engine.cpp`
`resolve_api`), i.e. it calls the pointer it pulled from libcuda directly, not
the preloaded alias. `libcumod_count.so` is therefore a **CUPTI driver-API
subscriber** (`CUPTI_CB_DOMAIN_DRIVER_API`, CBIDs `cuModuleLoad` /
`cuModuleLoadData` / `cuModuleLoadDataEx` / `cuModuleLoadFatBinary`) — CUPTI
traces the driver entry point regardless of how the caller resolved it. The
memo §4 / §7-decision-3 allowed "wrapper **or** CUPTI subscriber"; CUPTI is the
only one without the dlsym blind spot.

**Scope of the counter.** It tracks the four `cuModuleLoad*` driver CBIDs. The
`pre-marlin-warmup` CUPTI count is 0 (`marlin_cupti.log`) — PyTorch 2.11/cu13
loads its own built-in kernels via the newer `cuLibrary*` driver API (different
CBIDs), not `cuModuleLoad*`. Marlin uses `cuModuleLoadData` *explicitly*
(`resolve_api` → `nvrtc_compile_to_module`), confirmed by the
`LOAD #1 cuModuleLoadData` logged between the warmup marks — so the counter
does catch Marlin's load, and "no `cuModuleLoad*` during churn" is the correct
test for Marlin. A hypothetical `cuLibrary*`-based reload would be out of the
counter's scope, but the Marlin engine code does not use that API.

**Test:** warm Marlin (`cipher_rt_marlin_engine_ensure_compiled()` — one NVRTC
compile), then drive **20 production green-context swaps** via
`cp54_pool.PoolBinding.build_green_ctx()` (destroy + recreate, the same path a
real pool resize takes), re-calling `ensure_compiled()` after every swap.

| evidence | value |
|---|---|
| Marlin warmup `ensure_compiled()` | rc=0; ~11.9 s cold NVRTC compile (first run), 111.6 ms warm (NVRTC disk cache) — `marlin_confirm.json` is the warm re-run |
| `cuModuleLoad*` during warmup | **1** (CUPTI `LOAD #1 cuModuleLoadData`) |
| `cuModuleLoad*` total, pre-churn → post-churn | 1 → 1 — **delta 0** |
| post-swap `ensure_compiled()` rc | 0 on all 20 swaps |
| post-swap `ensure_compiled()` latency | ≤ 0.016 ms (typ. ~0.003 ms) |
| green-ctx swap latency | ~1.0 ms, max 1.61 ms |

| assert | result |
|---|---|
| **A1** `ensure_compiled()` one-shot (rc=0, µs latency, no recompile) | **PASS** |
| **A2** module stable / primary-context-bound | **PASS** |
| **A3** no `cuModuleLoad*` during the 20 green-ctx swaps | **PASS** |
| **A4** per-swap latency bounded | **PASS** |

**VERDICT: PASS.** The Marlin cubin is compiled and `cuModuleLoadData`-loaded
**exactly once per process**, at warmup, while only the primary context exists
(no green context had been created yet) — so the module is primary-context-
resident. Across 20 destroy/recreate green-context swaps the `cuModuleLoad*`
count never moves off 1, and `ensure_compiled()` is an idempotent ~3 µs no-op.
This is the code-derived prior (memo §4) confirmed empirically: `g_marlin_state`
is a one-shot `std::atomic`; green-context churn is a different subsystem and
never touches `g_marlin_module`. CP 5.3's partition-awareness (`grid = green_sm`
sizing; green-bound launch when a green ctx is active) re-uses that same
once-loaded module — it does not reload it.

## Analysis (1.5D) — is churn cheap enough to ignore in the arbitration policy?

**Yes — green-context churn is operationally free; the CP 5.4 arbitration policy
needs no resize rate-limiting or amortization.**

- A partition arrival or departure triggers **one** synchronous pool-resize,
  measured at **~1.7 ms**.
- A Phase B N=8 decode round is ~1.86 s wall (Step 1.3b' executor logs). One
  resize is therefore **~0.09 % of a round**. Even a pathological tenant
  arriving *and* leaving every single round (2 resizes/round) is ~0.18 %.
- **Threshold:** for resize churn to breach the campaign's ±3 % noise gate a
  single resize would have to cost **> ~55 ms**; the measured 1.7 ms is **~32×
  under** that line. The arbitration policy can treat a resize as free.
- **Marlin adds zero churn cost** — the cubin is never recompiled or reloaded on
  a green-context swap (1.5B-4). The NVRTC compile (~11.9 s cold / ~0.1 s warm
  cache) is a one-time per-process cost, paid before steady state, unaffected
  by arbitration.

PARTITION/SHARED tenants contribute **no** churn at all — their green context is
create-once-never-refresh (`cipher_rt_green_ctx.c`); the POOL is the only
churning party, and its churn is the ~1.7 ms above.

## Anchors

Unchanged — measurement step. kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`. No substrate source was
modified: all new files are throwaway Step 1.5 measurement harnesses
(`cp54_s15_*.py`, `cumod_count.c`/`libcumod_count.so`). `cp54_pool.py` was
**not** modified (Step 1.4's `CIPHER_POOL_MAX_GROUPS` knob was unused here —
the resize bench drives real partitions, not the cap knob).

## Evidence

`cipher-fusion-evidence/cp_5_4/step1_5/` — harnesses, `bare_floor.json`,
`resize_path.json`, `marlin_confirm.json`, this report. Packaged as
`cp_5_4_step1_5_evidence.tar.gz` (+ `.md5`).

## Adjudication ask

Phase 1.5 is complete: churn cost is measured (~1.7 ms/resize, ~32× under the
noise threshold), the Marlin recompile question is confirmed NO (4/4 asserts
PASS), no stop-condition fired. Recommend **close Step 1.5**. Remaining CP 5.4
steps: 1.6 (mixed-deployment), 1.7 (failure-modes), 1.8 (closure).
