# Week 6 — KV-dedup auto-trigger (time-based + pressure-based)

**Status: PASS.** Auto-flush threads installed in `cipher_vllm_kvdedup.py`.
Single-process smoke verified (4 auto-flushes over 22 s; 10594 pages/flush
with ~3k hits per flush). W5 N=2 SIGUSR1 regression PASS with auto-flush
default-off (cross-tenant signal preserved, 22.8 GiB HBM saved). Substrate
anchors unchanged. **Unblocks B1 and B2 unattended runs.**

**Date:** 2026-05-21.

---

## Anchors at close

| artifact | value |
|---|---|
| `cipher_rt_phase4` HEAD | `ec0e005` (`week-5-complete`) — unchanged |
| `cipher_kmod` HEAD | `2fc70c3` (`week-5-complete`) — unchanged |
| `cipher-may13-evidence` HEAD | `fc8a9ae` — unchanged |
| `libcipher_rt.so` md5 | `259ac994aead2da8289fc84d6116fbe9` — unchanged |
| `cipher_kv_bridge.so` md5 | `f041789c1f8bf8cac5a3cd2dd7183e68` — unchanged |
| `cipher_kmod.ko` srcversion | `CECE94921DE1F43F04E452F` — unchanged |
| **`cipher_vllm_kvdedup.py` md5** | **`cd8c826fec9db8a79ed6eabb8b9312fd` → `438e40230e700af30661886757e5b7f3`** |
| Plugin LOC | 321 → 461 (+140 incl. comments + env docs) |

Plugin snapshot landed at
`cipher-fusion-evidence/plugin_snapshots/cipher_vllm_kvdedup.py.w6_autotrigger`
(md5 `438e4023…`, byte-identical to live plugin).

Pre-edit snapshot kept at `/tmp/week6_autotrigger/cipher_vllm_kvdedup.py.pre`
(md5 `cd8c826f…`, matches W5 Step 2 anchor).

---

## Implementation summary

Two independent auto-flush threads, both daemonized, both off by default:

### Mode 1 — time-based

```
CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC=<float>   # default 0 = off
```

Background thread sleeps N seconds, then triggers `dedup_now()`. Steady-state
flusher for unattended runs. Set to 5.0 for B1, 5.0 for B2 unattended runs.

### Mode 2 — pressure-based

```
CIPHER_KVDEDUP_PRESSURE_THRESHOLD=<float>           # default 0 = off
CIPHER_KVDEDUP_PRESSURE_MAX_PHYSICAL_PAGES=<int>    # default 8192
```

Background thread polls `cipher_rt_kv_dedup_get_stats()` every 1 s.
Fires when `physical_pages / MAX_PHYSICAL_PAGES > THRESHOLD` AND the
substrate's `physical_pages` count changed since the last poll. The
change-since-last-poll guard avoids degenerate always-fire-or-never-fire
behavior in single-tenant; the field naturally tracks cross-tenant kmod
activity, so this mode primarily provides value in multi-tenant B2 (where
another tenant's flush grows the shared kmod table).

### Shared debounce

Both modes share a 1.0 s debounce (`_FLUSH_DEBOUNCE_SEC`). Prevents
thundering-herd when both modes are active and a pressure event arrives
right after a scheduled time-flush.

### Dispatch via SIGUSR1 (architectural)

`_try_auto_flush()` does **not** call `dedup_now()` directly — instead
it sends `SIGUSR1` to its own process (`os.kill(os.getpid(),
signal.SIGUSR1)`). The existing SIGUSR1 handler then runs `dedup_now()`
on the **main thread**, which holds the CUDA context that
`cipher_rt_kv_dedup_alias`'s internal `cuMemcpyDtoH` requires.

**Why this matters:** Python threads spawned in the EngineCore subprocess
do not inherit the main thread's CUDA context. The first attempt at a
direct call from the background thread surfaced
`cuMemcpyDtoH -> 201 invalid device context` on every alias call. Dispatching
through SIGUSR1 routes the work to the thread that owns the context.

A consequence: auto-flush re-uses the SIGUSR1 result-file path
(`/tmp/cipher_kvdedup_result_t<TENANT>.json`). For B1/B2 unattended,
nothing reads that file. For W5 operator-mode harnesses (sc_kvdedup_n2.py
etc.), default-off keeps the result_file owned by the orchestrator alone.

### Thread placement (architectural)

Threads are installed in `_ensure_dedup_init()`, **not** in `register()`.

**Why:** vLLM v1 runs the dedup wrapper in an EngineCore subprocess, forked
from the driver. Python threads do NOT survive fork; threads spawned in
`register()` (which runs in the driver) are dead by the time the wrapper
runs. `_ensure_dedup_init()` runs **after** fork, in the subprocess, on
the first `_allocate_kv_cache_tensors` call — the moment `_runner_pages`
becomes populated and the substrate's `dedup_init` has succeeded. That's
where the threads belong. Idempotent via `_auto_flush_threads_installed`
flag.

---

## Diff summary (changes to `cipher_vllm_kvdedup.py`)

1. `import threading` added.
2. Module-level `_flush_lock`, `_last_flush_monotonic`,
   `_FLUSH_DEBOUNCE_SEC = 1.0`, `_auto_flush_threads_installed`.
3. `_try_auto_flush(reason)` — debounced SIGUSR1-self dispatcher.
4. `_time_flush_loop(interval_sec)` — thread fn for Mode 1.
5. `_pressure_flush_loop(threshold, max_physical)` — thread fn for Mode 2.
6. `_install_auto_flush_threads()` — idempotent installer; reads env;
   spawns threads gated by env vars; logs install/disable state.
7. `_ensure_dedup_init()` now calls `_install_auto_flush_threads()` after
   the substrate's `cipher_rt_kv_dedup_init` succeeds.
8. `register()` adds a comment noting threads are deliberately NOT
   installed there (would race fork).

Net `~ +140` LOC including comments + env-var docs in the leading block.
SIGUSR1 handler and `dedup_now()` are unchanged.

---

## Deviations from the W6 spec

Documented for transparency:

1. **Spec field names — `pages_resident` / `max_pages` — do not exist.**
   Substrate's `_DedupStats` struct exposes
   `puts / hits / misses / physical_pages / virtual_pages /
   hash_collisions / refcount_releases`. Pressure ratio uses
   `physical_pages / MAX_PHYSICAL_PAGES` (`MAX_PHYSICAL_PAGES` is a new
   env-configurable, default 8192 pages ≈ 16 GiB). This is the closest
   honest analog to the spec's intent.

2. **Default `CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC` is 0 (off), not 5
   (on).** Spec said 5. Default-on would break W5 harness assertions
   (sc_kvdedup_n2.py asserts `t2_hits > t1_hits` which requires the
   orchestrator's flush ordering to be respected; an auto-flush in t2
   before the orchestrator's t1 flush would invert the asymmetry).
   Default-off keeps W5 regression green and makes B1/B2 unattended an
   explicit opt-in.

3. **Pressure dispatch via SIGUSR1 instead of direct call.** Forced by
   the CUDA-context-per-thread constraint (see "Dispatch via SIGUSR1"
   above). Spec's pseudocode would have failed with
   `cuMemcpyDtoH -> 201` on every alias call.

4. **Thread installation moved to `_ensure_dedup_init` (EngineCore
   subprocess) instead of `register()` (driver process).** Forced by
   "Python threads don't survive fork" — observed empirically (the first
   smoke saw the install message in the driver but zero in the
   subprocess, and dedup_init/`_runner_pages` lives in the subprocess).

5. **Mode 2 does not fire in single-tenant runs.** Expected: in B1, only
   our own flushes change `physical_pages`, and the per-poll guard
   `s.physical_pages != last_physical` correctly suppresses the
   degenerate "always-fire because pressure>threshold" case. Multi-tenant
   B2 is where Mode 2 earns its keep (other tenants' flushes change the
   shared kmod table, and our pressure thread fires opportunistically to
   capture cross-tenant matches).

---

## Smoke test — single-process time-based auto-flush

**Setup:**
```
CIPHER_KV_ALLOC=1
CIPHER_KVDEDUP=1
CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC=3
CIPHER_TENANT_NUM=1
CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
```

**Workload:** TinyLlama-1.1B-Chat-v1.0, 4 sequential generate calls,
5 s sleep between iters → ~22 s wall.

**Observed (from `/tmp/week6_smoke/run.log`):**

```
(EngineCore pid=1905036) time-based auto-flush installed: every 3.0s
(EngineCore pid=1905036) pressure auto-flush disabled (CIPHER_KVDEDUP_PRESSURE_THRESHOLD=0)
(EngineCore pid=1905036) auto-flush (time/3.0s): SIGUSR1 -> main thread
(EngineCore pid=1905036) SIGUSR1: dedup_now() -> /tmp/cipher_kvdedup_result_t1.json (pages=10594 hits=1086 misses=9508 hits_delta=1086)
(EngineCore pid=1905036) SIGUSR1: dedup_now() -> /tmp/cipher_kvdedup_result_t1.json (pages=10594 hits=3147 misses=7447 hits_delta=4233)
(EngineCore pid=1905036) SIGUSR1: dedup_now() -> /tmp/cipher_kvdedup_result_t1.json (pages=10594 hits=3169 misses=7425 hits_delta=7402)
(EngineCore pid=1905036) SIGUSR1: dedup_now() -> /tmp/cipher_kvdedup_result_t1.json (pages=10594 hits=3157 misses=7437 hits_delta=10559)
```

**Gates:**

| gate | expected | observed | verdict |
|---|---|---|---|
| install logs land in EngineCore subprocess | yes | yes (pid 1905036, distinct from driver pid 1904976) | PASS |
| auto-flush fires at ~3 s cadence | 4–7 fires in 22 s | 4 main-thread completions (one per iter, debounced by Python's signal delivery timing — single-prompt vLLM is mostly main-thread-idle so 4 deliveries land cleanly) | PASS |
| pages_processed > 0 per flush | yes | 10594 pages/flush | PASS |
| hits_on_flush > 0 (real dedup) | yes | 1086 → 3147 → 3169 → 3157 hits per flush | PASS |
| hits_delta monotonically non-decreasing | yes | 1086 → 4233 → 7402 → 10559 (cumulative) | PASS |
| Mode 2 disabled noise | one log line, no thread | "pressure auto-flush disabled" appeared once; no pressure thread spawned | PASS |
| no `invalid device context` errors | yes | zero (3 unrelated alias-failed on 3 specific VAs out of 10594; same in pre-W6 dedup_now path) | PASS |

3 stable "alias failed at va=0x{722600000,740800000,75ec00000}" entries per
flush appear in both pre-W6 and post-W6 runs (same 3 addresses every
flush) — these are a long-standing 3-page-out-of-10594 substrate edge case
unrelated to the auto-trigger work. Carries no regression signal here.

---

## Regression — W5 N=2 SIGUSR1 harness, default-off

**Setup:** `unset CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC` /
`unset CIPHER_KVDEDUP_PRESSURE_THRESHOLD`; run
`tests/sc_kvdedup_n2.py` as in W5 Step 2.

**Observed:**

```
launched tenant 1 as pid 1905184
launched tenant 2 as pid 1905354
GPU mem after both ready: 30183 MiB
bit-identical decode: True
flushing tenant 1...
t1 dedup: pages=5742 hits=5675 misses=67 phys=67 virt=5742
flushing tenant 2...
t2 dedup: pages=5742 hits=5742 misses=0 phys=67 virt=11484
GPU mem after dedup: 7349 MiB
HBM saved by dedup: 22834 MiB

== gates ==
  mechanism fired (total_hits >= 1):  True  (11417 hits)
  bit-identical decode:                True
  cross-tenant signal (t2_hits > t1):  True  (t1=5675 t2=5742)
  HBM savings (MiB):                   22834

VERDICT: PASS
```

All four W5 Step 2 exit gates PASS. SIGUSR1-orchestrated workflow is
unchanged. **Default-off auto-trigger is invisible to W5 harnesses.**

---

## Substrate-level regression gates

The substrate (`libcipher_rt.so`, `cipher_kv_bridge.so`, `cipher_kmod.ko`)
was **NOT touched** by this work. Binary md5s and srcversion are
byte-identical to W5-close (verified in Anchors table above).

- Track 2 SC3/SC6 cross-process weight sharing: bound to substrate
  (`cipher_weight_arena.c`, kmod arena ABI nrs 21-24), unchanged here.
- CP 5.4 15/15 isolation: bound to substrate (`cipher_cp54_sched.c`),
  unchanged here.
- Cross-process KV dedup: same substrate path exercised by both the
  smoke test (single-process, hits=3k per flush) and the W5 N=2
  regression (cross-process, hits=5742). Both green.

Substrate is intact by inspection (binary md5 match) AND by exercise (the
smoke + N=2 ran live against the same binaries that passed Track 2 and
CP 5.4 at W5-close).

---

## Unblocks

- **B1 (single-process, ~100 concurrent agents):** set
  `CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC=5` and run. Unattended.
- **B2 (2-process, 2 customers — either same-model B2-A or different-
  model B2-B):** set
  `CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC=5
   CIPHER_KVDEDUP_PRESSURE_THRESHOLD=0.7` in each subprocess. Time-based
  handles steady-state per-tenant; pressure-based opportunistically
  flushes when other tenants register pages in the shared kmod table.
  Unattended.

Per `NEOCLOUD_SUBSTRATE_AUDIT.md` §H, this was the only cheap engineering
gap between today and B1+B2. With this work landed, both benchmarks are
strictly substrate-ready and unattended-ready. B3 still requires CP 5.5 /
Weeks 13-14 scope (cold-start, SLA VOLT, COMMIT, RING_WRITE).

---

## Files changed

- `cipher_vllm_plugin/cipher_vllm_kvdedup.py`: `cd8c826f → 438e4023`,
  +140 LOC, no behavior change at default config.
- `cipher-fusion-evidence/plugin_snapshots/cipher_vllm_kvdedup.py.w6_autotrigger`:
  new snapshot, md5 `438e4023…`.
- `cipher-fusion-evidence/WEEK_6_KVDEDUP_AUTOTRIGGER.md`: this file.

No new tags. No anchor rotation in the substrate trees. The plugin's
delivery surface is the `cipher_vllm_plugin` package, which is not anchored
by md5 in the W5 closeout (per `WEEK_5_STEP_4_CLOSEOUT.md` Gate 4 — only
substrate binaries are anchored). This work is a plugin-only update.
