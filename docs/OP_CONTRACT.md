# CIPHER Op Contract — Regression-Safety Invariants

**Status:** binding for all ops added after 2026-04-15 (post Op 26 LOOP).
**Purpose:** 10 ops are green (7.76× peak, 694 TFLOPS, max_diff=0.000000). That
result is load-bearing. Every new op must prove it cannot regress it — not by
argument, but by structural conformance to the six invariants below.

Conformance = safe to merge behind its default-OFF env flag.
Violation of any invariant = op must live in a separate DSO or a Stage 3
worker thread physically isolated from Stage 0/1 memory.

---

## The Six Invariants

### I1. Default OFF
- Op is disabled unless its `CIPHER_<NAME>` env var is `on` / `1` / `ON`.
- Disabled path is exactly: one `std::atomic<int>::load(memory_order_relaxed)`
  plus a branch. ≤3 cycles on x86 TSO / ARM equivalent.
- No other work — no syscalls, no allocations, no CUDA, no logging — on the
  disabled path.

### I2. Pure Stage 1 observer
- Op reads `CipherRingEntry` via its `*_observe(const CipherRingEntry*)` hook.
- Op writes only to its own private `g_<name>_*` state (per-session slot array,
  counters, atomics).
- Op does not write to the ring, the look-aside buffer, SENSE state, SHIELD
  state, or any other op's state.
- Reads of cross-op state (e.g. CONTINUITY may read SENSE classification) are
  OK — only writes are forbidden.

### I3. No critical-path writes to shared state
- Any Stage 2 ARBITRATE hints are writes to a *hint table* that ARBITRATE
  polls. In v1, ARBITRATE logs only — it does not act.
- Real actuation (SM rebalance, frequency steering, power gating, KV capture,
  etc.) requires an explicit v2 plan with separate user approval.

### I4. No allocations after init
- All per-op memory is fixed-size arrays, allocated at `_init()` time
  (typically `MAX_SESSIONS=1024` slots).
- `observe()` path does zero `malloc`, `new`, `realloc`, `mmap`, or
  container growth (`std::vector::push_back` etc).
- If the op needs unbounded storage, that storage lives behind a ring buffer
  or gets flushed by Stage 2 — never allocated on the hot path.

### I5. No CUDA / syscalls on the observe path
- No `cudaMalloc`, `cudaMemcpy`, `cuLaunchKernel`, `nvmlDeviceGet*`, no
  `open`, no `read`, no `write`, no `clock_gettime` — except:
  - `clock_gettime(CLOCK_MONOTONIC_RAW, ...)` is permitted (SENSE uses it).
    ~30 ns; measured cost, acceptable on Stage 1.
- Reports (`_report()`) may call `fopen`/`fprintf`/`fclose` — but `_report()`
  runs off the hot path (called by the workload at end-of-run, or by Stage 2).
- All CUDA/NVML access belongs in Stage 2 background threads or in dedicated
  probe utilities called off the hot path.

### I6. Acceptance gate = FULL regression
- New op's own test (`test_<name>.py`) must pass.
- Then **all prior op tests must pass with the new op ON**: run each prior
  gate with `CIPHER_<NEW_NAME>=on` exported. Zero tolerance for flake.
- If any prior gate fails, the op ships default-OFF with its test green but
  is not considered "integrated" — it is quarantined until root-cause is
  resolved. No regression ever masked.

---

## Conformance check (mechanical)

For a new op, grep its source file for these forbidden patterns:
```
cudaMalloc | cudaMemcpy | cudaMemset       # I5
malloc\(   | new        | realloc          # I4 (on observe path)
nvmlDevice                                 # I5
g_cipher_sense | g_cipher_shield | g_...   # I2 (writes to other ops)
```
And confirm:
- `*_init()` reads env, returns early when unset.
- `*_observe()` first line is the `g_<name>_enabled.load(relaxed)` branch.
- Test file runs 3+ workloads via subprocess + LD_PRELOAD + JSON report.
- Regression run: all prior `test_*.py` pass with `CIPHER_<NEW>=on`.

If all pass → conformant, safe to merge.
If any fails → op does not ship in-process. Isolate per §"Isolation" below.

---

## Isolation (when an op cannot conform)

Some ops genuinely need to write (KV snapshots, signed receipts, trace
export). Those do not belong on Stage 1. Two isolation tiers:

**Tier A — Stage 3 worker thread.** Separate pthread, reads ring buffer
read-only, owns its own memory arena. May call CUDA/NVML. Cannot write to
Stage 0/1 state. Feeds its own JSON report via the workload-driven
`_report()` call or a periodic flush.

**Tier B — Separate DSO.** Compiled independently, loaded via a second
`LD_PRELOAD` slot or dlopen'd by Stage 2. Crash in Tier B does not crash
`libcipher_rt.so`. This is the correct home for anything doing heavy host
I/O, network calls, or cryptographic signing.

---

## Remaining 12 ops — v1 observer scopes

All v1 scopes below conform to I1–I6 and ship in-process.
v2 items are deferred, require separate approval, and (where noted) belong
in Tier A/B isolation.

### Phase 3 — NCCL (remaining 1)

**NCCL P2P CPU Proxy** — DEFERRED outright (needs libibverbs + multi-node;
cannot build on single-pod). No v1 scope. Will be planned fresh in a
multi-node session.

### Phase 4 — Agentic AI (remaining 2)

**Op 19 CONTINUITY** — incremental KV state checkpoint tracking.
- **v1 (observer):** per-session bookkeeping of which (layer, head, page) KV
  regions *would* be checkpointed based on observed GEMM shapes and attention
  kernel launches. Emit a "snapshot manifest" JSON. No memcpy, no pinned
  memory, no CUDA. Reads SENSE classification to gate manifest generation to
  AGENT_AUTONOMOUS sessions.
- **v2 (Tier A):** actual KV page capture to host pinned arena in a Stage 3
  worker. Bounded ring buffer with eviction. Requires approval + own plan.

**Op 27 PIPELINE** — multi-agent session correlation.
- **v1 (observer):** cluster session fingerprints (from SENSE) by shape-ring
  Jaccard similarity; emit a pipeline-graph JSON naming upstream/downstream
  session pairs. Pure reads of SENSE state. No action.
- **v2:** cross-session priority inheritance (when upstream is ARBITRATE-
  protected, downstream gets same band). Defer to ARBITRATE v2.

### Phase 5 — Compliance & Observability (9 ops)

**Op 17 PREDICT** — proactive L2 preloading.
- **v1 (observer):** track which shapes would benefit from L2 persist based
  on dispatch history. Emit a preload-candidate JSON. No `cudaStreamSet`
  calls.
- **v2 (Tier A):** actually apply `cudaAccessPolicyWindow` in Stage 3.

**Op 18 RECEIPT** — per-session signed proof of compute.
- **v1 (observer):** accumulate per-session FLOP counts + dispatch hash
  chain in private state. Emit unsigned receipt JSON at `_report()`.
- **v2 (Tier B):** HMAC/Ed25519 signing belongs in a separate DSO — crypto
  calls and key material must not live in `libcipher_rt.so`. Ship as
  `libcipher_receipt.so` dlopen'd by Stage 2.

**Op 16 GUARD** — KV cache privacy enforcement.
- **v1 (observer):** score per-session KV sharing risk (shape overlap
  between sessions within a time window). Emit a risk-report JSON.
- **v2 (Tier A):** actual KV region isolation / scrubbing in Stage 3.

**Op 21 DETERMINISM** — reproducible dispatch sequence.
- **v1 (observer):** per-run dispatch-sequence hash chain. Emit a
  fingerprint JSON. Compare across runs offline.
- **v2 (Tier A):** in-run divergence alerts + forced replay ordering —
  real ordering control goes in Stage 3 / ARBITRATE v2.

**Op 23 CARBON** — per-session carbon certificate.
- **v1 (observer):** accumulate per-session joules (from NVML snapshots
  already collected by telemetry, read-only). Multiply by static
  grid-intensity constant. Emit JSON.
- **v2 (Tier B):** live grid-intensity API fetch + signed certificate →
  separate DSO for the same reason as RECEIPT.

**Op 24 FAIRNESS** — kernel-level tenant FLOP quota.
- **v1 (observer):** per-tenant FLOP accumulation (tenant = session
  fingerprint from SENSE). Emit fairness-violation flags in JSON.
- **v2:** actual throttling via ARBITRATE band demotion — defer to ARBITRATE
  v2 plan.

**Op 25 TOPOLOGY** — NVLink topology inference.
- **v1 (observer):** infer NVLink peer graph from observed `ncclAllReduce`
  latencies (already captured). One-shot at init + refresh at `_report()`.
- **v2:** topology-aware algorithm hints into NCCL tuner — ABI already
  exists, so v2 is small; still separate approval.

**Op 28 TRACE** — kernel-level execution trace export.
- **v1 (observer):** ring-buffered trace slice (last N events) flushable via
  `_report()`. Bounded, no allocations on hot path.
- **v2 (Tier A):** continuous streaming export to disk/network — Stage 3
  worker owns the flush path.

**Op 29 COMPLY** — regulatory compliance artifact generation.
- **v1 (observer):** aggregate RECEIPT + CARBON + FAIRNESS + DETERMINISM
  reports into a compliance bundle JSON at `_report()`. Pure reads of sibling
  op state (read-only crossing of I2 is permitted).
- **v2 (Tier B):** signed compliance artifact + external attestation —
  same separate-DSO pattern as RECEIPT.

---

## Precedent: this is how the 10 green ops shipped

- SHIELD v1 wrote band hints to a table ARBITRATE logs-only. SM rebalance is
  v2, explicitly deferred.
- VOLT + HIBERNATE v1 detect and log; actuation is probe-gated and SKIPs on
  the current pod. That's conformant — no action taken in-process.
- PULSE v1 scores hardware faults, cap 2. Signal 2 (sentinel compare)
  deferred — exactly because it would have violated I3.
- LOOP v1 (today) writes a runaway score + hint. Real throttle is v2.

The pattern is: **all detection in v1; all actuation in v2 with isolation if
required.** That is what kept the 10-op stack regression-free across every
build.

---

## Rule of thumb

> If you're unsure whether an op conforms, read its `observe()` top to
> bottom. If every line is an atomic load, an arithmetic op, an array
> index, or a counter increment — it conforms. If any line is a malloc,
> a CUDA call, a syscall, or a write to someone else's global state —
> it doesn't, and it belongs in Tier A or B.

---

## Lessons

- **Always verify wiring survived pod migration.** CLAUDE.md claims are only
  valid if confirmed by grep on the current pod. 2026-04-15: state docs
  claimed Op 26 LOOP was wired and passing regression; grep of
  `cipher_10ops_impl.cpp` showed zero references to `cipher_loop_init` /
  `cipher_loop_observe`. The symbols existed in `libcipher_rt.so` but
  `g_enabled` stayed 0, silently making `cipher_loop_report()` a no-op.
  Repaired alongside Op 19 CONTINUITY wiring.
