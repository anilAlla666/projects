# CIPHER Logic Audit — Wave 4: Observability Tier (16 observers, 17 TUs)

**Scope.** Deep logic-level audit of the 16 observer ops in
`cipher-may13-evidence/src/`. The op family is the "observability tier"
of the May-13 build: 16 operators that read events off the Stage-1 ring
buffer (or one of two parallel side channels) and emit telemetry,
counters, billing entries, or alerts. Total LOC across the 17 source
files: 3,863 (`wc -l` verified). Companion to Wave 1
(`CIPHER_LOGIC_AUDIT_WAVE_1_CLASSIFIER.md` — classifier brain),
Wave 2 (`CIPHER_LOGIC_AUDIT_WAVE_2_ACTUATORS.md` — userspace substrate
`cipher_rt_phase4` + `libcipher_v2`), and Wave 3
(`CIPHER_LOGIC_AUDIT_WAVE_3_KMOD.md` — `cipher_kmod` 0.4.8).

**Method.** Each file is audited under nine sections: PURPOSE /
PUBLIC SURFACE / CONTROL FLOW / STATE / CONCURRENCY / DEPENDENCIES /
LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED / CONTRIBUTION TO SYSTEM /
FUSION POINTS. Citations are file-relative line numbers. Wave 4 adds
six per-file emphases on top of the nine sections:

1. **Signal** — the trigger condition the observer watches for.
2. **Output category** — counter / telemetry / billing entry / alert / log.
3. **Hot-path cost** — per-event, per-tick, per-launch.
4. **Fusion mode** — exactly one of the five carried below.
5. **Classifier-input coupling** — what Wave 1 output the observer would
   consume; what its fallback today is.
6. **ABI/IOCTL touchpoints** — which `/dev/cipher` ioctls it calls;
   which `cipher_tenant_snapshot` fields it reads.

**Fusion classes** carried over from Waves 2/3:
PORT-AS-IS = compile with no logic change.
SHIM-REQUIRED = add adapter from classifier output, no behavioural change.
REFACTOR-REQUIRED = substantive code change.
VESTIGIAL = retired; not in OBJS.

**Fusion modes** (Wave 4 adds a fifth to the task brief's four):
- `passthrough-priority-0` — registers as a passive observer on the
  matmul or attn substrate registry at priority 0. The AUDIT template.
- `background-thread-polling` — long-running thread that wakes on a
  cadence, reads kmod state or NVML, emits.
- `kmod-resident` — requires kernel-side state mutation; must move into
  the kmod.
- `hybrid` — kmod-resident counter or shared-memory ring + userspace
  polling reader; the split is named.
- **`stage1-ring-consumer`** (new, Wave 4) — the dominant pattern in
  this wave. The observer is `void`-returning, takes a
  `const CipherRingEntry*`, runs in the Stage-1 shadow thread inside
  `cipher_10ops_impl.cpp::stage1_shadow()`. The observer is event-driven
  off a 65,536-entry SPMC ring populated by Stage 0 in the cuLaunchKernel
  interceptors. No substrate registry registration; no return value
  channel. Justified as a fifth mode because it has different timing
  (off-the-hot-path-but-still-event-driven, on a single shared thread),
  different failure semantics (a slow observer backpressures the ring,
  not the launch), and a different surface (`CipherRingEntry`, not
  `cipher_rt_matmul_call` / `cipher_rt_attn_call`).

---

## The Wave-4 wiring map (verified by inspection)

There are three observation lanes in the May-13 build:

| Lane | Surface | Wiring site | Observers in Wave 4 |
|---|---|---|---|
| **L1 Stage-1 ring** | `const CipherRingEntry*` (`include/cipher_10ops.h` L23-37; 128-byte cache-line-aligned record with `kernel_class`, `grid_*`, `block_*`, `func_ptr_hash`, `params_hash`, `output_hash`, `timestamp_ns`, `sequence`) | `cipher_10ops_impl.cpp::stage1_shadow()` L509-560 (the in-tree shadow thread; loops on `ring.read_seq_s1`, sleeps 500 µs when idle per L408-409) | 13 observers: shield, sustain, thermostat, pulse, hibernate, loop, continuity, pipeline, guard, trace, fairness, carbon, receipt; comply consumes the others' aggregates (no hot-path work) |
| **L2 Stage-2 cadence** | poll function called periodically from `stage2_background()` (L574+) and on demand | Same TU; same thread that consumes `ring.read_seq_s2` | thermostat::poll, pulse::evaluate, hibernate::poll — companions to L1 observers (same TU as the L1 hook) |
| **L3 cublasGemmEx shim** | direct fast-path call from `cipher_intercept_cudart.cpp` L860-883 | `libcipher_hook.so` cudart shim (NOT the Wave-2 `cipher_rt_matmul_dispatch` substrate) | exactly one: `cipher_fairness_shm` (companion TU to `cipher_fairness`); records every GEMM via shm; the `should_yield()` predicate can introduce a 20 µs `usleep` for the heavy tenant |
| **L4 NCCL tuner / nccl.cpp** | `cipher_straggler_observe(bytes, dur_ns, algo)` resolved by `dlsym(RTLD_DEFAULT, ...)` from `cipher_nccl.cpp` L287 | `libnccl-tuner-cipher.so` plus `cipher_nccl.cpp` per-bucket timing | exactly one: `cipher_straggler` (Phase 3) |

This wiring discovery is the most important Wave-4 finding for the
fusion plan: **none** of the Wave-4 observers sits on the Wave-2
`cipher_rt_matmul_dispatch` or `cipher_rt_attn_dispatch` substrate
registries. The Wave-2 blocking risk (HANDLED structurally discarded
on attn — `cipher_rt_attn_dispatch.cpp` L321/359/398) **does not
apply to Wave 4**: every observer here returns `void`. The classifier
fusion plan's "register the classifier observer at priority 5 on both
substrates" lives in Wave 2's domain — these observers, if ever
re-homed onto Wave 2's substrate registries, would land at the AUDIT
priority-0 template position (no HANDLED claim) and be safe. Stated
once here; not repeated per file.

## The Wave-4 ABI / IOCTL surface

**None of the 16 observers calls a `/dev/cipher` ioctl directly.**
The observers are userspace-only telemetry by design. Where they need
hardware state, they read it through one of three side channels:

1. The `g_cipher.liquid.device->hw.gpu_temp_c` field updated by Stage 11
   thermal feedback (read by THERMOSTAT only; `cipher_thermostat.cpp`
   L112-113).
2. NVML directly via `dlopen(libnvidia-ml.so.1, RTLD_LAZY|RTLD_NOLOAD)`
   (PULSE for ECC counters L130-141; HIBERNATE for the power-limit
   write probe L67-78).
3. The TLS `cipher_sense_current_session()` set by Op-13 SENSE (read by
   SHIELD, SUSTAIN, FAIRNESS, CARBON, RECEIPT, GUARD).

This is a structural property of the design: the observability tier
sits above the kmod. The kmod's `cipher_tenant_snapshot` (Wave 3
`nr 8`) and `cipher_gpu_state` (Wave 3 spine) are read by the
*classifier* (Wave 1) and the *actuators* (Wave 2), then exposed to
observers either via `cipher_sense_*()` (TLS), the shared liquid
state (`g_cipher.liquid`), or — in `cipher_fairness_shm`'s case —
a POSIX shm region carved out by the observer itself.

`cipher_fairness_shm` carries its own `/cipher_fairness` shm region
(magic `0xC1F4C1F4`, 64-slot table, ~2.6 KiB, `cipher_fairness_shm.cpp`
L19-20, L37, L107-108). This region is **not** the kmod's `tenant_snapshot`;
it is a cross-process userspace structure carved by `shm_open` +
`mmap` at constructor priority 114 (L225-226).

## The Wave-4 cross-wave dependencies

- **COMPLY → DETERMINISM, FAIRNESS, CARBON, GUARD, RECEIPT, TOPOLOGY**.
  `cipher_comply.cpp::cipher_comply_report` L46-82 pulls aggregates
  from six sibling observers. `cipher_determinism` is NOT in Wave 4
  (out of scope — it's in the next wave); treated as an external API.
- **SHIELD → SENSE, green_ctx::cipher_sm_set_priority**. SHIELD writes
  band hints into the Op 5 ARBITRATE → Op 9 ORCHESTRATE plane via
  `cipher_sm_set_priority`. This is the same priority-band API audited
  in Wave 2's `cipher_rt_green_ctx.c`. SHIELD is *not* a kmod ioctl
  caller — `cipher_sm_set_priority` is purely userspace, writing into a
  green-ctx-side priority table.
- **SUSTAIN → SENSE**. The AGENT-relax exception
  (`cipher_sustain.cpp` L171-175) reads `cipher_sense_get_type(sid)`
  and triples the slope threshold for `AGENT_AUTONOMOUS` sessions with
  decode_step_idx > 500.
- **CONTINUITY → SENSE**. Manifest gate is AGENT-only
  (`cipher_continuity.cpp` L165-166).
- **PIPELINE → SENSE**. Pretty-prints the SENSE classification per
  session and per edge endpoint (`cipher_pipeline.cpp` L174-178, L207-216).
- **THERMOSTAT → cipher.h::g_cipher.liquid.device**. The thermal source
  of truth (`cipher_thermostat.cpp` L112-113) is the silicon model
  field written by Stage 11 thermal feedback per CLAUDE.md. Comment at
  L110-111 flags the linkage explicitly ("anonymous-namespace `extern`
  would give it internal linkage and break the link").
- **PULSE → NVML (RTLD_NOLOAD only)**. `cipher_pulse.cpp` L130-131
  explicitly does *not* dlopen — it only attaches to NVML if some
  other layer (Stage 11) already loaded it. ECC signal is conditionally
  disabled when telemetry layer is off.
- **HIBERNATE → NVML (full load, then RTLD_NOLOAD on subsequent)**.
  In contrast to PULSE, HIBERNATE actively `dlopen`s NVML to call the
  `nvmlInit_v2` and power-limit family. The probe at L94-100 writes the
  current value back to detect whether write actuation is permitted —
  on the pod this returns `NVML_ERROR_NOT_SUPPORTED` so the observer
  falls through to "DEGRADED" mode (detection only, no power writes).
- **STRAGGLER → NCCL bucket telemetry**. Driven by `cipher_nccl.cpp`
  L287 resolving `cipher_straggler_observe` via `dlsym(RTLD_DEFAULT, ...)`.

---

# Per-file audits

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_carbon.cpp

### PURPOSE
Op 23 — v1 per-tenant carbon estimate (L1). Accumulates
`grid × block` "work units" per session and converts to grams-CO₂
at report time using two tunable env vars
(`CIPHER_CARBON_J_PER_UNIT`, `CIPHER_CARBON_GCO2_PER_KWH`).
File header L1.

### PUBLIC SURFACE
- `int cipher_carbon_init(void)` — L62; env-gated `CIPHER_CARBON`.
- `void cipher_carbon_observe(const CipherRingEntry* ev)` — L84.
- `unsigned cipher_carbon_session_count(void)` — L105.
- `void cipher_carbon_report(void)` — L113.

### CONTROL FLOW
`init`: idempotent CAS on `g_initialized` (L63); reads env vars
(L65-74); logs banner. Defaults: `g_j_per_unit = 1e-9` (L29),
`g_gco2_per_kwh = 400.0` (L30).

`observe`: enabled-load (L85); `cipher_sense_current_session()`
gate — silent return if `sess == 0` (L87-88); compute
`units = ∏(grid_{xyz}, block_{xyz})` with 1-fill on zero dims
(L90-97); probe/insert open-address table by `mix64(session_fp)`
(L99); `work.fetch_add(units)` and `launches.fetch_add(1)` relaxed
(L101-102).

`probe_or_insert` (L39-58): linear probe MAX_TENANTS slots; CAS
on empty key; concurrent insert losers re-load the key and re-check
for the same key (L54). On miss returns -1; caller drops silently
(I4).

`report`: walks the 256-slot table; computes `gco2 = work *
J/unit * (gCO2/kWh / 3.6e6 J/kWh)` (L120-128); maintains a top-32
by gCO2 via min-replace (L131-137); insertion-sort the survivors
in descending order (L139-143); emit
`/tmp/cipher_carbon_report.json` with `session_count`,
`j_per_unit`, `gco2_per_kwh`, `total_gco2`, plus the top-32 array.

### STATE
- `Tenant g_table[256]` (L23) — `{key, work, launches}` atomics, 64-byte
  aligned.
- `g_enabled`, `g_initialized`, `g_tenant_count` (L25-27).
- `g_j_per_unit`, `g_gco2_per_kwh` (L29-30) — non-atomic doubles, set
  exactly once in `init`.

### CONCURRENCY
Multiple Stage-0 launch threads write to the ring; one Stage-1
shadow thread consumes. CARBON is called only from the latter,
so observe() is single-threaded by construction. The atomics
on `key`/`work`/`launches` are defensive belt-and-suspenders
for the open-address insert; the CAS on `key` (L47-48) is the
only place where contention could happen, and only between the
shadow thread and itself (it can't, given single-consumer).

The `g_j_per_unit` / `g_gco2_per_kwh` doubles are written exactly
once in `init` before any observer fires, and read non-atomically
in `report` — safe because `init` happens-before
`stage1_shadow` is started by `cipher_10ops_impl.cpp` (init at
L920+, shadow-thread spawn later).

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L557 (observe), L936 (init).
  `cipher_comply.cpp` L50, L70 (session_count for the COMPLY report).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session()` — TLS read.
- ioctls: NONE. No NVML. No CUDA.
- file I/O: writes `/tmp/cipher_carbon_report.json` at report time only.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L29 `g_j_per_unit = 1e-9` joules per work-unit is a placeholder
  documented as such in the env-var name (`J_PER_UNIT`). With the
  default value, a 4096³ GEMM (≈ 1e10 work units) is 10 J — not a
  realistic energy figure. The honest framing in the header L1 says
  "v1 estimate". Not a code bug.
- The `units = ∏(grid, block)` model ignores per-kernel ops/byte;
  flagged as a v1 approximation by the comment "v1 per-tenant
  carbon estimate" at L1.
- `total_gco2` accumulates across only the top-32 survivors (`for i in
  [0, n)` at L153), not the full MAX_TENANTS table — this is correct
  because `n` ranges up to MAX_TENANTS during table iteration in
  L123-138 (the `n` accumulator at L131 is the **count of populated
  slots**, not just the top-N). Re-reading L131-137: when `n <
  REPORT_TOP_N` the slot is appended; once full it replaces the min.
  But `total_gco2` is updated *every* slot pass at L129. So the
  total is correct; the top-N filter is on output only. Not a drift.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY. CARBON contributes to COMPLY's gating
(`cipher_comply.cpp` L60 uses `receipt_sessions >= 1`, not carbon
specifically) and to the customer-facing telemetry stream. No
hot-path actuation; no v1 consumer beyond the COMPLY aggregator.

### FUSION POINTS

**Signal.** Every Stage-0 ring entry. The session-attribution gate
(L87-88) drops events without a SENSE-resolved session. Carbon
volume is proportional to `grid × block`.

**Output.** Billing entry (gCO₂ per session); reported on demand.
Per-tenant counter (`work`, `launches`) is on-line.

**Hot-path cost.** Per-event, single-threaded on the shadow thread:
one TLS read (sense_current_session), one mix64 hash + linear probe
(amortized ~1 cache line at low fill), one atomic fetch-add on
`work`, one on `launches`. Estimate ~50 ns/event at low fill,
~200 ns at high fill — never blocking the producer (the
producer is `stage1_shadow`'s consumer, not the CUDA hot path).

**Fusion mode: `stage1-ring-consumer`.** Defends: the observer is
strictly off the CUDA-launch hot path; it sits on the shadow
thread reading a SPMC ring. The work-unit model and dollar/CO₂
output have no place in either the matmul or attn substrate
registry. Migration to either substrate (Wave 2) would be a
strict regression in two ways: (a) it would put the
`grid × block` multiplication on the launch hot path; (b) it
would force every cuBLAS/SDPA call to take the session-TLS hash.

**Classifier-input coupling.** Today reads only
`cipher_sense_current_session()`. A v1.5 fusion would consume
classifier-output `tenant_billing_class` (corporate vs free-tier vs
research) and weight `g_j_per_unit` by the tenant's compute SKU —
this is a one-field SHIM, not a refactor.

**Fallback today.** No classifier; one global `g_j_per_unit` and
one global `g_gco2_per_kwh`, both ENV-tunable at init.

**ABI/IOCTL touchpoints.** NONE. No `cipher_tenant_snapshot` read.

**Fusion class: PORT-AS-IS** for v1. SHIM-required for v1.5
(tenant-class-aware energy intensity).

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_fairness.cpp

### PURPOSE
Op 24 — v1 per-tenant work-quota observer (L1). Same open-address
table shape as CARBON; adds an overrun-flag latch and a
`CIPHER_FAIRNESS_QUOTA` env tunable (default 5e9 work units, L18).

### PUBLIC SURFACE
- `int cipher_fairness_init(void)` — L66.
- `void cipher_fairness_observe(const CipherRingEntry* ev)` — L88.
- `unsigned cipher_fairness_tenant_count(void)` — L119.
- `unsigned cipher_fairness_overrun_count(void)` — L123.
- `void cipher_fairness_report(void)` — L131.

### CONTROL FLOW
Identical scaffold to CARBON's `probe_or_insert` (L42-62) — the
hash function, table layout, and CAS-insertion logic are byte-for-
byte parallel. The Wave-3 invariant "no allocation on hot path"
applies. The key behavioural difference is in `observe` L106-116:
after the fetch-add on `work`, the code checks whether the work
unit count crossed the quota gate:

```
uint64_t prev_work = t.work.fetch_add(units, std::memory_order_relaxed);
...
uint64_t q = g_quota.load(std::memory_order_relaxed);
if (prev_work < q && (prev_work + units) >= q) {
    uint32_t prev_flag = t.overrun_flag.exchange(1, std::memory_order_relaxed);
    if (prev_flag == 0) {
        g_overrun_count.fetch_add(1, std::memory_order_relaxed);
    }
}
```

Edge-triggered: exactly one global overrun count per tenant
(the per-tenant flag is set monotonically; the global counter
only increments on the 0→1 transition). The `prev_work < q &&
(prev_work + units) >= q` form is the correct cross-the-line
detection that uses the relaxed fetch-add return value to avoid
a second load on the relaxed counter.

`report` (L131-184) does a top-32 by work (L144-156), then
emits `/tmp/cipher_fairness_report.json` with quota,
tenant_count, overrun_count, plus the top-32 sessions array.

### STATE
- `Tenant g_table[256]` (L27) — `{key, work, launches, overrun_flag}`.
- `g_enabled`, `g_initialized`, `g_quota`, `g_tenant_count`,
  `g_overrun_count` (L29-33).

### CONCURRENCY
Same as CARBON. Single-consumer on Stage-1 shadow thread. The
overrun-flag latch is a benign double-CAS (the
`fetch_add(work)` returns the pre-add value, and the cross-the-line
check is computed from that — race-free).

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L554 (observe), L935 (init).
  `cipher_comply.cpp` L55, L56, L75 (tenant_count, overrun_count).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session()`.
- ioctls: NONE. No NVML. No CUDA.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L18 `DEFAULT_QUOTA = 5e9`: same work-unit model as CARBON. v1
  estimate, env-overridable.
- The fairness signal is **only** about gating overrun *detection*;
  there is no actuation (no throttling, no demotion of the offending
  tenant). The `cipher_fairness_shm` companion TU handles cross-process
  throttling via `should_yield()` and is the only fairness TU that
  performs back-pressure.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY + COMPLY input. The `overrun_count` is one of
COMPLY's four hard-gate fields (`cipher_comply.cpp` L62: `&&
(fair_overruns == 0)`).

### FUSION POINTS

**Signal.** Same as CARBON (per-event, gated on SENSE session).
The trigger condition is the edge crossing of cumulative work
through the quota line.

**Output.** Counter (`overrun_count` per session), with a single
global increment per first-time overrun. Logged once at report
time.

**Hot-path cost.** Same as CARBON; one extra branch on the
`prev_work < q && (prev_work + units) >= q` check, one
infrequent atomic exchange. ~60 ns/event.

**Fusion mode: `stage1-ring-consumer`.** Same defence as CARBON.
Per-tenant quota detection has no place in the matmul/attn
substrate; it is a higher-level policy signal.

**Classifier-input coupling.** Today reads
`cipher_sense_current_session()`. v1.5 would consume
classifier-output `tenant_quota_units` and overwrite the global
quota per tenant — a small SHIM. The fairness *enforcement* lives
in `cipher_fairness_shm`'s `should_yield()` predicate, not here.

**Fallback today.** No classifier; single global quota set at init
from `CIPHER_FAIRNESS_QUOTA`.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** + SHIM-required for per-tenant quotas.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_fairness_shm.cpp

### PURPOSE
The cross-process actuator companion to `cipher_fairness.cpp`.
Carves a POSIX shm region `/cipher_fairness` (64 tenant slots,
~2.6 KiB) and exports a `should_yield()` predicate consumed
*on the cublasGemmEx hot path* by the `cipher_intercept_cudart.cpp`
shim (L858-883). This is the only observer in Wave 4 that sits
on a hot path.

### PUBLIC SURFACE
- `int cipher_fairness_shm_init(void)` — L59; constructor priority 114
  (L225-226).
- `int cipher_fairness_shm_enabled(void)` — L112.
- `int cipher_fairness_shm_register(void)` — L116; returns slot id.
- `void cipher_fairness_shm_record_gemm(void)` — L160; per-GEMM.
- `int cipher_fairness_shm_should_yield(void)` — L170.
- `int cipher_fairness_shm_self_calls(uint64_t* out_calls,
   uint64_t* out_yields)` — L196.
- `void cipher_fairness_shm_shutdown(void)` — L207; destructor (L228-229).

### CONTROL FLOW

`init` (L59-110): CAS on `g_started` (L60-61); env-gate on
`CIPHER_FAIRNESS` (L63-67); `shm_open(/cipher_fairness, O_CREAT|O_RDWR,
0666)` (L69); `ftruncate` to `sizeof(CipherFairnessShm)` (L75); mmap
PROT_READ|PROT_WRITE MAP_SHARED (L81); close fd post-mmap (L89);
**CAS on magic** for first-mapper initialization
(L92-104) — exactly one process initializes the slot array, others
just attach.

`register` (L116-158): two passes over slot table. First pass
(L125-138) looks for an existing slot matching this process's
`tenant_id` (env `CIPHER_TENANT_ID` or PID) and `active==1` — claims
it on hit (re-registration after restart). Second pass
(L140-155) CAS-allocates a new slot. If both fail, log "all slots
full" (L156) and return -1. MAX_TENANTS = 64 (L20). The
`HARNESS_LIMITATIONS.md` flags that above 64 tenants slot
registration fails silently with the workload continuing —
documented limit.

`record_gemm` (L160-168): the hot-path call. Lazy registers on
first call (L162-164); atomic `fetch_add(1)` on `gemm_calls`;
release-store on `last_active_ns`. Two atomic ops + one
clock_gettime per call.

`should_yield` (L170-194): walks all 64 slots; for each
**recently-active** slot (active==1 AND `now - last_active_ns
< 2 s`, L178-182), accumulates `gemm_calls`. Computes
`avg = total / n_active`. Returns `1` iff `my_calls > 2*avg &&
my_calls > 1024` (L189). On a yield, `yield_count.fetch_add(1)`
on the caller's slot.

`shutdown` (L207-223): writes summary line; releases the slot
(`active=0`, `tenant_id=0` via release-stores); `munmap`. The
slot is *not* zeroed beyond `active`/`tenant_id` — the next
allocator overwrites the body.

### STATE
- `CipherFairnessShm* g_shm` (L39) — pointer to mmap'd region.
- Per-process: `g_my_slot`, `g_my_tenant_id`, `g_enabled`, `g_started`
  (L40-43).
- Per-process file-scope: `MAX_TENANTS=64` (L20), `FAIRNESS_MAGIC=
  0xC1F4C1F4u` (L19), `IDLE_THRESH_NS = 2e9` (L178).

### CONCURRENCY
**This is the only Wave-4 observer with cross-process concurrency.**
Multiple processes mmap the same region; the slot table is
manipulated entirely with `std::atomic` operations (every field of
`TenantSlot` is atomic, L23-29).

The `register` and `should_yield` walks are wait-free (single-pass
over a small fixed array); the per-call hot path is two atomic ops.
The `init` race uses CAS on `magic` (L93) to elect the initializer
— first-mapper-wins, others skip the zero-fill at L95-103. Standard
posix-shm initialisation pattern.

### DEPENDENCIES
- INBOUND: `cipher_intercept_cudart.cpp` L867-870 (resolves via
  `dlsym(RTLD_DEFAULT, ...)`), L873 (should_yield), L882
  (record_gemm). Constructor priority 114 (L225) auto-fires before
  the cipher_intercept_cudart shim is exercised.
- OUTBOUND: shm_open, mmap, ftruncate, close, munmap; clock_gettime.
- ioctls: NONE.
- NO CUDA, no NVML.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `CIPHER_DRIVER_LEVEL_BUILDS.md` L36-38 documents the wire as
  "cublasGemmEx shim" — code matches.
- Yield threshold `my > 2*avg && my > 1024` (L189): a tenant must
  be more than 2× the active-average AND have done at least 1024
  GEMMs to be throttled. Both bounds are hard-coded; documented in
  the design memo as the v1 floor.
- L18 (`DEFAULT_QUOTA` in `cipher_fairness.cpp`) and `cipher_fairness
  _shm.cpp` L189 share the name "fairness" but **are not aligned**:
  the per-process FAIRNESS counts work units (`grid × block`); the
  cross-process shm counts cublasGemmEx call frequency. They detect
  different things and are independent observers.
- L121 default tenant_id is PID; documented at the header L8-9.
- L156 silent log on "all slots full" — `HARNESS_LIMITATIONS.md`
  documents the bump-to-256 plan for the next observer-cleanup pass.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING in multi-tenant mode.** This is the only userspace
mechanism in the May-13 build that provides cross-process
back-pressure on cublasGemmEx without going through the kmod. In
single-tenant runs it does nothing (n_active ≤ 1 → return 0
at L186). In multi-tenant runs (the WL01-WL24 suite, Phase A) it
caps the heavy tenant at 20 µs/yield (intercept_cudart L874).

### FUSION POINTS

**Signal.** Every cublasGemmEx call from any process attached to
the shm. The trigger is per-tenant call frequency vs the
recently-active average.

**Output.** Counter (`gemm_calls`, `yield_count` per slot) +
side-effect (the `usleep(20)` in the calling shim).

**Hot-path cost.** `should_yield()` walks 64 atomic slots — 64
acquire-loads. At 20 ns/load, ≈ 1.3 µs per cublasGemmEx — non-
trivial. `record_gemm()` is two atomic ops plus a `clock_gettime`
— ≈ 100 ns. Combined per-GEMM cost ≈ 1.4 µs at 64-tenant fill,
which is well below cublasGemmEx's own per-call overhead but is
the largest fixed observer cost in Wave 4.

**Fusion mode: `kmod-resident` candidate, hybrid recommended.**
This is the single Wave-4 observer where the kmod offers a strict
win:
- The kmod's `tenant_snapshot` (Wave 3 nr 8) already holds
  per-tenant launch counts and a `last_seen_ns` field.
- The kmod's CP54 scheduler (Wave 3 ioctls 13-20) already moves
  tenants between SM partitions, with the very back-pressure
  semantics that the user-space `usleep(20)` is approximating.
- A `CIPHER_FAIRNESS_SHOULD_YIELD` ioctl returning a single
  signed-byte is *one* syscall vs 64 acquire-loads. The kmod can
  also use proper PIDR introspection (no env var spoof) to bind
  tenant identity.

The honest recommendation is `hybrid`:
- Move the slot table into kmod state (Wave 3-style additive
  ioctl `nr 25 CIPHER_FAIRNESS_REGISTER`, `nr 26
  CIPHER_FAIRNESS_RECORD_GEMM`, `nr 27 CIPHER_FAIRNESS_SHOULD_YIELD`).
- Keep the read-path (`should_yield`) in userspace as a fast-path
  via the existing CIPHER `tenant_snapshot` ioctl, decorated with
  the new fairness fields.

**Classifier-input coupling.** Today: PID-keyed slots, no
classifier. The classifier would supply a `tenant_class` per slot
(corp / free / research) that weights `should_yield` thresholds.
SHIM-required.

**Fallback today.** Cross-process POSIX shm; 64 slots; CAS-elected
initializer.

**ABI/IOCTL touchpoints.** NONE today. Recommended move to ioctls
25-27 in fusion v1.5.

**Fusion class: REFACTOR-REQUIRED for v1.5** (kmod migration).
PORT-AS-IS for v1 (works as-is; just be aware of the per-GEMM 1.4 µs
cost). Severity: requires-mitigation — the per-GEMM walk is the
worst hot-path cost in Wave 4.

**Wave-2 cross-check.** Since fairness_shm wires through
`cipher_intercept_cudart.cpp` (the `libcipher_hook.so` cudart
shim), **not** the Wave-2 `cipher_rt_matmul_dispatch` substrate,
the Wave-2 HANDLED-discard finding is not relevant here. If the
fusion plan re-homes fairness_shm onto the matmul substrate
(rather than the cudart shim), it would register at priority 0
AUDIT-template — passthrough-by-contract, no HANDLED claim — and
the Wave-2 finding still does not apply.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_receipt.cpp

### PURPOSE
Op 18 — v1 per-session signed proof of compute. Builds an
order-preserving FNV-1a chain over `params_hash` per session, then
emits HMAC-SHA256(chain || launches || order) at report time. The
output is a verifiable "I did N kernels for tenant X" receipt usable
in a v1.5 billing pipeline. File header L1.

### PUBLIC SURFACE
- `int cipher_receipt_init(void)` — L103.
- `void cipher_receipt_observe(const CipherRingEntry* ev)` — L123.
- `unsigned cipher_receipt_session_count(void)` — L145.
- `void cipher_receipt_report(void)` — L165.

### CONTROL FLOW
`init` (L103-121): env-gated; reads `CIPHER_RECEIPT_KEY` as a 64-char
hex string (L113-114) — if present and parses to 32 bytes, uses it as
a fixed HMAC key (L115); otherwise seeds 32 bytes from `/dev/urandom`
(L86-99). Critical: receipts produced with the random fallback are
verifiable only by the producer itself (which holds the ephemeral
key in process memory).

`observe` (L123-143): SENSE-session gate (L126-127); probe-or-insert
by session (L130); on first insert, initialize `chain = FNV_OFFSET`
(L56), set `first_ts_ns` and `order` (L58-61). The `order` is the
session's monotonic insertion rank — used as the receipt's index in
the per-process stream. On every event, the chain is advanced by
`(chain ^ mix64(params_hash)) * FNV_PRIME`, with a compare-exchange
loop for atomicity (L137-140). Note: **ASLR-stable** by design — only
`params_hash` is mixed in, not `func_ptr_hash` (which depends on
loader-time address randomization). Commented L133.

`report` (L165-225): walk 256 slots, top-32 by `launches`. For each
survivor, build the 20-byte canonical MAC input `chain ‖ launches ‖
order` (L197-200) and HMAC-SHA256 it via OpenSSL EVP (L203). Hex-
encode the 32-byte MAC (L204-205). Emit JSON to
`/tmp/cipher_receipt_report.json` with session_fp, chain, launches,
order, first/last ts, and the HMAC.

The MAC input is deliberately **reproducible**: it excludes session_fp
(which depends on init timing) and timestamps. Two runs with the same
key + same kernel sequence produce the same HMAC.

### STATE
- `Tenant g_table[256]` (L31) — `{key, chain, launches, first_ts_ns,
  last_ts_ns, order}`. Atomics for the four mutable fields.
- `g_hmac_key[32]` (L37) — set exactly once in init.
- `g_enabled`, `g_initialized`, `g_tenant_count` (L33-35).

### CONCURRENCY
Single-consumer on Stage-1 shadow thread. The compare-exchange loop
at L137-140 is defensive belt-and-suspenders for a future
multi-consumer scenario; in v1 it can be a relaxed store.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L560 (observe), L937 (init).
  `cipher_comply.cpp` L49, L60, L69 (session_count).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session()`;
  `openssl/hmac.h::HMAC` with `EVP_sha256()`.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L88-99 seed_random_key: reads 32 bytes from `/dev/urandom`; on
  failure falls back to time-XOR `0xdeadbeefcafebabe` with mix64
  PRF — deterministic enough to compile, not crypto-grade. The
  fallback is reachable only when `/dev/urandom` open fails, which
  on a normal Linux pod doesn't happen. Honest framing.
- "HMAC-SHA256(key, chain_fnv || launches || order)" advertised at
  L193 matches the L197-203 code exactly.
- The 32-byte HMAC key is held *in process memory only* — there is
  no key escrow, no rotation. v1 limitation.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY + COMPLY input. The `session_count >= 1` field
is the COMPLY "receipt present" gate (`cipher_comply.cpp` L60).
RECEIPT is the only Wave-4 observer that produces a cryptographic
artifact — billing rails would consume it.

### FUSION POINTS

**Signal.** Every Stage-0 ring entry, gated on SENSE session.
The chain advances per kernel; receipts emit at report time.

**Output.** Billing entry (HMAC-SHA256 receipt per session at
report time) + counter (`launches` per session, on-line).

**Hot-path cost.** Per-event: one TLS read, one probe, one
mix64, one CAS-loop on the chain (one iteration in practice
since this is single-consumer). ~60 ns. No HMAC on the hot path
— HMAC is only computed at report time on the report thread.

**Fusion mode: `stage1-ring-consumer`.** Defends: the
`params_hash` mixing must happen on every event; no place for
it on the substrate registries. Receipts are end-of-session
artifacts, not hot-path actuation.

**Classifier-input coupling.** Could consume a
`tenant_billing_key_id` per session to select the HMAC key from a
keyring; would let multi-tenant runs produce per-tenant
verifiable receipts. SHIM.

**Fallback today.** Single global key, env-overridable.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** for v1; SHIM for per-tenant keying
in v1.5.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_guard.cpp

### PURPOSE
Op 16 — cross-session residency leak detector. Detects when the same
`params_hash` (proxy for "same kernel shape, same input shapes")
appears in two different sessions within `RESIDENCY_NS = 2 s`. The
hypothesis is that if session A's data is still in L2 when session B
launches the same kernel shape, B could be reading A's residue —
classic cross-session leak in a shared GPU. File header L1-9.

### PUBLIC SURFACE
- `int cipher_guard_init(void)` — L83.
- `void cipher_guard_observe(const CipherRingEntry* ev)` — L99.
- `unsigned cipher_guard_leak_count(void)` — L132.
- `unsigned cipher_guard_session_count(void)` — L136.
- `void cipher_guard_report(void)` — L151.

### CONTROL FLOW
`observe` (L99-130): pick the key as `params_hash || func_ptr_hash`
(L103); SENSE-session gate (L106-107); pick `t` from
`ev->timestamp_ns` or now (L109); detect session change at the
*observer* level (L112-116) to count distinct sessions. Probe key
into 4096-slot open-address table (L57-79); read the entry's
`session_fp` and `last_ts_ns` (L121-122) — note these are
**non-atomic** plain fields; the table is single-consumer so a
torn read is impossible from the shadow thread. The leak gate is

```
if (prev_sess != 0 && prev_sess != sess
    && t > prev_ts && (t - prev_ts) < RESIDENCY_NS) {
    e.leak_count++;
    g_leak_events.fetch_add(1, std::memory_order_relaxed);
}
```

L123-127. Then write back the new session/timestamp pair (L128-129).

`probe` (L57-79): linear-probe MAX_SHAPES=4096; on table-full,
**overflow to slot 0** (L78) — the only Wave-4 hash-table that
explicitly fails to a sentinel slot rather than dropping silently.
The comment "rare in v1" is accurate (4096 distinct shapes is large).

### STATE
- `Entry g_table[4096]` (L35) — `{key, session_fp, last_ts_ns,
  seen_count, leak_count}`. Only `key` is atomic; rest plain.
- `g_enabled`, `g_initialized`, `g_shape_count`, `g_leak_events`,
  `g_last_session_seen`, `g_session_count` (L37-42).

### CONCURRENCY
Single-consumer Stage-1. The plain non-atomic fields in `Entry` are
safe under this assumption. The `key.compare_exchange_strong` is
defensive — would matter only under a multi-consumer scenario.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L545 (observe), L931 (init).
  `cipher_comply.cpp` L51, L52, L59, L71 (session_count, leak_count).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session()`;
  `clock_gettime(CLOCK_MONOTONIC_RAW)`.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `RESIDENCY_NS = 2 s` (L25) is a heuristic for "L2 still warm".
  H100 L2 is 50 MiB; 2 s is generous. Documented as v1.
- The leak detection is *probabilistic*: a recurring kernel shape
  in two sessions does not prove a leak. The output is a "leak
  candidate" stream; investor framing in COMPLY would treat
  `leak_count > 0` as cause for review.
- `cipher_comply.cpp` L62 hard-gates `compliance_ok` on
  `guard_leaks == 0` — this is the only Wave-4 observer whose
  output flips compliance.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY + COMPLY hard-gate. GUARD is the canonical
"cross-tenant safety" observer; its zero-leak invariant is the
load-bearing signal that the multi-tenant configuration is sound.

### FUSION POINTS

**Signal.** Same `params_hash` (kernel shape) reappearing in
distinct sessions within 2 s.

**Output.** Counter (leak_count per shape, plus a global
`g_leak_events`) + alert (compliance flip).

**Hot-path cost.** Per-event: TLS read, mix64+probe of a 4096-slot
table (~2-3 cache misses at low fill), one branch, a couple of
plain stores. ~80 ns/event.

**Fusion mode: `stage1-ring-consumer`.** GUARD's signal requires
cross-session correlation, which depends on having a session-fp
already attributed. Putting it on the substrate registry would
gate it on the substrate's TLS plumbing, which is fragile.

**Classifier-input coupling.** Could consume a
`tenant_isolation_zone` per session — leaks within the same
tenant's zone are benign and could be filtered. SHIM.

**Fallback today.** Single global `RESIDENCY_NS`.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** for v1. Note: when L2 partitioning
(Phase 5 CP 5.3 partition-aware Marlin) is active, the leak rate
should structurally drop; GUARD remains the right canary.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_comply.cpp

### PURPOSE
Op 29 — v1 compliance pack. Aggregates the six sibling observer
outputs (RECEIPT, CARBON, GUARD, DETERMINISM, FAIRNESS, TOPOLOGY)
and emits a single `compliance_ok` boolean + a structured JSON
report. No per-event work; report-time aggregator.

### PUBLIC SURFACE
- `int cipher_comply_init(void)` — L23.
- `void cipher_comply_observe(const CipherRingEntry* ev)` — L38;
  **explicit no-op** (L39 `(void)ev`).
- `int cipher_comply_ok(void)` — L42.
- `void cipher_comply_report(void)` — L46.

### CONTROL FLOW
`init` (L23-36): standard env-gate + banner.
`observe`: explicit no-op. The aggregation is report-only.
`report` (L46-82): pulls six sibling counters (L49-57); computes
the hard gate `compliance_ok = (guard_leaks == 0) && (receipt_sessions
>= 1) && (determ_count > 0) && (fair_overruns == 0)` (L59-62);
emits `/tmp/cipher_comply_report.json` with all six fields.

### STATE
- `g_enabled`, `g_initialized`, `g_last_ok` (L18-20) — three
  atomics.

### CONCURRENCY
N/A on the hot path (observe is a no-op). The report function is
single-thread.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L938 (init); not wired into the
  observe stream (deliberately — the observe is `(void)ev`).
- OUTBOUND: RECEIPT (`session_count`), CARBON (`session_count`),
  GUARD (`session_count`, `leak_count`), DETERMINISM
  (`hash`, `count`), FAIRNESS (`tenant_count`, `overrun_count`),
  TOPOLOGY (`device_count`).
- **Cross-wave dependency**: `cipher_determinism_*` symbols are not
  in Wave 4's 17 files; they live in `cipher_determinism.cpp` which
  belongs to a sibling wave. Treat as external API.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The hard-gate at L59-62 omits CARBON and TOPOLOGY. Carbon
  numbers don't have a pass/fail definition in v1; topology
  device_count is informational. Documented by absence.

### CONTRIBUTION TO SYSTEM
v1 COMPLIANCE AGGREGATOR. COMPLY's `compliance_ok` boolean is the
single externally-facing "this run is compliant" bit.

### FUSION POINTS

**Signal.** None on the hot path — aggregator only.

**Output.** Telemetry (the JSON report) + counter (`last_ok`).

**Hot-path cost.** ZERO. `observe` is `(void)ev; return`.

**Fusion mode: `stage1-ring-consumer`** by registration, though
its observe is a no-op. The aggregator pattern is the reason it
appears in Wave 4 at all.

**Classifier-input coupling.** None today. v1.5 could read a
`tenant_compliance_profile` to vary the hard-gate per tenant.

**Fallback today.** Hard-gate constants in the report function.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS.** The only Wave-4 observer that is
trivially port-as-is because it has no hot-path work to move.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_loop.cpp

### PURPOSE
Op 26 — agentic runaway detection. Three signals per session:
S1 shape-cycle repetition (autocorrelation over a 64-deep shape ring),
S2 decode burn-rate vs the session's first-100-event baseline,
S3 prefill drought (>500 events since last prefill, gated by S1).
Score = S1+S2+S3, range 0..3. Score ≥ 2 ⇒ runaway, latched
("sticky-high"). File header L1-10.

### PUBLIC SURFACE
- `int cipher_loop_init(void)` — L131.
- `void cipher_loop_observe(const CipherRingEntry* ev)` — L148.
- `int cipher_loop_get_score(uint64_t fingerprint)` — L231.
- `unsigned cipher_loop_session_count(void)` — L242.
- `unsigned cipher_loop_runaway_count(void)` — L246.
- `void cipher_loop_report(void)` — L250.

### CONTROL FLOW
`observe` (L148-229): begins by reading the ring entry's timestamp
and updating `g_last_ts_ns` (L152-154). If the gap from the
*global* last event exceeds 200 ms (`IDLE_RESET_NS`, L27), or no
slot is currently active, a new session slot is allocated
(L157-168). The seed is `fnv1a64(timestamp ^ func_ptr_hash)` —
ensures distinct sessions get distinct seeds.

After session selection, the per-event work splits into four parts:
1. shape ring push (L175-178): the params_hash (or func_ptr_hash if
   that's zero) is shifted into a 64-deep ring at `shape_head`.
2. prefill/decode accounting (L181-196): unpacks `(M,K,N)` from
   `params_hash` if the high nibble is `0xC` (the M-K-N packing
   marker); `batch = min(M,N)`; decode if 1 ≤ batch ≤ 8, prefill if
   batch ≥ 256.
3. score evaluation (L199-228): every 16 events after warmup
   (100 events). Calls `detect_period` (L95-114) which materializes
   the last W shapes into a buffer (L98-101) and searches for a
   repeating period 1..MAX_PERIOD requiring MIN_CYCLES (3) full
   repetitions (L103-112). Sets `s1_cycle`, `s2_burn`, `s3_drought`
   monotonically.
4. score is only raised, never lowered ("sticky-high", L217-218).
   On 0→≥2 transition, increment `g_runaway_count` and stderr-log
   the diagnosis (L219-226).

### STATE
- `LoopSession g_slots[1024]` (L52) — per-session state including
  the 64-deep shape ring and the four counters.
- `g_enabled`, `g_initialized`, `g_session_count`, `g_runaway_count`,
  `g_current_slot`, `g_current_fp`, `g_last_ts_ns` (L54-60).

### CONCURRENCY
Single-consumer Stage-1. The atomic `fingerprint.compare_exchange_strong`
at L75-76 is the allocation-race guard; in single-consumer it always
succeeds with `expected==0`. The `g_last_ts_ns.exchange` at L154 is
the inter-event delta computation, also race-free under single-
consumer.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L533 (observe), L927 (init).
- OUTBOUND: standard libc.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L4-5 documentation: "Score = S1+S2+S3 (0..3)" matches L215 code.
- L8-10: "sticky-high" — matches L217 (`if (newscore > oldscore)`).
- The `detect_period` algorithm at L95-114 is correct
  autocorrelation by element-equality — robust to non-numeric
  shape hashes.
- `IDLE_RESET_NS = 200 ms` (L27) — same constant as SENSE; consistent
  cross-observer.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY + agentic-policy signal. LOOP's runaway flag
would feed a Stage 2 ORCHESTRATE demotion path; v1 logs only.

### FUSION POINTS

**Signal.** Per-event Stage-1 ring entries; trigger is the
combination of (period detected) + (decode/prefill ratio > 500) +
(events_since_prefill > 500 with cycle).

**Output.** Alert (stderr `RUNAWAY detected`) + counter
(`runaway_count`, per-session `score`).

**Hot-path cost.** Per-event: ring push, M/K/N unpack, integer
ops. Every 16 events: autocorrelation over up to 64 entries
(O(MAX_PERIOD × MIN_CYCLES) = O(24) comparisons per period
candidate, ≤ 8 candidates = O(192) per evaluation, amortized
12 comparisons per event). ~100 ns/event amortized.

**Fusion mode: `stage1-ring-consumer`.** Defended: per-event
shape-ring update + amortized autocorrelation is inherently
Stage-1, not substrate-side.

**Classifier-input coupling.** Could consume a
`session_max_decode_count` per session (different budget for
research vs production agents). SHIM.

**Fallback today.** Hard-coded gates (BURN_RATIO_MIN=500,
PREFILL_DROUGHT_MIN=500). No classifier.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** + SHIM for per-tenant budgets.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_pipeline.cpp

### PURPOSE
Op 27 — per-session bounded shape set; pairwise Jaccard similarity
across sessions at report time. Edge tagged upstream/downstream by
`first_ts_ns`. Inferred RAG/agent-chain topology. File header L1-5.

### PUBLIC SURFACE
- `int cipher_pipeline_init(void)` — L101.
- `void cipher_pipeline_observe(const CipherRingEntry* ev)` — L117.
- `unsigned cipher_pipeline_session_count(void)` — L143.
- `unsigned cipher_pipeline_edge_count(void)` — L147.
- `void cipher_pipeline_report(void)` — L151.

### CONTROL FLOW
`observe` (L117-141): same idle-reset session-allocation pattern as
LOOP / CONTINUITY (L120-135). Per-event: increment `event_count`,
`insert_shape(s, params_hash || func_ptr_hash)`. `insert_shape`
(L76-85) is linear search; if shape exists return; else if `n_shapes
< 32` append; else **silently drop on full** (I4 contract). The
session shape-set is therefore the first 32 distinct shapes the
session sees — a sketch, not a complete set.

`report` (L151-239): collects up to 256 allocated session indices
(L155-161). Emits a session list with SENSE-classification names
(L174-189). Then walks all `O(n²)` session pairs (L197-229);
computes Jaccard `|A ∩ B| / |A ∪ B|` per pair (L87-97);
emits edges with `jaccard ≥ 0.5` (L202).

For each edge, the upstream is the session with earlier `first_ts_ns`
(L203-204) — directionality inferred from temporal precedence.

### STATE
- `PipelineSession g_slots[1024]` (L33) — `{fingerprint, first_ts_ns,
  last_ts_ns, event_count, n_shapes, shapes[32]}`.
- `g_enabled`, `g_initialized`, `g_session_count`, `g_edge_count`,
  `g_current_slot`, `g_last_ts_ns` (L35-40).

### CONCURRENCY
Single-consumer Stage-1. Shape-set is non-atomic; safe under
single-consumer.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L539 (observe), L929 (init).
- OUTBOUND: `cipher_sense.h::cipher_sense_get_type` (L174, L207-208).
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L5: "tagged upstream/downstream by first_ts_ns" — matches L203.
- L23: `JACCARD_THRESHOLD = 0.5f` — fixed threshold; documented at
  the constant.
- `REPORT_MAX_SESSIONS = 256` (L22) — pairwise cap is O(256²) =
  65 K comparisons at report time. Each comparison is up to 32²
  =1024 element equality checks (L90-94). Worst case ≈ 67 M ops
  per report — still well below 1 ms on modern hardware.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY of inter-session structure. PIPELINE produces
the only Wave-4 output that is a *graph* (sessions × edges); the
others are per-session scalars.

### FUSION POINTS

**Signal.** Per-event shape insertion; session pairing detected
at report time, not on the hot path.

**Output.** Telemetry (JSON graph: sessions, edges with Jaccard
similarity).

**Hot-path cost.** Per-event: linear scan of up to 32 shapes,
optional append. ~40 ns at low fill, ~150 ns at saturation.

**Fusion mode: `stage1-ring-consumer`.** Defended.

**Classifier-input coupling.** SENSE classification is already
used for pretty-printing (L174). A v1.5 could consume a
`tenant_pipeline_id` to pre-group sessions known to belong to the
same logical pipeline. SHIM.

**Fallback today.** Pure inference; no a-priori labelling.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS.**

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_trace.cpp

### PURPOSE
Op 28 — bounded kernel-trace exporter. Fixed 8192-entry ring; one
record per event; flushes to JSONL at report time. The only
Wave-4 observer that emits **per-event** structured output. File
header L1-3.

### PUBLIC SURFACE
- `int cipher_trace_init(void)` — L36.
- `void cipher_trace_observe(const CipherRingEntry* ev)` — L50.
- `uint64_t cipher_trace_written(void)` — L68.
- `uint64_t cipher_trace_dropped(void)` — L73.
- `void cipher_trace_report(void)` — L77.

### CONTROL FLOW
`observe` (L50-66): increment `g_written` (relaxed); if `idx >=
CAPACITY` (8192), increment `g_dropped` and silently return
(L54-57); else copy the relevant ring-entry fields into a local
`Rec` slot (L58-65). The Rec is 56 bytes (`sequence`,
`timestamp_ns`, `func_ptr_hash`, `params_hash` = 32 bytes; plus
`kernel_class` + 6×`grid/block` u32 = 28 bytes; with alignment
that's a hair under 64). Total static footprint:
8192 × 64 ≈ 512 KiB.

`report` (L77-116): writes `/tmp/cipher_trace.jsonl` line-per-
event with all 10 fields, plus a summary `/tmp/cipher_trace_report.json`
with `written`, `dropped`, `capacity`, and a path pointer.

### STATE
- `Rec g_buf[8192]` (L27) — fixed-capacity ring.
- `g_enabled`, `g_initialized`, `g_written`, `g_dropped`
  (L29-32).

### CONCURRENCY
Single-consumer Stage-1. The `g_written.fetch_add(1)` returns the
pre-add index, so two consumers would each write to distinct
slots (it's slot-safe), but record content fields are plain — a
multi-consumer would need a CAS-published `Rec.sequence` watermark.
v1 single-consumer makes this moot.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L551 (observe), L934 (init).
- OUTBOUND: stdio.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-3 header matches code.
- The 8192-entry bound is hard-coded (L15). For a multi-tenant
  long-running deployment, this fills in milliseconds and the
  remainder is logged as `dropped`. Honest cap; documented by
  emitting `dropped` in the report.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY. TRACE is the per-event firehose. Useful for
post-hoc analysis and shape-distribution histograms.

### FUSION POINTS

**Signal.** Every Stage-0 ring entry (no gate).

**Output.** Log (`/tmp/cipher_trace.jsonl`) + counters (`written`,
`dropped`).

**Hot-path cost.** Per-event: one atomic fetch_add, one branch,
one 56-byte structured copy. ~30 ns/event when bucket is unfilled,
~10 ns once dropped (early-out at L54).

**Fusion mode: `stage1-ring-consumer`.** Defended.

**Classifier-input coupling.** None today. A v1.5 could mask
high-frequency low-value kernel classes (filtered) and keep
substitution-candidate kernels — would extend the effective
capacity. SHIM.

**Fallback today.** Drop-on-full.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS.**

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_continuity.cpp

### PURPOSE
Op 19 — per-session attention-region tracking + manifest gate. The
v2 design intent: at session checkpoint time, the manifest tells a
Tier-A worker what KV regions to snapshot. v1: tracking only; no
KV capture, no pinned memory. File header L1-5.

### PUBLIC SURFACE
- `int cipher_continuity_init(void)` — L114.
- `void cipher_continuity_observe(const CipherRingEntry* ev)` — L130.
- `unsigned cipher_continuity_session_count(void)` — L177.
- `unsigned cipher_continuity_manifest_count(void)` — L181.
- `void cipher_continuity_report(void)` — L185.

### CONTROL FLOW
`observe` (L130-175): same idle-reset session allocation as LOOP
and PIPELINE. Per-event: increment `event_count` and
`attn_events_total`; compute `region_key = fnv1a64(func_ptr_hash ||
grid_x*grid_y)` (L64-68); `upsert_region(s, key, sequence)`.

`upsert_region` (L90-110): linear scan of 32 regions; on hit update
`last_seq`, increment two counts (`count`, `count_since_snap`); on
miss insert into first empty slot; on full silently drop (I4).

Every `SNAPSHOT_EVERY=500` events (L21), AND if the session is
classified as `AGENT_AUTONOMOUS`, fire a manifest: increment
`manifest_count`, record `last_manifest_seq`, clear
`count_since_snap` across all 32 regions (L162-174). This is the
"v2 snapshot trigger" that today only counts but in v2 would
actually drive KV snapshot.

### STATE
- `ContinuitySession g_slots[1024]` (L42) — `{fingerprint,
  first/last_ts_ns, attn_events_total, event_count, manifest_count,
  last_manifest_seq, regions[32]}`. Region = `{key, first_seq,
  last_seq, count, count_since_snap}`.
- `g_enabled`, `g_initialized`, `g_session_count`,
  `g_manifest_count`, `g_current_slot`, `g_current_fp`,
  `g_last_ts_ns` (L44-50).

### CONCURRENCY
Single-consumer Stage-1.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L536 (observe), L928 (init).
- OUTBOUND: `cipher_sense.h::cipher_sense_get_type` (L165) —
  the manifest gate.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L155-156 comment: "v1 tracks *all* dispatch events (KV state
  advances on every decode, not only attention kernels)". The
  variable name `attn_events_total` is misleading — it counts all
  events, not just attention. The semantic is "events between
  manifests"; the field name is a v1 leftover from a design draft.
  Cosmetic but worth noting.
- The manifest gate (L162-166) explicitly restricts manifests to
  AGENT sessions — sensible: human-interactive sessions don't have
  the long-context KV that warrants Tier-A checkpointing.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY of session checkpoint cadence. CONTINUITY
counts manifest opportunities; v2 would do the actual checkpoint.

### FUSION POINTS

**Signal.** Per-event ring; gate fires every 500 events for
AGENT sessions.

**Output.** Counter (`manifest_count` global + per-session) +
telemetry (per-session region list at report).

**Hot-path cost.** Per-event: ring push + 32-slot linear scan.
~80 ns/event. Plus an occasional SENSE classification read +
zero-fill of 32 region count_since_snap fields every 500 events.

**Fusion mode: `stage1-ring-consumer`.** Defended.

**Classifier-input coupling.** Already reads
`cipher_sense_get_type`. A v1.5 could consume a
`tenant_checkpoint_cadence` to vary `SNAPSHOT_EVERY` per session.
SHIM.

**Fallback today.** Fixed `SNAPSHOT_EVERY = 500`; AGENT-only.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** for v1. REFACTOR-required for v2
(actual KV checkpoint) — out of Wave-4 scope.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_pulse.cpp

### PURPOSE
Op 22 — hardware fault early warning. Stage 1: per-(M,K,N) Welford
on inter-event ITL with a long baseline frozen at n=100. Stage 2:
NVML ECC delta read at evaluation. Score = (sig_drift?1:0) +
(sig_ecc?1:0), max=2. Score 1 → INFO log; score ≥ 2 → JSON alert.
Signal 2 (per-substitution max_diff) **deferred** — requires a
Stage-0 sentinel hook in Op 3 SUBSTITUTE which is not yet wired.
File header L1-9.

### PUBLIC SURFACE
- `int cipher_pulse_init(void)` — L305.
- `void cipher_pulse_observe(const CipherRingEntry* ev)` — L327.
- `void cipher_pulse_evaluate(void)` — L360.
- `void cipher_pulse_force_evaluate(void)` — L361.
- `int cipher_pulse_score(void)` — L363.
- `unsigned cipher_pulse_alert_count(void)` — L364.
- `const char* cipher_pulse_severity(void)` — L365.
- `void cipher_pulse_inject_drift_for_shape(uint32_t M, uint32_t K,
   uint32_t N, double mean_inflation_factor)` — L369.
- `void cipher_pulse_inject_ecc_delta(uint64_t)` — L384.
- `void cipher_pulse_clear_injection(void)` — L388.
- `void cipher_pulse_report(void)` — L393.

### CONTROL FLOW
`observe` (L327-358): dispatch counter increment (L330);
non-GEMM events skipped (L331); unpack `(M,K,N)` from
`params_hash` packed via the `0xC...` marker (L82); `find_or_alloc`
of a 64-slot hash table with 8-probe linear probing (L93-118);
on first event for the shape, just record `prev_ts_ns` and exit
(L342-345); on subsequent events, compute `itl = now -
prev_ts_ns`, Welford-update `mean_ns` and `M2_ns`. **At
n == BASELINE_FREEZE_N (= 100)**, snapshot `mean_ns` and
`sqrt(M2_ns/n)` into `long_baseline_mean` / `long_baseline_sd`
— the baseline is *frozen* (L353-357). Subsequent updates
continue to refine `mean_ns` but never touch the baseline.

`evaluate_impl` (L276-301): rate-limited by `EVAL_DISPATCH_GATE =
1000` dispatches between evaluations (L280); `try_resolve_nvml()`
(idempotent, L283); evaluate signal 1 drift (L177-200) and signal 3
ECC (L202-216); score = sum; re-alert dedup at 60 s of same severity
(L296-298); emit alert via `emit_alert` (L227-274).

`emit_alert` (L227-274): score 1 → append JSONL line to
`/tmp/cipher_pulse.log`; score ≥ 2 → write
`/tmp/cipher_pulse_alert.json` with severity "MEDIUM" (and a
`score_ceiling_v1: 2` flag — explicitly documents that
CRITICAL is unreachable in v1).

NVML resolve `try_resolve_nvml` (L126-175): RTLD_NOLOAD only — does
NOT load NVML if the telemetry layer hasn't. If NVML present:
`nvmlDeviceGetHandleByIndex_v2` + `nvmlDeviceGetMemoryErrorCounter`
(types CORRECTED, VOLATILE, DEVICE_MEMORY). On NVML failure /
unsupported, ECC signal is disabled and the score is just (drift?1:0)
ceiling 1 (not the score=2 max).

### STATE
- `PulseShape g_shapes[64]` (L51) — `{key, M, K, N, prev_ts_ns,
  mean_ns, M2_ns, n, long_baseline_mean, long_baseline_sd,
  inject_factor}`.
- `PulseState g_pulse` (L68) — `{dispatch_count, last_eval_count,
  ecc_baseline, ecc_inject, score, alert_count, last_alert_ns,
  last_severity, nvml_resolved, nvml_handle, nvml_device,
  fn_get_ecc}`.

### CONCURRENCY
Single-consumer Stage-1 for `observe`. The `evaluate*` functions
are called from Stage-2 or on demand; they share `g_pulse`'s
atomics (dispatch_count, score, alert_count, last_severity) with
the observer. Release/acquire ordering on the score store at L291
makes the eval write visible to any reader of `cipher_pulse_score`.

The NVML resolution at L126-175 is gated by `nvml_resolved` (L127)
— first-caller-wins. Subsequent calls early-exit. The `fn_get_ecc`
field is plain (non-atomic); safe because the resolver is
idempotent under single Stage-2 caller.

`last_alert_ns` is plain (L60); written only inside `emit_alert`
which is only called from `evaluate_impl`. Safe.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L524 (observe), L923 (init).
- OUTBOUND: dlopen(libnvidia-ml.so.1, RTLD_LAZY|RTLD_NOLOAD);
  `nvmlDeviceGetHandleByIndex_v2`, `nvmlDeviceGetMemoryErrorCounter`.
- ioctls: NONE. Uses NVML directly.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L7-9 explicitly defers Signal 2 (max_diff degradation per
  substitution); requires Stage-0 sentinel hook in Op 3
  SUBSTITUTE. The deferred-reason JSON field at L262-263 documents
  this explicitly.
- L29-32 NVML constants are inlined "matches NVML headers; we
  don't include them" — to avoid the NVML header dependency. Risk:
  if NVML changes a value, the observer reads the wrong counter.
  v1 has a comment but no version check. Cosmetic risk.
- L130-131 RTLD_NOLOAD-only: PULSE will not activate NVML on its
  own. If `cipher_thermal_feedback` (Stage 11) hasn't loaded NVML
  by the time PULSE's first `try_resolve_nvml` fires, ECC signal
  is permanently disabled. There is no retry. The framing is
  honest ("telemetry off?") at L133-135.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY (drift + ECC). Score ≥ 2 is intended as the
operator-page-trigger. Today emits to JSON; no Stage-0 wiring.

### FUSION POINTS

**Signal.** Per-event ITL drift on hot shapes (Stage 1) +
periodic ECC counter check (Stage 2).

**Output.** Alert (`/tmp/cipher_pulse_alert.json` at score ≥ 2;
JSONL `/tmp/cipher_pulse.log` at score 1) + counters.

**Hot-path cost.** Per-event observe: M/K/N unpack + 8-probe
find_or_alloc + Welford update. ~70 ns/event at low fill; the
`find_or_alloc` returns the **first probed slot** on contention
(L114) which can cause hash collisions — acceptable since the
slot table is only 64 entries.

Eval cost: 64-slot scan for drift (L180-200) + 1 NVML call
(L206-209). NVML calls are typically 10s of µs; PULSE rate-limits
to once per 1000 dispatches.

**Fusion mode: `hybrid`** — Stage 1 ring-consumer (observe) +
Stage 2 background-polling (evaluate). The two halves share the
PulseState struct.

**Classifier-input coupling.** Could read a
`shape_is_hot_path` boolean from the classifier to focus the
Welford on the highest-value shapes and reduce 64-slot scan
cost. SHIM.

**Fallback today.** All 64 slots equal weight; no classifier.

**ABI/IOCTL touchpoints.** NONE. PULSE could in principle consume
the kmod's per-tenant launch counters from `tenant_snapshot`
(Wave 3 nr 8) to detect cross-tenant drift, but does not today.

**Fusion class: PORT-AS-IS** for v1 (hot-path cost is the lowest
of the per-event observers because of the kernel_class gate). The
Signal 2 wiring is a REFACTOR in a different op (Op 3 SUBSTITUTE),
not in PULSE itself.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_straggler.cpp

### PURPOSE
Phase 3 — local v1 straggler detection. Per-AllReduce telemetry,
written to per-rank JSONL files (`/tmp/cipher_straggler_rank_<R>.jsonl`).
Local slowdown detector via per-bucket EMA. Algorithm hint biasing
ORCHESTRATE toward RING when sustained slow. Cross-rank attribution
deferred to an offline aggregator (`tools/straggler_aggregate.py`).
File header L1-10.

### PUBLIC SURFACE
- `int cipher_straggler_init(void)` — L139; returns mode enum.
- `void cipher_straggler_observe(uint64_t bytes, uint64_t dur_ns,
   int chosen_algo)` — L178.
- `int cipher_straggler_local_slowdown_active(void)` — L238.
- `unsigned cipher_straggler_event_count(void)` — L241.
- `int cipher_straggler_my_rank(void)` — L244.
- `int cipher_straggler_world_size(void)` — L245.
- `const char* cipher_straggler_status_string(void)` — L246.
- `int cipher_straggler_algo_hint(uint64_t bytes)` — L255.
- `void cipher_straggler_inject_dur(uint64_t bytes,
   uint64_t dur_ns_inflated)` — L263.
- `void cipher_straggler_clear_injection(void)` — L267.
- `void cipher_straggler_report(void)` — L271.

### CONTROL FLOW
`init` (L139-176): tri-state mode env `CIPHER_STRAGGLER` —
`OBSERVE` / `ACTIVE` / unset (= OFF) (L143-150). Reads
`RANK / OMPI_COMM_WORLD_RANK / SLURM_PROCID / PMI_RANK` for
rank id (L77-85); reads the world-size pendants for world size
(L86-93). Truncates per-rank JSONL log on (re)start (L161-163).

`observe(bytes, dur_ns, algo)` (L178-236): the unique-in-Wave-4
signature — driven NOT by a CipherRingEntry but by NCCL bucket
telemetry. Called from `cipher_nccl.cpp` L287 (via dlsym).

Per-call:
1. mode-off short-circuit (L181); bytes==0 reject (L182).
2. `bucket_index_for(bytes)` (L71-75) → 0..6 by powers of 2/16.
3. Test injection consumed once (L188-191).
4. ns_per_byte = dur / bytes (L193).
5. EMA update with `alpha = 0.10` (L195-200). Bootstraps to the
   first observed value at n==0 (L195-196).
6. Slowdown detection (L203-232):
   - If calls > 5 AND ema_before > 0 AND ns_per_byte > 1.5 ×
     ema_before, increment `consec_slow`, zero `consec_normal`.
   - If `consec_slow >= 5`, raise the global slowdown flag
     atomically; first 0→1 transition increments the event
     counter and (if 60 s since last alert) emits the JSON alert
     (L213-223).
   - Otherwise the inverse: increment `consec_normal`, on
     reach 5 with `slowdown_active==1` clear the flag (L228-231).
7. Always emit a JSONL telemetry line (L234-235).

### STATE
- `StragState g_strag` (L62) — `{mode (atomic), slowdown_active
  (atomic), event_count (atomic), my_rank, world_size,
  rank_log_path[128], buckets[7], last_alert_ns}`.
- `BucketStats` (L43-50) — `{ema_ns_per_byte, calls, consec_slow,
  consec_normal, inject_dur_ns}`.

### CONCURRENCY
NCCL operates on a single rank thread; `cipher_straggler_observe`
is called serialised per rank. The atomics on `mode`,
`slowdown_active`, `event_count` are belt-and-suspenders for the
read-side API (multi-thread queries of `local_slowdown_active`,
`algo_hint`).

### DEPENDENCIES
- INBOUND: `cipher_nccl.cpp` L287 (resolves via
  `dlsym(RTLD_DEFAULT, "cipher_straggler_observe")`); also called
  from `cipher_10ops_impl.cpp` L926 (init only).
- OUTBOUND: clock_gettime, fopen/fprintf.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-10 honest framing: cross-rank attribution NOT implemented,
  offline aggregator path documented.
- The bucket boundaries (L31-39) match NCCL conventions and
  `cipher_nccl.cpp::g_buckets`.
- `SLOWDOWN_FACTOR = 1.5` (L25), `RUN_GATE = 5` (L26),
  `CLEAR_RUN = 5` (L27): all documented as v1 heuristics.
- `algo_hint(bytes)` returns RING (algo=1) without inspecting
  bytes (L259-261) — the comment "RING is the safe tail choice"
  documents this as a deliberate v1 simplification.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY + ORCHESTRATE-bias signal. The `algo_hint` is
the only Wave-4 observer that returns an actuation suggestion
(int return), not just void+counters. Today the consumer is
ORCHESTRATE / NCCL tuner v4.

### FUSION POINTS

**Signal.** Per-AllReduce bucket-relative slowdown sustained
over ≥ 5 consecutive calls.

**Output.** Alert (`/tmp/cipher_straggler_alert.json`) + log
(`/tmp/cipher_straggler_rank_*.jsonl`) + algorithm hint (int
return).

**Hot-path cost.** Per AllReduce call (NOT per cuLaunch): one
bucket-index lookup (7 comparisons), one EMA update, two
fopen+fprintf per call (the JSONL line — every call). The fopen
churn is the heaviest cost in Wave 4 — `O(IO)`/AllReduce ≈ 10 µs.
At 1 AllReduce/forward-pass for distributed training this is
negligible; for very small AllReduces it could dominate.

**Fusion mode: `background-thread-polling` companion to NCCL
callbacks**, though strictly the observer is event-driven by
NCCL completion, not by a CipherRingEntry. The taxonomy in
this wave treats it as its own thing — "NCCL-driven observer"
— but for fusion purposes the closest match is
`background-thread-polling` because it runs on the NCCL completion
thread and has no Stage-1 ring dependency.

**Classifier-input coupling.** Could consume a
`session_collective_priority` to weight slow alerts. SHIM.

**Fallback today.** Env-driven rank + world size; hard-coded
gates.

**ABI/IOCTL touchpoints.** NONE today.

**Fusion class: PORT-AS-IS** for v1. The fopen-per-AllReduce
cost is a soft regression for very small AllReduces; v1.5 should
buffer to a ring and flush in a background thread — small
REFACTOR.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_topology.cpp

### PURPOSE
Op 25 — v1 NVLink/PCIe peer adjacency. Static inference at init
using `cudaGetDeviceCount` + `cudaDeviceCanAccessPeer(i, j)`.
Observer is a no-op. File header L1-4.

### PUBLIC SURFACE
- `int cipher_topology_init(void)` — L26.
- `void cipher_topology_observe(const CipherRingEntry* ev)` — L57;
  **explicit no-op** (L58-59).
- `unsigned cipher_topology_device_count(void)` — L62.
- `void cipher_topology_report(void)` — L66.

### CONTROL FLOW
`init` (L26-55): env-gate; on enable, queries `cudaGetDeviceCount`
(L37), then double-loop `cudaDeviceCanAccessPeer(i, j)` populating
the 16×16 adjacency matrix (L42-49). Records `g_device_count`.
On any CUDA failure, `dev_count = 0` and the matrix stays all-zero.

`observe`: no-op.

`report`: emits matrix + edge count (L86-89).

### STATE
- `g_adj[16 × 16]` (L22) — uint8_t adjacency.
- `g_enabled`, `g_initialized`, `g_device_count` (L19-21).

### CONCURRENCY
N/A; `init` runs single-threaded before observation; observation
is no-op.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L933 (init).
  `cipher_comply.cpp` L57, L77 (device_count).
- OUTBOUND: `cudaGetDeviceCount`, `cudaDeviceCanAccessPeer`.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-4 "static inference: init-time" matches L42-49.
- The matrix is upper-bounded at 16 devices (L17). H100 pods rarely
  exceed 8.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY (informational). COMPLY reads `device_count` for
the report (not the gate).

### FUSION POINTS

**Signal.** None at runtime — static inference only.

**Output.** Telemetry (JSON matrix + edge count).

**Hot-path cost.** ZERO. `observe` is `(void)ev; return`.

**Fusion mode: `stage1-ring-consumer`** (by registration; observe
is no-op).

**Classifier-input coupling.** None. Topology is a hardware
fact, not a classifier output.

**Fallback today.** N/A.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS.**

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_thermostat.cpp

### PURPOSE
Op 20 — temperature-driven workload regulation. Stage 1: per-shape
Welford on inter-event ITL (64-entry table, 30-event baseline).
Stage 2: polls every 500 ms, combines `gpu_temp_c > 75 °C AND
rising` with `any shape > +10 % over baseline` to set the
`aggressive` flag. Clear gate: `temp < 72 °C AND no shape > +5 %`.
Flag is currently log-only — Stage-0 consumption is deferred.
File header L1-6.

### PUBLIC SURFACE
- `int cipher_thermostat_init(void)` — L144.
- `void cipher_thermostat_observe(const CipherRingEntry* ev)` — L166.
- `void cipher_thermostat_poll(void)` — L238.
- `void cipher_thermostat_force_poll(void)` — L239.
- `int cipher_thermostat_aggressive_active(void)` — L241.
- `unsigned cipher_thermostat_event_count(void)` — L245.
- `unsigned cipher_thermostat_drift_shape_count(void)` — L249.
- `void cipher_thermostat_inject_temp(float celsius)` — L253.
- `void cipher_thermostat_inject_drift_for_shape(...)` — L257.
- `void cipher_thermostat_clear_injection(void)` — L272.
- `void cipher_thermostat_report(void)` — L277.

### CONTROL FLOW
`observe` (L166-191): kernel_class == 0 (GEMM) only (L168); unpack
M/K/N (L171); `find_or_alloc` 8-probe (L75-99); on first event,
just record `prev_ts_ns` (L179-181); on subsequent, Welford-update.
At `n == BASELINE_FREEZE_N = 30` (L25), freeze `baseline_mean_ns`
(L190).

`thermostat_poll_impl` (L193-236): once per 500 ms unless
`bypass_cadence` (L197-200); `read_temp()` returns the
`g_cipher.liquid.device->hw.gpu_temp_c` field set by Stage 11
thermal_feedback (L112-113), with the test-injection override
in front (L108-109). Two signals:
- `sig_temp = (curr > 75 AND curr > prev)` (L207). Note the
  rising-temp gate prevents flapping at steady state.
- `sig_drift = any_shape_drifting(1.10, ...)` (L210), counting
  shapes whose `effective_mean / baseline > 1.10`.

Set/clear rules:
- Set: `sig_temp AND sig_drift` AND not already aggressive → set
  to 1, increment event_count, log (L215-223).
- Clear: aggressive=1 AND `curr < 72 °C AND no shape > +5 %` → set
  to 0, log (L224-235).

The hysteresis (fire 75/+10%, clear 72/+5%) is intentional to
prevent thrashing.

### STATE
- `ShapeWelford g_shapes[64]` (L38) — same shape as PULSE's.
- `ThermoState g_thermo` (L50) — `{last_temp_c, prev_temp_c,
  inject_temp_c, aggressive, event_count, drift_shape_count,
  last_poll_ns}`. Atomic floats for thread safety.

### CONCURRENCY
`observe` is single-consumer Stage-1. `poll_impl` is Stage-2 or on
demand. Both touch the shape table — `observe` writes via the
8-probe `find_or_alloc`, `poll` reads via `any_shape_drifting`.
The non-atomic `mean_ns` / `M2_ns` reads in the poll are
strictly racy with the observer's updates — but the poll is
order-of-magnitude-tolerant (it's checking ratios), so a transient
inconsistent read is benign.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L521 (observe), L922 (init).
- OUTBOUND: `g_cipher.liquid.device->hw.gpu_temp_c` (set by Stage
  11 thermal_feedback per CLAUDE.md). The comment at L110-111
  explicitly flags the extern linkage requirement.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-6 header matches.
- L157 "v1: aggressive flag set but not yet consumed by Stage 0" —
  flagged.
- The `cipher.h` extern coupling at L110-114 makes THERMOSTAT
  the most tightly-bound observer to the Stage 11 thermal
  feedback infrastructure. If `g_cipher.liquid` is not
  initialized, `read_temp` returns 0.0f (L115), the temp signal
  will never fire, and the observer effectively degrades to
  drift-only.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY of thermal-correlated drift. The aggressive flag
is the canonical "throttle the substitute-actuator aggressiveness"
signal but is not yet consumed.

### FUSION POINTS

**Signal.** Temperature crossing 75 °C while rising, AND any
GEMM shape with mean ITL drifted +10 % over baseline.

**Output.** Counter (`event_count`, `drift_shape_count`) + alert
(stderr log on flag transition).

**Hot-path cost.** Per-event observe (GEMM only): M/K/N unpack
+ 8-probe + Welford update. ~70 ns/event.

Poll cost: 64-slot scan + one temp read. ~3 µs every 500 ms.

**Fusion mode: `hybrid`** — Stage 1 ring-consumer (observe) +
Stage 2 background-polling (poll). Same architecture as PULSE.

**Classifier-input coupling.** Could read
`silicon_thermal_class` from a classifier to discriminate "this
H100 is a known hot-runner" vs nominal silicon. SHIM.

**Fallback today.** Hard-coded 75/72 °C gates.

**ABI/IOCTL touchpoints.** NONE. Could consume the kmod's per-
device thermal field from `cipher_gpu_state` (Wave 3 spine) if a
new tenant_snapshot field exposed it — would obviate the
`g_cipher.liquid` direct read.

**Fusion class: PORT-AS-IS** for v1; SHIM-required to consume
the aggressive flag downstream.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_shield.cpp

### PURPOSE
Op 14 — Stage 1 latency-protection monitor. For HUMAN_INTERACTIVE
sessions: writes `CIPHER_BAND_PROTECTED` via
`cipher_sm_set_priority` (the only Wave-4 observer that *writes*
an actuation hint), tracks per-session ITL P50/P95 over a 20-event
rolling window, counts >2σ jitter events, raises
`oracle_aggressive` when P95/P50 > 3.0. File header L1-10.

### PUBLIC SURFACE
- `int cipher_shield_init(void)` — L150.
- `void cipher_shield_observe(const CipherRingEntry* ev)` — L169.
- `void cipher_shield_arbitrate_scan(void)` — L199; thin wrapper
  around `cipher_sm_priority_scan_log()`.
- `unsigned cipher_shield_jitter_event_count(void)` — L204.
- `unsigned cipher_shield_smoothed_session_count(void)` — L208.
- `unsigned cipher_shield_priority_assignments(void)` — L212.
- `void cipher_shield_report(void)` — L216.

### CONTROL FLOW
`observe` (L169-197): SENSE-session gate (L173-174); 8-probe
`find_or_alloc` keyed on session id (L62-88); pick `now` from
`ev->timestamp_ns` or clock_gettime (L179-184).

**Protection 1 (band assignment)** (L188-194): if not yet
priority-assigned and SENSE classification == HUMAN_INTERACTIVE,
call `cipher_sm_set_priority(sid, CIPHER_BAND_PROTECTED)`,
flip the per-slot `priority_assigned` flag, increment
`g_priority_total`.

`update_itl` (L90-146) — the per-event ITL machinery:
1. compute itl from `now - last_ts_ns` (L91-92).
2. Welford-update `itl_mean_ns` and `itl_M2_ns` (L94-97).
3. push `now` into 20-entry rolling window
   (`ts_window[head]`, L100-102).
4. once 5+ samples available: materialize gaps from the window
   (L107-117); sort (L119); compute `p50` and `p95` (L120-121).
5. **Jitter (Protection 2)**: if `itl >= 5 × p50`, increment
   `jitter_events`; 0→1 transition on `cache_aggressive` triggers
   `g_jitter_total` increment (L127-132). The comment L123-127 is
   the load-bearing rationale: a simple Welford-2σ collapses under
   bimodal "fast burst + slow gap" patterns; the median-relative
   gate is robust to that.
6. **Smoothing (Protection 3)**: if `p95/p50 > 3.0`, set
   `smoothed=1` and `oracle_aggressive=1` (L135-141). Sticky.

### STATE
- `ShieldSession g_shield[256]` (L49) — `{session_id (atomic),
  ts_window[20], head, full, last_ts_ns, itl_mean_ns, itl_M2_ns,
  n_itl, jitter_events, smoothed, oracle_aggressive (atomic),
  cache_aggressive (atomic), priority_assigned}`.
- `g_jitter_total`, `g_smoothed_total`, `g_priority_total`
  (L53-55) — globals.

### CONCURRENCY
Single-consumer Stage-1 in `observe`. The two atomic flags
`oracle_aggressive` and `cache_aggressive` are exported for v2
hot-path readers (Op 3 SUBSTITUTE in the cache case, the oracle
in the smoothed case).

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L513 (observe), L920 (init).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session`,
  `cipher_sense_get_type`; `cipher_green_ctx.h::cipher_sm_set_priority`,
  `cipher_sm_get_priority`, `cipher_sm_priority_scan_log`.
- ioctls: NONE directly; `cipher_sm_set_priority` is the
  green-ctx priority-band API audited in Wave 2 — that's a
  userspace table, not a kmod ioctl.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-10 header lists 3 protections with explicit v1 deferral on
  P2 and P3 ("flag set but not yet consumed").
- L122-127 comment on jitter algorithm choice — careful design
  documentation.
- L188-194: P1 actuation (band write) is **already wired** today.
  This is the only Wave-4 observer with active actuation.

### CONTRIBUTION TO SYSTEM
v1 ACTUATION (P1 band write) + OBSERVABILITY (P2/P3 jitter
counts). SHIELD is the only Wave-4 observer that closes the loop
to an actuator (the green-ctx priority table).

### FUSION POINTS

**Signal.** Per-event ITL; P1 fires once per session on first
HUMAN_INTERACTIVE classification. P2 fires per-event on
`itl >= 5×p50`. P3 fires once per session on `p95/p50 > 3.0`.

**Output.** Side-effect (band write) + counter (jitter,
smoothed, priority assignments) + alert (none today — would be
the Op-3 / oracle hot-path consumer).

**Hot-path cost.** Per-event:
- session find_or_alloc: 8-probe hash, ~20 ns at low fill.
- Welford: 4 doubles, ~10 ns.
- Window push + sort: 20-element radix-irrelevant sort, ~100 ns
  per event (the sort is the dominant cost).
- Band write (one-shot per session): ~20 ns into the green-ctx
  table.

Total ~140 ns/event for HUMAN_INTERACTIVE sessions; ~60 ns for
others (no SENSE-classified type → no band write, no sort either
since we still update Welford but skip the gap analysis only
once window has 5 events — actually the sort runs unconditionally
under L107).

Actually re-reading L107: `if (s.full || s.head >= 5)` — the sort
runs as soon as 5 events are buffered. So the cost is paid on
event #5 onward for *all* sessions, not just human-interactive.
This is fine; just want to call out the cost.

**Fusion mode: `stage1-ring-consumer`** + a single actuator
write (the band table). The actuator is userspace-resident.

**Classifier-input coupling.** Already reads SENSE classification
(L189). A v1.5 could also read a `tenant_latency_sla` to override
the BURST_P95_P50_GATE and JITTER_SIGMA_GATE per session. SHIM.

**Fallback today.** SENSE classifier is required; without it
P1 never fires.

**ABI/IOCTL touchpoints.** NONE. `cipher_sm_set_priority` is
userspace-only.

**Fusion class: PORT-AS-IS** for v1 (already wired). SHIM for
per-tenant SLA thresholds in v1.5. The P2/P3 wiring into Op 3 /
oracle is REFACTOR in those ops, not here.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_sustain.cpp

### PURPOSE
Op 15 — KV-cache pressure detector. Per session, maintains a 20-pt
rolling window of `(decode_step_idx, ITL_ns)` and recomputes a
closed-form OLS slope every 10 decode events. When `slope >
50,000 ns/step`, sets `sustain_compress_flag` (atomic). The flag
is for a v2 Op 3 SUBSTITUTE hook (KV compression actuation); v1
detection only. File header L1-9.

### PUBLIC SURFACE
- `int cipher_sustain_init(void)` — L109.
- `void cipher_sustain_observe(const CipherRingEntry* ev)` — L129.
- `unsigned cipher_sustain_pressure_event_count(void)` — L186.
- `unsigned cipher_sustain_compress_flag_count(void)` — L190.
- `double cipher_sustain_last_slope_ns_per_step(uint64_t)` — L194.
- `void cipher_sustain_report(void)` — L206.

### CONTROL FLOW
`observe` (L129-184): kernel_class == 0 only (L132); unpack M/K/N
(L134-136); compute `batch = min(M,N)`, skip if batch > 8 (L139)
— decode-only.

Session find_or_alloc by `cipher_sense_current_session()` (L141-143);
8-probe hash, sticky per session.

Per decode-event:
1. Compute `itl = now - prev_ts_ns` (L147-152).
2. Push `(decode_step_idx, itl)` into 20-slot rolling window
   (L156-161). Step idx is a u16 — wraps at 65 535 decodes.
3. Increment `decode_step_idx` and `steps_since_eval` (L160-161).
4. Every 10 decodes, run `regress_slope_ns_per_step` (L90-105) —
   closed-form OLS over the window. Stores `last_slope`.
5. AGENT exception (L171-175): if session is AGENT_AUTONOMOUS and
   `decode_step_idx > 500`, multiply threshold by 3.0. Rationale:
   long agent reasoning has structurally higher KV pressure and
   shouldn't trip the gate.
6. If `slope > threshold`, increment `pressure_events` and
   `g_pressure_total`; 0→1 transition on `sustain_compress`
   increments `g_compress_flag_total` (L177-183).

### STATE
- `SustainSession g_sustain[256]` (L42) — `{session_id, prev_ts_ns,
  decode_step_idx, step_buf[20], itl_buf[20], head, full,
  steps_since_eval, last_slope, sustain_compress (atomic),
  pressure_events}`.
- `g_sustain_enabled`, `g_sustain_initialized`, `g_pressure_total`,
  `g_compress_flag_total` (L44-47).

### CONCURRENCY
Single-consumer Stage-1. The slope-result `last_slope` is plain;
read by the `cipher_sustain_last_slope_ns_per_step` accessor —
inconsistent reads are tolerable for a diagnostic.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L518 (observe), L921 (init).
- OUTBOUND: `cipher_sense.h::cipher_sense_current_session`,
  `cipher_sense_get_type`.
- ioctls: NONE.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L1-9 header matches code.
- L25 `SLOPE_THRESHOLD_NS = 50000.0` — "0.5 ms per 10 steps". v1
  heuristic.
- L26 `AGENT_RELAX_FACTOR = 3.0` and L27 `AGENT_DECODE_GATE = 500`
  — the AGENT-relax window; matches spec annotation at L26.
- The Welford-less OLS at L90-105 uses long-double accumulation
  to control catastrophic cancellation under the n×Σxy − Σx Σy
  formula. Defensive.

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY (slope) + v2 SUBSTITUTE-actuation signal
(`sustain_compress` flag). Today no Stage-0 reader.

### FUSION POINTS

**Signal.** Per-decode-event slope of ITL over the last 20
decodes; trigger is slope > 50 µs/step (relaxed 150 µs/step for
long-running agents).

**Output.** Counter (`pressure_events` per-session +
`g_pressure_total`; `g_compress_flag_total` for 0→1 sessions).

**Hot-path cost.** Per-event (decode-only): 8-probe find, window
push, integer arithmetic, every 10 events the OLS (20-element
sums, ~200 ns). Decode is comparatively rare (1/step) so the
cost is dominated by find_or_alloc. ~70 ns/decode-event.

**Fusion mode: `stage1-ring-consumer`.** Defended; per-decode
slope is event-driven on the ring.

**Classifier-input coupling.** Already reads SENSE
classification for the AGENT relax. A v1.5 could read a
`tenant_kv_compression_class` to vary the threshold per tenant.
SHIM.

**Fallback today.** SENSE is consulted; if absent, the AGENT
exception just doesn't fire.

**ABI/IOCTL touchpoints.** NONE.

**Fusion class: PORT-AS-IS** for v1; SHIM for per-tenant
thresholds; REFACTOR in Op 3 SUBSTITUTE to consume the
`sustain_compress` flag.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_hibernate.cpp

### PURPOSE
Op 31 — execution-idle SM power gating (Path B). Stage 1:
per-event last-dispatch-timestamp + pre-engage notify. Stage 2:
periodic poll evaluates idle gap and (when permitted) writes
`nvmlDeviceSetPowerManagementLimit`. On the May-13 pod the
write-probe returns `NVML_ERROR_NOT_SUPPORTED`, so HIBERNATE
falls into DEGRADED mode (observability only, no power write).
File header L1-8.

### PUBLIC SURFACE
- `int cipher_hibernate_init(void)` — L174.
- `void cipher_hibernate_observe(const CipherRingEntry* ev)` — L205.
- `void cipher_hibernate_poll(void)` — L223.
- `void cipher_hibernate_force_poll(void)` — L224.
- `int cipher_hibernate_actuation_supported(void)` — L226.
- `int cipher_hibernate_idle_gate_flag(void)` — L229.
- `unsigned cipher_hibernate_idle_event_count(void)` — L232.
- `unsigned cipher_hibernate_pre_engage_count(void)` — L235.
- `const char* cipher_hibernate_status_string(void)` — L238.
- `void cipher_hibernate_inject_idle_gap_ms(unsigned ms)` — L243.
- `void cipher_hibernate_clear_injection(void)` — L246.
- `void cipher_hibernate_report(void)` — L250.

### CONTROL FLOW

`init` (L174-203): env-gate; optional threshold override
`CIPHER_HIBERNATE_THRESHOLD_MS` clamped to [1,1000] (L185-189);
`probe_actuation()` (L94-100) — try resolving NVML, then write
the current power-limit back to itself. The "no-op write" is a
permissions probe: if the kernel denies the write, fail
gracefully. On success, `g_hib.actuation_supported = 1`.

`resolve_nvml` (L65-91): tries NOLOAD first
(L67-68), then full load (L69-70). On full load, calls
`nvmlInit_v2` (L72-73). Caches `nvmlDeviceGetHandleByIndex_v2`,
`nvmlDeviceGetPowerManagementLimit`,
`nvmlDeviceGetPowerManagementLimitConstraints`,
`nvmlDeviceSetPowerManagementLimit` (L74-78). Computes
`reduced_power_mw = base × 0.30`, clamped to `constraint_lo_mw`
(L86-90).

`observe` (L205-221): record `last_dispatch_ns` (L209-211); if
`idle_gate_flag` was raised, drop it and (if actuation supported)
restore base power **synchronously on the event** (L213-220).
This is the "pre-engage notify": the work has resumed, so undo
the power-cut before the work hits the GPU.

`poll_impl` (L139-170): rate-limited to 10 ms cadence (L142-145).
Reads `last_dispatch_ns`, computes `gap = now - last`, compares
to threshold (default 5 ms). State machine:
- Was idle (`was_idle == 1`), still idle (`is_idle == 1`): no-op.
- Was active, now idle: raise flag, increment `idle_event_count`;
  if actuation supported, install signal handlers (L160), write
  `reduced_power_mw` (L161).
- Was idle, now active: drop flag; if actuation supported, write
  `base_power_mw` (L167).
- Was active, still active: no-op.

Signal handlers (L109-119): on `SIGTERM/SIGINT/SIGSEGV/SIGABRT/SIGBUS`,
write `base_power_mw` back (so a crash doesn't leave the GPU power-
gated), then `SIG_DFL` and re-raise. `atexit_restore` (L103-107)
does the same on normal exit.

### STATE
- `HibState g_hib` (L57) — large struct containing all atomics +
  cached NVML pointers + power constraint cache.

### CONCURRENCY
`observe` is single-consumer Stage-1. `poll_impl` runs on Stage-2
or on demand. They share `last_dispatch_ns`, `idle_gate_flag`, and
the NVML write path.

The signal handlers are async-signal-safe by construction: they
touch only cached pointers and atomic loads. Critically, the
handlers do *not* dlsym (would be unsafe in a signal handler);
the symbols are pre-resolved at init.

### DEPENDENCIES
- INBOUND: `cipher_10ops_impl.cpp` L530 (observe), L925 (init).
- OUTBOUND: dlopen libnvidia-ml.so.1 (RTLD_NOLOAD then full);
  `nvmlInit_v2`, `nvmlDeviceGetHandleByIndex_v2`,
  `nvmlDeviceGetPowerManagementLimit{Constraints}`,
  `nvmlDeviceSetPowerManagementLimit`. `sigaction` for the
  cleanup handlers.
- ioctls: NONE. (Could use kmod CIPHER_SET_CLOCK_MHZ nr 10 to
  drop clocks rather than power-limit — different actuation
  vector. See FUSION POINTS.)

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L7-8 explicit pod-degraded framing: NOT_SUPPORTED → observability
  only.
- The 30 %-of-TDP `reduced_power_mw` (L86) is a v1 floor; spec'd
  in the header comment.
- The pre-engage notify (L213) is the load-bearing decision: it
  fires synchronously on the *next* dispatch event after idle,
  meaning the first event after wake-up pays the power-limit-restore
  cost on the producer's path. That cost is one NVML call (~tens of
  µs). For latency-critical first-token-time, this is significant.
  Documented in the file header (L4 "pre-engage notify").

### CONTRIBUTION TO SYSTEM
v1 OBSERVABILITY (idle event count) + ACTIVE actuation when
permitted. On the May-13 pod, DEGRADED. On a host with NVML
write permission, this is the second Wave-4 observer with active
actuation (after SHIELD).

### FUSION POINTS

**Signal.** `now - last_dispatch_ns > threshold_ms` (default 5 ms)
sampled every 10 ms.

**Output.** Counter (`idle_event_count`, `pre_engage_count`) +
side-effect (power-limit write on supported hosts).

**Hot-path cost.** Per-event observe: one release-store on
`last_dispatch_ns`, one exchange on `idle_gate_flag` (almost
always 0→0, fast path). On the rare 1→0 transition, one NVML
write — ~tens of µs. Tail latency hit on first-event-after-idle
when actuation is supported.

**Fusion mode: `hybrid`** — Stage 1 ring-consumer (observe + pre-
engage notify) + Stage 2 background-polling (poll_impl).

**Classifier-input coupling.** Could consume a
`session_min_idle_threshold_ms` per session — research workloads
that have known long pauses could be allowed to idle-gate aggressively;
HUMAN_INTERACTIVE could be forbidden. SHIM.

**Fallback today.** Single global `CIPHER_HIBERNATE_THRESHOLD_MS`
env var.

**ABI/IOCTL touchpoints.** NONE today. Could use the kmod's
`CIPHER_SET_CLOCK_MHZ` (Wave 3 nr 10) as an alternative
actuation vector — that ioctl is unprivileged (T4.3.2), so on
pods where NVML power-limit-write is denied, the kmod-clock
write would still let HIBERNATE actuate. This would also let
HIBERNATE benefit from the same SIGTERM/SIGINT cleanup
infrastructure shared with VOLT (Wave 2 cipher_rt_volt.c L241,
which uses the same 5-signal cleanup pattern).

**Fusion class: REFACTOR-required for v1.5** to add the
kmod-clock fallback actuation path. PORT-AS-IS for v1.

---

# Cross-File Synthesis (Wave 4 specific)

## The four-lane observation map

| Lane | Threading | TUs | Surface | Failure mode |
|---|---|---|---|---|
| **L1 Stage-1 ring** | dedicated shadow thread consuming `ring.read_seq_s1` | shield, sustain, thermostat (observe-half), pulse (observe-half), hibernate (observe-half), loop, continuity, pipeline, guard, trace, fairness, carbon, receipt, comply, topology | `const CipherRingEntry*` (128-byte record, 14 fields) | slow observer back-pressures the ring; producer drops via `cipher_ring_write` returning 0 |
| **L2 Stage-2 cadence** | Stage-2 background thread + force-poll APIs | thermostat::poll, pulse::evaluate, hibernate::poll | poll function (no arg or `bool force`) | rate-limited internally; missed polls just defer the next decision |
| **L3 cublasGemmEx shim** | caller thread of cublasGemmEx (PyTorch's compute thread) | fairness_shm | direct call from `cipher_intercept_cudart.cpp` L867-883 via `dlsym(RTLD_DEFAULT, ...)` | slow `should_yield()` walk delays GEMM dispatch |
| **L4 NCCL completion** | NCCL's callback thread | straggler | `void cipher_straggler_observe(uint64_t bytes, uint64_t dur_ns, int chosen_algo)` via `dlsym` from `cipher_nccl.cpp` L287 | per-call fopen+fprintf adds 10s of µs |

## Hot-path cost inventory (per event/call)

| Observer | Lane | Cost | Dominant component |
|---|---|---|---|
| comply | L1 (no-op) | 0 ns | observe is `(void)ev` |
| topology | L1 (no-op) | 0 ns | observe is `(void)ev` |
| trace | L1 | ~30 ns | atomic fetch_add + 56-byte memcpy |
| carbon | L1 | ~50 ns | TLS read + 256-slot hash probe |
| fairness | L1 | ~60 ns | same as carbon + edge-detect on quota |
| receipt | L1 | ~60 ns | mix64 + CAS-loop on chain |
| guard | L1 | ~80 ns | 4096-slot hash probe |
| pulse | L1 | ~70 ns | M/K/N unpack + 8-probe + Welford |
| thermostat | L1 | ~70 ns | same as pulse |
| sustain | L1 | ~70 ns | 8-probe + window push (decode-only) |
| pipeline | L1 | ~40-150 ns | 32-element linear shape scan |
| loop | L1 | ~100 ns | shape ring + amortized autocorrelation |
| continuity | L1 | ~80 ns | 32-element linear region scan |
| shield | L1 | ~140 ns | sort of 20-element gap window |
| hibernate | L1 | ~20 ns + ~10 µs on idle→active | atomic store + (rare) NVML write |
| **fairness_shm** | **L3** | **~1.4 µs/GEMM** | **64-slot acquire-load walk** |
| straggler | L4 | ~10 µs/AllReduce | fopen+fprintf JSONL line |

The **L3 fairness_shm cost is the largest single observer cost in
Wave 4** and is the only one that lands on a hot compute path (every
cublasGemmEx, not amortized over 16 events). It is also the only
observer that imposes *back-pressure* via `usleep(20)`. The
recommended `hybrid` migration (move the slot table to kmod,
keep the read path in userspace) would cut the per-GEMM cost to
a single ioctl call.

## Patterns shared across observers

Five idioms recur across the 17 TUs:

### 1. The env-gate + idempotent init

```c
extern "C" int cipher_X_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_X");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) std::fprintf(stderr, "[CIPHER OpXX] X enabled — ...\n");
    return on;
}
```

Verified in carbon L62-82, fairness L66-86, receipt L103-121,
guard L83-97, comply L23-36, loop L131-146, pipeline L101-115,
trace L36-48, continuity L114-128, pulse L305-325, straggler
L139-176, topology L26-55, thermostat L144-164, shield L150-167,
sustain L109-127, hibernate L174-203. **16/17 observers use this
pattern**; only `cipher_fairness_shm.cpp` differs (it has a
constructor-priority autoinit at L225-226 and does the actual
shm carving inline at L59-110).

### 2. The open-address hash + atomic CAS insert

```c
int probe_or_insert(uint64_t key) {
    uint64_t h = mix64(key);
    for (unsigned p = 0; p < MAX_TENANTS; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_TENANTS - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(...)) {
                /* init slot */
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key)
                return (int)i;
        }
    }
    return -1; /* full — drop silently per I4 */
}
```

The exact same `mix64` constants (`0xff51afd7ed558ccdULL`,
`0xc4ceb9fe1a85ec53ULL` — splitmix64) appear in carbon L33-36,
fairness L36-39, receipt L40-43, guard L45-48. The 0-key-as-empty
sentinel is universal. **GUARD differs in one subtle way** at
L77-78: on table-full it overflows to slot 0 (vs the others'
silent drop). Documented in GUARD's section above.

### 3. The 8-probe slot_idx + find_or_alloc

```c
inline unsigned slot_idx(uint64_t key) {
    uint64_t mix = key ^ (key >> 32);
    return (unsigned)((mix * 0x9FB21C651E98DF25ULL) >> 58) % MAX_SHAPES;
}
ShapeWelford* find_or_alloc(uint32_t M, uint32_t K, uint32_t N) {
    if (M == 0 || N == 0) return nullptr;
    uint64_t key = pack_key(M, K, N);
    unsigned start = slot_idx(key);
    for (unsigned probe = 0; probe < 8; ++probe) {
        unsigned i = (start + probe) & (MAX_SHAPES - 1);
        uint64_t cur = g_shapes[i].key.load(std::memory_order_acquire);
        if (cur == key) return &g_shapes[i];
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_shapes[i].key.compare_exchange_strong(...)) {
                /* init */
                return &g_shapes[i];
            }
            return &g_shapes[i];  /* lose race; return current entry */
        }
    }
    return nullptr;
}
```

Used by the per-shape Welford observers (pulse L93-118 with mix
constant `0x9FB21C651E98DF25ULL`, thermostat L75-99 with mix
constant `0xD1B54A32D192ED03ULL`) and the per-session find_or_alloc
(shield L62-88 with mix constant `0xBF58476D1CE4E5B9ULL`, sustain
L54-81 with mix constant `0x94D049BB133111EBULL`). The 8-probe
limit is the same across all four; the magic multiplier differs
to spread collisions across observers. The bound-of-8 is
explicit ("drop silently if 8 probes don't find") — same I4
discipline as the open-address tables but with a smaller probe
budget (8 vs MAX_TENANTS).

### 4. The idle-reset session allocation

```c
uint64_t prev = g_last_ts_ns.exchange(t, std::memory_order_relaxed);
int slot = g_current_slot.load(std::memory_order_relaxed);
bool start_new = (slot < 0) ||
                 (prev != 0 && t > prev && (t - prev) > IDLE_RESET_NS);
if (start_new) {
    uint64_t seed = fnv1a64((const uint8_t*)&t, sizeof(t),
                            0xcbf29ce484222325ULL ^ ev->func_ptr_hash);
    slot = allocate_slot(seed);
    ...
}
```

`IDLE_RESET_NS = 200 ms` is the same in loop (L27), pipeline (L21),
continuity (L22), matching SENSE per the comments. Used by loop
L156-168, pipeline L125-135, continuity L138-149. A fourth
candidate (sustain) does NOT use this idle-reset; sustain keys
on `cipher_sense_current_session()` directly (L141), which
implicitly inherits SENSE's session boundary detection.

### 5. The Welford ITL accumulator

```c
double itl = (double)(now - prev_ts_ns);
prev_ts_ns = now;
n++;
double d = itl - mean_ns;
mean_ns += d / n;
M2_ns   += d * (itl - mean_ns);
```

Used by pulse L350-352, thermostat L187-189, shield L94-97.
Numerically stable; constant-memory. The pulse and thermostat
variants freeze the baseline (`long_baseline_mean`,
`baseline_mean_ns`) at a specific n (100 for pulse, 30 for
thermostat) — the baseline is the snapshot at warmup; subsequent
samples refine `mean_ns` but never touch the baseline. SHIELD's
Welford does *not* freeze a baseline because the gap-based
P50/P95 supersedes it.

These five idioms together constitute a Wave-4 "observer DSL".
Future observers should land in this style; the fusion plan
preserves them.

## Cross-wave findings worth flagging

### F4-1. Wave-2 HANDLED-discard does not apply to Wave 4

The Wave 2 finding (HANDLED structurally discarded by attn
trampolines, `cipher_rt_attn_dispatch.cpp` L321/359/398) is a
blocking risk *only* for substrate-registry actuators that return
HANDLED to claim substitution. **Every Wave-4 observer returns
void**; none is registered on the `cipher_rt_matmul_dispatch` or
`cipher_rt_attn_dispatch` registries. The Wave-2 finding is moot
for Wave 4 today.

If the fusion plan ever re-homes an observer onto a substrate
registry (e.g., to move fairness_shm onto the matmul dispatch as
a priority-0 AUDIT-template), the move is still safe: priority-0
AUDIT is passthrough-by-contract (`cipher_rt_audit.c` L101
matmul, L117 attn — verified Wave 2). It does not claim HANDLED,
so the attn HANDLED-discard does not affect it. **Stated once
here.**

### F4-2. ABI surface is empty for the entire observability tier

None of the 16 observers (17 TUs) calls a `/dev/cipher` ioctl.
This is consistent with the design intent: the kmod's role is
hardware-state aggregation (BAR0, GPU state, partition allocator,
weight arena); the observability tier sits above it and reads
state through TLS (`cipher_sense_current_session`), shared
globals (`g_cipher.liquid`), NVML directly (PULSE, HIBERNATE),
or POSIX shm (fairness_shm). **No Wave-3 ioctl is on the
observability hot path.**

A consequence: the Wave-3 ABI rule [[cipher-abi-rule]] ("additive
only; reserved nrs return -ENOSYS; new ioctls take fresh nrs") is
not exercised by any current Wave-4 observer. Future observers
that need kmod state should land at fresh nrs.

### F4-3. The Stage-1 ring is the load-bearing fan-out point

All 13 L1 observers fan-out from a *single point* in
`cipher_10ops_impl.cpp` (L509-560). They are called in fixed
order on every ring entry consumed by the Stage-1 shadow thread.
The compounded per-event cost is roughly the sum of the L1 column
above: ~800 ns/event with all 13 enabled. At 48 K launches/s
(M4 baseline per CLAUDE.md), that's a ~4 % CPU consumption on one
core for observability — not on the GPU hot path, but visible.

A v1.5 fusion that reorders these calls or makes them concurrent
on multiple consumer threads (the ring's read_seq_s1 supports
SPMC) would let the L1 observers run independently. Today they
share one consumer.

### F4-4. Two observers have active actuators today

Of the 16, only SHIELD (`cipher_sm_set_priority` band write, L190)
and HIBERNATE (NVML power-limit write when permitted, L161) have
active actuation. The other 14 set flags that are documented
as "not yet consumed by Stage 0" — these are detection-only,
awaiting v2 Op 3 SUBSTITUTE / oracle wiring.

The fusion plan can land the consumer side (Op 3, oracle) without
touching Wave-4 observers — they already expose the right atomic
flags for downstream readers. This is the major positive surprise
in Wave 4: the observability tier is **ready** for the actuation
wiring, even though only 2/16 observers are wired today.

### F4-5. cipher_fairness vs cipher_fairness_shm are sibling, not redundant

Despite the shared "fairness" name and `CIPHER_FAIRNESS` env var,
the two TUs detect different things:
- `cipher_fairness.cpp` (L1) counts work units (`grid × block`)
  per session within a single process and flags quota overrun.
- `cipher_fairness_shm.cpp` counts cublasGemmEx call frequency
  per process across a cross-process shm region and back-pressures
  the heaviest tenant via `usleep(20)`.

They share the `CIPHER_FAIRNESS` env var (cipher_fairness.cpp
L69 and cipher_fairness_shm.cpp L63) — which means enabling one
enables both. This is by-design coupling: in single-process runs
the shm observer does nothing (n_active ≤ 1 at L186), and the
per-process observer carries the load; in multi-process runs the
shm observer kicks in. **No conflict; no redundancy.**

### F4-6. PULSE's RTLD_NOLOAD coupling

PULSE explicitly uses `RTLD_NOLOAD` (cipher_pulse.cpp L130-131) —
it will not load NVML on its own. If `cipher_thermal_feedback`
(Stage 11) hasn't loaded NVML by the time PULSE's first
`try_resolve_nvml` fires, the ECC signal is permanently disabled
(`fn_get_ecc = nullptr`) and PULSE caps at score 1.

This is a documented degradation path (L132-136 stderr message)
but it does mean PULSE's effectiveness depends on Stage 11
booting first. The init priority chain in CLAUDE.md (111 for
thermal_feedback) is below the Wave-4 observer init order (which
happens later in `cipher_10ops_impl.cpp`), so the ordering is
correct in practice. Worth noting because any v1.5 init-order
refactor that reorders these would break PULSE's ECC signal.

### F4-7. HIBERNATE's pre-engage notify is on the producer thread

`cipher_hibernate.cpp::cipher_hibernate_observe` L213-220: when
the gate is raised and a new event arrives, the NVML power-limit
restore runs **synchronously on the producer's thread before the
event is processed by Stage 1**. The cost is one NVML call (~tens
of µs) on the first event after idle. For latency-critical
first-token-time workloads this is a non-trivial regression on
wakeup.

The architecture choice is correct (you want to restore power
*before* the actual GPU launch hits, not after) but the
producer-thread placement should be flagged in the fusion plan:
if a v1.5 splits Stage 0/1 across more cores, the pre-engage
should move to the Stage-1 reader to keep the producer fast.

---

## Specific anchors — verified or refuted

Numeric constants and behaviour claimed by the task prompt or by the
file headers themselves, verified or refuted against the source.

| Claim | Verdict | Cite |
|---|---|---|
| 17 source files, 3,863 LOC total | VERIFIED | wc -l confirmed |
| All 16 observers are wired into `stage1_shadow` at L509-560 | VERIFIED | cipher_10ops_impl.cpp L509-560 |
| `CipherRingEntry` is 128 bytes, cache-line aligned, 14 fields | VERIFIED | cipher_10ops.h L23-37 |
| `CIPHER_RING_SIZE = 65536` | VERIFIED | cipher_10ops.h L72 |
| fairness_shm sits on cublasGemmEx hot path via `cipher_intercept_cudart.cpp` | VERIFIED | cipher_intercept_cudart.cpp L858-883; CIPHER_DRIVER_LEVEL_BUILDS.md L36 |
| fairness_shm `MAX_TENANTS = 64`, magic `0xC1F4C1F4` | VERIFIED | cipher_fairness_shm.cpp L19-20 |
| straggler is driven from `cipher_nccl.cpp` via dlsym, not from the Stage-1 ring | VERIFIED | cipher_nccl.cpp L287 |
| PULSE uses RTLD_NOLOAD only; will not load NVML on its own | VERIFIED | cipher_pulse.cpp L130-131 |
| HIBERNATE actively dlopens NVML (full load if NOLOAD fails) | VERIFIED | cipher_hibernate.cpp L67-70 |
| HIBERNATE on May-13 pod is DEGRADED (NVML write probe returns NOT_SUPPORTED) | VERIFIED | cipher_hibernate.cpp L94-100 + L191-202 |
| THERMOSTAT reads `g_cipher.liquid.device->hw.gpu_temp_c` directly | VERIFIED | cipher_thermostat.cpp L112-113 |
| SHIELD writes `CIPHER_BAND_PROTECTED` via `cipher_sm_set_priority` for HUMAN_INTERACTIVE sessions | VERIFIED | cipher_shield.cpp L189-194; cipher_green_ctx.h L121 |
| LOOP `BURN_RATIO_MIN = 500`, `PREFILL_DROUGHT_MIN = 500`, `SHAPE_WIN = 64`, `MAX_PERIOD = 8`, `MIN_CYCLES = 3` | VERIFIED | cipher_loop.cpp L23-30 |
| PIPELINE `JACCARD_THRESHOLD = 0.5`, `MAX_SESSIONS = 1024`, `SHAPES_PER_SESSION = 32`, `REPORT_MAX_SESSIONS = 256` | VERIFIED | cipher_pipeline.cpp L19-23 |
| CONTINUITY `SNAPSHOT_EVERY = 500`, manifest gated AGENT_AUTONOMOUS-only | VERIFIED | cipher_continuity.cpp L21, L165-166 |
| GUARD `MAX_SHAPES = 4096`, `RESIDENCY_NS = 2 s`, table-full overflows to slot 0 | VERIFIED | cipher_guard.cpp L23-25, L77-78 |
| RECEIPT uses HMAC-SHA256 over `chain ‖ launches ‖ order`, omits session_fp and timestamps | VERIFIED | cipher_receipt.cpp L196-203 |
| CARBON default `J_per_unit = 1e-9`, `gCO2_per_kWh = 400` | VERIFIED | cipher_carbon.cpp L29-30 |
| FAIRNESS default quota = 5e9 work units | VERIFIED | cipher_fairness.cpp L18 |
| TRACE capacity = 8192 entries, ~512 KiB total static footprint | VERIFIED | cipher_trace.cpp L15 |
| THERMOSTAT temp gates 75/72 °C, drift gates +10 %/+5 %, poll period 500 ms, `BASELINE_FREEZE_N = 30` | VERIFIED | cipher_thermostat.cpp L21-26 |
| PULSE score ceiling 2 (CRITICAL unreachable in v1); ECC delta gate 10; eval gate 1000 dispatches; re-alert period 60 s; `BASELINE_FREEZE_N = 100` | VERIFIED | cipher_pulse.cpp L22-28 + L257-258 + L317-322 |
| SUSTAIN window 20, eval every 10, threshold 50 µs/step, AGENT relax ×3 at decode > 500 | VERIFIED | cipher_sustain.cpp L22-27 |
| SHIELD ITL_WINDOW = 20, JITTER_SIGMA_GATE = 2.0 (unused in current code; replaced by 5× p50 gate), BURST_P95_P50_GATE = 3.0 | VERIFIED | cipher_shield.cpp L27-29 + L127 |
| HIBERNATE default threshold 5 ms, poll cadence 10 ms, reduced target = 30 % of TDP | VERIFIED | cipher_hibernate.cpp L24-25 + L86 |
| STRAGGLER: 7 buckets matching NCCL conventions, EMA α=0.10, slowdown factor 1.5, RUN_GATE = 5 sustained calls, RING is the hard-coded slow-tail fallback | VERIFIED | cipher_straggler.cpp L23-30 + L259-261 |
| TOPOLOGY observe is a no-op; init-time only via `cudaGetDeviceCount` + `cudaDeviceCanAccessPeer` | VERIFIED | cipher_topology.cpp L42-49, L57-60 |
| COMPLY observe is a no-op; compliance gate at report time = `(guard_leaks==0) && (receipt_sessions>=1) && (determ_count>0) && (fair_overruns==0)` | VERIFIED | cipher_comply.cpp L38-40, L59-62 |
| COMPLY depends on `cipher_determinism_*` which is out-of-Wave-4 scope | VERIFIED + flagged | cipher_comply.cpp L53-54 calls into a TU not in Wave 4's 17-file list |

All 28 anchors verified. No drifts beyond the cosmetic/documented
ones called out in the per-file sections.

## v1 fusion summary (Wave 4 observers)

| File | Action | Severity | Rationale |
|---|---|---|---|
| `cipher_carbon.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant `g_j_per_unit` from classifier `tenant_billing_class` |
| `cipher_fairness.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant `g_quota` from classifier `tenant_quota_units` |
| `cipher_fairness_shm.cpp` | REFACTOR (medium) | requires-mitigation | Move slot table to kmod (proposed ioctls nr 25/26/27); keep userspace read path. Worst per-call hot-path cost in Wave 4 (~1.4 µs/GEMM). |
| `cipher_receipt.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant keying via classifier `tenant_billing_key_id` |
| `cipher_guard.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant `tenant_isolation_zone` to filter benign within-tenant repeats |
| `cipher_comply.cpp` | PORT-AS-IS | none | Pure aggregator; zero hot-path work |
| `cipher_loop.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant `session_max_decode_budget` |
| `cipher_pipeline.cpp` | PORT-AS-IS + SHIM | minor | Add `tenant_pipeline_id` to pre-group sessions |
| `cipher_trace.cpp` | PORT-AS-IS + SHIM | minor | Add kernel-class mask from classifier to drop low-value events |
| `cipher_continuity.cpp` | PORT-AS-IS + SHIM | minor | Add per-tenant `tenant_checkpoint_cadence`; v2 actual KV checkpoint is REFACTOR in a different op |
| `cipher_pulse.cpp` | PORT-AS-IS + SHIM | minor | Add `shape_is_hot_path` to focus Welford. Signal-2 wiring is REFACTOR in Op 3, not here |
| `cipher_straggler.cpp` | PORT-AS-IS | none-minor | Buffer fopen to background ring for v1.5 (current fopen-per-AllReduce cost is the second-largest in Wave 4) |
| `cipher_topology.cpp` | PORT-AS-IS | none | Static; no hot-path work |
| `cipher_thermostat.cpp` | PORT-AS-IS + SHIM | minor | Add `silicon_thermal_class` to vary gates per silicon |
| `cipher_shield.cpp` | PORT-AS-IS + SHIM | minor (already wired) | Add per-tenant `tenant_latency_sla` to vary P95/P50 gate. Only Wave-4 observer with active actuation today |
| `cipher_sustain.cpp` | PORT-AS-IS + SHIM | minor | Add `tenant_kv_compression_class`. Op-3 SUBSTITUTE consumer is REFACTOR there |
| `cipher_hibernate.cpp` | REFACTOR (small) | minor | Add kmod-clock fallback actuation via Wave-3 nr 10 `CIPHER_SET_CLOCK_MHZ` for pods where NVML write is denied |

## Per-fusion-mode rollup

| Fusion mode | Count | Files |
|---|---|---|
| `stage1-ring-consumer` (pure) | 10 | carbon, fairness, receipt, guard, comply, loop, pipeline, trace, continuity, topology |
| `hybrid` (Stage-1 + Stage-2) | 3 | thermostat, pulse, hibernate |
| `stage1-ring-consumer` + active actuator | 1 | shield (band write) |
| `stage1-ring-consumer` (decode-only filter) | 1 | sustain |
| L3 cublasGemmEx-shim (kmod-resident recommended) | 1 | fairness_shm |
| L4 NCCL-completion observer | 1 | straggler |

## The load-bearing fact (re-verified, Wave 4)

The Wave-4 observability tier is *ready* to consume classifier output:

- **A canonical hot-path surface** — `const CipherRingEntry*` with 14
  fields including `params_hash`, `func_ptr_hash`, `kernel_class`,
  `grid_*`, `block_*`, and `timestamp_ns`. The classifier brain
  (Wave 1) can decorate the ring entry with additional fields (e.g.,
  `session_band`, `tenant_quota_units`) via a TLS-aligned overlay
  without touching the observer surface.

- **Five shared idioms** — env-gate+init, open-address+CAS,
  8-probe slot_idx, idle-reset session alloc, Welford ITL —
  documented in the synthesis section. Any classifier-output
  observer should land in this style.

- **Atomic flags already exported for downstream actuators** —
  SHIELD's `oracle_aggressive` and `cache_aggressive`, SUSTAIN's
  `sustain_compress`, THERMOSTAT's `aggressive`, LOOP's `score`,
  HIBERNATE's `idle_gate_flag`. The actuation consumer side (Op 3
  SUBSTITUTE, the oracle) can land without touching the observer
  TUs.

- **One actuation already wired** — SHIELD's
  `cipher_sm_set_priority` (Wave 2 green-ctx priority table). The
  v1 fusion can claim this is the canonical "observer → actuator"
  wiring template.

What is missing for v1 classifier-driven fusion:

- **No per-tenant overrides today.** Every observer has a single
  global threshold/quota/gate. SHIMs (one or two new fields per
  observer) suffice.
- **fairness_shm is the only observer with non-trivial hot-path
  cost** (~1.4 µs/GEMM). Migration to kmod (proposed nrs 25/26/27)
  is the single REFACTOR that yields a measurable win.
- **Two observers (SHIELD jitter / smoothing, SUSTAIN compress)
  set flags that no current actuator consumes.** This is a
  REFACTOR in Op 3 / oracle, not in the observers — Wave-4
  scope ends at the flag.

The honest engineering posture for v1: port all 16 observers
PORT-AS-IS; add the listed SHIMs only as their downstream
consumers come online; defer the fairness_shm kmod migration
to v1.5; defer the HIBERNATE kmod-clock fallback to v1.5.
The ABI surface remains untouched (the Wave-3 ioctl table
gets the proposed nr 25-27 additions only if and when
fairness_shm is migrated).

This wave's contribution to the fusion bill of materials is
the smallest of the four waves: most of these observers are
already in the right shape for v1. The classifier brain wired
through Wave 1 supplies the per-tenant fields the SHIMs
need; the actuators wired through Wave 2 supply the
consumption side of the atomic flags. Wave 3's ABI remains
the right interface; Wave 4 observers do not touch it today
and should not be forced to.
