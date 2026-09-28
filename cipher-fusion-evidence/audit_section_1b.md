## Section 1b — The 21 overlay ops

Source root: `/home/ubuntu/cipher-may13-evidence/src/`. Each of the 21 overlay op
files (canonical op numbers 13–33) was read in full. Classification is grounded
strictly in the code on disk; doc/memory claims are used only where the code
confirms them, and contradictions are called out.

### Firing evidence — scope note

The only valid stress firing record is
`/home/ubuntu/cipher-may13-evidence/stress2/per_op_validation.json` (referenced by
`OPS_VALIDATION_REPORT.md`). That file tests a set of 21 ops that is **not** the
canonical overlay-21: it includes runtime modules (PERSIST_ENGINE, KV_REDIRECT,
GRAPH_ENGINE, SUBSTITUTE_V2, KOOPMAN, WEIGHT_SHARE, FLOW_PATTERNS, FLOW_RECORDER,
FUSION_RESIDUAL, ARBITRATE) that are NOT overlay ops. The intersection with the
canonical overlay-21 is exactly **11 ops**: PREDICT, GUARD, RECEIPT, CONTINUITY,
DETERMINISM, CARBON, TOPOLOGY, LOOP, PIPELINE, TRACE, COMPLY. Those 11 carry a
FIRES/SILENT verdict from the JSON. The other 10 overlay ops (SENSE, SHIELD,
SUSTAIN, THERMOSTAT, PULSE, FAIRNESS, VOLT, HIBERNATE, STRAGGLER, NCCL_P2P) were
never exercised by that run and are reported **UNKNOWN**.
The second run, `per_op_v2.log`, aborted at `import torch` (NumPy 1.x/2.x ABI
crash) before any observe events flowed; its `[CIPHER OpNN] ... enabled` lines are
init banners only and are NOT firing evidence.

### Table

| Op# | Name | File | Class | File:Line evidence | Stress firing | Notes |
|-----|------|------|-------|--------------------|---------------|-------|
| 13 | SENSE | cipher_sense.cpp | WORKING | cipher_sense.cpp:171-273 `cipher_sense_observe` (session boundary detection, M-based prefill/decode discrimination), :129-150 `classify` | UNKNOWN | Not in per_op_validation.json. Real session-classification state machine; downstream consumers (SHIELD/SUSTAIN/GUARD/etc.) read `cipher_sense_current_session`/`get_type`. |
| 14 | SHIELD | cipher_shield.cpp | PARTIAL | cipher_shield.cpp:90-146 `update_itl` (real Welford + P50/P95 jitter detection); header :7-10 and init banner :162-164 state "cache_aggressive/oracle_aggressive flags set but not yet consumed by Stage 0" | UNKNOWN | ITL/jitter detection is real and the Protection-1 priority hint at :188-194 calls `cipher_sm_set_priority`; Protections 2 and 3 are detection-only, flags never consumed — observational where an actuator is intended. |
| 15 | SUSTAIN | cipher_sustain.cpp | PARTIAL | cipher_sustain.cpp:90-105 OLS slope regression, :177-183 sets `sustain_compress`; header :5-6 and init banner :122 state "sustain_compress flag set but not yet consumed by Stage 0" | UNKNOWN | KV-pressure slope detector is real; the compress flag it produces has no consumer — observational only. |
| 16 | GUARD | cipher_guard.cpp | WORKING | cipher_guard.cpp:99-130 `cipher_guard_observe` (cross-session params_hash reuse within 2 s residency window → leak count) | FIRES | per_op_validation.json: `shape_count=0 leak_count=0` — observer engaged, zero leaks at the 50-tok workload (single session, nothing to attribute). Real detector. |
| 17 | PREDICT | cipher_predict.cpp | WORKING | cipher_predict.cpp:160-179 shape hot-count tracking; :193-245 `cipher_predict_observe_ptr` promotes hot pointers to the persist engine via dlsym'd `cipher_persist_engine_register` | FIRES | per_op_validation.json: `shape_count=7, candidate_count=3, ptr_seen_count=153`. Real actuation path into the persistence engine. |
| 18 | RECEIPT | cipher_receipt.cpp | WORKING | cipher_receipt.cpp:123-143 FNV chain accumulation per session; :165-225 report emits real HMAC-SHA256 (OpenSSL `HMAC`/`EVP_sha256`) over chain‖launches‖order | SILENT | per_op_validation.json: `report=[]` — no tenant/session attributed at 50 tok (session fp not finalized), not a code defect. HMAC proof-of-compute logic is real. |
| 19 | CONTINUITY | cipher_continuity.cpp | PARTIAL | cipher_continuity.cpp:130-175 per-session region tracking + manifest counter; header :3-5 states "No KV capture, no pinned memory, no CUDA calls. v1 observer" | FIRES | per_op_validation.json: `session_count=7, manifest_count_total=0` — region accounting works, but no manifest is ever written (v2 Tier-A checkpoint worker not present). Observational where a checkpointer is intended. |
| 20 | THERMOSTAT | cipher_thermostat.cpp | WORKING | cipher_thermostat.cpp:166-191 per-shape Welford on ITL; :193-236 poll combines temp + drift gates and sets `aggressive`; consumed downstream by cipher_volt.cpp:321 `cipher_thermostat_aggressive_active()` | UNKNOWN | Init banner :157 says "aggressive flag set but not yet consumed by Stage 0" — that comment is **stale**: VOLT reads the flag and uses it to force frequency reduction (`thermo_force`, cipher_volt.cpp:321-336). Real detector with a real consumer. |
| 21 | DETERMINISM | cipher_determinism.cpp | WORKING | cipher_determinism.cpp:50-63 `cipher_determinism_observe` mixes ordered params_hash stream into a 64-bit FNV+splitmix accumulator | FIRES | per_op_validation.json: `dispatch_hash=935cae3589c6fd6d, dispatch_count=36450`. Complete and self-contained. |
| 22 | PULSE | cipher_pulse.cpp | PARTIAL | cipher_pulse.cpp:327-358 per-shape Welford baseline; :177-216 Signal 1 (ITL drift) + Signal 3 (NVML ECC) real; header :7-8, alert text :261-263 and banner :318-322 state Signal 2 deferred, "v1 SCORE CEILING = 2 (CRITICAL unreachable)" | UNKNOWN | Two of three fault signals implemented; Signal 2 (per-substitution max_diff) explicitly not wired — needs a Stage-0 sentinel hook in Op 3. Incomplete by the code's own admission. |
| 23 | CARBON | cipher_carbon.cpp | WORKING | cipher_carbon.cpp:84-103 per-session grid*block work accumulation; :113-169 report converts joules→gCO2 with env-tunable factors | FIRES | per_op_validation.json: `session_count=0, j_per_unit=1e-09, gco2_per_kwh=400` — report fields present; session_count=0 because no session attributed at 50 tok. Real per-tenant carbon estimator. |
| 24 | FAIRNESS | cipher_fairness.cpp (also cipher_fairness_shm.cpp) | WORKING | cipher_fairness.cpp:88-117 `cipher_fairness_observe` (per-tenant work-unit accumulation, quota overrun flag) | UNKNOWN | Not in per_op_validation.json. Open-address tenant table + quota detection is complete. `cipher_fairness_shm.cpp` is a separate POSIX-SHM cross-process variant; the canonical Op-24 logic here is self-contained and complete. |
| 25 | TOPOLOGY | cipher_topology.cpp | WORKING | cipher_topology.cpp:26-55 init-time `cudaGetDeviceCount` + `cudaDeviceCanAccessPeer` adjacency build; observe is intentionally a no-op (:57-60, "v1: topology is static") | FIRES | per_op_validation.json: `device_count=1, adjacency=[[0]], edge_count=0`. Static inference is complete and correct for the design (no-op observer is by design, not a stub). |
| 26 | LOOP | cipher_loop.cpp | WORKING | cipher_loop.cpp:148-229 `cipher_loop_observe`: 3-signal runaway scoring (S1 period autocorrelation :95-114, S2 burn ratio, S3 prefill drought) with sticky-high latch | FIRES | per_op_validation.json: `session_count=8, runaway_count=7, period=5`. Detector fires and latches. Complete. |
| 27 | PIPELINE | cipher_pipeline.cpp | WORKING | cipher_pipeline.cpp:117-141 per-session bounded shape-set; :151-239 report emits pairwise Jaccard edges with upstream/downstream tagging | FIRES | per_op_validation.json: `session_count=6, edge_count=10, jaccard=1.0`. Complete. |
| 28 | TRACE | cipher_trace.cpp | WORKING | cipher_trace.cpp:50-66 bounded-ring append; :77-116 report flushes JSONL + summary | FIRES | per_op_validation.json: `written=8192, dropped=32308, capacity=8192`. Complete bounded exporter (drop-on-full is by design). |
| 29 | COMPLY | cipher_comply.cpp | WORKING | cipher_comply.cpp:46-82 `cipher_comply_report` aggregates RECEIPT/CARBON/GUARD/DETERMINISM/FAIRNESS/TOPOLOGY counters into a single ok/not-ok verdict; observe is intentionally no-op (:38-40, "aggregation-only") | SILENT (FIRES as designed) | per_op_validation.json: `compliance_ok=false` — verdict computed correctly; false because upstream RECEIPT/FAIRNESS report zero sessions at 50 tok. Aggregation logic is complete; the no-op observer is by design, not a stub. |
| 30 | VOLT | cipher_volt.cpp | PARTIAL | cipher_volt.cpp:353-366 real AI classifier; :307-347 `actuate_impl` calls `nvmlDeviceSetGpcClkVfOffset`; :123-132 startup probe — header :1-7 declares "Path B production-ready, pod-degraded": on this pod NVML clock-set returns NOT_SUPPORTED → DEGRADED classifier-only mode | UNKNOWN | Full actuation code path exists (atexit/signal restoration, ±200 MHz offset). On the documented pod (driver 580.105.08, non-root) actuation is unsupported, so VOLT runs classifier-only — actuator present but gated off in this environment. |
| 31 | HIBERNATE | cipher_hibernate.cpp | PARTIAL | cipher_hibernate.cpp:139-170 `poll_impl` idle-gap detection + `nvmlDeviceSetPowerManagementLimit` call; :94-100 startup probe — header :6-7 declares "Pod-degraded: probe returns NOT_SUPPORTED → power-limit writes skipped" | UNKNOWN | Idle-detection state machine is real and observable via counters; the power-limit actuator is gated off when the NVML write probe fails (the documented pod state). Actuator present but degraded. |
| 32 | STRAGGLER | cipher_straggler.cpp | PARTIAL | cipher_straggler.cpp:178-236 real per-bucket EMA slowdown detector + `cipher_straggler_algo_hint` :255-261; header :8-10 and alert JSON :126-129 state "Cross-rank rank attribution: NOT IMPLEMENTED" | UNKNOWN | Local slowdown detection and the RING algo hint are real and wired (cipher_nccl.cpp:283-290 calls `cipher_straggler_observe`). The cross-rank attribution half of the op is explicitly not implemented — deferred to an offline aggregator script. Incomplete by the code's own admission. |
| 33 | NCCL_P2P | cipher_nccl.cpp (+ cipher_nccl_bpf.cpp, cipher_nccl_neural.cpp, cipher_nccl_tuner.cpp, cipher_nccl_v4.cpp) | PARTIAL | cipher_nccl.cpp:34-75 real CfC/rule policy `select_policy`; :198-291 live `record_decide`/`record_feedback` with per-bucket EMA; :21-29 `cipher_nccl_init` sets `ebpf_active=false` — "eBPF requires root + kernel module; simulated here (CPU stub mode)" | UNKNOWN | Algorithm-selection policy and feedback EMA are real and fire from the ncclAllReduce shim. The eBPF P2P-routing path — the "P2P" in NCCL_P2P — is simulated, not active; `policy_map` is an in-process struct, not a real eBPF map write. The CLAUDE.md build state itself records "NCCL P2P CPU Proxy DEFERRED (multi-node)". |

### Ops classified STUB or MISSING

**None.** No overlay op is classified STUB or MISSING. Every one of the 21 files
contains substantive, executing logic. The `cipher_topology_observe` and
`cipher_comply_observe` no-op bodies are deliberate design choices (static
inference / aggregation-only ops), not empty stubs — both ops have complete
init/report logic that does real work.

### Summary of non-WORKING classifications (PARTIAL)

Eight ops are PARTIAL — real logic present but incomplete or with the actuator
gated off / detection-only where an actuator is intended:

- **14 SHIELD** — Protections 2/3 are detection-only; `cache_aggressive`/`oracle_aggressive` flags set but never consumed (cipher_shield.cpp:7-10, 162-164).
- **15 SUSTAIN** — `sustain_compress` flag produced but has no consumer (cipher_sustain.cpp:5-6, 122).
- **19 CONTINUITY** — region tracking only; no manifest is ever written, no KV capture, no checkpoint worker (cipher_continuity.cpp:3-5; firing record `manifest_count_total=0`).
- **22 PULSE** — Signal 2 explicitly deferred, severity score ceiling hard-capped at 2 / CRITICAL unreachable (cipher_pulse.cpp:7-8, 261-263, 318-322).
- **30 VOLT** — full DVFS actuator code present but runs classifier-only / DEGRADED on this pod where NVML clock-set is NOT_SUPPORTED (cipher_volt.cpp:1-7, 123-132).
- **31 HIBERNATE** — idle detector real, but power-limit actuator gated off when the NVML write probe fails — the documented pod state (cipher_hibernate.cpp:6-7, 94-100).
- **32 STRAGGLER** — local slowdown detection real; cross-rank attribution NOT IMPLEMENTED, deferred to an offline aggregator (cipher_straggler.cpp:8-10, 126-129).
- **33 NCCL_P2P** — algorithm-selection policy real; the eBPF P2P-routing path is simulated/CPU-stub, not active (cipher_nccl.cpp:21-29).

Final count: **WORKING 13, PARTIAL 8, STUB 0, MISSING 0** (21 total).
