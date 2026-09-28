# R.C v1 — PER-GPU FLEET RELIABILITY DETECTION (signal fusion: R.A SDC ⊕ W.6 cohort ⊕ NVML)

**Date:** 2026-06-09. **Model:** mistralai/Mistral-7B-v0.1 fp16 (cached). **Stack:** vLLM 0.20.2 / torch
2.11 / CUDA 13, real KV cache, continuous batching, real FlashAttention, real token generation, `enforce_eager`.
**GPU:** NVIDIA H100 80GB HBM3, `GPU-993dd5d1-740e-ce8f-7c5e-404188d4b950`, PCI `00000000:07:00.0`, driver
`580.105.08`, node `192-222-53-2`. **Clock:** requested/locked 1980 MHz during measurement (achieved
~1830 MHz under `SwPowerCap` during compute — disclosed in the throttle reasons; a power-cap, not a fault) → `-rgc` at
exit (this report written post-reset). **Telemetry consumed:** NVML via `pynvml` (DCGM/`dcgmi` is **NOT installed on this pod** —
stated, not faked; the consumed counters are ECC volatile/aggregate uncorrected+corrected, XID critical-error
events, throttle reasons, power, SM clock, and per-process compute-procs).

**Discipline:** frozen anchor `libcipher_rt.so` md5 `2edba0d2…` **UNCHANGED entry+exit**; no vLLM source edit, no
monkeypatch; the SDC signal is a SCRATCH `LD_PRELOAD` cuBLAS-interception shim **copied** from
`ra_realvllm/detector_shim.c` and extended **only** in `rc_pergpu/` (ra_realvllm/, ra_e2e/, etc. md5 UNCHANGED); the
substrate runtime `libcipher_rt.so` is deliberately **NOT loaded** into vLLM (would activate its CUDA interception and
contaminate the eager measurement); `CUDA_INJECTION64_PATH` unset, `VLLM_PLUGINS=""`, `VLLM_USE_DEEP_GEMM=0`; vLLM
children reaped; GPU exclusive (`compute-procs=[]` at each baseline). **Evidence:** `rc_pergpu/{sdc_shim.c,
vllm_driver.py, rc_monitor.py, clean_*, inject_*, probe_*, ENTRY_MD5.txt, EXIT_MD5.txt}`.

---

## BOTTOM LINE

**Per-GPU reliability detection by signal fusion is REAL; fleet aggregation is schema-ready** (the per-GPU event is
defined and emitted; the multi-GPU aggregation service is unbuilt). On real vLLM serving Mistral-7B, R.C fuses three
signals into ONE per-GPU verdict: (1) the **R.A SDC detector** (online GEMM correctness — the signal NVML/DCGM lack),
(2) the **W.6 kmod cohort registry** (which tgid/context held the GPU + co-resident count), and (3) **NVML host
telemetry** (ECC/XID/throttle/power/clock — corroboration). An injected persistent SDC produced a **DEGRADED** verdict
with the failing GPU id, the **decode step the fault began** (detected step 64, confirmed step 72, in a *single run*
that read clean before step 64), a **descriptive process/context tag consistent across all three signals** (`tgid
3320702`, cmdline `VLLM::EngineCore`, W.6 fingerprint), and severity. The clean run produced **0 DEGRADED events** (0
SDC detections over its 2,176 clean GEMM-checks — 4,352+ including the probe's clean co-resident context; ECC/XID
counter-clean; even a benign `SwPowerCap` throttle did **not** trigger a false verdict). The fleet **aggregation
schema** is defined and
emitted (`rc.v1`). **What CIPHER adds over DCGM/NVML alone:** an *online correctness* signal — the fused verdict flags
DEGRADED for a GEMM-output corruption while ECC/XID read clean (correctness-checking is orthogonal to hardware
counters). **What is honestly NOT claimed:** this is per-GPU not per-tenant (v2); ONE GPU validated (multi-GPU scale +
the aggregation service are unbuilt); the eager hosting carries the documented Path-1 **+51% eager tax**; the injected
fault is a *software* flip so NVML-clean is *expected* (not proof CIPHER caught a real hardware SDC DCGM missed);
deterministic mercurial-core faults stay blind to same-GPU recompute. **Localization PROBE:** observation-localization
is **CLOSED-WITH-BOUND** (a clean context **concurrently executing GEMMs on the same GPU stayed 0/514** while the
injected context went DEGRADED — verified co-resident at the fault instant by the fault-window timestamp-join, `n_resident=2`
across all 92 samples; the descriptive-not-attributive boundary holds); causation-localization (true per-tenant
attribution, v2) is **WALL-WITH-MECHANISM** (software in-process injection cannot emulate a shared-hardware fault;
same-GPU recompute is blind to the deterministic case).

---

## STEP-0 / RULE-4 PRE-REGISTRATION (fusion logic + false-positive surface, written before building)

**Three signals consumed (none rebuilt):**

| Signal | Source (consumed) | Contributes | Runs in |
|---|---|---|---|
| **SDC correctness** | R.A detector shim (extended copy, LD_PRELOAD) — intercepts `cublasGemmEx`, every N steps recomputes each linear GEMM independently and compares `‖D′−C‖` (T=0) | the **online-correctness** signal DCGM/NVRx lack; per-event {gpu, step, ordinal/k, residual, **pid/tid**} | inside the vLLM worker |
| **Co-residence** | **W.6 kmod cohort registry**, ioctl NR 31 `CIPHER_COHORT_QUERY` (verified live) | the **descriptive context tag**: which tgid(s) held the GPU + co_resident count + per-tgid fingerprint | shim heartbeat (worker) + monitor query-only |
| **Host telemetry** | **NVML** (DCGM absent): ECC vol/agg uncorr+corr delta, XID events, throttle, power, SM clock, compute-procs(pid→mem) | corroborate/contradict; the **SDC-fired-but-NVML-clean** case = CIPHER's added value | external monitor |

**Fusion rule (pre-registered, then measured):**
- `DEGRADED` ⟺ the SDC detector fired (residual > 0 under T=0) **persistently** (≥2 consecutive check-steps for the
  same gemm-ordinal) **OR** 1 detection + corroborating ECC-uncorrected↑/XID. Severity `critical`.
  - SDC + (ECC-uncorr↑ OR XID) → DEGRADED, **corroborated (hardware)** *(branch specified + implemented as
    `persist≥2 OR (1 detection AND (ECC-uncorr↑ OR XID))` in `rc_monitor.py`; UNEXERCISED here — `xids=[]` and ECC
    delta=0 in all three runs — and verified **inert**: re-deriving all three verdicts with vs without the XID term is
    bit-identical, so the measured verdicts stand)*
  - SDC + NVML-clean → DEGRADED, **uncorroborated_by_hardware_counters** ← the orthogonality case CIPHER adds
- `HEALTHY` ⟺ no SDC event over the window. **NVML-only anomalies (correctable ECC, power-cap throttle) NEVER set
  DEGRADED-corruption** — they emit a separate `host_health` annotation. *This rule is what keeps the clean-run
  DEGRADED count at 0.*

**False-positive surface (pre-registered):** (a) the SDC signal's own FP is already 0 at T=0 (bit-identical clean
recompute, R.A: 0/5,376); (b) the **new fusion FP risk** — an NVML-only benign event mislabeled as corruption — is
eliminated by "only SDC sets DEGRADED-corruption"; (c) a single comparator blip is guarded by the ≥2-consecutive-check
persistence rule. **Predicted:** clean run emits 0 DEGRADED; the injected persistent SDC produces exactly 1 DEGRADED with
a non-trivial onset. **Architecture:** post-hoc CLOCK_MONOTONIC timestamp-join (no live FIFO) — the shim appends
per-check records; the monitor samples NVML+cohort on a ~0.5 s timeline; fusion joins by time after the run; clock
domains are CLOCK_MONOTONIC (shim) / `time.monotonic()` (monitor, = CLOCK_MONOTONIC on Linux) / kmod `ktime_get_ns`
(monotonic-compatible); correlation window = the full persistent-fault span with an onset-window cross-check.

---

## (A) CLEAN RUN — 0 DEGRADED, 0 FALSE POSITIVES  *(DONE clause 2, clean half)*

`clean_verdict.json` / `clean_fusion.json` (N=8, no injection, eager Mistral-7B, batch 8, 64 decode tok):

| Quantity | Value |
|---|---|
| **Verdict** | **HEALTHY** |
| SDC detections | **0** over **2,176** GEMM-checks (17 check steps); watched-ordinal residual = 0 at every check |
| `max_clean_residual` | **0** (T=0 bit-identical recompute) |
| ECC delta (vol+agg, uncorr+corr) | **0 / 0 / 0 / 0** |
| XID critical events | **none** |
| Throttle reasons seen | `0x0`, `0x1` (GpuIdle), **`0x4` (SwPowerCap)** — benign; did **NOT** trigger DEGRADED |
| Cohort `n_resident` | **1** (the vLLM worker, self-registered via the shim heartbeat) |
| pid alignment | SDC pid == NVML compute pid == W.6 cohort tgid == 3320192 ✓ |

The clean run is the false-positive proof: thousands of clean GEMM-checks, exactly 0 SDC detections, and a benign
`SwPowerCap` throttle that the fusion rule correctly did **not** escalate to a corruption verdict. (Combined with the
probe's clean co-resident context, **0 FP over 4,352+ clean GEMM-checks**; the inject run additionally read
`max_clean_residual=0` on all non-injected GEMMs.)

## (B) INJECTED PERSISTENT SDC — DEGRADED verdict with onset + descriptor  *(DONE clauses 1 & 2)*

`inject_verdict.json` (N=8, **RV_ONSET=64**, inject layer-0 down_proj ordinal 3 / k=14336, persistent bit-14 flip,
stream-synced). The onset guard makes the fault begin *mid-run*, so a single run shows the clean→DEGRADED transition:

```
 step : residual : detection        ← watched ordinal 3 (down_proj), one record per check step
    0..56 :   0.000 : 0   (8 clean checks — GPU reads clean before the fault)
   64 : 322.750 : 1   ← FAULT ONSET detected
   72 :  26.281 : 1   ← verdict CONFIRMED (2nd consecutive detect-check, persistence rule)
   80..128 : >0   : 1   (9 detect-checks total, max_detect_residual = 322.75, max_clean_residual = 0)
```

**Per-GPU verdict record:**
- **verdict** = DEGRADED, **severity** = critical
- **gpu** = index 0, uuid `GPU-993dd5d1-…`, pci `00000000:07:00.0`, node `192-222-53-2`
- **fault_onset** = { decode_step_detected **64**, decode_step_confirmed **72**, wall_clock `2026-06-09T13:09:41Z`,
  covered_op `linear_gemm:ordinal3:k14336` }
- **signal_sources.sdc_detector** = fired, observer pid 3320702, 9 detections, persistence 9, 8 clean checks before
  onset, max_residual 322.75; coverage_note = linear projections (qkv/o/gate_up/down) via cublasGemmEx,
  attention(FlashAttn)+lm_head uncovered; threshold T=0
- **signal_sources.nvml** = ECC delta 0/0/0/0, XID none, telemetry_source "NVML (DCGM/dcgmi absent)"
- **corroboration** = `uncorroborated_by_hardware_counters`
- **process_context.descriptor** = { tgid **3320702**, cmdline **`VLLM::EngineCore`**, fingerprint
  `0xc1dec0de00000001`, descriptor_source **`nvml_compute_procs+w6_cohort`**, seen_by_nvml true, in_w6_cohort true } —
  **DESCRIPTIVE** (what was running), not attributive
- **co_resident_count** = 1

This is the single fused determination the goal asked for: *"GPU 0 is producing silent corruption, it began at decode
step 64 (13:09:41Z), the process/context running was tgid 3320702 (VLLM::EngineCore)."* The descriptor is
**consistent across three independent signals** — the SDC-observer pid, the NVML compute pid, and the W.6 cohort tgid
are the **same** 3320702 — each attesting a *distinct fact* about that pid (it produced the corrupt GEMM / it is a real
GPU compute process `VLLM::EngineCore` / it is a registered tenant). This is **identity-corroboration** (three sources
agree on *which process was running when it fired*), **not attribution-corroboration**: at `n_resident=1` only one
process is present, so the agreement is non-discriminating about *cause* (it confirms the tag, not who is to blame).
Multi-tenant disambiguation is examined in the probe (§E) and is by-construction there.

## (C) WHAT CIPHER ADDS OVER DCGM/NVML ALONE — and the honest framing

The DEGRADED verdict fired with **ECC=0, XID=none, throttle benign** — i.e. CIPHER flagged a GEMM-output corruption
that **no hardware counter recorded**. The defensible claim: **correctness-checking (independent recompute) is
orthogonal to hardware counters.** For the class of GEMM-output SDC that does not trip ECC/RAS (non-deterministic
compute corruption, certain mercurial-core arithmetic errors), the fused verdict surfaces DEGRADED where DCGM/NVRx —
which read ECC/XID/RAS — would report the GPU healthy. NVRx/DCGM watch the *hardware*; CIPHER's dispatch-boundary
detector watches the *answer*.

**The honesty boundary (do not over-read):** the fault injected here is a **software** bit-flip of a GEMM output, so
NVML reading counter-clean is *expected* — no hardware event occurred — and this run is **not** a demonstration that
CIPHER caught a *real silicon* SDC that DCGM missed. What it demonstrates is that the correctness signal is *independent
of and complementary to* the hardware-counter signal, shown with a flip that NVML correctly shows counter-clean (the
real NVML values were captured, never fabricated). The **corroborated branch** (SDC + ECC-uncorr↑/XID) is *specified in
the fusion rule but UNEXERCISED* — injecting a real ECC/XID would require hardware fault injection we do not have, and
fabricating one is forbidden. **Division of labor (stated):** deterministic "mercurial-core" faults that a same-GPU
recompute reproduces stay **blind** to the SDC signal (R.A caveat) — *there* DCGM RAS may catch some via hardware
counters, which is exactly why the fusion keeps NVML as a parallel source rather than discarding it.

## (D) FLEET AGGREGATION SCHEMA — the fleet-consumable event record  *(DONE clause 3)*

`rc.v1` per-GPU reliability event (emitted to `*_verdict.json`; a fleet collector aggregates by **`gpu.uuid`**, which is
fleet-unique unlike the ordinal):

```json
{ "schema_version":"rc.v1", "event_type":"gpu_reliability_event",
  "gpu":{ "node","gpu_index","uuid","pci_bus_id","name","driver" },
  "verdict":"DEGRADED|HEALTHY", "severity":"critical|warning|info",
  "fault_onset":{ "decode_step_detected","decode_step_confirmed","wall_clock_utc","mono_ns","covered_op" },
  "signal_sources":{
     "sdc_detector":{ "fired","observers":[{pid,max_residual,n_detections,persistence,n_clean_checks_before_onset}],
                      "coverage_note","threshold" },
     "nvml_ecc":{ "vol_uncorr_delta","vol_corr_delta","agg_uncorr_delta","agg_corr_delta" },
     "nvml_xid":{ "events":[] }, "nvml_throttle_reasons_seen":[], "nvml_power_clock":{},
     "telemetry_source":"NVML (DCGM/dcgmi absent on this pod)" },
  "corroboration":"uncorroborated_by_hardware_counters|corroborated_ecc_uncorrected|corroborated_xid",
  "process_context":{ "descriptor":{tgid,cmdline,fingerprint,descriptor_source,seen_by_nvml,in_w6_cohort},
                      "co_resident_count","co_resident_tgids":[], "nvml_compute_pids_seen":[],
                      "note":"DESCRIPTIVE not attributive; per-GPU; per-tenant is v2" },
  "scope_caveats":[ ... ] }
```

A fleet layer consumes a stream of these per (node, gpu.uuid): a HEALTHY heartbeat per GPU and a DEGRADED event when a
GPU starts producing silent corruption, carrying onset + descriptor + corroboration for triage/quarantine. The
**aggregation service and multi-GPU rollout are explicitly UNBUILT** here — this run validates the per-GPU event the
service would consume, not the service.

## (E) LOCALIZATION PROBE (v2-preview) — observation CLOSED-WITH-BOUND / causation WALL-WITH-MECHANISM

`probe_fusion.json` / `probe_verdict.json`: two co-resident vLLM contexts (`gpu_memory_utilization`=0.4 setpoint;
measured **35.2 GB each (44% of 80 GB)**, both fit, total 70 GB), **long 4096-token decodes** so their execution windows
overlap, inject p0 only, p1 clean. **Concurrent co-execution was verified by the pre-registered fault-window
timestamp-join** (not inferred from an end-of-run snapshot): taking p0's detection wall-times (steps 2200→4104, a
**46.6 s** fault window) and pulling every `probe_timeline.jsonl` sample inside it — **all 92 samples show
`n_resident=2` with BOTH contexts (`3323689`, `3323884`) present in BOTH the W.6 cohort AND NVML compute-procs
simultaneously**. The two contexts were genuinely running GEMMs on the same GPU throughout p0's entire fault window.

| Context | injected | pid | SDC detections | clean checks | verdict role |
|---|---|---|---|---|---|
| p0 | yes (ord 3) | 3323689 | **239** | clean before onset | **DEGRADED observer** |
| p1 | no | 3323884 | **0** | **0 / 514** | co-resident & concurrently executing, **NOT blamed** |

The verdict **localized the SDC observation to tgid 3323689**, **listed 3323884 as co-resident** (`co_resident_count=2`,
verified at the fault instant), and **did not blame it** — the clean context's detector read **0/514 even while
concurrently executing GEMMs on the same GPU as the corrupting context for 46.6 s**. **Co-residence does NOT smear the
SDC signal** across contexts, because cuBLAS interception is per-process: each context's detector sees only its own
dispatch stream. ⇒ **observation-localization CLOSED-WITH-BOUND.** *(An earlier probe attempt with a 20 s launch
stagger + short decode left p1 still loading during p0's brief fault window — n_resident was 1 at the fault instant; the
timestamp-join caught it, and the run was redone with overlapping long decodes so the co-residence is real at the fault
instant, not an end-of-run artifact.)*

**The bound (honest, per the v2 risk):** this is partly *by construction* — the software flip is injected into p0's own
GEMM output, so p0's detector necessarily observes it and p1's does not; this proves the *descriptive-not-attributive
boundary holds*, not that v2 per-tenant *attribution* is reachable. **Causation-localization is WALL-WITH-MECHANISM:**
a *shared-hardware* fault (a bad SM/HBM column) would be observed by *whichever context's kernel landed on the faulty
unit*, and the per-process SDC detector would attribute it to the **observing** context, not the **causing** one;
same-GPU recompute is additionally blind to the deterministic case. Software in-process injection cannot emulate a
shared-hardware fault, so v1 cannot separate causation — that is the precise v2 risk, de-risked here: **per-tenant blame
needs a causation signal the dispatch-boundary SDC detector does not carry.**

## (F) HOSTING COST (named, not this run's problem to solve)

The SDC signal is **eager-hosted** (a cuBLAS interceptor sees every steady-state decode GEMM only under
`enforce_eager`; under cudagraph the GEMMs replay via graph launch and the interceptor never fires). The documented R.A
costs apply unchanged: **+51% eager tax** (cudagraph 1289 vs eager 629 tok/s) + **~+3% detector-marginal @N=45**.
R.C's fusion *additions* over the R.A detector are **negligible**: the W.6 heartbeat is ≤1 ioctl/sec at step
boundaries; the monitor is a separate process sampling NVML+cohort every 0.5 s (no GPU work); per-check event emission
is 17 file-appends for one watched ordinal. The cudagraph-on path (removing the eager tax via Path-1 in-graph
injection) is WALL-WITH-MECHANISM per the prior arc (vLLM owns+re-processes its decode cudagraph; external structural
mutation is rejected) — **out of scope here and documented.**

---

## HONEST SCOPE STATEMENT

- **Per-GPU, NOT per-tenant.** v1 answers "GPU X is producing silent corruption, here is when and what was running." The
  process/context tag is **descriptive** (what ran), not **attributive** (who caused it). Per-tenant attribution is v2
  (causation WALL above).
- **ONE GPU validated.** The per-GPU detection+fusion+emission primitive is validated on one H100, plus the schema a
  fleet service consumes. **Multi-GPU scale and the aggregation/alerting service are UNBUILT** and named as such.
- **Eager hosting tax** +51% (Path-1 cudagraph-on walled) — documented, not solved here.
- **Blind spots (named, not dropped):** deterministic mercurial-core faults (same-GPU recompute reproduces them — DCGM
  RAS may catch some); attention-internal SDC (Gap-3, FlashAttn not cuBLAS); lm_head (excluded); NCCL/collective/
  multi-GPU. At short context the checked linear projections are >99% of decode FLOPs; the uncovered fraction
  (attention+lm_head) grows with context length.
- **Telemetry:** NVML real (ECC/XID/throttle/power/clock/compute-procs); **DCGM/dcgmi absent on this pod** — no DCGM
  values fabricated. The corroborated-by-hardware branch is specified but unexercised (no real hardware fault).

## FORK-1 INTEGRITY (md5, entry vs exit)

| File | entry | exit | match |
|---|---|---|---|
| `cipher_rt_phase4/libcipher_rt.so` (anchor) | `2edba0d2136f8ede4713d90a8f7cd55f` | `2edba0d2136f8ede4713d90a8f7cd55f` | ✅ |
| `ra_e2e/common.py` | `8276b0814020a8d9167a0edb2da58e85` | = | ✅ |
| `ra_e2e/run_e2e.py` | `2de53982eaf1aef14234edd8a5f6819a` | = | ✅ |
| `ra_e2e/calib.json` | `ade20e25357d108423210b4c2fea3ac9` | = | ✅ |
| `ra_e2e/trace.json` | `7d55b9120d05a74249db3dcc7618bb4d` | = | ✅ |
| `ra_e2e/run_e2e_summary.json` | `e5ac91b2d54407ab85f1d3ee82251d57` | = | ✅ |
| `ra_e2e/throughput.json` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | = | ✅ |
| `ra_e2e/detect.json` | `cefa60651cc8cb56885923164c070a4c` | = | ✅ |
| `ra_realvllm/detector_shim.c` (copied, not edited) | `0c10c1eeca0d04ea5ce041e74ab43b71` | = | ✅ |
| `ra_realvllm/vllm_decode.py` | `77f75eb9ae2ce1636d60c7c21aad8dd6` | = | ✅ |

`diff(ENTRY_MD5, EXIT_MD5)` = empty. New work confined to `rc_pergpu/`.

## GUARDRAILS (exit)

Anchor `2edba0d2…` unchanged (entry+exit md5). No vLLM source edit; scratch shim copied+extended in `rc_pergpu/` only;
`libcipher_rt.so` NOT loaded into vLLM; `CUDA_INJECTION64_PATH` unset, `VLLM_PLUGINS=""`, `VLLM_USE_DEEP_GEMM=0`. No
telemetry fabricated (DCGM absent → NVML only, stated). GPU exclusive at each baseline; clock 1980 during measurement →
`-rgc` at exit (report post-reset); no compute procs left; vLLM EngineCore workers reaped.

## BOTTOM LINE (restated)

Per-GPU reliability detection via signal fusion is **REAL**; fleet aggregation is **schema-ready** (the per-GPU event
is defined + emitted; the aggregation service is unbuilt): R.A SDC ⊕ W.6 cohort ⊕ NVML → one per-GPU verdict with
failing GPU id, fault onset, a descriptive context tag consistent across all three signals, and severity; injected SDC
caught and surfaced; 0 FP on clean; schema emitted. **CIPHER adds an online-correctness signal orthogonal to DCGM/NVML
hardware counters** (flags GEMM-output SDC that ECC/XID miss). **Walls/scope:** per-GPU not per-tenant (causation
WALL); ONE GPU validated, fleet scale + aggregation service UNBUILT; eager +51% hosting tax (Path-1 walled);
mercurial-core/attention-internal/NCCL blind. The localization probe makes v2 per-tenant attribution's risk precise:
the dispatch-boundary detector localizes *observation* cleanly but does not carry *causation*.
