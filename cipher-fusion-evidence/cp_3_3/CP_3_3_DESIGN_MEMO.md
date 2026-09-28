# CP 3.3 — Design Memo: continuous hardware FLOP counting + per-tenant MFU

**Date:** 2026-05-15. **Status:** DESIGN — awaiting approval. No C is written
until this memo is approved.

CP 3.3 canonical scope: "PMU programming for continuous FLOP counting." Mandate
elaboration: real hardware FLOP counters, continuous, confirmed against NVML,
**per-tenant MFU as a free byproduct of existing telemetry**.

---

## 0. Architecture (recap of the parked audit)

The audit (`PARKED_AUDIT_NOTES.md`) established: kmod-direct FLOP counting is
blocked — the GH100 SM perfmon PRI register map is not published
(`hopper/gh100/dev_perf.h` is a 29-line stub), and writing un-mapped PRI
registers risks GPU hangs. The supported continuous hardware-counter path on
H100 is **CUPTI PM Sampling** (`libcupti.so.2025.1.1`, `cupti_pmsampling.h` —
both present). PM Sampling is device-wide and process-independent; Range
Profiling is per-CUDA-context and cannot sample other processes' work, so PM
Sampling is required, not optional.

**Hybrid design — two components, one new kmod TU:**

```
  ┌─────────────────────┐   CUPTI PM Sampling    ┌──────────────┐
  │  cipher_flopd        │◀──(HW counters)───────│  H100 GPU    │
  │  (root userspace     │                        │  PM units    │
  │   daemon, ~10 Hz)    │──NVML(util,clk,pwr)───▶│              │
  └──────────┬──────────┘                         └──────────────┘
             │ ioctl CIPHER_SUBMIT_FLOP_SAMPLE  (nr 11, ~10 Hz, cold path)
             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  cipher_kmod  (new TU: cipher_flops.c)                    │
  │  • device FLOP-series ring  (owns the continuous series)  │
  │  • per-tenant attribution via existing launches_total     │
  │  • /proc/cipher/flops   • ioctl CIPHER_QUERY_FLOPS (nr 12) │
  └─────────────────────────────────────────────────────────┘
```

The hardware-counter *read* happens in `cipher_flopd` because CUPTI is the only
access NVIDIA provides; the kmod **owns the continuous FLOP series and derives
the per-tenant metric**. That is the honest reading of "from cipher_kmod."

**Why per-tenant MFU is "free."** The kmod *already* keeps per-tenant
`launches_total` in `struct cipher_pid_stats` (fed by the existing CUPTI
launch-callback path, ioctl nr 7). CP 3.3 adds exactly one device-wide number —
achieved FLOPs/s — and the per-tenant split falls out of launch counts the kmod
already has. No new per-tenant instrumentation. That is the line that matters:
real-time per-tenant MFU as a byproduct of telemetry already flowing.

---

## 1. CUPTI PM Sampling API surface used

Confirmed present in `/usr/include/cupti_pmsampling.h` +
`cupti_profiler_host.h` (CUPTI 2025.1.1, `CUPTI_API_VERSION 26`). All CUPTI
calls live in `cipher_flopd` (userspace); `libcupti.so` is `dlopen`'d so a
missing/old CUPTI degrades gracefully (see §5).

**Config-image build (Profiler Host API — done once at startup):**
- `cuptiProfilerHostInitialize` — host object for the GH100 chip.
- `cuptiProfilerHostGetSupportedMetrics` / `cuptiProfilerHostGetBaseMetrics` —
  **enumerate which FLOP metrics are actually PM-samplable on this GH100** (the
  metric-set decision is validated here at build STEP 1, not assumed).
- `cuptiProfilerHostConfigAddMetrics` — add the selected FLOP metrics.
- `cuptiProfilerHostGetConfigImageSize` + `cuptiProfilerHostGetConfigImage` —
  emit the opaque config-image blob (`pConfig`).

**PM Sampling lifecycle (device, continuous):**
- `cuptiPmSamplingEnable` — bind a `CUpti_PmSampling_Object*` to device 0.
- `cuptiPmSamplingGetCounterAvailability` — counter-availability image.
- `cuptiPmSamplingSetConfig` — `CUpti_PmSampling_SetConfig_Params{ pConfig,
  configSize, hardwareBufferSize, samplingInterval (ns), triggerMode,
  hwBufferAppendMode }`. `samplingInterval` ≈ 1 ms hardware sample; `triggerMode`
  = interval-based.
- `cuptiPmSamplingGetCounterDataSize` + `cuptiPmSamplingCounterDataImageInitialize`
  — allocate/init the counter-data image the decoder fills.
- `cuptiPmSamplingStart` — GPU begins sampling its PM units into the HW buffer.
- **Decode loop (~10 Hz in `cipher_flopd`):** `cuptiPmSamplingDecodeData`
  (drain HW buffer → counter-data image) → `cuptiPmSamplingCounterDataGetSampleInfo`
  (per-sample timestamps/ranges) → `cuptiProfilerHostEvaluateToGpuValues`
  (counter-data → metric values).
- `cuptiPmSamplingStop` / `cuptiPmSamplingDisable` on shutdown.

**Metrics selected (validated against `GetSupportedMetrics` at build STEP 1):**
- *Primary — exact instruction counts (if PM-samplable):*
  `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` (×2 FLOP),
  `_fadd_pred_on.sum`, `_fmul_pred_on.sum` (×1), and the `h*` fp16 variants.
- *Tensor:* `sm__inst_executed_pipe_tensor.sum` (or the GH100 HMMA op metric),
  converted with the documented per-HMMA FLOP factor.
- *Fallback if instruction counters are not PM-samplable on this GH100:*
  pipe-active metrics `sm__pipe_fma_cycles_active`,
  `sm__pipe_tensor_cycles_active` + `sm__cycles_elapsed` → achieved-FLOP
  estimate. Decision rule is in the memo; the build records which path was taken.
- `achieved_FLOPs/s` = FP instruction FLOPs + tensor FLOPs, integrated over the
  sample interval. Both an FP component and a tensor component are submitted to
  the kmod separately so the split is visible.

---

## 2. cipher_kmod ioctl interface for FLOP queries

New TU `cipher_flops.c`. Two new ioctls on `/dev/cipher`, magic `'C'`,
**additive only** — next-free nrs 11 and 12; reserved nrs 2/3/4 remain
`-ENOSYS`; the ABI rule (`cipher-abi-rule`) is honored.

**`CIPHER_SUBMIT_FLOP_SAMPLE` — nr 11, `_IOW`, CAP_SYS_ADMIN.**
Writer: `cipher_flopd` only (CAP check, same gate as `SUBMIT_GPU_STATE`).
Payload:
```c
struct cipher_flop_sample {
    __u64 timestamp_ns;        /* daemon CLOCK_MONOTONIC at sample */
    __u64 device_fp_flops;     /* FP-pipe FLOPs in this interval   */
    __u64 device_tensor_flops; /* tensor-pipe FLOPs in this interval */
    __u64 interval_ns;         /* span this sample integrates over */
    __u32 sm_clock_mhz;        /* NVML, same instant — for cross-check */
    __u32 sm_util_pct;         /* NVML, same instant — for cross-check */
    __u32 power_mw;            /* NVML, same instant */
    __u32 source_seq;          /* daemon monotonic seq; gap-detect */
};
```
On receipt the handler (cold path, ~10 Hz): pushes the sample to the device
FLOP ring; walks the pid hashtable computing per-tenant launch deltas and
attributed FLOPs (§4 algorithm); updates per-tenant fields. ~150 tenants × a
few words — microseconds, off the kprobe hot path entirely.

**`CIPHER_QUERY_FLOPS` — nr 12, `_IOWR`, unprivileged read.**
For programmatic consumers (the Prometheus exporter, arbitration). Returns a
fixed-size header (latest device sample + derived device MFU) and, if the
caller supplies a buffer, up to `max` `struct cipher_flop_tenant` rows
(pid, tgid, tenant_id, launch_share_ppm, attributed_flops_per_s, mfu_ppm).
Snapshot copy under the hashtable lock; process context only.

The kmod **owns the continuous series**: a fixed ring of the last 256 device
samples (`struct cipher_flop_sample`), ~25 s at 10 Hz — the literal "continuous
FLOP counting" the CP names. Ring + per-tenant state live in `cipher_flops.c`,
spinlock-guarded; readers use the lock (cold path).

---

## 3. /proc/cipher/flops layout

New read-only seq_file `/proc/cipher/flops` (alongside `stats`, `bar0_state`,
`gpu_state`). Plain text, two blocks:

```
# CIPHER FLOP telemetry — schema 1
device  ts_age_ms=83  fp_tflops=12.4  tensor_tflops=ְ301.7  total_tflops=314.1
device  mfu_pct=31.76  hfu_pct=47.59  sm_clock_mhz=1980  sm_util_pct=88  power_w=512
device  series_samples=256  series_span_s=25.6   # the continuous ring

# per-tenant  (attributed: device_flops × launch-share)
PID     TGID    TENANT            LAUNCH_SHARE%  FLOPS/s        MFU%
181234  181200  mistral-tenant-A  41.2           1.294e14       13.09
181251  181200  mistral-tenant-B  33.8           1.062e14       10.74
181270  181255  mistral-tenant-C  25.0           7.852e13       7.94
...
total                              100.0          3.141e14       31.76
```

Header block = the device-wide hardware truth (from CUPTI). Per-tenant block =
device FLOPs split by each tenant's launch share. `MFU% = attributed_flops /
989e12 × 100`; `HFU%` vs the pod's 660 TFLOPS cap. The seq_file walks the pid
hashtable under RCU; cold path.

Honest label in the file header: per-tenant FLOPs are **attributed**
(device-wide measured × launch-count share), not a hardware per-context
measurement — CUPTI PM Sampling is device-wide. This is proportional
attribution; it is gate-sufficient and is the "free byproduct" — precise
per-context hardware FLOP accounting is out of CP 3.3 scope.

---

## 4. NVML cross-validation methodology

**NVML has no FLOP field** — it cannot directly ground-truth a FLOP counter.
The cross-validation is therefore two-tier, and the memo states this honestly
rather than implying NVML measures FLOPs:

**Tier 1 — consistency (NVML as the "is the GPU doing work" oracle).**
Every `cipher_flop_sample` carries NVML `sm_util_pct`, `sm_clock_mhz`,
`power_mw` captured at the same instant. Gate check: across the telemetry
stream the CUPTI achieved-FLOP rate must track `sm_util_pct × sm_clock_mhz`
monotonically — high when the GPU is busy at high clock, ≈0 when NVML reports
idle. A FLOP series that stays high while NVML says idle (or vice-versa) fails
the gate. Quantified: Pearson r between `total_flops` and `sm_util×clock`
across a mixed workload, gate r ≥ 0.95.

**Tier 2 — absolute accuracy (analytical microbenchmark as ground truth).**
NVML cannot do this; an analytically known kernel can. Run a fixed fp16 GEMM of
known size (e.g. 4096³ → 2·4096³ = 137.44 GFLOP/call) for a known call count C.
The kmod's device FLOP ring, integrated over the run window, must equal
`C × 137.44 GFLOP` within tolerance. Gate: integrated CUPTI FLOPs within **±10%**
of the analytical count (±10% absorbs the tensor-pipe conversion factor and
PM-sampling quantization; tightened if the instruction-count metric path is
available, which is exact for the FP component).

Both tiers run in the CP 3.3 gate harness; both results go in the report.

---

## 5. Failure modes

1. **CUPTI library / API version drift.** PM Sampling is a CUDA-12.6+ API; the
   `_Params` structs are versioned by `_STRUCT_SIZE`. Mitigation: `cipher_flopd`
   `dlopen`s `libcupti.so`, resolves PM-sampling symbols by name, and checks
   `CUPTI_API_VERSION`. Missing symbols / version mismatch → daemon logs once,
   exits cleanly. The kmod is unaffected: `/proc/cipher/flops` shows
   `device  status=no-flop-source` and the per-tenant FLOP/MFU columns read
   `n/a`. Launch counting and all other telemetry keep working.

2. **Profiling privilege denied.** `/sys/module/nvidia/parameters/` was empty on
   this pod, so `NVreg_RestrictProfilingToAdminUsers` could not be pre-read.
   `cipher_flopd` runs as **root** (it is a standalone telemetry daemon, not the
   `CUDA_INJECTION64_PATH` user-process path — root is acceptable). If CUPTI
   still returns `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` / `NOT_SUPPORTED`, the
   daemon logs the exact error once and degrades exactly as failure mode 1
   (kmod shows `status=privilege-denied`). The gate report records the observed
   privilege state as a finding either way.

3. **Sampling-rate vs overhead tradeoff.** Two distinct costs:
   - *Kmod hot path:* **zero added.** CP 3.3 touches only cold paths — a new TU,
     two new ioctls (called ~10 Hz by the daemon), a new proc file. The kprobe
     handler on `nvidia_unlocked_ioctl` gets **no new instructions**; this is
     verified by diffing the handler pre/post. The ">1% hot-path overhead" gate
     is met by construction.
   - *GPU/workload cost:* PM Sampling runs on the GPU's PM hardware; NVIDIA
     documents it as low- but non-zero-overhead on the sampled workload. Faster
     `samplingInterval` → more HW-buffer pressure + more daemon decode CPU.
     Chosen operating point: ~1 ms hardware sample, decode/submit at ~100 ms
     (10 Hz). The gate measures workload overhead with PM Sampling on vs off
     (a fixed GEMM loop) and reports it; target ≤1% on the workload.

4. **(noted) Per-tenant attribution is proportional, not per-context.** Already
   stated in §3 — recorded here as a known modelling limitation, not a bug. If a
   tenant launches few but very heavy kernels its FLOP share is under-counted
   relative to a tenant with many light kernels. Gate-sufficient for real-time
   MFU; precise per-context FLOP accounting is a separate effort.

---

## 6. Build plan (atomic STEP), gate, anchors

**Build (single atomic STEP, executed only after approval):**
1. Enumerate PM-samplable FLOP metrics via `cuptiProfilerHostGetSupportedMetrics`
   on this GH100; lock the metric set; record which path (exact instruction
   counts vs pipe-active fallback) was taken.
2. `cipher_flopd.c` — root daemon: CUPTI PM Sampling + NVML + the nr-11 ioctl.
3. `cipher_kmod`: new TU `cipher_flops.c` (ring, attribution, nr 11/12 handlers,
   `/proc/cipher/flops`); `cipher_flop_sample`/`cipher_flop_tenant` +
   ioctl defs in `cipher_ioctl.h`; new fields in `struct cipher_pid_stats`
   (`launches_at_last_flop`, `attributed_flops_per_s`); wire into `cipher_dev.c`,
   `cipher_proc.c`, `cipher_main.c`, `Kbuild`.
4. Build clean; rmmod/insmod; gate.

**Gate (all must pass for CP 3.3 SHIPPED):**
- (a) Continuous device FLOP series present and kmod-owned — `/proc/cipher/flops`
  shows a populated 256-sample ring.
- (b) Per-tenant FLOPs + MFU visible at `/proc/cipher/flops`, real time, under a
  live multi-tenant Mistral workload.
- (c) NVML cross-check Tier 1: r ≥ 0.95 vs `sm_util×clock`. Tier 2: integrated
  FLOPs within ±10% of an analytical GEMM count.
- (d) Kmod hot path: kprobe handler byte-unchanged (diff); workload overhead
  with PM Sampling on vs off ≤1%.

**Anchors:** `55ab8c0c` + `86618c30` frozen — untouched. Current
`cipher_kmod.ko` md5 `b263ad30453d9f620d4f279627c7258e` is the pre-CP-3.3
baseline; it is copied **outside** the build dir before rebuild (kbuild `clean`
globs `*.ko`). The CP 3.3 report records the fresh post-build `.ko` md5 as the
new working baseline; `b263ad30…` is preserved as the CP 3.3 rollback point.

---

**Approval requested.** No C is written until this memo is approved. On
approval, the build is one atomic STEP, then the gate, then one report.
