# CP 3.3 — parked audit findings (work paused 2026-05-15)

Status: **PARKED.** CP 3.3 (PMU programming for continuous FLOP counting) work
was stopped mid-audit by discipline reset. Phase 0 (CP 0.4 / 0.5 / 0.6) must
close and be adjudicated first. This note preserves the audit-before-build
findings so CP 3.3 resumes without re-investigation.

## Audit findings (verified on disk, 2026-05-15)

1. **Kmod-direct hardware FLOP counting is structurally blocked.**
   `~/ext/open-gpu-kernel-modules/src/common/inc/swref/published/hopper/gh100/dev_perf.h`
   is a **29-line stub** — one define (`NV_PERF_PMMSYSROUTER_NUM_USER_STREAMING_CHANNELS 9`),
   no SM perfmon / PM-counter PRI register map. No GR/SM activity-counter
   headers are published for GH100 either. A kmod cannot read FLOP counters
   off BAR0 without that map; writing unknown PRI registers risks GPU hangs.
   This is code-level evidence of the kmod-direct gap — NOT non-buildability.

2. **CUPTI PM Sampling is present** — the supported continuous, device-wide,
   hardware-counter FLOP path on H100:
   - `/usr/include/cupti_pmsampling.h`, `/usr/include/cupti_profiler_host.h`,
     `/usr/include/cupti_profiler_target.h` all present.
   - `/usr/lib/x86_64-linux-gnu/libcupti.so.12` present; `libcupti-dev` installed.
   - `cupti_profiler_target.h` exposes the Range Profiling API too, but Range
     Profiling is per-CUDA-context — it cannot continuously sample a device
     running *other processes'* workloads. PM Sampling is process-independent
     and is therefore **required**, not optional, for device-wide continuous
     FLOP telemetry.

3. **Profiling privilege:** `/sys/module/nvidia/parameters/` is empty (0
   entries) on this pod — `NVreg_RestrictProfilingToAdminUsers` could not be
   read. Plan: run the FLOP agent as **root** (it is a standalone telemetry
   daemon, not the CUDA_INJECTION64_PATH user-process path — root is
   acceptable). Confirm empirically: a non-privileged CUPTI profiler call
   returns `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` if restricted.

4. **Existing CUPTI code does NOT cover this.** `libcipher_v2/cipher_cupti.c`
   (136 lines) uses the CUPTI **callback** API (`cuptiSubscribe` +
   `cuptiEnableCallback`) — it counts kernel launches and flushes them via
   `CIPHER_SUBMIT_LAUNCH_STATS` (ioctl nr 7). The Profiler / PM Sampling API
   is a separate CUPTI subsystem; the launch-callback code cannot be extended
   to read hardware counters. A fresh PM Sampling agent is needed.

5. **Kmod already has the per-tenant attribution basis.**
   `struct cipher_pid_stats` (cipher_internal.h) carries `launches_total`
   ("from LAUNCH_STATS, CUPTI snapshot") and `grid_ops_total` per tenant.
   Per-tenant FLOP attribution = `device_flops × (tenant launches_total /
   Σ launches_total)` — proportional, gate-sufficient. Precise per-tenant
   hardware FLOP accounting is a separate multi-week effort; out of CP 3.3.

## Decided architecture (hybrid)

- **`cipher_flopd`** — root userspace daemon. CUPTI PM Sampling of the H100's
  FP/tensor instruction counters (e.g. `sm__sass_thread_inst_executed_op_
  {ffma,fadd,fmul}_pred_on`, tensor HMMA counters) at a fixed cadence
  (~100 ms). Computes achieved FLOPs/s. Also reads NVML (sm_util, clock,
  power) for the cross-check.
- **New additive kmod ioctl** (fresh nr, ABI additive-only rule) —
  `CIPHER_SUBMIT_FLOP_SAMPLE`-style. Kmod holds a continuous FLOP-series ring
  buffer; the kmod **owns the series**.
- **Per-tenant MFU** — kmod derives per-tenant FLOPs from device FLOPs ×
  launch-share, MFU = achieved / peak; exposed via `/proc/cipher` + the
  tenant snapshot.
- **Gate:** continuous FLOP series exists, kmod-owned; NVML cross-check shows
  the FLOP-rate tracks `sm_util_pct × sm_clock_mhz`.

## Resume discipline (per 2026-05-15 reset)

CP 3.3 resumes only AFTER CP 0.6 is adjudicated. Then full atomic STEP:
audit (this note) → design memo → build → gate → one report. Kmod baseline:
save the current `.ko` outside the build dir before rebuild (kbuild clean
wipes `*.ko`); record the new artifact md5 in the CP 3.3 report; anchor
`b263ad30…` is the pre-CP-3.3 baseline.
