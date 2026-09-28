# CIPHER Phase 3 — GPU-state Spine Telemetry

## Date / Environment
- Date: 2026-05-13
- Pod: Lambda H100 SXM5, 192.222.53.2
- Kernel: 6.8.0-1046-nvidia
- Driver: 580.105.08, CUDA 12.8.93, CUPTI 12.8.90
- Module version: cipher_kmod 0.3.1, srcversion B1AF5E2AAB2FB170A661E2A

## Goals Status

- MFU 5%→15-25%: PENDING Phase 4 (substrate landed here)
- TPW 2.96×→5-7×: PENDING Phase 4 (substrate landed here)
- Multi-tenant 8-16→60-150: PENDING Phase 5 (substrate landed here)
- O(1) Koopman: DEFERRED
- Driver-level LD_PRELOAD-free deployment: ACTIVE (CUDA_INJECTION64_PATH proven)

Phase 3's job was telemetry substrate. Phase 4 actuators consume this substrate.
Phase 5 (Green Contexts) builds on it for multi-tenant density.

## What Shipped

### Layer A — BAR0 reader (Task 2)
- cipher_kmod direct register access via pci_iomap, no nvidia.ko mediation
- Read-only enforcement (Refinement 2: no iowrite/memcpy_toio/raw_write symbols in module)
- Real silicon read confirmed: PMC_BOOT_0=0x180000a1, PMC_BOOT_1=0x00000000
- Live reads on PMC_INTR(0/1) and PBUS_INTR_STATUS
- Field decode (architecture, implementation, revision) deferred to Phase 6
  (one H100 = one data point, cannot validate field boundaries from single sample)
- Exposed via /proc/cipher/bar0_state

### Task 3 — Stable ABI extension
Three new ioctls on /dev/cipher, magic 'C':

- nr 5 SUBMIT_GPU_STATE      CAP_SYS_ADMIN, device-wide gauges
- nr 6 SUBMIT_PROCESS_UTIL   CAP_SYS_ADMIN, per-PID telemetry
- nr 7 SUBMIT_LAUNCH_STATS   anti-spoofed (pid+tgid must match current), per-tenant launches

Reserved nrs 2/3/4 (SNAPSHOT/RESET/GET_VERSION) remain -ENOSYS per cipher-abi-rule.

17/17 test cases pass (3 happy + 3 sudo-required happy + 6 non-root EPERM/ENOSYS/ENOTTY + 5 sudo EINVAL).

### Layer B — cipher-gpustate daemon (Task 4)
Userspace process, 21 KB binary, libnvidia-ml only.

- Polls NVML at 250 ms (drift-corrected deadline)
- Device-wide state via nvmlDeviceGet{Power,Temp,Clock,Util,Memory}Info
- Per-process util via nvmlDeviceGetProcessUtilization with two-call dynamic buffer pattern
- Submits via the three Task 3 ioctls
- Signal-handled clean shutdown (SIGINT/SIGTERM)
- CLI: --device, --interval-ms, --verbose, --once

Observed NVML quirks documented:
  - nvmlDeviceGetProcessUtilization with single-call fixed buffer fails INSUFFICIENT_SIZE
    on pods with many historical CUDA contexts, even with timestamp window
  - Two-call dynamic buffer pattern (utilization=NULL → query size → alloc → re-query) is
    the documented workaround
  - NVML reports per-process samples for recently-exited PIDs (stale cache, ~minutes)

### Layer C — libcipher_v2 CUPTI integration (Task 5)
CUDA injection library loaded via CUDA_INJECTION64_PATH.

- InitializeInjection2() calls REGISTER_TENANT (Phase 2 functionality, preserved)
- cipher_v2_cupti_init() subscribes a CUPTI callback at cuInit time
- Callback registered for CUPTI_CB_DOMAIN_RUNTIME_API:
    cudaLaunchKernel_v7000 (cbid 211)
    cudaLaunchKernelExC_v11060 (cbid 430)
  And CUPTI_CB_DOMAIN_DRIVER_API:
    cuLaunchKernel (driver path for TensorRT etc.)
- Per-launch atomic increment of g_launches_total
- Flush every 256 launches via SUBMIT_LAUNCH_STATS ioctl
- No flush thread (would break anti-spoof identity match);
  callback fires on workload thread, gettid()/getpid() naturally match

Observed nvidia.ko quirk: when CUPTI subscribes against an active CUDA context,
nvidia.ko logs NVRM: refcntRequestReference_IMPL: Failed to enter state 1, status: 0x56.
Benign — workload completes cleanly. Documented for record.

### Layer D — cipher-exporter Prometheus surface (Task 6)
Python 3 stdlib, no third-party deps. http.server on :9402.

Endpoints:
  GET /metrics   Prometheus exposition format
  GET /health    "ok" or "no cipher_kmod"
  *              404

Metrics:
  Device-wide (8 gauges): power, temp, clocks, util, fb_used, age
  Per-tenant (6 series): sm_util, mem_util, ioctl_total, launches_total, age, telemetry_age
  Module uptime: 1 gauge

## End-to-End Demo Receipt (Stage 5b)

Two-tenant TinyLlama-style workload, 30 seconds, single H100.

Mid-run /metrics output:
  cipher_gpu_power_watts{device="0"} 576.262        ← workload (idle 70 W)
  cipher_gpu_sm_util_pct{device="0"} 100             ← saturated
  cipher_gpu_sm_clock_mhz{device="0"} 1980           ← max boost

  cipher_tenant_sm_util_pct{tenant="tenant-A",...} 40
  cipher_tenant_sm_util_pct{tenant="tenant-B",...} 60     ← live split

  cipher_tenant_launches_total{tenant="tenant-A",...} 512
  cipher_tenant_launches_total{tenant="tenant-B",...} 256  ← CUPTI accounting

Workload fairness: 658 vs 657 iters (0.15% delta), proving CUDA's fair scheduler
is what the per-tenant SM% split is observing.

Stability across the full run:
  - Tenant A exit 0, Tenant B exit 0, Daemon exit 0, Exporter exit 0
  - Taint stable at 12288
  - 0 ioctl errors
  - 0 INSUFFICIENT_SIZE warnings
  - Zero kernel WARN/oops/BUG
  - All fallback md5s unchanged
  - Peak GPU power 615 W (well within H100 SXM5 700 W envelope)

## Known Limitations / Phase 4+ Scope

1. PMC_BOOT_0 field decode deferred to Phase 6 (need multi-architecture data points)
2. NVRM refcnt warning on CUPTI subscribe is benign but documented for future investigation
3. cipher-gpustate ghost PID cleanup: NVML's stale-sample-cache can cause
   /proc/cipher/stats to show a row for an exited PID for ~minutes.
   Phase 6 scope: reap stale telemetry > N seconds old.
4. grid_ops_total in LAUNCH_STATS is wired but always 0 (parser for kernel
   launch param structs not implemented; deferred)
5. Layer A live-reads only on read; Phase 4+ may want a kernel-thread for
   polling-style sampling

## Files Shipped

cipher_kmod/                         Kernel module v0.3.1
  cipher_main.c
  cipher_dev.c (REGISTER_TENANT + 3 SUBMIT ioctls)
  cipher_ioctl.h (public ABI)
  cipher_bar0.c (Layer A)
  cipher_proc.c (/proc/cipher/{stats,bar0_state,gpu_state})
  cipher_internal.h
  Makefile

cipher_gpustate/                     Userspace daemon
  cipher_gpustate.c
  Makefile
  cipher-gpustate.service

libcipher_v2/                        CUDA injection lib
  cipher_inject.c
  cipher_tenant.c
  cipher_cupti.c (Phase 3 Task 5 addition)
  cipher_v2_internal.h
  Makefile

cipher_exporter/                     Prometheus exporter
  cipher-exporter.py
  Makefile
  cipher-exporter.service

Tests:
  /tmp/cipher_test_happy.c
  /tmp/cipher_test_root.c
  /tmp/cipher_test_negative.c

## Build / Run Pipeline

cd cipher_kmod && make
sudo insmod cipher_kmod.ko
sudo chmod 666 /dev/cipher

cd cipher_gpustate && make
sudo ./cipher-gpustate --verbose &

cd cipher_exporter
python3 cipher-exporter.py --port 9402 --bind 0.0.0.0 &

Then workloads:
CIPHER_TENANT_ID=<name> \
  CUDA_INJECTION64_PATH=$PWD/libcipher_v2/libcipher_v2.so \
  <your CUDA application>

## Strategic Read

Phase 3 is substrate. The numbers visible in /metrics today are observation,
not optimization. Phase 4 (persistent kernel routing) and Phase 5 (Green
Contexts) consume this substrate to drive MFU/TPW/density improvements. The
product story for neoclouds: "Look, we attribute GPU work to tenants in real
time, at kernel-launch granularity, via Prometheus. Now imagine Phase 4
actuators using this signal to optimize."
