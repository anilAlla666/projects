# Phase 3 — GPU-State Spine + BAR0 + PMU Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four complementary telemetry surfaces on top of Phase 2 (BAR0 reads in cipher_kmod; NVML/GPM daemon; CUPTI callbacks in libcipher_v2; Prometheus exporter) such that per-tenant GPU MFU is observable from a Grafana dashboard while Phase 1/1.5/2 functionality is preserved and total overhead stays <2 % on TinyLlama smoke.

**Architecture:** Phase 2's `/dev/cipher` ABI grows **additively** with three new ioctls (nrs 5/6/7) that accept GPU-state, per-process util, and CUPTI launch-counter payloads from userspace daemons. cipher_kmod 0.3.0 adds a read-only BAR0 mapper alongside the existing kprobe + tenant infrastructure. libcipher_v2 0.3.0 optionally `dlopen`s libcupti and registers a kernel-launch callback. Two new userspace daemons (cipher-gpustate in C/Python, cipher-controller in Python) feed the kernel and serve `/metrics`.

**Tech Stack:** Linux 6.8 kernel modules (Kbuild), gcc, NVML (libnvidia-ml 580.105.08), CUPTI (libcupti.so.12), Python 3.10 (http.server + ctypes), Prometheus text format, Grafana dashboard JSON.

---

## Environment confirmed (Phase 3a — research)

- Pod: Lambda H100 80GB SXM5 @ `0000:07:00.0`, driver 580.105.08, kernel 6.8.0-1046-nvidia.
- BAR0 sysfs resource: `0x6002000000`–`0x6002ffffff` (16 MB exactly), 64-bit memory mapped (flags `0x14220c`).
- NVML GPM symbols present: `nvmlGpmMetricsGet`, `nvmlGpmSampleAlloc/Get/Free`, `nvmlGpmQueryDeviceSupport`, `nvmlDeviceGetProcessesUtilizationInfo`, plus the basic `nvmlDeviceGet{UtilizationRates,PowerUsage,Temperature,ClockInfo,MemoryInfo}`.
- CUPTI: `libcupti.so.12` at `/lib/x86_64-linux-gnu/`; header at `/usr/include/cupti.h`.
- Go is **not** installed → Layer D uses Python `http.server` per the spec's documented fallback.
- Phase 2 source lives inside `cipher-phase2-evidence/` (the live tree was archived). Task 1 restores it to `/home/ubuntu/cipher_kmod/` and `/home/ubuntu/libcipher_v2/`.

## Spec ambiguities (resolved)

- **Path discrepancy:** Spec writes `/workspace/cipher_kmod/...`; actual project home is `/home/ubuntu/`. Plan uses `/home/ubuntu/` throughout.
- **Layer D language:** Go absent on pod. Plan uses Python http.server (explicitly allowed by spec, "rewrite in Go in Phase 7").
- **cipher-gpustate language:** Plan uses **C** (consistent with rest of stack, links `-lnvidia-ml` cleanly, lowest poll-overhead). Python was an option but C is cheaper and the spec prefers consistency.
- **GPU util storage (design choice 1):** Plan picks option (a) — add `sm_util_pct` field directly to `cipher_pid_stats` (4 extra bytes per entry, simplest lookup).
- **Polling interval (design choice 5):** Plan keeps 250 ms (spec default). 100 ms produced stale samples in prior research; 500 ms is too coarse for the Grafana refresh.

## Plan refinements (post-approval, 2026-05-13)

**Refinement 1 — No auto-rollback in smoke script.** `phase_3_smoke.sh` MUST NOT rmmod-and-roll-back automatically on failure. On any failure: log everything, leave the system in the failing state, alert the operator with a clear message. Manual rollback only. (Rationale: spurious failures could lose hours of work if the script eagerly tears down state.)

**Refinement 2 — BAR0 code-level read-only enforcement.** `cipher_bar0.c` must contain:
- Top-of-file comment: `"READ-ONLY. WRITES TO BAR0 WILL CORRUPT NVIDIA.KO STATE. PHASE 3 IS OBSERVATION ONLY."`
- Only `ioread32` (no `iowrite32`, no `__raw_writel`, no `memcpy_toio`).
- A `static inline u32 cipher_bar0_rd32(const void __iomem *base, u32 offset)` helper that takes `const void __iomem *` — `const` makes write-through impossible at the type level. All register reads in the file use this helper.

**Refinement 3 — Layer C overhead measurement protocol.** After Layer C ships (Task 5), measure overhead with vs without CUPTI:
- Run TinyLlama smoke 5 times with `CIPHER_V2_NO_CUPTI=1` (CUPTI disabled).
- Run TinyLlama smoke 5 times with default (CUPTI active).
- Compare median tps; compute delta percentage.
- **If delta > 2 %:** document the overhead, flip the default — CUPTI becomes opt-in via `CIPHER_V2_ENABLE_CUPTI=1`, and `CIPHER_V2_NO_CUPTI` becomes unnecessary.
- **If delta ≤ 2 %:** keep CUPTI on by default; `CIPHER_V2_NO_CUPTI=1` remains the opt-out.
- Report the numbers regardless.

---

## File Structure

### cipher_kmod 0.3.0 (`/home/ubuntu/cipher_kmod/`)

- Create: `cipher_bar0.c` — Layer A BAR0 mapper + reader (read-only, `pci_iomap` of BAR0, reads 6 well-known offsets).
- Modify: `cipher_main.c` — bump `MODULE_VERSION` to `"0.3.0"`, init/cleanup hooks for BAR0 mapper, banner update.
- Modify: `cipher_ioctl.h` — add nrs 5/6/7 and three new ABI structs.
- Modify: `cipher_internal.h` — extend `cipher_pid_stats` with `sm_util_pct`, add `cipher_gpu_state_slot[MAX_GPUS]` global, add CUPTI counter fields.
- Modify: `cipher_dev.c` — three new ioctl handlers with CAP_SYS_ADMIN / anti-spoof checks.
- Modify: `cipher_proc.c` — extend `/proc/cipher/stats` (new columns), add `/proc/cipher/gpu_state` and `/proc/cipher/bar0_state` files.
- Modify: `Kbuild` — add `cipher_bar0.o` to objs.

### libcipher_v2 0.3.0 (`/home/ubuntu/libcipher_v2/`)

- Create: `cipher_cupti.c` — optional dlopen of libcupti, callback subscriber, lockless atomic counters, 1-second flush thread.
- Modify: `cipher_inject.c` — call `cipher_cupti_init()` after tenant registration (best-effort, never blocks).
- Modify: `cipher_v2_internal.h` — declare cupti hooks + counter struct.
- Modify: `Makefile` — bump version, optional CUPTI flag, link `-ldl -lpthread`.

### cipher_gpustate (`/home/ubuntu/cipher_gpustate/`, new)

- Create: `cipher_gpustate.c` — main loop, NVML init, 250 ms poll, builds `cipher_gpu_state` + `cipher_process_util` payloads, submits via ioctl.
- Create: `Makefile` — links `-lnvidia-ml -lpthread`.
- Create: `cipher-gpustate.service` — systemd unit (documented but not auto-installed; Phase 3 invokes manually).

### cipher_controller (`/home/ubuntu/cipher_controller/`, new)

- Create: `cipher_controller.py` — Python http.server on :9402, parses `/proc/cipher/stats` + `/proc/cipher/gpu_state` + `/proc/cipher/bar0_state`, emits Prometheus text exposition format.
- Create: `grafana_dashboard.json` — 7-panel dashboard (per-tenant SM util, per-tenant launch rate, GPU tensor util, GPU power, HBM util, RM_ALLOC:RM_FREE ratio, per-tenant ioctl breakdown).
- Create: `README.md` — minimal: how to start, port, scrape URL.

### Smoke + close-out (`/home/ubuntu/`)

- Create: `phase_3_smoke.sh` — full-stack integration test driver (insmod 0.3.0, start cipher-gpustate, run TinyLlama smoke with CUDA_INJECTION64_PATH+CIPHER_TENANT_ID, scrape :9402/metrics, compare tps to Phase 2 baseline, rmmod, restore 0.2.0 if anything fails).
- Create: `PHASE_3_NOTES.md` — close-out prose, 6 sections, ~250 lines.
- Create: `cipher-phase3-evidence.tar.gz` + sha256.

---

## Task Decomposition (7 high-level tasks)

### Task 1: Restore working tree + version stamp + .ko fallback preservation

**Files:**
- Modify: `/home/ubuntu/cipher_kmod/cipher_main.c` (after restore, bump version)
- Preserve: `/home/ubuntu/cipher_kmod/cipher_kmod.ko.v0.2.0`

- [ ] **Step 1.1** Restore Phase 2 working tree
  ```bash
  cp -r /home/ubuntu/cipher-phase2-evidence /home/ubuntu/cipher_kmod_phase2_snapshot  # immutable reference
  mkdir -p /home/ubuntu/cipher_kmod /home/ubuntu/libcipher_v2
  cp /home/ubuntu/cipher-phase2-evidence/*.c /home/ubuntu/cipher_kmod/
  cp /home/ubuntu/cipher-phase2-evidence/*.h /home/ubuntu/cipher_kmod/
  cp /home/ubuntu/cipher-phase2-evidence/Kbuild /home/ubuntu/cipher_kmod/
  cp /home/ubuntu/cipher-phase2-evidence/Makefile /home/ubuntu/cipher_kmod/
  cp -r /home/ubuntu/cipher-phase2-evidence/libcipher_v2/* /home/ubuntu/libcipher_v2/
  cp /home/ubuntu/cipher-phase2-evidence/cipher_kmod.ko /home/ubuntu/cipher_kmod/cipher_kmod.ko.v0.2.0
  ```
- [ ] **Step 1.2** Verify build of unchanged Phase 2 (sanity)
  ```bash
  cd /home/ubuntu/cipher_kmod && make clean && make
  modinfo cipher_kmod.ko | grep -E "version|depends|license"
  ```
  Expected: `version: 0.2.0`, `depends:` (empty), `license: GPL`.
- [ ] **Step 1.3** Bump version to 0.3.0 in cipher_main.c (`MODULE_VERSION`, banner pr_info, MODULE_DESCRIPTION)
- [ ] **Step 1.4** Build, verify 0.3.0 modinfo. Module fails-to-load is fine — only the file is built.
- [ ] **Step 1.5** Checkpoint: report build output. **Wait for go before insmod.**

### Task 2: Layer A — BAR0 reader (`cipher_bar0.c` + `/proc/cipher/bar0_state`)

**Files:**
- Create: `/home/ubuntu/cipher_kmod/cipher_bar0.c` (~150 lines)
- Modify: `/home/ubuntu/cipher_kmod/cipher_internal.h` (declare `cipher_bar0_init/exit/read_state`)
- Modify: `/home/ubuntu/cipher_kmod/cipher_main.c` (call init/exit)
- Modify: `/home/ubuntu/cipher_kmod/cipher_proc.c` (`bar0_state` seq_file)
- Modify: `/home/ubuntu/cipher_kmod/Kbuild` (add `cipher_bar0.o`)

Implementation contract:
- `cipher_bar0_init()`: `pci_get_device(0x10de, PCI_ANY_ID, NULL)` → `pci_iomap(pdev, 0, 16*1024*1024)`. If either fails, log `pr_warn` and set `bar0_base = NULL`. Module load still succeeds (Layer A degrades to disabled).
- `cipher_bar0_read_state(struct cipher_bar0_snapshot *out)`: with NULL guard on `bar0_base`, `ioread32` at offsets 0x0, 0x4, 0x88000, 0x20200, 0x20208. Also `ioread32` PMC_BOOT_0 to decode chip family using a small static table (GH100 = 0x180, AD100 = 0x190, etc. — derived from gpu-admin-tools).
- `cipher_bar0_exit()`: `pci_iounmap(pdev, bar0_base)`; `pci_dev_put(pdev)`.
- `/proc/cipher/bar0_state` seq_file: lazy refresh on every cat (16 register reads × ~60ns = ~1µs, no caching needed). Format per spec.

- [ ] **Step 2.1** Write `cipher_bar0.c` complete with NULL-guards, `pr_info` banner, decoder table.
- [ ] **Step 2.2** Wire into `cipher_main.c` (init after device-create, exit before device-destroy).
- [ ] **Step 2.3** Add `bar0_state` proc file in `cipher_proc.c` (parallel to existing `stats` seq_file).
- [ ] **Step 2.4** Update `Kbuild` objs list.
- [ ] **Step 2.5** Build. Run sparse if available.
- [ ] **Step 2.6** Checkpoint: ask for go, then `insmod cipher_kmod.ko`, `cat /proc/cipher/bar0_state`, verify PMC_BOOT_0 high nibble decodes to GH100 family, dmesg shows no errors, `rmmod cipher_kmod` clean. Report the values back.

### Task 3: ABI extension — three new ioctls + kernel handlers

**Files:**
- Modify: `/home/ubuntu/cipher_kmod/cipher_ioctl.h` (~+60 lines)
- Modify: `/home/ubuntu/cipher_kmod/cipher_internal.h` (~+30 lines: extend `cipher_pid_stats`, add `cipher_gpu_state_slot[MAX_GPUS]`)
- Modify: `/home/ubuntu/cipher_kmod/cipher_dev.c` (~+200 lines: three new handlers)

ABI added (verbatim from spec, only renaming for consistency):

```c
struct cipher_gpu_state {
    __u32 device_index;
    __u32 reserved_pad;
    __u64 timestamp_us;
    __u32 sm_util_pct;
    __u32 mem_util_pct;
    __u32 tensor_util_pct;
    __u32 fp16_util_pct;
    __u32 fp32_util_pct;
    __u32 fp64_util_pct;
    __u32 dram_bw_util_pct;
    __u32 power_mw;
    __u32 temperature_c;
    __u32 sm_clock_mhz;
    __u32 mem_clock_mhz;
    __u64 hbm_used_bytes;
    __u64 hbm_total_bytes;
    __u32 reserved[8];
};
struct cipher_process_util {
    __u32 pid;
    __u32 reserved_pad;
    __u64 timestamp_us;
    __u32 sm_util_pct;
    __u32 mem_util_pct;
    __u32 enc_util_pct;
    __u32 dec_util_pct;
};
struct cipher_launch_stats {
    __u32 pid;
    __u32 tgid;
    __u64 timestamp_us;
    __u64 kernel_launches;
    __u64 mem_allocs;
    __u64 mem_frees;
    __u64 streams_created;
    __u64 reserved[4];
};
#define CIPHER_SUBMIT_GPU_STATE    _IOW('C', 5, struct cipher_gpu_state)
#define CIPHER_SUBMIT_PROCESS_UTIL _IOW('C', 6, struct cipher_process_util)
#define CIPHER_SUBMIT_LAUNCH_STATS _IOW('C', 7, struct cipher_launch_stats)
```

Handler contracts:
- `SUBMIT_GPU_STATE`: `if (!capable(CAP_SYS_ADMIN)) return -EPERM;` `copy_from_user`; range-check every `*_pct ≤ 100` and `device_index < MAX_GPUS` (16); store into `cipher_gpu_state_slot[device_index]` under a spinlock; return 0.
- `SUBMIT_PROCESS_UTIL`: same CAP check; range-check pcts; hashtable lookup on `payload.pid`; if found, update `sm_util_pct` field on stats entry and bump a `gpu_util_updates` counter; if not found, return `-ENOENT` (silent, expected for transient PIDs).
- `SUBMIT_LAUNCH_STATS`: NO CAP check (any tenant can submit its own counts); anti-spoof: `if (payload.pid != task_pid_vnr(current) || payload.tgid != task_tgid_vnr(current)) return -EPERM;` Update stats entry counters (launches/allocs/frees/streams).

- [ ] **Step 3.1** Extend `cipher_ioctl.h` with structs + new `_IOW` macros + clear comment block about additive ABI rule (reserved nrs 2/3/4 remain `-ENOSYS`; new 5/6/7 are live).
- [ ] **Step 3.2** Extend `cipher_internal.h`: `MAX_GPUS=16`, declare `static struct cipher_gpu_state_slot gpu_state_slots[MAX_GPUS];` + spinlock; extend `cipher_pid_stats` with `sm_util_pct`, `mem_util_pct`, `kernel_launches`, `mem_allocs`, `mem_frees`, `streams_created`, `last_gpu_util_ns`.
- [ ] **Step 3.3** Add three handlers to `cipher_dev.c` switch in `cipher_dev_ioctl`. Reuse existing copy_from_user helper.
- [ ] **Step 3.4** Build. Manually verify symbol table contains the three new handler functions.
- [ ] **Step 3.5** Smoke test the ABI from kernel side only (no daemon yet): a tiny C program in /tmp does `open("/dev/cipher")` + three ioctls (one happy, two intended to fail EPERM/EINVAL). Verify dmesg + return codes.
- [ ] **Step 3.6** Checkpoint: report ioctl test results.

### Task 4: Layer B — `cipher-gpustate` C daemon

**Files:**
- Create: `/home/ubuntu/cipher_gpustate/cipher_gpustate.c` (~400 lines)
- Create: `/home/ubuntu/cipher_gpustate/Makefile`

Implementation contract:
- `main()`: parse `--interval-ms` (default 250) and `--once` flags; `nvmlInit_v2`; `nvmlDeviceGetCount_v2`; for each device, `nvmlGpmQueryDeviceSupport` and record whether GPM is available; install `SIGTERM`/`SIGINT` handler that sets a global `keep_running = 0`.
- Poll loop: for each device, call the basic NVML APIs (always work), then if GPM supported allocate two `nvmlGpmSample_t`, take samples 250 ms apart, `nvmlGpmMetricsGet` for the 7 metrics in the spec; if GPM not supported, fill GPM fields with `nvmlDeviceGetUtilizationRates` (sm only) and zero the rest. Build `cipher_gpu_state`, `ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &state)`.
- `nvmlDeviceGetProcessesUtilizationInfo` → loop processes, `ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu)` per pid. Ignore `-ENOENT` (PID unregistered).
- Cleanup: `nvmlShutdown` on SIGTERM. Exit 0 on clean shutdown.
- Failure modes:
  - NVML init fails → log `errno`+NVML error string, exit 1.
  - `/dev/cipher` open fails → log + exit 1.
  - ioctl returns -EPERM → log once + continue (privilege issue; operator fixes).

- [ ] **Step 4.1** Write `cipher_gpustate.c` with the structure above. Include `#include "../cipher_kmod/cipher_ioctl.h"` so the daemon shares the ABI header.
- [ ] **Step 4.2** Write `Makefile`: `cipher-gpustate: cipher_gpustate.c ; gcc -O2 -Wall -Wextra -o $@ $< -lnvidia-ml -lpthread`.
- [ ] **Step 4.3** Build. Run `./cipher-gpustate --once --interval-ms 250` standalone first (cipher_kmod NOT loaded) — verify it cleanly errors on `/dev/cipher`. Then with 0.3.0 loaded, verify it runs.
- [ ] **Step 4.4** Run for 5 s with kmod loaded: `cat /proc/cipher/gpu_state` shows current sm_util/tensor_util/power/temp/clocks/hbm. Compare against `nvidia-smi` for sanity (sm_util within ±5 % of nvidia-smi).
- [ ] **Step 4.5** Checkpoint: post `/proc/cipher/gpu_state` snapshot + nvidia-smi snapshot side by side.

### Task 5: Layer C — CUPTI callbacks in libcipher_v2

**Files:**
- Create: `/home/ubuntu/libcipher_v2/cipher_cupti.c` (~250 lines)
- Modify: `/home/ubuntu/libcipher_v2/cipher_v2_internal.h` (~+15 lines)
- Modify: `/home/ubuntu/libcipher_v2/cipher_inject.c` (~+10 lines call out)
- Modify: `/home/ubuntu/libcipher_v2/Makefile` (bump VERSION, add -ldl -lpthread, optional NO_CUPTI flag)

Implementation contract:
- `cipher_cupti_init()`:
  - Check env `CIPHER_V2_NO_CUPTI` — if set, return 0 (clean opt-out).
  - `dlopen("libcupti.so.12", RTLD_LAZY)`. On failure: log + return 0 (graceful disable).
  - `dlsym` `cuptiSubscribe`, `cuptiEnableDomain`, `cuptiEnableCallback`. Any NULL → disable.
  - `cuptiSubscribe(&handle, &cipher_callback, NULL)` (callback function below).
  - `cuptiEnableDomain(1, handle, CUPTI_CB_DOMAIN_DRIVER_API)` — single call enables all driver cbids in the domain.
  - Start a pthread `cipher_cupti_flush_thread` that sleeps 1 s and submits the current counter snapshot via `CIPHER_SUBMIT_LAUNCH_STATS`.
- `cipher_callback(domain, cbid, cbinfo)`:
  - Filter: `if (cbinfo->callbackSite != CUPTI_API_ENTER) return;`
  - Switch on cbid: increment one of four atomic counters using `__atomic_fetch_add` with `__ATOMIC_RELAXED`.
- Flush thread:
  - Every 1 s, read counters with `__atomic_load_n`, build `cipher_launch_stats`, ioctl. Counters are cumulative (not deltas) — kernel side computes its own deltas.
- Shutdown: no explicit teardown — process exit collects everything. (CUPTI unsubscribe is fragile; ignoring is acceptable for an injection library.)
- Performance: target <100 ns per callback (atomic increment + branch). Spec budget is ~100 µs/token on 1500 kernel launches.

- [ ] **Step 5.1** Write `cipher_cupti.c`. Use dlopen — do NOT link directly against libcupti. Header-include cupti.h purely for symbolic constants and struct layout.
- [ ] **Step 5.2** Wire into `cipher_inject.c`: after tenant registration (existing call to `cipher_register_tenant_if_set()`), call `cipher_cupti_init()`. Both are best-effort; either failing must NOT prevent CUDA from initializing (always return 1 from `InitializeInjection`).
- [ ] **Step 5.3** Build libcipher_v2.so 0.3.0. Verify `nm` shows the cupti init symbol; `ldd` does NOT show libcupti (because of dlopen).
- [ ] **Step 5.4** Smoke test: `LD_PRELOAD=` not needed; instead `CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so CIPHER_TENANT_ID=phase3-test python -c "import torch; torch.cuda.synchronize()"`. dmesg should show the REGISTER_TENANT + SUBMIT_LAUNCH_STATS ioctls firing. `/proc/cipher/stats` should show the test PID with non-zero LAUNCHES column.
- [ ] **Step 5.5** Checkpoint: report stats column output for the test PID.

### Task 6: Layer D — Prometheus exporter (Python http.server)

**Files:**
- Create: `/home/ubuntu/cipher_controller/cipher_controller.py` (~250 lines)
- Create: `/home/ubuntu/cipher_controller/grafana_dashboard.json` (~5 KB)
- Create: `/home/ubuntu/cipher_controller/README.md` (~30 lines)

Implementation contract:
- `cipher_controller.py`: `http.server.BaseHTTPRequestHandler`, `do_GET` serves `/metrics`. On every request: open `/proc/cipher/stats`, `/proc/cipher/gpu_state`, `/proc/cipher/bar0_state`; parse the seq_file output (defined format from cipher_proc.c); emit Prometheus text exposition format.
- Metric names (Prometheus convention, snake_case + `_total` suffix for counters):
  - `cipher_kmod_info{version="0.3.0"}` (gauge 1)
  - `cipher_bar0_pmc_boot_0{device="0"}` (gauge, hex value as decimal)
  - `cipher_gpu_sm_util_pct{device="0"}`
  - `cipher_gpu_tensor_util_pct{device="0"}`
  - `cipher_gpu_fp16_util_pct`/`fp32`/`fp64`
  - `cipher_gpu_dram_bw_util_pct`
  - `cipher_gpu_power_milliwatts`
  - `cipher_gpu_temperature_celsius`
  - `cipher_gpu_sm_clock_mhz`/`mem_clock_mhz`
  - `cipher_gpu_hbm_used_bytes`/`hbm_total_bytes`
  - `cipher_tenant_sm_util_pct{tenant="<id>",pid="<n>"}`
  - `cipher_tenant_kernel_launches_total{tenant="<id>",pid="<n>"}`
  - `cipher_tenant_mem_allocs_total`/`mem_frees_total`/`streams_created_total`
  - `cipher_tenant_ioctl_total{tenant="<id>",op="rm_control"}` (existing per-family counters from /proc/cipher/stats)
- `grafana_dashboard.json`: 7 panels per spec, Prometheus datasource, 30-s refresh.
- `README.md`: how to run (`python3 cipher_controller.py`), how to scrape, port.

- [ ] **Step 6.1** Write `cipher_controller.py`. Parse stats line-by-line — proc emitter is deterministic, no regex; use `str.split()` and explicit indexing.
- [ ] **Step 6.2** Test parser independently: `cat /proc/cipher/stats | python3 -c "from cipher_controller import parse_stats; import sys; print(parse_stats(sys.stdin.read()))"`.
- [ ] **Step 6.3** Start daemon on :9402. `curl localhost:9402/metrics | head -30` returns Prometheus format.
- [ ] **Step 6.4** Validate with promtool (`promtool check metrics`) if available; otherwise visually inspect HELP/TYPE/LABEL lines for spec compliance.
- [ ] **Step 6.5** Write `grafana_dashboard.json`. Hand-craft (no Grafana export needed) — the structure is well-documented.
- [ ] **Step 6.6** Checkpoint: 30-line metrics sample.

### Task 7: Integration smoke + close-out

**Files:**
- Create: `/home/ubuntu/phase_3_smoke.sh` (~150 lines)
- Create: `/home/ubuntu/PHASE_3_NOTES.md` (~250 lines, 6 sections per Phase 2 template)
- Create: `/home/ubuntu/cipher-phase3-evidence.tar.gz` + sha256

Smoke contract (phase_3_smoke.sh):
1. Pre-flight: phase_2_smoke baseline tps must be on file (already in cipher-phase2-evidence). If not, abort.
2. Insmod cipher_kmod 0.3.0. Verify `cat /proc/cipher/bar0_state` shows non-zero PMC_BOOT_0.
3. Start cipher-gpustate in background. Verify `/proc/cipher/gpu_state` populates within 1 s.
4. Start cipher-controller. Verify `curl :9402/metrics | grep cipher_gpu_sm_util_pct` returns a numeric value.
5. Run TinyLlama smoke (existing Phase 1.5.2 protocol script) with `CUDA_INJECTION64_PATH` and `CIPHER_TENANT_ID=phase3-smoke`. Capture tps.
6. Compare new tps vs Phase 2 baseline. Compute overhead %. **PASS** if ≤ 2 %.
7. Verify /proc/cipher/stats shows the tenant with non-zero LAUNCHES + non-zero SM_UTIL.
8. Verify /metrics endpoint shows `cipher_tenant_kernel_launches_total{tenant="phase3-smoke"} > 0`.
9. Re-run all Phase 1/1.5/2 regression assertions (REGISTER_TENANT still works; reserved nrs still return -ENOSYS; tainted kernel state).
10. Cleanup: kill daemons, rmmod 0.3.0, insmod 0.2.0 + verify ABI compat, rmmod 0.2.0.

- [ ] **Step 7.1** Write `phase_3_smoke.sh`. Idempotent (cleanup hook on any failure). Error trap → automatic rmmod 0.3.0 → insmod 0.2.0 (insurance rollback).
- [ ] **Step 7.2** Execute smoke with everything wired. Report all 10 checkpoints PASS/FAIL.
- [ ] **Step 7.3** Measure overhead (Phase 1.5.2 protocol). If >2 %, identify culprit (likely Layer C CUPTI) and either optimize the atomic batching or recommend opt-in default; do NOT just disable to hit the budget without explanation.
- [ ] **Step 7.4** Write `PHASE_3_NOTES.md` — six sections: Outcome, Artifacts shipped, Integration test ledger (10 rows), Performance, Risk register status, Next-phase setup. Match prose style of `PHASE_2_NOTES.md`.
- [ ] **Step 7.5** Build evidence tarball:
  ```bash
  tar czf /home/ubuntu/cipher-phase3-evidence.tar.gz \
      cipher_kmod cipher_gpustate cipher_controller libcipher_v2 \
      PHASE_3_NOTES.md phase_3_smoke.sh \
      smoke_*.log
  sha256sum /home/ubuntu/cipher-phase3-evidence.tar.gz
  ```
  Confirm prior 4 tarballs (may13, phase1, phase1.5, phase2) still on disk untouched.
- [ ] **Step 7.6** Final report to user: tps delta, all 10 checkpoints, evidence sha256, fallback verification.

---

## Risk Register (running)

| Risk | Mitigation | Status |
|---|---|---|
| pci_iomap blocked by nvidia.ko exclusive claim | Try pci_iomap first; if -EBUSY, fall back to `ioremap_wc` on BAR0 phys addr from sysfs | Watch in Task 2 |
| GPM not supported on driver 580.105.08 | `nvmlGpmQueryDeviceSupport` checked at startup; fall back to basic util-rates | Watch in Task 4 |
| CUPTI conflicts with concurrent Nsight | Document opt-out env var `CIPHER_V2_NO_CUPTI=1`; refuse to subscribe twice | Watch in Task 5 |
| 2 % overhead budget exceeded | Layer C is likely culprit — atomic batching is already lockless; if still hot, make CUPTI opt-in via env var | Measure in Task 7 |
| Reading wrong BAR0 offsets corrupts GPU state | Whitelist of safe offsets in cipher_bar0.c; absolutely no writes | Build into Task 2 |
| Root-required daemon is security concern | Document for Phase 7 capability rework | Note in PHASE_3_NOTES.md |

---

## Discipline Gates Checklist (per cipher-phase-discipline memory)

- [ ] Plan approval before any code (this document, awaiting your "go")
- [ ] Build verification with `modinfo` output reported BEFORE every insmod
- [ ] Explicit go before each insmod and each rmmod
- [ ] `cipher_kmod.ko.v0.2.0` preserved before 0.3.0 build (Task 1.1)
- [ ] All Phase 1/1.5/2 smoke tests pass with full 0.3.0 stack active (Task 7.2 step 9)
- [ ] Total Phase 3 overhead < 2 % vs Phase 2 baseline (Task 7.3)
- [ ] GPL license preserved; depends:empty preserved (Task 1.4, Task 7.2)
- [ ] PHASE_3_NOTES.md + evidence tarball + sha256 (Task 7.4–7.6)
- [ ] All five insurance tarballs intact on disk (Task 7.5 final verification)

---

## Self-Review (post-write)

**Spec coverage:** All four layers (A BAR0, B NVML/GPM, C CUPTI, D Prometheus) mapped to Tasks 2/4/5/6. ABI (3 new ioctls) covered by Task 3. Discipline (.ko preservation, version bump) covered by Task 1. Integration + overhead measurement + close-out covered by Task 7. Risk register pre-populated.

**Placeholder scan:** No TBDs. Every code instruction includes either a struct definition, a function signature, or a concrete command. Performance budget and rollback procedure both specified concretely.

**Type consistency:** `cipher_gpu_state`, `cipher_process_util`, `cipher_launch_stats` struct names appear identically in Tasks 3, 4, 5. Ioctl macro names `CIPHER_SUBMIT_GPU_STATE` / `_PROCESS_UTIL` / `_LAUNCH_STATS` identical in Tasks 3 (definition) and 4/5 (callers). MAX_GPUS = 16 in Task 3, referenced in Task 4 (`device_index < MAX_GPUS`). Versions: cipher_kmod 0.3.0, libcipher_v2 0.3.0, all consistent.
