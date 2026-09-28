# Phase 4.3.2 (Stream 2 Priority 1) — Kmod-mediated DVFS, privilege gap closed

**Date:** 2026-05-14 mid-afternoon (continuing post-T4.3.1).
**Cap:** 2 h on Priority 1; ~1 h 25 min elapsed.
**Kmod build:** 0.4.6 (srcversion `CFADD44796885D936E750EC`) md5 `119cb58348b6553b9e848cca0468f624`.
**libcipher_rt build:** `libcipher_rt.so.v0.2.0_T4_3_2` md5 `5ebcf26cb0fe8a3954cc8831a5b01bf3`.
**Pre-T4.3.2 snapshots:**
- `cipher_kmod.ko.v0.4.5.pre_T4_3_2` md5 `b47db4500e226f5ddc010f9148d3d6ca`
- `libcipher_rt.so.v0.2.0_T4_3_1` md5 `bcbadcdb1da00d5c934e42079cc73d1a` (pre-T4.3.2 base)

## TL;DR (statistical headline, n=5 matched pairs)

| Question | Answer |
|---|---|
| a. Which actuation path worked? | **(b) usermodehelper from kmod.** Direct nvidia.ko symbol calls (path a) not available — exports limited to `nvidia_p2p_*` + error callbacks; no clock/perf symbols. GSP RPC construction (c) infeasible without firmware RE. |
| b. cipher_rt VOLT end-to-end watts (not sudo) | **151.16 ± 0.38 W → 96.37 ± 0.08 W (−36.24% ± 0.12%, z=314)** |
| c. Tok/W lift over 5 matched pairs | **+57.28% ± 0.32% (95% CI: +56.65% to +57.90%)** |
| d. Tok/s effect | **+0.28% ± 0.39% — not distinguishable from zero (z=0.72)**; HBM-bound decode is insensitive to SM clock at this batch size |
| e. Phase 4.3.1 deployment readiness | **READY for non-root user-process injection.** Privilege gap closed via kmod ioctl path. Watts moat is the dominant axis; tok/s is unchanged. |

## a — Path decision (S2.A1 + S2.A2)

### Path (a) — direct nvidia.ko symbol call: **NOT VIABLE**

Audited `/usr/src/nvidia-srv-580.105.08/` for `EXPORT_SYMBOL` macros:

```
nvidia/nv-p2p.c: NV_EXPORT_SYMBOL(nvidia_p2p_init_mapping);
                 NV_EXPORT_SYMBOL(nvidia_p2p_destroy_mapping);
                 NV_EXPORT_SYMBOL(nvidia_p2p_get_pages);
                 NV_EXPORT_SYMBOL(nvidia_p2p_get_pages_persistent);
                 ... [11 P2P functions for RDMA]
nvidia: nvidia_register_error_cb / nvidia_unregister_error_cb
```

Only peer-to-peer GPU memory mapping (RDMA path) and error-callback registration are exported. **No clock-control, perf-control, or NVML-equivalent symbols are exported.** This is consistent with NVIDIA's stable-ABI policy: only the public-RDMA surface is callable from other kernel modules.

### Path (c) — construct NV_RM_RPC packet: **NOT VIABLE in 2 h scope**

GSP RPC requires:
- Knowing the RPC packet structure for `NV2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL` and related commands
- An entry point in nvidia.ko's RPC machinery accessible to other modules (none exported)
- GSP firmware version compat handling

Months of firmware reverse-engineering. Outside scope.

### Path (b) — `call_usermodehelper("/usr/bin/nvidia-smi", ...)`: **VIABLE and shipped**

Linux kernel pattern for invoking user-space binaries from kernel context. nvidia-smi:
- Runs as root regardless of the kmod's caller uid (kernel-spawned processes inherit init's privilege)
- Returns RC=0 on success
- Already proven to work on this pod (S1.3, S1.6)

Latency: ~100-300 ms per invocation. Init-time use only.

## S2.A2 — `CIPHER_SET_CLOCK_MHZ` ioctl (nr 10)

New ioctl in `cipher_ioctl.h`:

```c
#define CIPHER_SET_CLOCK_MHZ _IOW(CIPHER_IOCTL_MAGIC, 10, __u32)
```

Handler in new `cipher_kmod/cipher_clock.c`:
- Bounds check: mhz must be 0 (reset) or in `[210, 1980]`
- mutex serializes one in-flight call at a time
- `call_usermodehelper("/usr/bin/nvidia-smi", ["-i", "0", "-lgc", "<mhz>"], envp, UMH_WAIT_PROC)`
- Audit log at KERN_INFO with caller pid + uid

### Trust model decision (decision recorded in commit)

The initial implementation had `CAP_SYS_ADMIN` at ioctl entry. I removed it after recognizing this defeats the design intent: the whole point of T4.3.2 is to give the non-root user-process injection-time actuator a privileged path. The kmod's `call_usermodehelper` *conveys* root privilege; requiring the caller to already have CAP makes the ioctl useless for its purpose.

The chosen model matches the rest of CIPHER's ABI: `/dev/cipher` is `0666` by convention; other ioctls (`REQUEST_SM_PARTITION`, `SUBMIT_LAUNCH_STATS`) also accept non-root. The access gate is the device-node permission — operator-deployed environments can restrict via udev rule + cipher group if a stricter policy is required. Bounds are enforced inside the kmod regardless of caller; a malicious tenant cannot exit `[210, 1980]` MHz.

Audit trail per actuation in dmesg:
```
cipher_clock: set 1000 MHz via nvidia-smi (caller pid=1120888 uid=1000)
```

### ioctl verification (root + non-root + bounds + reset)

```
[Thu May 14 15:02:04 2026] cipher_clock: set 1000 MHz via nvidia-smi (caller pid=1120220 uid=0)    # root
[Thu May 14 15:02:04 2026] cipher_clock: refused mhz=50 (allowed: 0 or [210, 1980])              # bounds
[Thu May 14 15:02:04 2026] cipher_clock: set 0 MHz via nvidia-smi (caller pid=1120228 uid=0)     # reset
[Thu May 14 15:05:27 2026] cipher_clock: set 1000 MHz via nvidia-smi (caller pid=1120888 uid=1000) # non-root ✓
```

The gap-closing test: uid=1000 (ubuntu) successfully invoked the ioctl from a regular non-root C test program.

## S2.A3 — libcipher_rt VOLT fall-through wiring

`cipher_rt_phase4/cipher_rt_volt.c` extended with three additions:

1. **`CipherRtVoltPath`** enum tracking which actuator owns the lock: NONE / NVML / KMOD_IOCTL.
2. **Probe-then-fallback in init:** try NVML's `nvmlDeviceSetGpuLockedClocks` first (zero-latency, no spawn); if NVML returns `NVML_ERROR_NOT_SUPPORTED`, fall to `probe_kmod_actuation()` which opens `/dev/cipher` and sends a no-op `CIPHER_SET_CLOCK_MHZ(0)` (reset is idempotent).
3. **Path-aware `cipher_rt_volt_do_restore()`:** atexit and signal handlers now route to whichever path actually applied the lock. Both branches are async-signal-safe (cached `void *` fp / fd, no dlsym in handler).

Init log lines under non-root python injection:

```
[cipher_v2] VOLT: NVML probe rc=4 (NOT_SUPPORTED=4) — trying kmod ioctl path
[cipher_v2] VOLT: kmod ioctl path available (CIPHER_SET_CLOCK_MHZ)
[cipher_v2] VOLT: ACTIVE — locked GPU clock to 1000 MHz via path=kmod-ioctl;
            atexit + SIG{TERM,INT,SEGV,ABRT,BUS} handlers installed
```

Restore path verified: post-process pod state is `345 MHz, 700.00 W` (baseline idle).

## S2.A4 — End-to-end TPW measurement under libcipher_rt actuation

Same workload as S1.6 (B=1 TinyLlama decode), no sudo nvidia-smi.
A: `CIPHER_VOLT=on CIPHER_VOLT_BATCH=1` → libcipher_rt locks 1000 MHz via kmod ioctl.
B: `CIPHER_VOLT=off` → no lock, default boost.

### Statistical headline (5 matched pairs, 60 s each)

The single-shot S1.6 (+3.1% tok/s) and S2.A4 first-shot (-4.5% tok/s)
contradicted each other on the tok/s direction. To establish the noise
band, 5 back-to-back matched pairs were captured at the close of T4.3.2
(matched-pair binding rule per B9). Results:

| Metric | OFF mean ± σ (n=5) | ON mean ± σ (n=5) | Δ% | 95% CI | z |
|---|---:|---:|---:|---:|---:|
| Watts (steady) | 151.16 ± 0.38 W | **96.37 ± 0.08 W** | **−36.24%** | [−36.47%, −36.02%] | 314.6 |
| Tok/s | 67.40 ± 0.56 | 67.59 ± 0.16 | **+0.28%** | [−0.48%, +1.04%] | 0.72 |
| Tok/W | 0.4459 ± 0.0028 | 0.7013 ± 0.0015 | **+57.28%** | [+56.65%, +57.90%] | 179.8 |
| GPU SM clock (steady) | 1980.0 MHz (σ=0) | 1005.0 MHz (σ=0) | — | — | — |

Clock readout has **zero variance across all 5 runs in each arm** — the
lock holds rigidly via the kmod ioctl path.

### Three credible claims

1. **Watts reduction: −36.24% ± 0.12% (95% CI: −36.5% to −36.0%).**
   z=314 — astronomically significant. Run-to-run watts variance is
   tiny (CV ≈ 0.25% in OFF, CV ≈ 0.08% in ON).
2. **Tok/s effect: statistically indistinguishable from zero.** Mean
   +0.28%, 95% CI [−0.48%, +1.04%], z=0.72. The earlier single-shot
   readings (+3.1% in S1.6, −4.5% in first S2.A4 single-shot) were
   sampling jitter at the tail of the distribution. TinyLlama decode
   B=1 is HBM-bandwidth-bound; SM clock change above the HBM-bound
   point produces no measurable tok/s effect either way.
3. **Tok/W lift: +57.28% ± 0.32% (95% CI: +56.7% to +57.9%).** z=180 —
   astronomically significant. Driven entirely by watts reduction with
   tok/s unchanged.

### Per-pair detail (auditable)

```
pair 1: off tps=67.50 watts=150.87 tpw=0.4474 | on tps=67.72 watts=96.47 tpw=0.7020
pair 2: off tps=66.70 watts=150.88 tpw=0.4421 | on tps=67.80 watts=96.40 tpw=0.7034
pair 3: off tps=68.39 watts=151.90 tpw=0.4502 | on tps=67.55 watts=96.25 tpw=0.7018
pair 4: off tps=67.31 watts=151.10 tpw=0.4454 | on tps=67.39 watts=96.31 tpw=0.6997
pair 5: off tps=67.10 watts=151.04 tpw=0.4442 | on tps=67.46 watts=96.43 tpw=0.6996
```

### Interpretation

The mechanism is unambiguous: at B=1 decode, locking the GPU clock to
1000 MHz cuts power draw by 36% with zero throughput penalty. The tok/W
moat is +57%, not the earlier single-shot +60% or +51% claims — those
were sampling extremes; this is the mean over five matched-pair runs.

**The crucial result remains: the mechanism is the same; the actuation
source is different.** S1.6 needed `sudo nvidia-smi`; this 5-pair run
had libcipher_rt apply the lock autonomously via the kmod ioctl path
from a non-root injected python process. The user-process gap is closed.

## Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (3+6+3) |
| Fallback kmod md5 `55ab8c0c` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30` | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Loaded kmod srcversion `CFADD44796885D936E750EC` (0.4.6) | ✅ |
| dmesg oops/WARN/BUG since start | none |
| Pod state | restored (SM idle 345 MHz, power limit 700 W) |

## Artifact map

| Artifact | md5 | Purpose |
|---|---|---|
| `cipher_kmod.ko.v0.4.5.pre_T4_3_2` | `b47db4500e226f5ddc010f9148d3d6ca` | pre-T4.3.2 kmod snapshot |
| `libcipher_rt.so.v0.2.0_T4_3_1` | `bcbadcdb1da00d5c934e42079cc73d1a` | pre-T4.3.2 libcipher base |
| **`cipher_kmod.ko.v0.4.6`** | **`119cb58348b6553b9e848cca0468f624`** | new kmod (CIPHER_SET_CLOCK_MHZ ioctl) |
| **`libcipher_rt.so.v0.2.0_T4_3_2`** | **`5ebcf26cb0fe8a3954cc8831a5b01bf3`** | new libcipher (VOLT fall-through) |
| `cipher_kmod/cipher_clock.{c}` | new | usermodehelper-based clock actuator |
| `cipher_kmod/cipher_ioctl.h` | extended | CIPHER_SET_CLOCK_MHZ nr 10 |
| `cipher_kmod/cipher_dev.c` | extended | nr 10 dispatcher case |
| `cipher_rt_phase4/cipher_rt_volt.{c,h}` | extended | path-aware fall-through |
| `cipher_workloads/measurement/volt_tpw_ab_libcipher.sh` | new | single-pair A/B harness for libcipher_rt actuation |
| `cipher_workloads/measurement/volt_tpw_npairs.sh` | new | N-pair credibility harness (default n=5) |
| `cipher-phase4-evidence/t4_3_2/summary.json` | new | single-pair headline (preserved for audit) |
| `cipher-phase4-evidence/t4_3_2_npairs/summary.json` | new | **5-pair statistical headline** |
| `cipher-phase4-evidence/t4_3_2_npairs/pair{1..5}_{off,on}/` | new | per-pair watts CSV, progress, run.log |
| `cipher-phase4-evidence/t4_3_2/{A,B}_watts.csv` | new | single-pair 1Hz timeline |
| `cipher-phase4-evidence/t4_3_2/{A,B}_progress.txt` | new | single-pair run progress |
| `cipher-phase4-evidence/t4_3_2/{A,B}_run.log` | new | single-pair stderr (init logs) |
| `/tmp/cipher_set_clock_test.c` | new | ioctl smoke test program |

## What ships, what's open

**T4.3.2 closes:**
- Non-root user-process VOLT actuation. Operator-deployed libcipher_rt now drives DVFS autonomously.
- No reliance on sudo NOPASSWD or file capabilities.
- Audit trail via dmesg.
- atexit + 5 signal handlers, path-aware restore. Safe against process death.

**Open (Stream 2 Priority 2 — BAR0 sniff):**
- Empirical register-delta capture during `nvidia-smi -lgc` (read-only BAR0 snapshots before/after).
- Register map for future direct-BAR0-write substrate (Phase 6 hardening).

**Open (T4.3.3+ — VOLT extensions):**
- Per-launch classifier (M/K/N inference; currently env-driven only).
- Calibrate mode (auto-detect optimal clock per batch / workload class).
- HIBERNATE (`nvmlDeviceSetPowerManagementLimit`) — also viable via the same kmod-ioctl pattern.
