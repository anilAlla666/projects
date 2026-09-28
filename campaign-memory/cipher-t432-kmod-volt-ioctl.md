---
name: cipher-t432-kmod-volt-ioctl
description: T4.3.2 closed the non-root NVML privilege gap via kmod-mediated CIPHER_SET_CLOCK_MHZ ioctl + libcipher_rt fall-through; same 36% watts reduction with no sudo dependency
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.3.2 (Stream 2 Priority 1, 2026-05-14 mid-afternoon) closed the
non-root user-process privilege gap surfaced in T4.3.1. The libcipher_rt
VOLT actuator now drives DVFS autonomously from inside an injected
non-root python process — no sudo, no helper, no pre-config.

**Architecture decisions:**

1. **Path audit (S2.A1):** nvidia.ko's only EXPORT_SYMBOL functions are
   `nvidia_p2p_*` (RDMA) and `nvidia_register_error_cb`. No clock/perf
   symbols exported. Direct kmod→nvidia symbol call (path a) and GSP
   RPC packet construction (path c) are infeasible. **Path (b)
   `call_usermodehelper("/usr/bin/nvidia-smi")` is the viable mechanism.**

2. **Trust model:** /dev/cipher remains 0666 (matching existing CIPHER
   ABI pattern; REQUEST_SM_PARTITION etc. also accept non-root). The
   initial CAP_SYS_ADMIN check was REMOVED because it defeats the
   design intent (the kmod's whole purpose is to convey privilege to
   non-root callers). Bounds enforced in kmod regardless: mhz must be
   0 or in [210, 1980]. Audit via dmesg KERN_INFO with caller pid+uid.

3. **Fall-through:** libcipher_rt VOLT init tries NVML first (zero-
   latency), falls through to kmod ioctl on NVML's NOT_SUPPORTED.
   Path-aware atexit + signal restore for both paths (async-signal-safe).

**Why:** every future libcipher_rt actuator (HIBERNATE for power-cap,
THERMOSTAT for thermal-feedback, etc.) that needs operator privilege
can follow the same pattern. The kmod ioctl path generalizes.

**How to apply:**
- New privileged actuators: add ioctl nr to cipher_ioctl.h, handler
  in cipher_kmod/, fall-through in libcipher_rt.
- Always bounds-check inside the kmod (the caller can lie).
- Audit-log every actuation with caller pid+uid for forensic trail.
- usermodehelper has ~100-300 ms spawn cost; init-time use only,
  not per-launch.

**Statistical headline (n=5 matched pairs, 60s each, libcipher_rt-driven actuation):**
- Watts: 151.16 ± 0.38 W → 96.37 ± 0.08 W (**−36.24% ± 0.12%**, z=314)
- Tok/s: 67.40 ± 0.56 → 67.59 ± 0.16 (**+0.28%**, z=0.72, not distinguishable from 0)
- Tok/W: 0.446 ± 0.003 → 0.701 ± 0.002 (**+57.28% ± 0.32%**, z=180)
- Clock: 1980 MHz → 1005 MHz (zero variance — lock holds rigidly)

The earlier single-shot +60% (S1.6) and +51% (first S2.A4) were sampling
extremes. The credible number is **+57.3% tok/W ± 0.3%** with 95% CI
[+56.7, +57.9].

Tok/s effect is statistically zero: TinyLlama decode B=1 is HBM-bound,
so SM clock above the HBM-bound point produces no throughput effect.
The TPW lift is entirely watts reduction.

Builds:
- `cipher_kmod.ko.v0.4.6` md5 `119cb58348b6553b9e848cca0468f624` (srcversion `CFADD44796885D936E750EC`)
- `libcipher_rt.so.v0.2.0_T4_3_2` md5 `5ebcf26cb0fe8a3954cc8831a5b01bf3`
- Tarball `cipher_rt_phase4_src_T4_3_2.tar.gz` md5 `be7e5ea2d5b81d2c5f5208d50f75d299`

ioctl nr 10 is the new ABI; matches CIPHER additive-only rule.

Linked: [[cipher-abi-rule]] (additive only), [[cipher-t431-volt-shipped]]
(the gap this closes), [[cipher-phase-discipline]] (7-item discipline).
