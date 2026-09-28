# Phase 4 T4.D.3 — GSP Boundary Investigation

**Date:** 2026-05-13
**Scope:** read-only inspection of NVIDIA's open-gpu-kernel-modules source tree at `/tmp/nvopen` (md5-stable since Phase 3 clone) to determine what CIPHER can or cannot do on GSP.

## Question

Can CIPHER run code on the GPU's GSP (GPU System Processor), or otherwise influence GSP-RM (Resource Manager firmware) directly, to gain finer-grained control than the Driver API allows?

## Verdict

**No.** GSP firmware is signed, the signature is verified by NVIDIA's booter, and there is no third-party plugin / extension mechanism in the GSP-RM interface. CIPHER's leverage stays at the host-side Driver API layer.

## What runs on GSP vs nvidia.ko

| Layer | Where it runs | What it does |
|---|---|---|
| GSP firmware (`GSP-RM`) | RISC-V (GA10x/AD102) or NVRISC-V (GH100, GB10x) processor on the GPU package | Resource manager: VBIOS, init, ECC, MIG layout, clock/power state machines, telemetry collection |
| nvidia.ko | Host CPU (Linux kernel) | Submits RM commands to GSP via shared-memory ring; mediates ioctls from `libnvidia-rm` userspace consumers |
| libnvidia-rm.so | Host userspace | RM API consumed by CUDA Driver, CUDA Runtime, NVML, etc. |
| CUDA Driver / Runtime | Host userspace | Application-level GPU programming surface |
| **CIPHER (Phase 1-4)** | Host userspace (`libcipher_v2.so`, `libcipher_rt.so`) + Host kernel (`cipher_kmod`) | GOT/PLT intercept of CUDA Driver/Runtime + kprobe on `nvidia_unlocked_ioctl` |

## GSP-RM source tree map (read-only refs)

Located at `/tmp/nvopen/src/nvidia/src/kernel/gpu/gsp/`:

| File | Role |
|---|---|
| `kernel_gsp.c` | Top-level GSP RM client — submits commands, reads responses |
| `kernel_gsp_booter.c` | Booter shim — loads + verifies GSP firmware blob, hands control off |
| `kernel_gsp_fwsec.c` | Firmware security checks |
| `kernel_gsp_trace_rats.c` | Trace ring access from GSP |
| `message_queue_cpu.c` | CPU-side end of the GSP message queue (RPC) |
| `arch/hopper/kernel_gsp_gh100.c` | H100-specific GSP init + RM-version pinning |
| `arch/ada/kernel_gsp_ad102.c` | AD102 (RTX 4090) variant |
| `arch/ampere/kernel_gsp_ga{100,102}.c` | A100, A6000 |
| `arch/blackwell/kernel_gsp_gb{100,200,202,10b}.c` | B100, B200, GB10b |
| `arch/turing/kernel_gsp_{tu102,booter_tu102,frts_tu102,vbios_tu102}.c` | TU102 Turing path |

## The signing check (where third-party access is blocked)

GSP firmware signature verification flows through `libspdm` (DMTF Security Protocol and Data Model — an industry-standard hardware-rooted signing protocol):

```
/tmp/nvopen/src/nvidia/src/libraries/libspdm/3.5.0/
├── include/internal/libspdm_requester_lib.h        — requester (host) side
├── include/internal/libspdm_responder_lib.h        — responder (device firmware) side
├── library/spdm_common_lib/libspdm_com_context_data.c — root cert chain
└── library/spdm_crypt_lib/libspdm_crypt_asym.c     — asymmetric verify
```

Signature verification happens at GSP boot, BEFORE GSP firmware starts executing application logic. The booter (`kernel_gsp_booter.c`) computes the firmware digest, the device-side SPDM responder verifies it against a hardware-fused root certificate, and only on success does the firmware begin running.

**The hardware-fused certificate is NVIDIA-controlled.** Third parties cannot:
- Sign their own firmware to replace `gsp_gh100.bin`
- Hot-patch GSP firmware at runtime (the firmware is loaded once at boot from a signed blob)
- Inject code into the running GSP through any documented or undocumented host-side interface

## Where the GSP firmware blob lives

```
/lib/firmware/nvidia/580.105.08/gsp_ga10x.bin   (Ampere — 100-series Tesla)
/lib/firmware/nvidia/580.105.08/gsp_tu10x.bin   (Turing)
```

H100 GSP firmware ships with `nvidia.ko` 580.105.08 — typically named `gsp_gh100.bin` or pulled from one of the architecture-specific files. The binary is opaque; no per-version source release.

**The signature check is enforced in BOTH host driver AND GSP boot ROM.** Even bypassing the host check (e.g., LD_PRELOAD'ing a modified `libnvidia-rm.so`) cannot get unsigned firmware to execute — the boot ROM independently verifies before transferring control.

## Whether nvidia.ko exposes a GSP-RM-command-injection interface

Reading `kernel_gsp.c` + `message_queue_cpu.c`:

- The CPU-side maintains a **CPU-RM ⇄ GSP-RM message queue** in shared memory
- Each command is structured per `NV_RM_COMMAND_*` (allocate resource, free, query state)
- The queue is written by `nvidia.ko` only; userspace cannot directly enqueue
- Userspace can request behaviors via `/dev/nvidiactl` ioctls (which is what `libnvidia-rm.so` does); `nvidia.ko` translates these into GSP-RM commands and enqueues them

**CIPHER's interception point** (kprobe on `nvidia_unlocked_ioctl` for observation + GOT/PLT hooks at Driver/Runtime layer for actuation) sits **above** this translation. We can:
- **Observe** every RM-bound ioctl (what we do today via cipher_kmod kprobe)
- **Decide not to dispatch** an ioctl (would require kprobe replacement, not just observation — Phase 6+ territory)
- **Reorder or batch** ioctls at the libnvidia-rm boundary (Phase 6+)

We CANNOT:
- Inject commands directly into the GSP message queue
- Modify the GSP-RM command schema
- Run any code on GSP itself

## Whether there's a "GSP debug" interface exposed in production

Grepping for debug surfaces:

```
/tmp/nvopen/src/nvidia/src/kernel/gpu/gsp/kernel_gsp_trace_rats.c
```

`RATS` = "Remote Attestation and Trace Server". It's a trace-readout interface from GSP to host — **read-only**. CIPHER could leverage this for additional telemetry (per-tenant GSP-side counters?), but it's unlikely to give controls we don't already have at the Driver layer.

Confirmed: no documented or undocumented production-mode write interface to GSP. The only GSP-touching production path is `nvidia.ko`'s message queue, which translates structured RM commands — not arbitrary code or arbitrary memory.

## Implications for Phase 4 / Phase 5 / Phase 6

| Phase | GSP-related work | Verdict |
|---|---|---|
| Phase 4 | None — current intercept points (Driver/Runtime + kprobe) are correct | ✓ Optimal layer for our intercepts |
| Phase 5 | Multi-tenant density via MIG + Green Context. Both are GSP-managed but configured via NVML/Driver API — not direct GSP access | ✓ Stays at NVML layer |
| Phase 6 | `cipher_rt_km` (kernel-space cipher runtime for sub-100ns decisions) | Would talk to GSP **through** `nvidia.ko`'s RM interface, same as today. Lower latency than userspace but no extra capabilities. |

**The GSP-RM interface is not a future depth source for CIPHER.** All optimization work continues at the existing host-side layers.

## Cross-reference to PHASE_4_DEPTH_AUDIT.md

This document supports the "GSP accessibility verdict: signed-only / partial-access / new-discovery" entry in the depth audit summary. The verdict is **signed-only**, with the precise reference being `kernel_gsp_booter.c` (host-side signature check) + libspdm 3.5.0 (cryptographic verification) + hardware-fused root certificate (not in source).

No source modifications were made during this investigation. `/tmp/nvopen` remains the read-only mirror from Phase 3.
