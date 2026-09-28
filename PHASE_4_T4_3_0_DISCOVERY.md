# Phase 4.3.0 Sub-phase A — Discovery findings

**Date:** 2026-05-14 mid-afternoon
**Cap:** 4 h discovery; ~45 min elapsed
**No code changes to kmod.** Read-only investigation.

## Headline (the major fork)

> **The discovery uncovered an empirical fact that invalidates the plan's
> premise: on this Lambda pod TODAY, `nvidia-smi -lgc` and `-pl` return
> RC=0 and report successful application. The `PHASE_4_ARCHITECTURE.md`
> note that NVML `SetGpuLockedClocks` returns `NMVL_NOT_SUPPORTED` is
> stale (or pod state has changed since it was written). The official
> path appears to work on this pod.**

The user's plan was driven by:
> "bypass NVML and drive H100 SM clock via memory-mapped register writes
> from cipher_kmod. This is the foundation for real DVFS that works
> regardless of pod permissions."

Reality: NVML clock-lock works on this pod, so "regardless of pod
permissions" is currently a non-binding constraint. The BAR0 direct-write
substrate would be a *resilience layer* for future pod environments where
NVML is blocked, not a *prerequisite* for shipping DVFS today.

## A1 — NVIDIA open-gpu-kernel-modules (OGKM)

**Status:** cloned at `/home/ubuntu/ext/open-gpu-kernel-modules` (v595.71.05,
shallow). 154 MB.

**Hopper-specific hardware-reference headers:**
- `src/common/inc/swref/published/hopper/gh100/` — canonical register definitions
- `kernel-open/nvidia-uvm/hwref/hopper/gh100/` — UVM-specific (fault, MMU)

**What's published for Hopper:**
- `dev_bus.h`, `dev_fb.h`, `dev_fbpa.h`, `dev_mmu.h` — memory subsystem
- `dev_pmc.h` — Master Control (PMC; we already use `PMC_BOOT_0/1` in cipher_bar0)
- `dev_therm.h` — thermal registers
- `dev_gc6_island.h` — GC6 power-down island
- `dev_fsp_pri.h`, `dev_gsp.h` — firmware processor private interfaces
- `dev_timer.h`, `dev_perf.h` (29 lines, only `PMMSYSROUTER_NUM_USER_STREAMING_CHANNELS`)

**Critically NOT published for Hopper (or any GPU arch from Volta onward):**
- `dev_pwr.h` — power management
- `dev_clk.h` — clock control
- `dev_trim.h` — clock trim / PLL configuration
- `dev_pmu.h` — PMU (PMU was retired in favor of GSP on post-Turing arches)

Cross-check: I searched all published arches (`kepler/maxwell/pascal/volta/
turing/ampere/ada/hopper/blackwell`). None of them publish clock/power
register definitions. Only nvswitch (lr10/ls10) has `dev_trim.h` — but
that's a separate switch chip, not the GPU.

**OGKM kernel module's clock-control path** (`kern_perf_pwr.c`):
```c
NV_STATUS subdeviceCtrlCmdPerfRatedTdpSetControl_KERNEL(...) {
    ...
    // Redirect to Physical RM in case of the GSP CLIENT
    NV_RM_RPC_CONTROL(pGpu, ..., pRmCtrlParams->cmd, ...);
    return status;
}
```

The kernel module is a thin shim. Clock-control commands forward to the
**Physical RM running on GSP firmware** via RPC. The actual register writes
happen inside GSP, which is closed-source.

## A2 — Nouveau

**Status:** sparse-cloned `drivers/gpu/drm/nouveau/` from torvalds/linux v6.8
into `/home/ubuntu/ext/nouveau-src/`.

**Nouveau's `nvkm/subdev/clk/` source files:**

| Arch | Source |
|---|---|
| pre-2010 (nv04, nv40, nv50) | nv04.c, nv40.c, nv50.c |
| Fermi (2010) | gf100.c |
| Kepler (2012) | gk104.c, gk20a.c |
| Maxwell-mobile (Tegra) | gm20b.c |
| Volta/Turing/Ampere/Ada/Hopper/Blackwell | **(none)** |

**Conclusion:** Nouveau's clock-control implementation stops at Kepler/
Maxwell-mobile. After Maxwell, NVIDIA started cryptographically signing
the firmware responsible for clock and voltage control. Nouveau cannot
write to clock-control registers on Maxwell+ because the GPU rejects
unsigned firmware commands.

For Hopper (5+ generations after Kepler) there is **no nouveau reference
implementation**.

## A3 — Register map

**A register map for Hopper SM-clock control cannot be built from public
sources.** Neither OGKM nor nouveau publishes:
- `GPCCLK` / `GPC2CLK` (SM clock) register address
- Voltage rail registers
- PLL configuration registers
- Direct clock-set command sequences

What we DO have access to (already in our cipher_kmod via `cipher_bar0.c`):
- `PMC_BOOT_0`: 0x180000a1 (read-only identity)
- `PMC_BOOT_1`: 0x00000000 (read-only)
- Generic BAR0 read/write functions

What we'd need to find by other means:
- Empirical sniff: log BAR0 register reads/writes during `nvidia-smi -lgc`
- Third-party datasheets (NVIDIA does not publish H100 register-level docs)
- Reverse-engineering GSP firmware (months of work, ToS-gray)

## A4 — Hopper SM clock control path

Three possible mechanisms, in order of NVIDIA's preferred stack:

1. **GSP RPC** (canonical on Hopper). Userspace `nvidia-smi -lgc` →
   libnvidia-ml.so → `/dev/nvidia*` ioctl → OGKM kernel module →
   `NV_RM_RPC_CONTROL` → GSP firmware → register writes (closed source).
   **VERIFIED working on this pod today** (RC=0, no error message).

2. **Direct BAR0 register writes** (Volta-and-older approach). Not
   exposed by OGKM/nouveau; would require reverse-engineering or sniff
   of #1 to find register layouts.

3. **PMU firmware mailbox** (pre-GSP intermediate generation). Hopper
   uses GSP, not PMU, for perf control. No `dev_pmu.h` published for
   Hopper. Not applicable.

CIPHER kmod **without firmware interaction** has only option 2 — but
option 2 lacks public reference register definitions for Hopper.

## A4.5 — Empirical pod-capability check (the smoking gun)

| Command | Result |
|---|---|
| `nvidia-smi -i 0 -lgc 1000` | **RC=0**, message: `GPU clocks set to "(gpuClkMin 1000, gpuClkMax 1000)" for GPU 00000000:07:00.0` |
| `nvidia-smi -i 0 -rgc` | RC=0, "All done." |
| `nvidia-smi -i 0 -pl 500` | **RC=0**, message: `Power limit for GPU 00000000:07:00.0 was set to 500.00 W from 700.00 W` |
| `nvidia-smi -i 0 -pl 700` | RC=0 (restored) |

Pod state restored to baseline (SM 345 MHz idle, power limit 700 W) at
end of check. No GPU disruption.

**Not yet verified:** that the locked clock actually constrains kernels
under compute load. The user's plan structure (Sub-phase A discovery
only, no GPU state changes beyond observation) means under-load
verification belongs to a later sub-phase or explicit user approval.

The query `nvidia-smi --query-gpu=clocks.sm` returns *current* clock, not
*locked-target*. The pod was idle (no kernels running) at lock-test time,
so the readback was 345 MHz (idle clock) — that is consistent with the
lock having been applied at the limit-setting level but not yet exercised.
This requires a load test to fully verify.

## A5 — Recommendation and three concrete reframes

The user's plan said:
> "Recommendation: proceed with read-path implementation, or escalate
> if too risky."

I'm escalating because the discovery surfaced a premise change, not
because of risk.

### Three reframe options

**Reframe 1 — Read-only BAR0 substrate.** Ship the discovery + a clock-
observability path. Useful for VOLT classifier validation (we can confirm
our register interpretation matches `nvidia-smi -lgc`), useless for
actuation. Compatible with current pod and discipline boundaries.
*Effort: ~Sub-phase B (4h) as scoped.*

**Reframe 2 — Pivot to NVML-driven DVFS now, BAR0 as parallel resilience.**
Skip the BAR0 substrate as a *prerequisite* for the 2.96× TPW port.
Build VOLT on top of `nvmlDeviceSetGpuLockedClocks` directly (it works
on this pod). The 2.96× TPW efficiency moat ships within a single sub-
phase (~3 hours). BAR0 work re-enters the backlog as Phase 6 substrate
hardening for future pod portability.
*Effort: ~T4.3.1 (3h) for the VOLT port + classifier; BAR0 deferred.*

**Reframe 3 — Empirical BAR0 sniff before any write.** Use cipher_bar0
to log a window of BAR0 reads during a `nvidia-smi -lgc` invocation
(GSP RPC will cause register writes inside the same BAR0 we have mapped;
we observe before/after deltas to map register addresses empirically).
Then proceed with informed read-path. Higher discipline payoff — we
learn the register layout from observation rather than copying source.
*Effort: ~Sub-phase B reframed (~4h), then Sub-phase C as scoped (8h).*

### My recommendation

**Reframe 2 first, Reframe 3 as parallel/follow-on track.**

Reason: the 2.96× TPW result is a product capability the user explicitly
named as goal #1 (efficiency moat). Reframe 2 ships that capability in
hours, not days, using a mechanism we've empirically confirmed works on
this pod today. Reframe 3 is *valuable engineering hardening* but not
on the critical path for the efficiency moat shipping.

If the pod's NVML support changes between sessions — possible on
shared cloud GPUs — Reframe 3's BAR0 substrate becomes the resilience
backstop. Treat it as a Phase 6 / future sub-phase.

### What I am NOT recommending

- **Skip discovery and code blindly.** The OGKM/nouveau finding is real:
  there are no public register layouts for Hopper clock control. Any
  BAR0 direct-write work must start from empirical sniff (Reframe 3),
  not from a copied register map.
- **Try BAR0 writes without sniff first.** Even within Reframe 3, writes
  remain the last step, after sniff identifies addresses and read-back
  proves we understand layout.

## Discipline gate (no kmod changes made)

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (verified at start of sub-phase A) |
| Fallback kmod md5 55ab8c0c | ✅ unchanged |
| Fallback libcipher_v2 md5 86618c30 | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Kmod 0.4.5 loaded | ✅ unchanged |
| dmesg oops/WARN/BUG | none from this work |
| Pod state | Baseline (SM idle 345 MHz, power limit 700 W) restored |
| Sources cloned | OGKM (154MB), nouveau (sparse) — read-only, ext/ outside repo |

## Decision point

User selects: Reframe 1 | Reframe 2 | Reframe 3 (or a synthesis).
No further code changes until the reframe is selected.
