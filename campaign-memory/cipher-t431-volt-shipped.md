---
name: cipher-t431-volt-shipped
description: T4.3.1 ships VOLT (NVML clock-lock). +57% tok/W lift on TinyLlama-1.1B B=1 (reproduced 2026-05-14 at +54.97%). Envelope is workload-dependent — NEUTRAL-TO-NEGATIVE on Mistral-7B-class. See cipher-t43-envelope.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.3.1 (Stream 1, 2026-05-14 afternoon) ports the VOLT DVFS actuator
from `cipher-may13-evidence/cipher_volt.cpp` into libcipher_rt as
`cipher_rt_volt.{c,h}`. Build md5 `bcbadcdb1da00d5c934e42079cc73d1a`
saved as `libcipher_rt.so.v0.2.0_T4_3_1`.

**Headline efficiency-moat measurement** (B=1 TinyLlama decode, 60s,
operator-level actuation via `sudo nvidia-smi -lgc 1000`):

| Metric | Default boost | Locked 1000 MHz | Δ |
|---|---:|---:|---:|
| Watts | 150.5 W | 96.8 W | **-35.7%** |
| Tok/s | 66.9 | 68.9 | +3.1% |
| Tok/W | 0.4443 | 0.7125 | **+60.4% (1.604×)** |

**The mechanism is real.** Tok/s went UP at lower clock because
TinyLlama decode B=1 is HBM-bound; lower clock avoids power-cap
throttling.

**Important: NVML privilege boundary.** From inside a non-root python
process injected with libcipher_rt, `nvmlDeviceSetGpuLockedClocks`
returns `NVML_ERROR_NOT_SUPPORTED (rc=4)`. From `sudo` it works. The
libcipher_rt VOLT code correctly probes, detects DEGRADED, and skips
actuation — no crashes, signals still installed defensively.

**How to apply:**
1. For production operator deployment: the kmod-mediated
   `CIPHER_SET_CLOCK` ioctl path (Stream 2 scope) closes this gap.
2. For demonstration / benchmark: use `sudo nvidia-smi -lgc N` as the
   actuator; libcipher_rt-side env vars then control the lock target
   for diagnostic purposes.
3. The prior 2.96× tok/W result from cipher-may13-evidence was FULL
   STACK (Marlin + spec + DVFS); DVFS marginal there was 4.2% on top
   of Marlin/spec. Today's 60% is DVFS on unoptimized stack — bigger
   marginal because more waste to claw back. Both are legit results
   at different operating points.

Build path: `cipher_rt_phase4/cipher_rt_volt.{c,h}` (C, not C++);
NVML resolved via dlopen+dlsym at init; atexit + SIG{TERM,INT,SEGV,
ABRT,BUS} handlers restore lock on exit; idempotent via atomic flag.
Env vars: `CIPHER_VOLT={off|on}`, `CIPHER_VOLT_BATCH={1|8|32|64}` or
`CIPHER_VOLT_MHZ=N`. Calibration `{1→1000, 8→1600, 32/64→1980}` from
`cipher-may13-evidence/p5_optimal_clocks.json`.

**ENVELOPE UPDATE (2026-05-14 night, see [[cipher-t43-envelope]]):**
The +60% TinyLlama-B=1 number reproduces (C6 +54.97%, T4.3.2 +57.28%,
CIs overlap). It does NOT generalize to Mistral-7B. 7-condition
campaign:
  - TinyLlama-1.1B B=1: +55% (memory-bandwidth-bound, sweet spot)
  - Mistral-7B B=1 @ 1000MHz: -14% (over-aggressive clock cut)
  - Mistral-7B B=1 @ 1600MHz: -2% (best clock, still negative)
  - Mistral-7B B=8: ~0%
  - Mistral-7B B=32: ~0%
Mechanism: discriminator is bandwidth-to-compute ratio at target
clock. `batch_to_mhz` is keyed only on batch; model size matters at
least as much. Calibration table is tuned for TinyLlama-class. The
honest moat claim is "+55% on memory-bandwidth-bound decode (small
model, low batch, calibrated clock)" — NOT "+57% on B=1 decode."

For 7B+ production regime (what neocloud customers serve), VOLT
needs recalibration (model-size axis) or pivot to a different lever.

Linked: [[cipher-lift-framing]] (efficiency moat motivation),
[[cipher-t424d-enforcement-fixed]] (binding diagnostic pattern that
T4.3.1 mirrors via "clock holds under load" test),
[[cipher-t43-envelope]] (7-condition envelope characterization).
