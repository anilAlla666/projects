# V0 GATE-1 increment g1.1: DVFS composed into the engine path (capture-safe) -- first integrated G1⊕G2 number

**2026-06-03. DVFS (CIPHER VOLT actuator) composed into the engine, CAPTURE-SAFE, at FAULT=0. First integrated
G1⊕G2 measurement: up to 1.53× tok/W vs the inc-4 fp16 substrate baseline, throughput-preserving.** NO `.so` change
(host-side `sudo nvidia-smi -lgc`; user NVML = NoPermission, as cipher_rt_volt.c flagged); inc-1+inc-2 modules
BYTE-IDENTICAL + gates PASS; deployed anchor 1f305ce6 (May-27) + staging (Jun-1) UNCHANGED. `CIPHER_RT_DISABLE_AUTO_INIT=1`.
Artifact: `cipher_inc4.py` (+ CIPHER_VOLT path + PowerSampler).

## Why DVFS composes where the other actuators don't

The engine path is pager + captured graph-decode; legacy actuators (Koopman cusolver, Marlin per-call ops) fire
capture-illegal ops that invalidate stream capture (inc-1/inc-3b graph-fragility). **DVFS is NOT a kernel** -- it is a
host-side clock-set call, so it runs BETWEEN waves (in the scheduler, OUTSIDE serve_wave's captured graph) ->
**capture-safe by construction**. Privileged (user NVML SetGpuLockedClocks = NoPermission) -> `sudo nvidia-smi -lgc`.

## Correctness FIRST -- KL=0 bit-exact (the clock changes timing, not numerics)

100-agent both-regimes load (the inc-4 trace), teacher-forced oracle (ratified), DVFS ON: **exact=95 near-tie=5
FAULT=0** (100/100), cross-model misroute neg-control DETECTED. Identical 0-FAULT at @1600 and @1000. DVFS does not
perturb any agent (it cannot -- clock-setting is bit-exact).

## The integrated number (first G1⊕G2 composed) -- a tok/W ↔ latency tradeoff curve

| config | SM clock (measured) | throughput | avg power | **tok/W (tok/J)** | vs baseline | P99 short/long |
|---|---|---|---|---|---|---|
| baseline (default boost) | **1980** | 398 tok/s | 285 W | 1.398 | 1.00× | 2.41s / 6.50s |
| DVFS @1600 (legacy calib) | 1611 | 380 | 234 W | 1.622 | **1.16×** | 2.35s / 7.33s |
| **DVFS @1000 (MEASURED optimum)** | 1012 | 357 | **168 W** | **2.134** | **1.53×** | 3.43s / 10.06s |

DVFS delivers **up to 1.53× tok/W** (285->168W, **−41% power**; throughput **−10%**) at FAULT=0. Near-iso-latency
(@1600) it is 1.16×; aggressive (@1000) it is 1.53× at +40% P99. The engine picks the operating point per SLA.

## Mechanism VERIFIED (not hypothesis -- bare-decode clock/power probe)

Ruled out the "baseline already auto-downclocked" alternative and the "power-modest, hand-wave" non-explanation:
- **Baseline decode BOOSTS to 1980 MHz** (B=4 sustained: avg 1980). Not auto-downclocked.
- **1980->1600 barely cuts power** (B=4: 199->190W) -- the legacy B<=8->1600 calibration is CONSERVATIVE for the
  engine's pure-decode path; the 1.16× came mostly from the B=1->1000 waves.
- **1980->1000 cuts ~25% power at iso-decode-throughput** (B=1: 187->146W @1.08× thru; B=4: 199->148W @1.01× thru ->
  tok/W 1.35-1.37× at the kernel). The engine-level −10% throughput is the prefill (compute-bound) + clock-switch
  + longer-decode tail; pure decode is iso.
- **Root: decode is SM-UNDERUTILIZED** (HBM-bandwidth + host-overhead-bound, per inc-2). The SMs are not the
  bottleneck, so their clock drops to 1000 MHz with ~no throughput loss -> the energy headroom. **The SAME property
  caps inc-2's per-step latency (host-bound) AND enables g1.1's energy lever** -- one bottleneck, two consequences.
  The lever is therefore a tok/W↔latency TRADEOFF (lower clock = less power but slower), not free.

(Honest scope of the claim: tok/W = served_tok/(wall·avgP) = tokens/Joule, same tokens both sides, internally
consistent (0.90×thru / 0.59×power = 1.53× ✓). Loop-average power blends active+idle -> only the tokens/Joule claim
is supported, not a standalone "decode power-cut %".)

## Non-regression + OFF semantics (honest)

DVFS default-OFF (CIPHER_VOLT unset -> NO clock calls). inc-4 gate VOLT=OFF = 100/100 FAULT=0 (preserved); inc-1+inc-2
modules BYTE-IDENTICAL + gates PASS; no `.so` change; anchor 1f305ce6 UNCHANGED; clock reset on exit (`-rgc`).
**"OFF byte-identical" = OUTPUT byte-identical** (FAULT=0; the PowerSampler thread is read-only and runs in both
arms, so the PROCESS is not identical, but numerics are unaffected -- stated precisely).

## Gate status (Mem #11) + STOP

1. Correctness FIRST: per-agent teacher-forced KL=0, FAULT=0 with DVFS ON, misroute DETECTED -- PASS.
2. Integrated tok/W (first G1⊕G2 composed): **1.16×→1.53×** vs inc-4 fp16 baseline (DVFS = between-wave, host-side,
   the only capture-safe-by-construction actuator).
3. NON-REGRESSION: inc-1..inc-4 pass with DVFS OFF; OFF output byte-identical; anchor UNCHANGED; clocks reset, subprocs reaped.

**STOP for g1.2 (Marlin density) decision.** g1.1 turns the fp16 substrate's energy axis ON: DVFS now runs WITH the
G1 multi-model mux in one process (the first compute-actuator ⊕ engine composition), 1.53× tok/W throughput-preserving.
Remaining gate-1 increments: g1.2 Marlin (has a CAPTURE_SAFE path -> density/int4); g1.3 Koopman (cusolver, likely
engine-INCOMPATIBLE -> probe-and-honestly-report). The integrated PRODUCT advances; FP8/Marlin density + the 4-model
ceiling + swap mechanism remain. Anil's call on g1.2.
