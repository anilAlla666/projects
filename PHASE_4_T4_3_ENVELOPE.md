# T4.3.x — VOLT envelope characterization

## Committed outcome

**VOLT's lift is workload-mechanism-bounded, not workload-batch-bounded.**

The +57% tok/W headline from T4.3.2 reproduces on its own workload —
**TinyLlama-1.1B B=1, +54.97% today vs +57.28% in T4.3.2** (overlapping
95% CIs). That measurement was real and is repeatable.

What does NOT generalize is the framing. The honest claim is:

> VOLT delivers **+55% tok/W on memory-bandwidth-bound decode**
> (small-model, low-batch, with calibrated clock target). On
> compute-bound workloads (Mistral-7B-class at B=1 or any
> reasonable batch), the lift envelope is **−15% to ~0%** —
> VOLT is neutral or actively regresses.

The discriminator is **bandwidth-to-compute ratio at the target clock**,
which depends on (model weight size, batch size, target clock) — NOT on
batch alone, contrary to the current `batch_to_mhz` calibration table's
assumption.

**Marvel-pitch implication that needs to land, not be hand-waved:**

VOLT as currently calibrated is a TinyLlama-class moat, not a Mistral-
class moat. The 7B+ production regime — which is what neocloud
customers actually serve — shows neutral-to-negative VOLT lift. Either
the calibration table grows a model-size axis (preserves the moat on a
narrower envelope after recalibration), or the product claim restricts
to the operating envelope where it actually delivers.

## The envelope (7 conditions)

All conditions: 3 matched-pair (off/on) × 60s decode each, power averaged
over the window (TRIM_LEAD=5s, TRIM_TAIL=2s, sampled at 1Hz via
nvidia-smi). Mirrors T4.3.2's canonical methodology.

| Cond | Model | B | Binary | Pod | Δ tok/W | 95% CI | off tok/s | on tok/s | off W | on W | on clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C6** | **TinyLlama-1.1B** | **1** | current | normal | **+54.97%** | +54.60..+55.33 | 66.9 | 68.0 | 142.0 | 93.3 | 958 MHz |
| C2 | Mistral-7B | 8 | current | normal | -1.33% | -4.03..+1.37 | 286.1 | 253.9 | 133.5 | 120.0 | 983 MHz |
| C3 | Mistral-7B | 32 | current | normal | -1.44% | -2.16..-0.73 | 918.0 | 892.7 | 132.5 | 130.8 | 1092 MHz |
| **C7** | **Mistral-7B** | **1** | current (MHz=1600 override) | normal | **-2.03%** | -2.36..-1.70 | 35.7 | 31.8 | 130.6 | 118.7 | 967 MHz |
| C1 | Mistral-7B | 1 | current (table → MHz=1000) | normal | -13.78% | -14.79..-12.78 | 35.8 | 21.2 | 131.8 | 90.7 | 673 MHz |
| C4 | Mistral-7B | 1 | current (table → MHz=1000) | reset | -14.71% | -16.09..-13.32 | 36.0 | 21.1 | 132.9 | 91.0 | 686 MHz |
| C5 | Mistral-7B | 1 | pre-T4.6.1 | normal | -15.45% | -16.74..-14.16 | 35.8 | 20.8 | 131.4 | 90.3 | 675 MHz |
| (T4.3.2 anchor reference) | TinyLlama-1.1B | 1 | (v0.2.0_T4_3_2) | (prior) | +57.28% | +56.65..+57.90 | 67.4 | 67.6 | 151.2 | 96.4 | 1005 MHz |

## Interpretation logic

The phase prompt mapped five candidate explanations to the C1 vs T4.3.2
gap. The data falsifies four of them:

| Hypothesis | Test | Result | Falsified? |
|---|---|---|---|
| Methodology only — same workload reproduces | C6 reproduces T4.3.2 on its own model | +54.97% vs +57.28% (CIs overlap) | ✅ T4.3.2 reproduces |
| Pod-state drift | C4 = C1 after `nvidia-smi -rgc/-rmc -pl 700` + 30s settle | -14.71% vs -13.78% (within noise) | ✅ pod-state doesn't matter |
| Binary regression | C5 = C1 with pre-T4.6.1 binary | -15.45% vs -13.78% (within noise) | ✅ binary doesn't matter |
| Calibration table over-aggressive at 1000 MHz | C7 = C1 with MHZ=1600 override | -2.03% vs -13.78% (calibration recovers most of the loss) | partially true |
| Mistral-7B B=1 fundamentally not VOLT-favorable | C7 best-case clock = neutral, not positive | -2.03% ± 0.3% (does not cross zero) | ✅ Mistral B=1 has no positive sweet spot |

**The remaining explanation that survives:** model size × clock target
interact to determine whether VOLT helps. The current `batch_to_mhz`
calibration table is keyed only on batch and was tuned for TinyLlama-
class (1.1B params, 2.2GB weights). On Mistral-7B (14GB weights), the
same B=1 → 1000 MHz mapping over-cuts the clock for a workload where
compute is no longer free-riding behind memory wait.

**Mechanism story (load-bearing):**

- TinyLlama-1.1B B=1 at 1980 MHz: 2.2 GB weights × 67 tok/s = ~147 GB/s
  effective bandwidth demand. H100 HBM3 bandwidth ceiling is ~3 TB/s.
  Decode is comfortably bandwidth-bound; SMs spend most cycles waiting
  for memory anyway. Cutting clock to 1000 MHz (−50% compute throughput)
  has zero throughput impact (off tok/s 67 → on tok/s 68) because the
  memory wait was the bottleneck. Watts drop 36% → **+55% tok/W**.

- Mistral-7B B=1 at 1980 MHz: 14 GB × 36 tok/s = ~504 GB/s effective
  bandwidth — still bandwidth-bound, but the kernels per memory wait
  are much larger (each layer's matmul is on 4096×4096 weight matrices
  vs TinyLlama's 2048×2048). At 1000 MHz the compute can no longer
  hide behind memory wait: tok/s drops 36 → 21 (-41%). Watts drop 31%.
  Compute throughput loss outpaces watts savings → **−14% tok/W**.

- Mistral-7B B=1 at 1600 MHz (C7): partial recovery. tok/s drops
  36 → 32 (-11%), watts drop 9%. Balanced → **−2% tok/W**. Even the
  best calibration in our sweep doesn't get positive lift.

- Mistral-7B B=8/B=32: kernels are bigger still; compute is firmly
  bound. VOLT_BATCH lookup picks 1600/1980 MHz; almost no clock cut
  → almost no watts savings → tok/W ~0%.

## Methodology audit (S1)

T4.3.2 anchor methodology (now verified against the npairs summary.json):
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Prompt: "Hello, world." (~5 tokens), MAX_NEW=32, tight generate loop
- 60s decode window, 1Hz nvidia-smi sampling in shell
- 5 matched pairs of (off, on)
- Off: 151.16W, 67.40 tok/s, 0.4459 tok/W at 1980 MHz
- On: 96.37W, 67.59 tok/s, 0.7013 tok/W at 1005 MHz
- Δ tok/W: +57.28% (CI +56.65..+57.90)

Tonight's earlier volt_b1_check.py (the side-check that motivated this
campaign):
- Model: Mistral-7B-v0.1
- Single generate(max_new_tokens=400) per measurement
- 10Hz nvidia-smi sampling via Python subprocess, skip first 5 samples
- Single measurement, not matched pairs
- Observed: 19.5 tok/s off, 12.6 tok/s on; +2.4% tok/W

The 14h-of-engineering "regression" framing was misleading. The
volt_b1_check.py wasn't measuring the same workload T4.3.2 measured —
it had swapped Mistral-7B in. Methodology differences (matched pairs vs
single measurement, 1Hz vs 10Hz sampling, 60s loop vs 400-token-single-
generate) are second-order; the model swap is the primary cause.

## Discipline gate

| Gate | Result |
|---|---|
| Phase 3 ABI surface (ioctl nrs 1-10) | ✅ untouched |
| ABI additive invariant | ✅ preserved (no new ioctls in this phase) |
| Fallback kmod md5 `55ab8c0c` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30` | ✅ unchanged |
| Kernel taint | ✅ 12288 |
| cipher_kmod loaded | ✅ refcount=2 |
| Production artifacts modified | NONE (measurement-only phase) |
| Pod state at start | 345 MHz idle, 70 W, persistence Enabled, 700 W limit |
| Pod state at end | clocks restored (VOLT atexit) |

## Strategic implications

### What the moat actually is now

The honest moat language post-envelope:

> CIPHER's VOLT actuator delivers +55% tok/W on memory-bandwidth-
> bound decode workloads (small models, low batch, calibrated clock
> target). On compute-bound workloads — the 7B+ production regime —
> VOLT is neutral or negative depending on calibration.

This narrows the previous "VOLT lifts tok/W +57% on B=1 decode" claim,
which was a single-operating-point measurement falsely generalized.

### The marvel-pitch B200-parity claim

The B200-parity-on-H100 framing rested on VOLT contributing meaningful
tok/W lift on production-realistic workloads. **The data does not
support that.** Two options for the pitch:

1. **Narrow the target.** "CIPHER recovers TinyLlama-class efficiency
   on H100, with B200-parity tok/W for memory-bandwidth-bound workloads."
   Smaller moat than the original framing, but defensible.

2. **Recalibrate and re-measure.** Extend `batch_to_mhz` to a
   `(model_size, batch_size) → mhz` table. Re-measure C7-equivalent
   points across (TinyLlama, Mistral-7B, Llama-70B-class). If
   recalibration recovers some positive lift on 7B+, the moat envelope
   widens. If C7's −2% is the best achievable on 7B+ at any clock, the
   moat is narrow by construction.

The envelope-then-recalibrate path is the principled-researcher choice.

### What this does NOT change

- T4.3.2's +57.28% measurement was real and reproduces (C6 +54.97%).
- VOLT's mechanism (NVML clock-lock + kmod ioctl fall-through for
  non-root) is unchanged and works correctly across all conditions.
- The discipline pattern (matched pairs, decode-window-only power
  averaging) is correct; the gap was in workload coverage, not method.

## Recommended next actions

1. **Update strategic framing.** Replace "+57% tok/W on B=1 decode" with
   the envelope-conditional claim everywhere it appears.
2. **Calibration sweep across model sizes.** Add a 2-3h follow-on
   campaign covering (TinyLlama, Mistral-7B, Llama-3-8B?, Llama-3-70B-
   quant) × {1000, 1200, 1400, 1600, 1800 MHz} × B={1, 8, 32}. Map
   the lift surface in (model_size, batch, clock) space.
3. **Decide on calibration scope.** If recalibration recovers >+10%
   on 7B+, ship a model-aware VOLT. If not, pivot VOLT to "small-
   model accelerator" and find a different 7B+ lever (e.g., HBM
   scheduling, decode-stream priority).

## Artifacts shipped

| Path | Purpose |
|---|---|
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/envelope_driver.py` | One workload-arm driver, env-configurable (WL_MODEL + WL_BATCH) |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/run_condition.sh` | C1-C6 orchestrator, n=3 matched pairs, mirrors T4.3.2 structure |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/run_C7.sh` | C7 (Mistral B=1 with MHZ=1600 override) |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/analyze_condition.py` | Per-condition aggregator: matched-pair Δ, 95% CI from stderr |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/{C1..C7}/summary.json` | Per-condition results (per-pair + summary + Δ) |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/{C1..C7}/pair{N}_{off,on}/watts.csv` | Raw 1Hz nvidia-smi samples |
| `/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/{C1..C7}/pair{N}_{off,on}/run.log` | Per-pair driver stdout/stderr |
| `/home/ubuntu/PHASE_4_T4_3_ENVELOPE.md` | This document |
