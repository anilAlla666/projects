# PRE-REGISTRATION — TPW = 2×: compose DVFS (power-cap) + 4-bit to hit 2× tokens/watt (2026-06-11)

GOAL: the directive is TPW = 2× tok/W. Test whether composing the two REAL, measured power levers reaches it:
(1) DVFS — power-capping the GPU (200-700W) forces a lower V/f point; throughput drops SUB-linearly with power
(power ∝ V²f, throughput ∝ f) so tok/W rises until throughput collapses — a sweet spot. (2) 4-bit weights — ~1.5×
more decode tokens per unit memory-traffic/work (MBU phase 1). Prediction: each ~1.4-1.6×; composed ≥ 2×.

## Setup
Real vLLM, Mistral-7B, cudagraph ON, decode-only steady state, batch 16 (realistic serving decode), T=0. Arms:
fp16 vs 4-bit (machete/gptq, vLLM default). Power limits swept {700(default),500,400,300,250,200}W via
`sudo nvidia-smi -pl`. Per run: decode tok/s (best-of-3) + mean power.draw sampled DURING the timed decode (~10 Hz)
⇒ tok/W = tok/s ÷ mean_power. Baseline = fp16 @ 700W (unoptimized). Target = ≥ 2× that tok/W.

## Predictions
1. fp16 tok/W rises as power cap drops (DVFS efficiency), peaking ~250-400W then falling as throughput collapses;
   peak DVFS-only gain ~1.4-1.6× (matches prior 1.53× committed).
2. 4-bit @ default power already > fp16 tok/W (fewer bytes/Joule); 4-bit @ optimal cap composes to ≥ 2× baseline.
3. The 2× point is a (precision, power-cap) pair, e.g. 4-bit @ 300-400W; if even the best pair < 2×, report the
   real ceiling honestly (do NOT cherry-pick a batch/ctx to manufacture 2×).

## Decision rule
Best (precision × power-cap) tok/W ≥ 2.0× the fp16/700W baseline ⇒ TPW=2× DELIVERED (green). Else report the
achieved multiple as the honest ceiling. Quality cost of 4-bit noted (unmeasured), DVFS is lossless (same outputs).

## Discipline
Anchor 2edba0d2 entry==exit (substrate not loaded; this measures the PHYSICS the substrate's VOLT+quant actuators
deliver — power-cap stands in for the VOLT DVFS actuator). Power limit RESET to 700W default at exit. Scratch in
tpw/. Clocks/power sampled per run. tok/W is lossless for DVFS; 4-bit carries an unmeasured quality cost.
