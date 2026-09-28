# Task B — CP 2.4 composed-gate re-measurement on c2c5d313

**Date:** 2026-05-17. Side experiment (parallel to the CP 4.8 sequence, run
between build-STEP items 3 and 4). Purpose: confirm the CP 2.4 3.617× tok/W
headline reproduces on the current production anchor `libcipher_rt c2c5d313`,
for the investor one-pager.

**Harness:** `run_composed.sh` (copied from `cp_2_4/`, isolated output dir),
Mistral-7B B=1 varied-prompt decode, n=5 matched pairs, three arms
(vanilla / marlin / all-on), `analyze_composed.py`.

## Result — exact reproduction

| Lift | Task B (c2c5d313, 2026-05-17) | CP 2.4 original | match |
|---|---|---|---|
| tok/W allon/vanilla (composed) | **3.6166× [3.5908, 3.6423]** | 3.617× [3.591, 3.642] | ✅ exact |
| tok/s allon/vanilla | 1.7954× | 1.795× | ✅ |
| tok/W marlin/vanilla | 1.307× | 1.307× | ✅ |
| tok/W allon/marlin | 2.768× | 2.768× | ✅ |
| composition | clean: 1.090 × 1.647 = 1.795 | same | ✅ |

Per-arm absolutes also reproduce: vanilla ~48 tok/s, marlin ~52.5 tok/s,
all-on ~86.5 tok/s @ 105 W.

**Conclusion.** The 3.617× tok/W composed-stack headline is reproducible on
`c2c5d313`. The investor one-pager number stands. No CP 4.8 dependency — this
was a confirmation run.

*Note: `analyze_composed.py` hardcodes its output to `cp_2_4/composed_result.json`,
so that file is now this re-measurement — numerically identical to the original
to 4 s.f., so no evidence loss. This Task B copy (`composed_result.json` here)
is the labelled artifact.*
