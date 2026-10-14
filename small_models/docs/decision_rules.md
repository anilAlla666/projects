# Decision rules, fixed before the test split or the real set is scored (8 Oct 2026, 15:05 BST)

## Setting

The spec pre-registered the **strict** setting: the judge sees the named failing test and a fixed half of the
library's other tests. After the dev results, the user directed that the model be general-purpose, usable by anyone
writing graders without a reference solution. A grader can run every test a repository already has, so the
**realistic** setting was adopted: the judge sees the named failing test and every existing test; only the task's
other failing tests stay hidden (on the real set, the reference fix's other new tests). This change was made before
any test-split or real-set example was scored by any judge model. Both settings are reported.

## Checkpoint and threshold (realistic)

1. Candidates: the epochs of the realistic round (`real1`), each scored on realistic dev with `scripts/threshold.py`.
2. Threshold rule: among thresholds 0.05 to 0.9, choose the one that maximises balanced accuracy on the realistic
   dev hard subset (every visible test passes), subject to overall realistic dev balanced accuracy at or above the
   realistic dev visible-tests-only baseline.
3. Checkpoint rule: the candidate with the highest hard-subset balanced accuracy under rule 2; ties go to higher
   overall dev balanced accuracy.
4. Calibration (temperature, abstain band) on realistic dev with `scripts/calibrate.py`.

## Strict fallback (already chosen on strict dev)

Round 4 epoch 0 at threshold 0.7 (strict dev overall 0.959 on the 2,007-example dev, hard subset 0.684).

## What is touched once

The synthetic test split and the 939 real agent fixes are scored once, by `scripts/final_realistic.sh`, with the
choices above. The 32B line in the synthetic test report was scored on strict-setting inputs and is labelled so.
