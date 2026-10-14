# Building the verifier model

*9 Oct 2026. Builds on `docs/verifier_model_case.md` (27 to 28 Sep), which stays the governing argument.*

## What the model is

Agents write graders. The verifier model makes sure each grader is correct in both directions (it passes every
correct solution, and fails wrong ones, near misses and hacks), says what is wrong, and repairs it, so grader writing
becomes autonomous: task in, trustworthy grader out, no human and no reference solution.

Its three capabilities (case document, section 9): **derive the goal** (a checkable definition of success with a
calibrated confidence for each part), **judge attempts** against it, **harden graders** (the near misses and shortcuts
a grader must reject). It is the outside evidence that moved grader accuracy from 0.52 to 0.80 to 0.88, made reliable.

## Acceptance criteria: every failure Verifier Bench measured

| Failure (M1, M1b, research rounds 1 to 5) | What the system must do | Measured by |
|---|---|---|
| Ordinary breakage: image digest, ENTRYPOINT, timeouts | build and run the grader in its real verifier environment before any verdict | grader runs on the pool without infrastructure errors |
| Invented requirements: 11 of 17 graders accepted zero correct attempts | every check traced to a statement of the task; checks a correct attempt fails are removed or corrected | TPR on held-out pools |
| Checking the metric, not the stated constraint | every stated constraint has a check | TNR on held-out pools |
| The writer's own solution was wrong and self-validation locked the error in (144 vs 88 minutes; bcu vs dia) | never validate against the writer's own solution or the unfixed code alone; independent evidence only | TPR on tasks where the writer's model cannot solve |
| Near misses and hacks pass (3 to 4 of 10 caught even with outside evidence) | generate near misses and shortcut attempts and require the grader to reject them | near misses caught; red-team success rate |
| Terminal-Bench and DeepSWE format (artifacts, separate verifier containers, shell tests) | work in that format, not only on Python repositories | everything above is measured on Terminal-Bench tasks |

The bar (case document, section 9): beat the writer alone (0.52), a panel of five existing models (0.80) and the task
author's own solution (0.83 to 0.88), on held-out tasks, plus near misses caught and red-team resistance.

## What 8 Oct taught (`~/verifier_model/docs/report_2026-10-08.md`)

- A model trained to imitate maintainer tests learns to invent: 90% of its tests failed every fix. The case document
  had already said to learn from outcomes, not imitation.
- A grader checked only against the unfixed code keeps broken tests. Every check needs a second side.
- Strong agents' graders hold real tests under the bad ones: removing the bad ones is worth up to +0.08 on Python
  tasks, and dropping tests that fail every attempt (no labels) +0.035.

## Build order, each step measured on held-out Verifier Bench tasks

| Step | Work | Cost | Done when |
|---|---|---|---|
| 1 | Restore the instrument on Hetzner: the five development pools from the local HUD copies; rebuild the held-out pools (mp-checkpoint-consolidation, photonic-waveguide-routing) from leaderboard trials; positive controls | free | official tests score 1.0 and trivial graders 0 on every pool |
| 2 | Held-out confirmation of the outside-evidence result (pre-registered, configs/heldout.json): base against outside evidence, 3 rollouts each | HUD rollouts, about $25 | mean BA at least 0.75 and 0.2 above base, or the premise is reported false |
| 3 | The verify loop with existing models: run the grader; independent solutions from several models; requirement tracing; generated near misses and hacks; verdict and repair; back to the writer | API, budget to agree | the loop's graders beat 0.80 on held-out tasks |
| 4 | Data engine: all Terminal-Bench 4.0 leaderboard attempts with labels (8,910 over 66 tasks), split by task; correct-by-construction tasks at scale | free download; solver attempts for new pools paid | train and held-out task sets fixed before training |
| 5 | Train the dedicated model on outcomes (grader quality on hidden pools, calibration, adversarial), against step 3's loop | GPU | beats step 3 on held-out tasks |
