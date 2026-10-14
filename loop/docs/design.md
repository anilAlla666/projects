# Verifier Bench: design

**Date:** 26 September 2026 · **Revision 2** (replaces revision 1, commit acf94b4, after HUD's answers) · **Owner:** Anil Kumar

## 1. Purpose

Measure how well agents write **verifiers**: the tests that decide whether a task was solved.

An RL environment is only as good as its verifier:
- a verifier that rejects valid work starves training of signal;
- one that accepts wrong work teaches the model to game it.

The goal has three parts:
1. measure how well agents write verifiers, against verifiers written by people (Terminal-Bench 4.0 and DeepSWE);
2. separate the effect of the **model** from the effect of the **harness** (the agent program around the model), by running the same model in different harnesses and different models in the same harness;
3. build a harness that makes agents write better verifiers, measured the same way.

**Core principle:** a verifier is never judged by reading it. It is judged only by what it does on solutions whose correctness is already known.

## 2. Decisions

| Topic | Decision | Source |
|---|---|---|
| Tasks | Terminal-Bench 4.0 first, then DeepSWE | HUD |
| Trusted grader | The task's own human-written tests. A grader is only trusted when a person wrote it | HUD |
| A good verifier | Agrees with the trusted grader **and** catches near misses and cheats; both are reported | HUD |
| What the writer sees | The task instruction and the task environment. Never the official tests, the reference solution or any attempt | HUD |
| Writers | GPT-6 Luna and Claude Sonnet 5 first, each in its own harness and both in one shared harness (§6.1) | HUD (models); owner (harness comparison) |
| Deliverables | A report, and a HUD taskset where "write the verifier" is the task | HUD |
| Budget | $500 to start | HUD |
| Where it runs | The Hetzner server (8 CPUs, 15 GB, Docker, Harbor 0.23.0) | Owner |

## 3. Data

### 3.1 Terminal-Bench 4.0

- **66 tasks.** Every task grades in a **separate verifier container**: the agent's declared output files (`artifacts`, e.g. `/app/`) are copied into a fresh container built from `tests/Dockerfile`, then `tests/test.sh` writes `/logs/verifier/reward.txt`.
- **Attempts:** the Harbor Hub leaderboard (`9f966760-00f1-424e-90f5-c964fb6f6091`) holds 27 runs × 330 attempts = 8,910 recorded attempts.
  - Status: 3,567 passed, 5,155 failed, 188 errored (indexed 26 Sep 2026).
  - Each attempt downloads with its artifacts, reward, per-test results (`verifier/ctrf.json`) and trace.
- **57 tasks** have at least 3 passing and 3 failing attempts. **7 tasks** never passed; they are excluded, because they have no known-correct solution.

### 3.2 DeepSWE

- **113 tasks** across 91 repositories and 5 languages, public on GitHub (`datacurve-ai/deep-swe`) in the same Harbor format.
- The graded artifact is the agent's diff, `/logs/artifacts/model.patch`.
- There is **no leaderboard on Harbor Hub**, so DeepSWE pools need our own attempts (§5.2). DeepSWE comes after Terminal-Bench (M2).

### 3.3 Two facts that shape everything

1. **A recorded attempt can be graded again with a different verifier.** `harbor job regrade <job dir> -p <task dir>` reruns the verifier of `<task dir>` on each recorded attempt's artifacts.
   - Regrading an Astra attempt with its original tests reproduced its reward (1.0).
   - So pools come from real leaderboard attempts: no agent reruns and no model spend.
2. **The published package pins a prebuilt verifier image with the original tests inside it.** In `[verifier.environment]`, `docker_image` points at that image, so a replaced `tests/` folder is silently ignored.
   - Measured: an "always fail" test script still scored 1.0 until the pin was removed; afterwards it scored 0.0.
   - The GitHub source does not pin; the Harbor Hub package does.
   - **Every regrade with a non-original verifier must remove the pin, and every run proves the swap took effect (§6.3).**

## 4. Core objects

### 4.1 Task

A Terminal-Bench or DeepSWE task directory exactly as published: `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`. We never edit the original.

### 4.2 Pool entry

A recorded attempt at a task, with a known label.

| Field | Meaning |
|---|---|
| `trial_id` | Harbor Hub trial id (or local id for our own attempts) |
| `source` | `pass` (a real attempt the official tests pass), `real_fail` (a real attempt they fail), `near_miss` (a real failed attempt that passed at least 80% of the official tests), `reference` (the task's own solution), `nop` (an agent that did nothing), `hack` (built to fool a verifier; M2) |
| `label` | `pass` or `fail` |
| `recorded_reward` | the reward on the leaderboard |
| `regraded_reward` | the reward when we rerun the original tests ourselves |
| `model`, `agent` | who produced the attempt |
| `ctrf` | passed and total official tests |

**Label rule:**
- an entry is kept only when the original tests, rerun by us with the image built from the folder, give the same verdict as the recorded reward;
- entries that disagree are dropped and counted, because they are flaky or environment-dependent;
- a `hack` is `fail` by construction; a `nop` is kept only if the official tests fail it.

**Pool per task (M1):** 10 `pass` from at least 4 different runs; 10 fails, near misses first; 1 `reference`; 1 `nop`.

### 4.3 Verifier

A verifier is a `tests/` folder with the same contract as the task's own:
- `tests/Dockerfile` builds the verifier container;
- the solver's artifacts arrive at their declared paths;
- `tests/test.sh` writes `1` or `0` to `/logs/verifier/reward.txt`.

The writer produces this folder. To grade with it, we copy the task, replace `tests/`, remove the pin (§3.3) and regrade the pool.

## 5. How pools are built

### 5.1 Terminal-Bench 4.0

1. **Index** every leaderboard attempt with its reward (`vb/index.py`).
2. **Select tasks** by rule, not by hand:
   - at least 10 passing and 10 failing attempts with no errors;
   - no GPU;
   - artifacts on the main container only;
   - verifier timeout of 900 s or less;
   - no open community issue saying the official tests are wrong.
3. **Download** the chosen attempts and build a Harbor job directory per task.
   - Whole-job download fails on these jobs, whose names contain `/`, so attempts are fetched one by one.
4. **Add `reference` and `nop`** entries with Harbor's `oracle` and `nop` agents: compute only, no model spend.
5. **Classify** failures as `near_miss` or `real_fail` from `ctrf.json`.
6. **Check labels** by regrading the whole pool with the original tests, unpinned; drop disagreements (§4.2).

### 5.2 DeepSWE and hacks (M2)

- **DeepSWE pools:** `reference` (the task's solution diff), `nop`, and real attempts from inexpensive agents we run ourselves.
- **Hacks** come from a red-team agent that sees a written verifier and tries to pass it with wrong work; the same agent serves the attack stage.

## 6. Execution

### 6.1 Writing a verifier

1. For each task we generate a **writer task** in Harbor format:
   - the same environment image;
   - an instruction: the task's instruction, plus the verifier contract (§4.3), plus the list of artifact paths;
   - an artifact `/verifier_out/`.
2. **Writers are model × harness pairs**, arranged so both effects can be read off:

   | | Native harness | Shared harness: `terminus-2` (Harbor's reference agent) |
   |---|---|---|
   | Claude Sonnet 5 | Claude Code | terminus-2 |
   | GPT-6 Luna | Codex | terminus-2 |

   - The **model effect** compares the two rows in the shared column.
   - The **harness effect** compares the two columns within a row.
   - All four reach the models through the HUD gateway (`inference.hud.ai`).
   - Our own harness improvements (M2 onward) enter as a new column and are compared the same way.
3. **Network:** the solver's machine reaches only the gateway. Claude Code and Codex run inside it and need the gateway; terminus-2 drives it from outside and needs nothing. GitHub and Hugging Face, where the official tests are public, are never reachable.
4. Two attempts per writer per task in M1.

### 6.2 Scoring a verifier

Regrade the task's pool with the verifier. Every reward becomes a verdict: `pass` if 1, `fail` if 0, `error` if no reward was written, the build failed or it timed out. An error counts as `fail` and is reported as flakiness.

For one verifier on one task, with $C$ the correct entries and $W$ the wrong ones:

$$
\text{TPR} = \frac{|\{s \in C : \text{pass}\}|}{|C|}, \quad
\text{TNR} = \frac{|\{s \in W : \text{fail}\}|}{|W|}, \quad
\text{BA} = \frac{\text{TPR} + \text{TNR}}{2}.
$$

**Reported:**
- TPR, TNR and BA;
- TNR per source: `real_fail`, `near_miss`, `nop`, and later `hack`;
- flakiness;
- the writer's cost in dollars.

Per writer, results are macro-averaged over tasks and attempts, and shown as the model × harness table above.

### 6.3 Baselines on every run (the run is void if one fails)

| Verifier | Must score |
|---|---|
| Original tests, unpinned (trusted) | BA = 1 on the kept pool (true by the label rule; recomputed as a check) |
| Always pass | TPR 1, TNR 0, BA 0.5 |
| Always fail | TPR 0, TNR 1, BA 0.5 |
| Network probe (passes only if GitHub is reachable while grading) | passes nothing |

- "Always fail" is the proof that the swap took effect. If it passes anything, the pin is still in force.
- "Network probe" is the proof that grading is offline (`[verifier] network_mode = "no-network"` in every task copy). Without that, a written verifier could download the public official tests and score perfectly without any skill.
- A leak scan also flags written verifiers that copy the official test files, carry their canary string, or name a fetch from GitHub or Hugging Face in their Dockerfile; flagged verifiers are excluded from their writer's numbers.

**Uncertainty:** each writer's BA is reported with a 95% bootstrap interval over task-attempts, and a model or harness effect is stated only where the intervals do not overlap. With five tasks in M1, the intervals will be wide.

### 6.4 Reproducibility and cost

- Runs go to `runs/<name>/<UTC stamp>_<git sha>/` and are never overwritten.
- Every writer trace, verifier folder and regrade result is kept.
- Model spend is capped per stage in the config; a smoke test on one task precedes any batch.
- Regrades cost compute only: about 2 minutes each, 6 at a time.

## 7. Milestones

| Milestone | Scope | Exit check |
|---|---|---|
| **M1** | Terminal-Bench 4.0, 5 tasks; pools from leaderboard attempts; four writers (2 models × 2 harnesses) write verifiers; scoring with baselines | baselines hold (§6.3); BA per writer per task with per-source TNR; the model × harness table filled |
| **M2** | DeepSWE pools; hacks from a red-team agent; attack and repair stages; about 20 tasks | at least one attack confirmed; repair measured |
| **M3** | 50+ tasks; the HUD taskset (writer tasks via `hud.integrations.harbor.adapt`, reward = BA on the hidden pool); harness improvements; the report | a HUD run reproduces the offline BA for the same verifier |

## 8. Disclosures

- **Contamination:** Terminal-Bench 4.0 and DeepSWE tests are public, so models may have seen them in training. Network blocking stops live lookup, not memory. The report says so.
- **Fetching at build time:** a verifier's Dockerfile builds with network access (it has to install packages). The leak scan catches a visible fetch of the official tests; a disguised one, such as a package that bundles them, is not claimed to be caught.
- **Label noise:** the human-written tests have community-reported problems. Tasks with known issues are excluded; the label rule removes attempts whose verdict does not reproduce.
- **The artifact boundary:** a verifier sees only the artifacts a task declares, as the official one does. Anything the solver did outside them is invisible to both.

## 9. Layout

```
verifier_bench/
  vb/            index.py select.py pool.py taskdir.py regrade.py scorer.py writer.py evaluate.py report.py util.py
  configs/       m1.json
  templates/     writer_instruction.md  baselines/{always_pass,always_fail}/
  tests/         unit tests (local); integration checks run on Hetzner
  docs/
```

On Hetzner the repo lives at `/root/vbench/repo`, and data at `/root/vbench/{meta,pools,verifiers,runs}`.
