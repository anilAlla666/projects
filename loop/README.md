# Verifier Bench, and a verifier for agent-written graders

Two things live here.

1. **Verifier Bench** measures how well AI agents write the graders (verifiers) for real tasks. An agent gets a
   Terminal-Bench 4.0 task and writes its `tests/`; the tests are scored on real leaderboard attempts whose pass or
   fail is already known from the human-written official tests. Agent-written graders score about **0.50**
   balanced accuracy (BA); the official ones score **1.00**.
2. **The verifier** (`python -m vb make`) takes a task and an agent-written grader (or has an agent write one), tests
   the grader on candidate results, repairs it, and ships it with a report saying what each check rests on and what
   needs a human.

**What the verifier does and does not do, measured (details and every run in `docs/research_intent_goal.md`):**
- It repairs graders when its evidence can be right: on two of the four untouched test tasks, 0.50 to 1.00 and 0.70
  to 0.95 (judge v0); on the six development and held-out graders, a mean of 0.70 to 0.90.
- It cannot verify what only the task author knows (hidden evaluation data). With judge v0 it inverted a grader when
  every source of evidence, its own judge included, shared one wrong model (0.45 to 0.05), and with judge v1 it
  loosened a perfect grader on a reading of the text (1.00 to 0.91). Judge v2 adds the rules and the guard below; on
  those two graders it gives 0.45 to 0.55 and 1.00 to 1.00.
- It is not perfect, and nothing here claims it is. Each grader ships with its report so a person can see which
  checks are verified and which are not.

## Run it

Everything runs on a Linux machine with Docker (this one, or another over ssh with `VB_HOST=user@host`). The model
calls go to the Anthropic API from wherever `ANTHROPIC_API_KEY` is set.

    python -m vb setup     --task T             # a bundled task (bundles/T): build its images
    python -m vb controls  --task T             # official tests 1.00; always pass, always fail, cheat, none 0
    python -m vb make      --task T [--grader D] [--cap-usd 15]
                                                # the verifier, end to end (below)
    python -m vb score     --task T --grader D  # a tests/ folder scored on the task's hidden pool of real attempts
    python -m vb reproduce [--task T]           # every saved grader, before and after the loop, rescored (free)

`vb make`:
1. **Grader.** Yours (`--grader path/to/tests`), or a Sonnet 5 agent writes one in the task's machine.
2. **Evidence.** Three solver agents of different models (Haiku 5.5, Sonnet 5, Sonnet 5.5) attempt the task in its
   own machine, with no network; their results become the candidates (`vb/solve.py`). Unlabelled: some may be wrong.
3. **Verify and repair** (`vb/verify_loop.py`, judge v2 by default, Opus 5.5, up to 3 rounds): run the grader exactly
   as the real grader runs it on every candidate, the empty attempt and the near misses; a judge decides, for every
   rejection, whether the attempt or the check is wrong; traces every requirement to a check and back; writes near
   misses; repairs the grader. Three rules keep it from making a grader worse:
   - **An expected value changes only when it is settled:** by a quoted sentence of the task, by a recomputation
     the task specifies, or by the task's own data, where one reading must beat every other out of sample by half and
     reproduce every labelled example the task gives. Otherwise it is *contested*: the original check is kept and
     reported, however many attempts disagree.
   - **A check that needs data the machine lacks** (the author's hidden evaluation set) is *unverified*, and no bar
     is loosened on a stand-in.
   - **The do-no-harm guard** (mechanical, not a prompt): a round's repair is kept only if, on the same evidence, the
     new grader agrees with the judge's own verdicts on more items than the old one and on none fewer; otherwise the
     old grader is restored.
4. **Report** (`report.md` beside the repaired grader): every check with its basis (task-text, recomputed,
   data-settled, format, contested, unverified). Nothing waits on a person: the grader ships and runs as is; the
   report says which checks rest on the task and which do not.

A spend cap covers the whole run. Typical cost: a few dollars per task.

**Tasks.** Harbor's layout, as Terminal-Bench 4.0 uses it: `instruction.md`, `task.toml` with `artifacts` and a
separate verifier environment, `environment/`, and `tests/Dockerfile` for the grading environment. Build one with
`python -m vb.hudtask --task T` (`VB_TB_ROOT` points at the tasks folder). With labelled attempts (a pool,
`python -m vb.pool`) the grader is scored before and after; with `--no-pool` it is written, verified and repaired
but not scored.

## Results

Balanced accuracy on each task's hidden pool of real, labelled leaderboard attempts (21 per task), scored once, never
seen by the loop. One run per arm.

RESULTS_TABLE

## What is in the repository

- `vb/`: the pipeline (pool building, task building, the writer harness, the probe, the verify loop, the solver
  stage, the requirement model's data).
- `gpu/`: training and generation for the requirement model (Qwen2.5-7B-Instruct + LoRA). Optional: in the
  ablation it was not shown to beat the untuned model or no model at all (step 6).
- `runs/`: every run: transcripts, evidence, each round's grader, results.
- `results/step3/`: the step 3 graders before and after, with their recorded scores (`vb reproduce`).
- `docs/research_intent_goal.md`: every experiment, pre-registered before it ran, and its result.
- `REPORT.md`, `SCORECARD.md`: Verifier Bench milestone 1 (agent graders at about 0.50).

## Limits, stated plainly

- **Hidden data.** A check that needs the author's private data cannot be verified; the report marks it unverified.
- **Shared mistakes.** If every solver and the judge make the same mistake, the evidence is wrong; judge v2 keeps a
  contested check rather than overwriting it, but cannot tell which side is right unless the task's data settles it.
- **Multi-container tasks.** A task whose environment needs a second service (docker-compose; ctr-optimization) is
  not supported: the writer, solvers and judge each get one machine, so the solvers produce nothing.
- **Judge variance.** The same judge on the same grader reached 0.85 in two runs and 1.00 in a third. One judge pass
  per round; a vote of several was cut for budget.
- **Sample.** 13 Terminal-Bench evaluation tasks, 2 fresh ones, one run per arm. No figure here is a rate with an
  interval.
