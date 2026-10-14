# Verifier Bench scorecard

**Milestone 1 · Terminal-Bench 4.0, 5 tasks, 2 models × 2 harnesses, and a second harness variant**

## The score

| grader | accepted correct | rejected wrong | balanced accuracy |
|---|---:|---:|---:|
| **Official tests (human-written)** | **55 / 55** | **51 / 51** | **1.00** |
| Luna · terminus-2 | 33 / 66 | 33 / 61 | 0.52 |
| Sonnet 5 · Claude Code | 22 / 55 | 31 / 51 | 0.50 |
| Luna · Codex | 11 / 55 | 41 / 51 | 0.50 |
| Sonnet 5 · terminus-2 | 0 / 55 | 51 / 51 | 0.50 (wrote a grader on 1 of 5 tasks in 30 min) |
| always pass / always fail (for reference) | 55 / 55 · 0 / 55 | 0 / 51 · 51 / 51 | 0.50 · 0.50 |

**No AI-written grader beat a coin flip overall. The best single grader scored 0.65.** Five tasks and one
attempt per writer are too few to rank models or harnesses. Full details are in `REPORT.md`.

## Second run: a stricter writing protocol (the "+protocol" harness)

Same tasks, same attempts, same models and agent programs. Only the writer's process changed:

- list every requirement the task states, then write one check per requirement;
- build its own correct solution and 2 or 3 flawed ones, and prove the grader passes the first and fails the rest;
- 90 minutes instead of 30.

The writer still saw only the task and its environment.

| writer | first run: accepted / rejected, BA | +protocol: accepted / rejected, BA | graders written | +protocol cost |
|---|---|---|---|---:|
| Sonnet 5 · Claude Code | 22/55 · 31/51 · 0.50 | 34/55 · 23/51 · **0.53** | 5 → 5 | $20.66 |
| Sonnet 5 · terminus-2 | 0/55 · 51/51 · 0.50 | 22/55 · 31/51 · **0.50** | 1 → 4 | $22.14 |
| Luna · Codex | 11/55 · 41/51 · 0.50 | 0/55 · 51/51 · **0.50** | 5 → 5 | $0.91 (reported) |
| Luna · terminus-2 | 33/66 · 33/61 · 0.52 | 11/55 · 40/51 · **0.49** | 5 → 4 | $0.92 (reported) |

**The protocol did not make the graders better.**
- **Scores didn't move:** 0.49 to 0.53, against 0.50 to 0.52 before. 11 of the 18 new graders still accept
  zero correct attempts.
- **Behaviour shifted, not quality:** Sonnet · Claude Code accepts more correct work but catches fewer wrong
  attempts, and Luna · Codex now rejects everything on every task.
- **More time helped only with finishing:** Sonnet · terminus-2 wrote graders on 4 of 5 tasks instead of 1,
  but they are no better.
- **Cost:** each Sonnet attempt cost 2.4 to 3 times as much. The best single grader scored 0.60.

## What we observed

1. **Agents write plausible graders that don't discriminate.** Every grader looks reasonable when read: it
   builds, parses the output, and runs real checks. Almost every one either accepts everything or rejects
   everything on the real attempts.
2. **Too lenient: they check the headline number and miss the stated constraint.** On
   `pretrain-shard-corruption` the task forbids approximate replacement data. Graders recomputed the target
   metric honestly and passed every near miss that hit the number with approximate data.
3. **Too strict: they invent requirements.** Examples:
   - "missing a required field";
   - "expected a surgical patch";
   - "preserve the other 19 shards";
   - made-up "plausible ranges".

   Several reject the task's **own reference solution**. Of the 17 graders, 11 accepted zero correct
   attempts.
4. **They break in ordinary ways:** a 64-character image digest copied wrong, `python3` called in an image
   without Python, a test step that fails when a project has no test files.
5. **Harness matters for finishing, not just quality.** Sonnet in terminus-2 ran out of time on 3 of 5 tasks and
   stopped early on a fourth. In Claude Code, Sonnet finished all 5. Luna finished everywhere.
6. **Cost is lopsided.** Sonnet spent $16 for 10 attempts; Luna's reported cost is a few cents (likely
   understated). Neither graded better.
7. **Human graders are not perfect either.**
   - `react-lead-form`'s grader ignores a requirement its own task states.
   - `uefi-bootkit`'s grader is timing-sensitive under load.
   - One leaderboard "fail" now passes.
   - 15 of 28 eligible tasks have open reports of wrong tests.

   Measuring against a human grader works only after checking it, which this pipeline does.

## Why the protocol did not help

From the writers' own notes on `pretrain-shard-corruption`, the task whose constraint the first run missed:

1. **Listing requirements worked.** Both Luna writers quoted the decisive constraint verbatim ("statistically
   similar, generated, or otherwise approximate replacement inputs are not acceptable").
2. **No writer had a correct solution to prove its grader on.** Real correct attempts reach a loss of about
   6.41.
   - Both Sonnet writers used the broken environment, or a synthetic stand-in, as "correct", so their graders
     pass everything.
   - Luna · Codex used the broken run and set a 6.5 ceiling; its own loss computation reads 6.509 on correct
     attempts that report 6.407, so it rejects them all.
   - Luna · terminus-2 attempted the repair, reached 6.53, and demands shards byte-identical to its repair, so it
     rejects them all too.
   - Self-validation then locked each mistake in.
3. **Their flawed versions broke only the easy requirements:** a step limit, a stopping rule, a hard-coded
   number, a duplicated shard. None used approximate replacement data, the mistake every real near miss makes,
   so no grader was tested on it.

**The lesson:** a writer can only prove its grader against solutions it can build. Where the hard requirement
is exactly what the writer cannot reproduce, a self-check protocol verifies the easy requirements and leaves
the hard one untested.

## What would make the graders better

Ordered by how directly each addresses what we saw:

1. **Validate against real known-good and known-bad work, not the writer's own.** The second run showed
   that self-made solutions are the weak link. Giving the writer a held-out validation signal would target
   exactly that. For example, a tool that runs its draft grader on a few real correct attempts and real near
   misses and returns only pass/fail counts. Those attempts must be separate from the ones the grader is scored
   on, or the benchmark would be tuning on its own test set. The writer would still never see the official
   tests. This goes
   beyond "task and environment only", so it needs HUD's agreement.
2. **Show it a real near miss to reject.** The one mistake the graders never caught is the one real solvers
   make; a writer that can see (or be tested against) one would have to find what separates correct from
   almost-correct.
3. **Derive checks from the stated requirements, and only from them.** Tried in the second run: the writers
   did capture the requirements, but on its own this did not move the score. Worth keeping as part of any
   harness.
4. **Make the writer run its grader in its real container before finishing.** The second run had writers run
   their tests on their own solutions, but not in the container the grader is actually built into. A
   build-and-run step would catch the mistyped digest, the missing interpreter and the ENTRYPOINT mistake, all
   ordinary failures that cost a whole grader.
5. **Give it time.** The second run confirmed this helps finishing (Sonnet · terminus-2 went from 1 to 4 graders
   written) but not quality on its own.
6. **Calibrated tolerances need a real correct run.** The second run required every threshold to come from the
   task or from measuring the writer's own correct solution. That is what produced Luna · Codex's 6.5 ceiling
   and Luna · terminus-2's tolerance: a threshold measured on a wrong solution inherits its error.

## Next steps and whether they are worth it

| step | worth it? | why |
|---|---|---|
| **A validation tool with real attempts** (run the draft grader on a few real correct attempts and near misses, kept apart from the scoring pools; return only counts) | **Yes, first, with HUD's agreement** | Both runs point to the same missing ingredient: a real signal of what correct and almost-correct look like. It adds information beyond "task and environment", so it is HUD's call |
| 3 attempts per writer and 10 to 15 tasks | Yes, once a harness moves the score | Two runs at 0.50 give nothing to rank. Scale when there is a difference to measure |
| HUD taskset (writing the grader is the task; reward = BA on the hidden pool) | Yes, a deliverable | Makes the benchmark trainable on HUD; the scores above say there is a lot of room to train into |
| DeepSWE | Yes, after the above | Code-diff tasks with test-suite verifiers; its pools need our own attempts |
| Deliberately adversarial wrong answers | Later | Graders don't yet separate real near misses from correct work |
