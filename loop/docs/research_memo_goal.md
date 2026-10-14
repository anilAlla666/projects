# Research memo: deriving the goal for a grader without a reference solution

Research agent's memo, 27 Sep 2026. One key source (arxiv.org/abs/2606.20158) was re-opened and its numbers
confirmed; the other citations are as the agent reported them.

## Ranked shortlist

1. **Solve independently, cross-check, abstain on disagreement.** Before writing any check, the writer solves the
   task itself 2 to 3 independent ways and encodes an assertion only where the solutions agree; where they disagree
   it looks for a decisive check stated in the task, or leaves that observable unasserted. Measure goal derivation
   directly: does the writer's grader accept the task's reference solution, and do its derived expected values match
   the official tests'. Ranked first because it attacks the diagnosed bottleneck and is scorable apart from BA.
2. **Self-test against single-requirement mutants and a second valid solution.** Mutate the writer's own derived
   solution one stated requirement at a time (the grader must reject each) and solve the task a second, different
   valid way (the grader must accept it). Precedent: Meta's ACH, concern-specific mutants
   (arxiv.org/abs/2501.12862; 571 production tests, 73% accepted by engineers); CANDOR, consensus-based oracle
   generation (arxiv.org/abs/2506.02943). The accept side punishes invented requirements.
3. **N-version consensus over whole grader-writing runs.** Several independent runs, majority vote per assertion.
   N-Version Programming with Coding Agents (Ron, Baudry, Monperrus, June 2026, arxiv.org/abs/2606.20158): 48 agent
   implementations, 1,000,000 inputs, mean failures 387.44 for single versions and 130.99 for voted triples, with
   "substantial common-mode failure" where the specification is hard or ambiguous. Best on top of method 1.
4. **Invariants mined from the writer's own solve, kept only if the task text justifies them** and a second valid
   solution satisfies them. A floor under 1 to 3, not a substitute.

## Survey notes

- **Test-oracle problem:** specified, derived and implicit oracles (Barr et al., TSE 2015); metamorphic testing
  checks relations between outputs when the answer itself is unknown; mined invariants describe what code does, not
  what it should do.
- **LLM tests without ground truth:** CodeT dual execution agreement (arxiv.org/abs/2207.10397); AlphaCode filtering
  and clustering by behaviour (arxiv.org/abs/2203.07814); ConVerTest (arxiv.org/abs/2602.10522); incoherence between
  independently sampled programs lower-bounds the error rate and is blind when samples share the bug
  (arxiv.org/html/2507.00057). Eureka and Text2Reward assume a ground-truth fitness function already exists, so they
  shape rewards toward a known goal rather than derive one.
- **Synthetic task pipelines:** SWE-smith (arxiv.org/abs/2504.21798) and R2E-Gym (arxiv.org/abs/2504.07164) build
  tasks backward from a known-correct artifact (working code, a merged commit). None derives a goal from task text
  and environment alone.
- **LLM oracles copy behaviour:** generated oracles tend to capture what the program does rather than what it should
  (arxiv.org/abs/2410.21136).

## Where deriving the goal is impossible, and what practitioners do

When the decisive check compares against information not present in the environment (our pretrain task's hashes of
the true records), no derivation from task and environment can recover it. Disagreement-based methods are also blind
when every attempt makes the same mistake. The field's answer: build the task backward from a correct artifact, or
require the author's solution before the task exists (Terminal-Bench requires an author-written solution script).
For a synthetic pipeline, that means the grader should be written with the solution the generator already holds.
