# Research memo: making agent-written graders capture task-author intent

Research agent's memo, 27 Sep 2026. Its top-ranked source (arxiv.org/abs/2606.08960) was re-opened and its numbers
confirmed; the other citations are as the agent reported them.

## Ranked shortlist

1. **Adversarial hacker/fixer loop against the grader.** After the writer produces its grader, a second agent tries
   to make an attempt pass it without meeting the task's constraints; each successful hack goes back to the writer
   to patch, and a solver pass confirms legitimate variants still pass. 2 to 3 rounds. Evidence: "Hardening Agent
   Benchmarks with Adversarial Hacker-Fixer Loops" (Zhong et al., June 2026, arxiv.org/abs/2606.08960): attack
   success 62% -> 0% on KernelBench (held-out exploit corpus), 39% -> 17% for a strong attacker on 77 Terminal-Bench
   tasks; 16% of audited benchmark tasks were hackable from the task description alone. Measure: BA on the pool, plus
   the success rate of a fresh red-team pass not seen during hardening.
2. **Contrastive near-miss examples instead of prose intent.** Give the writer one or two labelled cases (an attempt
   that passes the obvious check but is wrong, with why). Rationale: Inverse Reward Design (arxiv.org/abs/1711.02827)
   treats a stated reward as evidence of intent only in its context; few-shot preference learning
   (arxiv.org/abs/2212.03363). Our own data: stating intent in words changed nothing. Measure: BA against the intent
   cells (0.51) and the reference cell (0.83).
3. **Self-generated counter-example gate.** For each derived requirement, the writer builds a plausible violation and
   must show its grader rejects it before submitting. Adjacent evidence: NL2Postcond (arxiv.org/abs/2310.01831),
   SpecGen (arxiv.org/abs/2401.08807), both judged by discriminative power. Untested on grader writing.
4. **Ambiguity detection by divergence, resolved from the environment.** Generate 2 or 3 independent graders or
   interpretations, flag the clauses where they disagree, resolve each by inspecting the environment. Mechanism from
   ClarifyGPT (arxiv.org/abs/2310.10996): MBPP-sanitized Pass@1 70.96% -> 80.80%.
5. **Evaluator stress test as a gate.** Perturb correct work in ways that must not change the verdict and reject
   graders that flip. Evaluator Stress Test (arxiv.org/abs/2507.05619): 78.4% precision, 81.7% recall detecting proxy
   gaming. Likely overlaps with method 1's solver-confirmation step.

## Survey notes

- **Specification gaming:** DeepMind's framing (satisfying the literal objective without the intended outcome);
  Goodhart variants need different countermeasures (arxiv.org/abs/1803.04585). Coding agents: EvilGenie
  (arxiv.org/abs/2511.21654) sees hard-coding and test editing by Codex and Claude Code; SpecBench
  (arxiv.org/abs/2605.21384) sees the held-out gap grow with code size; reward hacking generalises to wider
  misalignment (arxiv.org/abs/2511.18397). "The Verification Horizon" (arxiv.org/abs/2606.26300): every verifier is
  a proxy for intent, never the intent.
- **Formalisation:** nl2spec, NL2Postcond, SpecGen all measure faithfulness behaviourally (does the spec separate
  known-good from known-bad), not by how the spec reads.
- **Rubrics:** Rubrics-as-Rewards (arxiv.org/abs/2507.17746), TICK checklists (arxiv.org/abs/2410.03608); both are
  validated against reference answers.
- **Inverse approaches:** Inverse Reward Design; Inverse Constitutional AI (arxiv.org/abs/2406.06560) extracts
  principles from labelled preferences, going from outcomes to principles rather than principles to checks.
- **Red-team loops:** Hack-Verifiable Terminal Bench (arxiv.org/abs/2608.22103) builds detectable exploits into
  Terminal-Bench-style tasks.
- **Clarifying questions:** models rarely surface ambiguity unprompted; ClarifyGPT's divergence check is the one
  mechanism with strong numbers. We have no live author, so questions must be answered from the environment.
- **Human author practice:** Terminal-Bench's guide ("a benchmark is designed to find out if it can"): validate
  outcomes not implementation, allow alternate correct solutions, guard against reward hacking
  (www.tbench.ai/news/writing-a-good-terminal-bench-task; arxiv.org/abs/2604.28093).

## Intent versus the goal

Most well-evidenced methods (spec generation, rubrics, the counter-example gate) sharpen a goal that is already known;
they do not discover one. Two families work without the goal being stated: contrastive examples (intent read off
labelled cases) and the hacker/fixer loop (the fixer only has to recognise that a hack occurred). Those two are the
informative next experiments for the harder condition, where no reference solution is given.
