# The Verify Loop and the Verifier Model

Everything built in the Verifier Bench research program: a loop that makes graders written by coding agents correct
(every correct solution passes, every wrong, hacked or near-miss solution fails), the self-evolving step that
teaches the loop from its own mistakes, every model we trained along the way with its weights, and the full research
log with every run.

**Start with the report:** [`REPORT.md`](REPORT.md).

**Every study, pre-registration and result:** [`loop/docs/research_intent_goal.md`](loop/docs/research_intent_goal.md).

## What is where

| Folder | What it is |
|---|---|
| [`loop/`](loop/) | The verify loop and everything around it: the measurement harness, the judge (versions v0 to v4), the do-no-harm guard, the self-evolving step, the 27B training and RL code, the specialist models' code, configs, tests and every run record. Python package `vb`. |
| [`loop/vb/evolve.py`](loop/vb/evolve.py) | The self-evolving step: post-mortem on labelled pools, rule proposer, versioned rulebook ([`loop/configs/rulebook.json`](loop/configs/rulebook.json)), paired gate, ledger ([`loop/runs/evolve/`](loop/runs/evolve/)). |
| [`small_models/`](small_models/) | The 116M attempt classifier trained from scratch (package `vm`) and the 7B grader writer (package `gw`), with their training scripts, configs, experiments and tests. |
| [`weights/`](weights/) | Every trained model's weights, stored with Git LFS: the 116M classifier, the 7B grader writer (5 rounds), the 7B requirement model, and the open 27B judge (6 training studies). |
| [`REPORT.md`](REPORT.md) | The report to HUD. |

## The system in one paragraph

A judge model (any frontier model, or our own) works inside a task's machine: it gathers evidence (attempts by other
agents, an empty attempt, near misses it writes), solves the task itself where it can, runs the agent's grader the
way the real grader runs, decides every disagreement from the task, repairs the grader, and keeps a repair only if a
mechanical guard confirms it agrees with more of the evidence and loses nothing. A hidden pool of labelled attempts,
never shown to the loop, measures the result. On tasks with known answers, the self-evolving step reruns the final
grader on every labelled attempt, names the check that misfired, has a model write a general rule for the judge, and
adopts the rule only if it raises scores on other tasks without lowering any.

## Results, in brief

| | Balanced accuracy |
|---|---|
| Agent-written graders on their own (70 rollouts, Terminal-Bench 4.0) | 0.50 to 0.52 |
| After the loop, six graders, frontier judge | 0.71 to **0.90** mean; 5 of 7 graders reach 1.00 |
| Open Qwen 3.8 27B as the judge, untrained, four held-out tasks | **0.82** against the frontier judge's 0.90, at $0 API |
| 116M classifier, held-out development set | 0.96 to 0.97 |

## Running it

See [`loop/README.md`](loop/README.md) and [`small_models/README.md`](small_models/README.md). The loop needs a Linux
machine with Docker (local or over ssh with `VB_HOST`), and a judge: `claude-*` (Anthropic API), `openrouter:<model>`
(any model on OpenRouter), `cc:<model>` (Claude Code headless on a subscription) or `local:<name>` (a vLLM server).

## Not included

- API keys and other credentials (none are in this branch).
- Base model weights (Qwen 3.8 27B, Qwen2.5-7B): download from their publishers; only our adapters are here.
- Raw task pools and leaderboard trial archives (tens of GB, rebuilt by `python -m vb.pool` and `python -m vb.harvest`).
- One training prompt file that quotes a sample private key from a public GitHub issue
  (`small_models/experiments/gw/eval_prompts.jsonl`, rebuilt by `small_models/scripts`).
