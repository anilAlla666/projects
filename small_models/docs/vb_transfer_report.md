# Verifier Bench transfer (8 Oct 2026)

**Question:** does the judge help on the Verifier Bench problem, deciding whether a Terminal-Bench attempt succeeded
when no test results are available?

**Data:** the Verifier Bench M1 pools, 106 real leaderboard attempts on 5 Terminal-Bench 4.0 tasks
(hof-topology-interpenetration, pretrain-shard-corruption, production-planning, react-lead-form, uefi-bootkit),
labelled by each task's official tests. Every label matched the Verifier Bench record (106 of 106).

**Input (`scripts/vb_adapter.py`):** the task instruction as the task text, the attempt's output files as the change
(text capped at 4,000 characters per file, binaries replaced by their size), and an evidence block that says no tests
are available. A control version replaces that line with the attempt's official test counts and one line per failing
test, the training format. The control uses oracle evidence and only checks the adapter.

**Checkpoint:** round 4, epoch 0 (strict setting), threshold 0.7.

| Version | Macro AUC | Macro balanced accuracy | What the judge did |
|---|---:|---:|---|
| No tests (the measurement) | 0.53 | 0.50 | Rejected every attempt (probabilities 0.000 to 0.002) |
| Control: official test results shown | 0.92 (1.00 on the 4 tasks with recorded counts) | 0.71 | Ranked passing above failing perfectly where counts exist; react-lead-form records no counts |

Per-task numbers: `experiments/vb/score_none.json`, `experiments/vb/score_control.json`. About 21 attempts per task
gives intervals of roughly plus or minus 0.2 on any per-task figure.

**Conclusion:** the judge interprets test evidence well, even on task types it never saw (the control), but it
cannot create evidence. Without tests it has no signal. Verifier Bench's difficulty is producing discriminating
evidence (writing the checks), so this model does not address it. For reference, the Verifier Bench graders scored
0.50 to 0.52 alone, 0.80 as a five-model panel and 0.83 to 0.88 with the author's solution; those graders could run
the environment, and this judge cannot.

**What would address it:** a model trained to write checks, scored by whether its checks separate known-good from
known-bad attempts. This judge could then read those checks' results.
