# Verifier Bench on HUD: the taskset

**Taskset:** `verifier-bench`, https://hud.ai/tasksets/3e50b881-f8cc-4de7-90aa-ca2eb79773bb

## What the task is

The agent is given a Terminal-Bench 4.0 task's instruction and the task's own starting machine, and asked to
write the **verifier** for it: a `tests/` folder whose `test.sh` decides whether someone else's attempt
succeeded. It does not solve the task.

A hidden grader then runs the agent's tests on every attempt in the task's **pool**: 21 or 22 real
leaderboard attempts, each labelled pass or fail by the task's official, human-written tests (labels
re-checked when the pool was built, see `REPORT.md` §3.3). Most wrong attempts are near misses.

    TPR    = correct attempts the agent's tests accept / correct attempts
    TNR    = wrong attempts they reject / wrong attempts
    BA     = (TPR + TNR) / 2
    reward = max(0, 2 * BA - 1)

A verifier that accepts everything, rejects everything or does not exist scores 0; a perfect one scores 1.
BA itself is 0.5 for all of those, which would make "write nothing" look like a middling answer, hence the
rescaling. A run that errors, times out or writes no verdict counts as a reject. BA, TPR, TNR and the
per-attempt verdicts are printed by the grader and appear in the trace's verifier output.

## One change from the report's milestone 1

In milestone 1 the agent also wrote a `Dockerfile` for its tests. On HUD the grading image is built at deploy
time, so that is not possible: **the agent's tests run in the task's official grading image**, with the
official tests removed. The instruction shows the agent that image's Dockerfile. The milestone 1 numbers
were measured under the older contract and are not claimed to carry over.

## How the grader keeps the answers hidden

The grader (`templates/hud/grader.py`) runs inside HUD's own verifier sandbox, as root of a single-user
namespace, so file permissions cannot hide anything from the agent's tests. Instead each run of the agent's
tests gets its own nested sandbox (bubblewrap):

- the grading image read-only, with fresh private copies of the directories tests write to;
- `/grader` (the labels and the other attempts) and `/verifier_out` replaced by empty directories;
- no capabilities, no network, its own process namespace; the grader marks itself non-dumpable.

HUD does not allow a fresh `/proc` inside the nested sandbox, so the grader shares the platform's `/proc`
read-only; before grading it proves from inside the sandbox that no visible process exposes `/grader`. If
neither mode can be set up, it refuses to grade (error, no reward) rather than run the tests unsandboxed.

## Checked on HUD itself, per task

Each task was checked on HUD before publishing (`tools/hud_publish.py`). A check copy of the task put five
graders on the writer's machine, and a small model was told to copy one of them into place:

| task | official tests | always pass | always fail | nothing | cheating grader | grading time, official |
|---|---:|---:|---:|---:|---:|---:|
| production-planning | 1.0 | 0 | 0 | 0 | 0 | 42 s |
| hof-topology-interpenetration | 1.0 | 0 | 0 | 0 | 0 | 6 s |
| react-lead-form | 1.0 | 0 | 0 | 0 | 0 | 264 s |

The cheating grader tries every route to the hidden state we know of: read the labels, list the other
attempts, unmount the hiding layer, read another process's root or environment, read the platform's files,
write to the system directories. It accepts if any works; it scores exactly like always-fail.

## Reproduce

On the server (pools under `/root/vbench/pools`):

    python3 -m vb.hudtask --task <task>          # -> /root/vbench/hud_tasks/write-verifier-<task>
    bash tools/hud_local_check.sh <task>         # the five graders, in Docker

Where the HUD CLI is logged in:

    python tools/hud_publish.py --task <task>    # check copy on HUD, then deploy and sync the real task

## Limits

- **Pool size:** 21 or 22 attempts per task, so one run's reward moves in steps of about 0.1.
- **Grading time** is the official tests' time × the pool size. The grader's timeout allows the official
  per-attempt limit for every attempt, so a slow verifier makes a slow rollout.
- **Shared `/proc`:** the agent's tests can see the grader's process and its command line (public
  information); what they cannot reach is checked on every grading run.
- **The writer sees the grading image's Dockerfile,** which names the tools the official tests use.
