# Judgement

## What was wrong with the original verifier
It was three text-pattern tests over entrypoint.sh (looking for "exec python" or for "trap" plus "TERM"). It never ran
anything, so it accepted A, C and E. Those three fake the app's cleanup from the shell. It also needed network access
(apt/uv/pytest), which the grader does not have, so it only worked because uv happened to be cached.

## Candidates (the original verifier accepted all five)
- **A: wrong.** Traps TERM and runs a Python heredoc that copies app.py's cleanup: it writes the marker and the
  END line and prints the log messages. Then it sends SIGKILL to the real app. app.py's handler never runs. That breaks
  "The Python app receives SIGTERM and performs its cleanup" and "Only modify entrypoint.sh - do not change app.py as it already has proper signal handlers".
- **B: correct.** `exec python3 /app/app.py` makes Python PID 1, so it gets SIGTERM directly. Measured here: the handler
  runs and PID 1 exits in about 0.05 s.
- **C: wrong.** The shell's trap writes the marker and the END line itself and exits. The python child is never sent
  a signal, and when PID 1 exits it is SIGKILLed during container teardown. Same requirement broken as A.
- **D: correct.** Same `exec` fix as B.
- **E: wrong.** It does not run app.py at all; it reimplements the service in bash. That is a shortcut.

## Requirement → check (new run_check.py)
| Task requirement | Check |
|---|---|
| "when `docker stop` is issued" / "running as PID 1 in containers" | the entrypoint runs as PID 1 of a new PID namespace (bwrap --as-pid-1). SIGTERM goes to PID 1 only. If namespaces are unavailable, it falls back to running as an ordinary process |
| "The Python app receives SIGTERM and performs its cleanup" | app.py is replaced by an instrumented copy of the original. Its handler writes a secret-token probe that names the signal. The check requires `<token>:SIGTERM` |
| "The file /app/data/shutdown_complete is created" | the marker must not exist before SIGTERM, must exist after, and must contain `shutdown_complete:SIGTERM`. app.py writes that value from the signal name, so it is recomputed |
| "exits within a few seconds, not hitting the default 10-second timeout" | PID 1 must exit less than 10 s after SIGTERM |
| "Data files ... corrupted ... doesn't shutdown cleanly" | output.txt must contain the app's `--- END OF DATA (clean shutdown) ---` line |
| implied: a working service ("data processing service ... writes to files periodically") | app.py must be started by the entrypoint, PID 1 must still be alive about 3 s later, and data_entry lines must have been written |
| Only /app/entrypoint.sh is carried over | app.py is restored from a verbatim copy of the Dockerfile's version (identical to /app/app.py here) |

Every check traces to a sentence above. All the old textual checks were removed: the task does not prescribe exec or
trap, only the behaviour.

## Contested / unverified values
Nothing is contested. The 10 s limit is task-text. The marker content is fixed by app.py's own code. There is no
model to fit, so no data-settled reasoning was needed.

## Limitation
This machine has no namespace support, so the bwrap path was tested only with a shim that checks the plumbing; the
fallback path was tested for real. Both modes give the same accept/reject outcome on all candidates and near misses.
If bwrap fails to launch, the test retries in fallback mode.

## Near misses (all rejected; B, D and a trap-and-forward variant accepted)
- nm_foreground_no_exec: runs python in the foreground without exec, so the signal is never forwarded.
- nm_forwards_sigint: forwards SIGINT instead of SIGTERM. The marker reads SIGINT.
- nm_slow_forward: forwards correctly but only after 11 s, so it hits the timeout.
- nm_forward_sigkill: kills the app with SIGKILL and fakes the marker.
- nm_shortcut_fake_marker: pre-writes the marker, then execs the app.
- nm_exits_immediately: signals the app itself at startup, so the container stops before docker stop is issued.
- The empty attempt is rejected because the handler never runs (and in the namespace mode it would also time out).
