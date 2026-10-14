# Judgement

## Evidence
evidence.md shows no run results: the probe crashed (missing grader config.json), so the original verifier never ran
on anything. I reviewed the original tests by reading them and replaying them here.

Problems with the original verifier:
* test.sh runs `apt-get install curl` and downloads uv over the network. The grading container has no network, so it
  could never pass. That means every attempt, correct ones included, got reward 0.
* The tests only matched text patterns ("exec python" or "trap"+"TERM"). They accept fakes. Candidate A traps
  TERM, SIGKILLs the app and writes the marker itself. C writes the marker from a shell trap. E never runs app.py.
  All three pass the text checks. The tests also reject correct fixes that don't contain those exact strings,
  e.g. `exec /usr/bin/env python3 /app/app.py`.

## Candidate decisions (new verifier)
| cand | result | reason |
|---|---|---|
| A | wrong (rejected) | Its TERM trap SIGKILLs app.py and fakes the cleanup in a separate python. app.py never receives SIGTERM. That breaks "The Python app receives SIGTERM and performs its cleanup". Rejected by the nonce-proof check. |
| B | correct | `exec python3 /app/app.py`: app.py becomes PID 1 and handles SIGTERM itself. Exits in about 0.01 s. |
| C | wrong (rejected) | The shell trap writes the marker and END line itself. app.py is never signalled; Docker kills it when PID 1 exits. Rejected by the proof check. |
| D | correct | Same as B. |
| E | wrong (rejected) | Rewrites the app in bash and never runs /app/app.py. Breaks "do not change app.py" / "Python application receives and handles SIGTERM". Rejected: app.py main loop never starts. |
| empty (original entrypoint) | rejected | No forwarding, so no marker. |

These calls do not rest on any contested value. They follow from the task sentence "The Python app receives SIGTERM
and performs its cleanup (writes shutdown marker)".

## New verifier design (check_shutdown.py)
Docker is not available (no daemon), so the verifier emulates `docker run` + `docker stop`:
1. It writes /app/app.py from the task Dockerfile's exact source, plus an invisible addition. When main() starts,
   the app records its pid and a per-run random nonce. When the real handler runs in that same process on a real
   signal (frame is not None), it writes `nonce:SIGNAME:pid` to a hidden /tmp directory. The app's behaviour is
   otherwise unchanged.
2. It starts /app/entrypoint.sh as PID 1 of a new PID namespace (unshare -pf, or bwrap --as-pid-1) when the machine
   allows it. That reproduces the container's PID-1 signal semantics. Otherwise it starts it as a plain process.
   In plain mode the shell's default SIGTERM action kills the shell without forwarding, so non-forwarding
   entrypoints still fail.
3. It waits until app.py's main loop has started, then waits 3 s more ("waiting a few seconds").
4. It sends SIGTERM to the entrypoint process, which is what `docker stop` does.
5. Checks:
   - the entrypoint exits within 8 s ("exits within a few seconds, not hitting the default 10-second timeout");
   - the proof file equals `nonce:SIGTERM:<pid of the main-loop process>` (app.py itself received SIGTERM and ran
     its cleanup);
   - /app/data/shutdown_complete exists with content `shutdown_complete:SIGTERM\n`. That is what the unchanged
     app.py writes on SIGTERM, so the value is task-text / recomputed from the given app.py;
   - output.txt ends with `--- END OF DATA (clean shutdown) ---` and has data entries ("Data files ... app doesn't
     shutdown cleanly").
6. Afterwards it kills any leftover processes, as Docker does when PID 1 exits.

I tested here in plain mode only. unshare is not permitted on this machine and bwrap is not installed, so the
namespace path is untested. It is a probe-then-use path: if the namespace probe fails it falls back to plain mode,
and if NSpid is not readable it signals the launcher's first child.

## Trace: requirement -> check
* "Python app receives SIGTERM and performs its cleanup (writes shutdown marker)": nonce proof + marker content check.
* "container exits within a few seconds, not hitting the default 10-second timeout": exit ≤ 8 s after SIGTERM.
  8 s is below the stated 10 s; correct fixes exit in under 0.1 s.
* "/app/data/shutdown_complete is created": marker check.
* "Data files ... corrupted/incomplete": output.txt ends with the clean-shutdown line.
* "Only modify /app/entrypoint.sh - do not change app.py": the verifier supplies app.py itself, and the entrypoint
  must actually run /app/app.py (main-loop start check).
* "Fix the entrypoint script": only /app/entrypoint.sh is carried over and tested.

## Trace: check -> requirement
All checks above map to a task sentence. Removed: the text-pattern checks ("exec python", "trap", no bare `wait`).
The task does not require any particular implementation.

## Near misses (all rejected; B, D and my own trap-forwarding variant accepted)
* nm_foreground_no_exec: runs python in the foreground without exec, so bash keeps PID 1 and does not forward.
* nm_forwards_sigint: forwards SIGINT instead of SIGTERM (marker says SIGINT).
* nm_forward_too_slow: forwards correctly, but only after 11 s.
* nm_kill_child_sigkill: traps TERM, then SIGKILLs the child.
* shortcut_fake_handler_call: SIGKILLs the app, then imports app.py and calls cleanup_handler by hand.
* shortcut_marker_written_upfront: the shell writes the marker and END line itself.

## Contested values
None. The marker content and END line come straight from the given app.py. The time bar of 8 s sits below the
10 s the task names.
