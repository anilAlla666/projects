# Judgement

## Original verifier
`test.sh` installed uv/pytest over the network and ran `test_outputs.py`, which only grep-ed
`/app/entrypoint.sh` for `exec` or `trap` and for absence of a bare `wait`. It accepted all five
candidates, including A, C and E, which never let the Python app receive SIGTERM: they `trap` SIGTERM
in the shell and fake the app's cleanup (write `shutdown_complete:SIGTERM` and the END line
themselves). E does not even run app.py. The static checks were replaced by a behavioural test.

## New verifier (tests/check_shutdown.py, run by test.sh, no network needed)
The grading container has no Docker daemon and (on this machine) namespaces are not permitted, so
`docker run` + `docker stop` is emulated:
* /app/entrypoint.sh is executed directly (as Docker's ENTRYPOINT does) with an instrumented copy of the
  original app.py (from the Dockerfile) at /app/app.py; the instrumentation only adds a line in
  `cleanup_handler` writing a random nonce + signal name to a random file in /app/data.
* After the app has written >=3 entries, "docker stop": SIGTERM is sent to the init process **only if it
  has a SIGTERM handler** (SigCgt bit), which is the kernel's rule for a PID-namespace init signalled from
  the parent namespace (exactly why the original entrypoint ignores `docker stop`). Then up to 10 s wait;
  then SIGKILL everything. When the init exits, all remaining processes are SIGKILLed (namespace teardown;
  harness is a child subreaper).
* Original app.py is restored afterwards.

## Candidates
| cand | verdict | reason |
|---|---|---|
| A | wrong | Shell traps TERM, writes marker/END line itself via an inline python script, SIGKILLs app.py. App's handler never runs ("python app receives SIGTERM and performs its cleanup" violated). Rejected by nonce check. |
| B | correct | `exec python3 /app/app.py`; app is init, handler runs, exits in ~0.05 s. |
| C | wrong | Same fake as A in shell; app.py never receives SIGTERM. Rejected by nonce check. |
| D | correct | Same as B. |
| E | wrong | Does not run app.py at all; shell re-implements app. Rejected by nonce check. |
Also accepted: my own trap-and-forward solution (`kill -TERM $child; wait`). Empty attempt (original
entrypoint) rejected: SIGTERM ignored, no exit before timeout, no marker.

## Trace: requirement -> check
* "The Python app receives SIGTERM and performs its cleanup (writes shutdown marker)" -> nonce file written by
  the app's own handler with `SIGTERM`; marker content `shutdown_complete:SIGTERM` (the value app.py writes,
  task-text/app code); output.txt last line is the clean-shutdown line written by the handler.
* "The container exits within a few seconds, not hitting the default 10-second timeout" -> init must exit
  before the 10 s docker stop timeout. Bar is the task-text 10 s; "a few seconds" not turned into a tighter
  number (would be an interpretation). Correct solutions exit in ~0.05 s.
* "/app/data/shutdown_complete is created" -> file existence check.
* "Fix the entrypoint script ... so that the Python application receives and handles SIGTERM" -> app must be
  started by the entrypoint and write data entries before stop; entrypoint must not exit on its own.
* "Only modify /app/entrypoint.sh" -> only that file is carried over; verifier uses original app.py content.

## Trace: check -> requirement
All checks above map to the listed sentences. Removed: grep for `exec|trap` and for bare `wait`
(task does not mandate a technique; also unsound — accepted fakes). Removed network/uv dependency (format).

## Contested
None. No numeric expected values beyond the task's own 10 s timeout and app.py's marker string.
