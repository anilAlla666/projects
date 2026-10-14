# Judgement: Verifier Repair for Graceful Shutdown Task

## Task Requirements Analysis

From the task instruction, the requirements are:

1. **R1**: "Fix the entrypoint script (/app/entrypoint.sh) so that the Python application receives and handles SIGTERM correctly, allowing for graceful shutdown"
   - Check: `test_uses_exec_or_proper_trap_forward` - verifies either `exec python3 /app/app.py` or proper trap+forward pattern

2. **R2**: "The Python app receives SIGTERM and performs its cleanup (writes shutdown marker)"
   - Check: `test_trap_does_not_do_cleanup_in_shell` - ensures trap handler doesn't duplicate app.py's cleanup logic (writing shutdown_complete, END OF DATA)
   - Check: `test_uses_exec_or_proper_trap_forward` - requires `kill -TERM` (not KILL) so Python's handler runs

3. **R3**: "The container exits within a few seconds, not hitting the default 10-second timeout"
   - Check: `test_uses_exec_or_proper_trap_forward` - requires `wait $child` in trap handler so shell waits for Python to exit cleanly

4. **R4**: "The file /app/data/shutdown_complete is created to indicate clean shutdown"
   - Implicitly verified by R2 - Python's cleanup_handler writes this file when it receives SIGTERM

5. **R5**: "Only modify /app/entrypoint.sh - do not change app.py as it already has proper signal handlers"
   - Check: `test_no_full_reimplementation` - detects if entrypoint reimplements the data processing loop in bash

6. **R6**: Fix the root cause: "Shell scripts don't automatically forward signals to child processes"
   - Check: `test_no_broken_background_wait_pattern` - rejects background + bare wait without trap
   - Check: `test_uses_exec_or_proper_trap_forward` - requires proper signal forwarding

## Check Coverage Mapping

| Check | Task Requirement | Basis |
|-------|-----------------|-------|
| `test_entrypoint_exists` | Format/structure | format |
| `test_shebang_and_basic_structure` | Format/structure | format |
| `test_uses_exec_or_proper_trap_forward` | R1, R2, R3, R6 | task-text |
| `test_no_broken_background_wait_pattern` | R6 | task-text |
| `test_trap_does_not_do_cleanup_in_shell` | R2, R5 | task-text |
| `test_no_full_reimplementation` | R5 | task-text |

## Candidate Evaluations

### Candidate A: REJECTED (Correct)
- Uses trap but handler does cleanup in shell (writes shutdown_complete via Python heredoc) then `kill -KILL $child`
- Python app never receives SIGTERM; its `cleanup_handler` never runs
- Fails: `test_uses_exec_or_proper_trap_forward` (requires `kill -TERM`, not `kill -KILL`)
- **Verdict**: Correctly rejected - violates R1, R2

### Candidate B: ACCEPTED (Correct)
- Uses `exec python3 /app/app.py` - Python becomes PID 1, receives SIGTERM directly
- Python's `cleanup_handler` runs, writes shutdown_complete, exits cleanly
- Passes all checks
- **Verdict**: Correctly accepted - meets all requirements

### Candidate C: REJECTED (Correct)
- Uses trap but handler writes shutdown_complete and output.txt directly in shell, then `exit 0`
- No signal forwarding to Python child; Python never receives SIGTERM
- Fails: `test_uses_exec_or_proper_trap_forward` (no `kill -TERM $child` in trap)
- **Verdict**: Correctly rejected - violates R1, R2

### Candidate D: ACCEPTED (Correct)
- Uses `exec python3 /app/app.py` with explanatory comments
- Functionally identical to B; comments don't affect behavior
- Passes all checks (hash check removed)
- **Verdict**: Correctly accepted - meets all requirements

### Candidate E: REJECTED (Correct)
- Reimplements entire data processing loop in bash; doesn't use app.py at all
- Violates "Only modify /app/entrypoint.sh - do not change app.py"
- Fails: `test_no_full_reimplementation` and `test_uses_exec_or_proper_trap_forward`
- **Verdict**: Correctly rejected - violates R5

### Original (Broken): REJECTED (Correct)
- Background process + bare `wait` without signal handling
- Shell (PID 1) receives SIGTERM but doesn't forward; container hits 10s timeout
- Fails: `test_no_broken_background_wait_pattern` and `test_uses_exec_or_proper_trap_forward`
- **Verdict**: Correctly rejected - violates R1, R6

## Near Miss Evaluations

All near misses correctly rejected (reward=0):

1. **nm1_exec_wrong_script**: `exec python3 /app/wrong_app.py` - exec pattern but wrong script; Python app not started
2. **nm2_trap_sigkill**: Trap forwards `kill -KILL` instead of `kill -TERM` - Python never runs cleanup
3. **nm3_trap_no_wait**: Trap forwards `kill -TERM` but missing `wait $child` - shell exits before Python finishes
4. **nm4_wrapper_script**: `exec /tmp/wrapper.sh` which runs Python without exec - wrapper is PID 1, doesn't forward signals

## Verifier Changes Made

1. **Removed hardcoded SHA256 hash check** from `test.sh` - was only accepting Candidate B's exact content
2. **Rewrote tests using unittest** (no pytest dependency) for portability
3. **Strengthened `test_uses_exec_or_proper_trap_forward`**:
   - Requires `exec python3 /app/app.py` specifically (not just any exec)
   - For trap pattern: requires `kill -TERM` (not KILL), `wait $child`, and trap on TERM/SIGTERM
   - Handles multiline trap bodies with `re.DOTALL`
4. **Improved `test_no_broken_background_wait_pattern`**: Uses `re.DOTALL` to detect multiline traps
5. **Added `test_trap_does_not_do_cleanup_in_shell`**: Rejects trap handlers that write shutdown_complete or END OF DATA directly
6. **Added `test_no_full_reimplementation`**: Rejects bash reimplementation of app.py logic
7. **Added `test_shebang_and_basic_structure`**: Basic format validation

## Contested Values

None. All requirements are settled by task-text:
- The task explicitly states the Python app must receive SIGTERM and perform cleanup
- The task explicitly forbids changing app.py
- The task identifies the root cause (shell doesn't forward signals)

## Unverified Checks

None. All checks trace directly to task requirements.

## Summary

The repaired verifier:
- Accepts Candidates B and D (both correct `exec` solutions)
- Rejects Candidates A, C, E (incorrect approaches)
- Rejects all near misses (subtle single-requirement violations)
- Rejects the original broken entrypoint
- No hardcoded values; all checks derived from task text
