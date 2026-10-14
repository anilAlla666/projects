# Grader report: tw-1168

Rounds: 1 · cost $0.00 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 4 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| test_entrypoint_exists | Implicit: entrypoint.sh must exist and be executable | format |
| test_shebang_and_basic_structure | Implicit: entrypoint.sh should have proper shebang | format |
| test_uses_exec_or_proper_trap_forward | Fix the entrypoint script so that the Python application receives and handles SIGTERM correctly; The Python app receives SIGTERM and performs its cleanup; The container exits within a few seconds; Shell scripts don't automatically forward signals to child processes | task-text |
| test_no_broken_background_wait_pattern | Shell scripts don't automatically forward signals to child processes (root cause identified in task) | task-text |
| test_trap_does_not_do_cleanup_in_shell | The Python app receives SIGTERM and performs its cleanup (writes shutdown marker); Only modify /app/entrypoint.sh - do not change app.py | task-text |
| test_no_full_reimplementation | Only modify /app/entrypoint.sh - do not change app.py as it already has proper signal handlers | task-text |
