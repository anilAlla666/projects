# Grader report: tw-1168

Rounds: 1 · cost $0.72 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 2 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| app.py main loop (the provided, unchanged app.py) is started by the entrypoint | Only modify /app/entrypoint.sh - do not change app.py; Fix the entrypoint script so that the Python application receives and handles SIGTERM | task-text |
| the process running app.py's main loop itself handles a real SIGTERM (per-run nonce proof) | The Python app receives SIGTERM and performs its cleanup (writes shutdown marker) | task-text |
| entrypoint (container PID 1) exits within 8 s of SIGTERM | The container exits within a few seconds, not hitting the default 10-second timeout | task-text |
| /app/data/shutdown_complete exists with content 'shutdown_complete:SIGTERM\n' | The file /app/data/shutdown_complete is created to indicate clean shutdown | recomputed |
| output.txt has data entries and ends with '--- END OF DATA (clean shutdown) ---' | Data files are sometimes corrupted/incomplete because the app doesn't shutdown cleanly | recomputed |
| test.sh writes 1/0 to /logs/verifier/reward.txt using only python3 (no network) | grader contract | format |
