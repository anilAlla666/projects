# Grader report: tw-1168

Rounds: 1 · cost $0.61 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 5 |
| recomputed | 0 |
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
| entrypoint runs app.py, which writes data entries, and does not exit on its own | Fix the entrypoint script (/app/entrypoint.sh) so that the Python application receives and handles SIGTERM correctly | task-text |
| app.py's own SIGTERM handler runs to completion (random nonce written by instrumented copy of original app.py, signal SIGTERM) | The Python app receives SIGTERM and performs its cleanup (writes shutdown marker) | task-text |
| /app/data/shutdown_complete exists with content shutdown_complete:SIGTERM | The file /app/data/shutdown_complete is created to indicate clean shutdown | task-text |
| output.txt ends with '--- END OF DATA (clean shutdown) ---' | The Python app receives SIGTERM and performs its cleanup | task-text |
| init exits before the 10 s docker-stop timeout after SIGTERM delivered with PID-1 semantics | The container exits within a few seconds, not hitting the default 10-second timeout | task-text |
| self-contained test.sh writing 0/1 to /logs/verifier/reward.txt, no network | grader runs bash /tests/test.sh | format |
