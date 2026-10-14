# Grader report: tw-1168

Rounds: 1 · cost $0.96 · candidates judged correct: B, D · judged wrong: A, C, E

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
| app.py (instrumented original) is started by the entrypoint, still running ~3s later, and writing data entries | The application is a Python-based data processing service in /app that writes to files periodically; do not change app.py | task-text |
| after SIGTERM to PID 1, app.py's own SIGTERM handler runs (secret probe file contains token:SIGTERM) | The Python app receives SIGTERM and performs its cleanup (writes shutdown marker) | task-text |
| PID 1 exits less than 10 s after SIGTERM | The container exits within a few seconds, not hitting the default 10-second timeout | task-text |
| /app/data/shutdown_complete absent before SIGTERM, present after with content shutdown_complete:SIGTERM | The file /app/data/shutdown_complete is created to indicate clean shutdown | recomputed |
| output.txt contains '--- END OF DATA (clean shutdown) ---' | Data files are sometimes corrupted/incomplete because the app doesn't shutdown cleanly | recomputed |
| test.sh writes 1/0 to /logs/verifier/reward.txt without network | grader contract | format |
