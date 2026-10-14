# Grader report: tw-recover-corrupted-sqlite-data

Rounds: 1 · cost $0.49 · candidates judged correct: A, B, C · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 4 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| /data/recovered.json exists and is valid JSON | export everything to /data/recovered.json | format |
| top level is a list of objects with exactly keys id,name,value; id/value int (not bool), name str | The output format should be: [{"id": 1, "name": "record1", "value": X}, ...] | format |
| ids ascending | sorted by id | format |
| exactly 15 records equal to id 1..15, name record<id>, value id*10 | recover all the missing data ... maybe 15 records total | recomputed |
| test.sh runs python3 check.py with no network/uv/pytest dependency | write 1 or 0 to /logs/verifier/reward.txt | format |
