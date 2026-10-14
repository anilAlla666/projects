# Grader report: tw-recover-corrupted-sqlite-data

Rounds: 1 · cost $0.34 · candidates judged correct: A, B, C · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 1 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| /data/recovered.json exists and parses as JSON | export everything to /data/recovered.json | format |
| top level is a list; each record is an object with exactly keys id,name,value; id/value int (not bool), name str | The output format should be: [{"id": 1, "name": "record1", "value": X}, ...] | format |
| ids in ascending order | sorted by id | task-text |
| exactly 15 records equal to id 1..15, name record<id>, value id*10, in order | recover all the missing data ... maybe 15 records total | recomputed |
| test.sh runs with python3 only (no network) and always writes 0/1 | grader: must write exactly 1 or 0 to /logs/verifier/reward.txt; no network | format |
