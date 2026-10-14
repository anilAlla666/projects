# Grader report: tw-recover-corrupted-sqlite-data

Rounds: 1 · cost $0.46 · candidates judged correct: A, B, C · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 1 |
| recomputed | 2 |
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
| recovered.json exists in /data, readable, valid non-empty JSON | export everything to /data/recovered.json | format |
| top level is a list of objects with exactly keys id,name,value | The output format should be: [{"id": 1, "name": "record1", "value": X}, ...] | format |
| id/value are non-bool integers, name non-empty string | The output format should be: [{"id": 1, "name": "record1", "value": X}, ...] | format |
| records sorted ascending by id | sorted by id | task-text |
| exactly 15 records, ids 1..15, no duplicates | recover all the missing data ... maybe 15 records total | recomputed |
| name == 'record<id>', value == id*10, sum 1200, spot checks 1/8/9/15 | recover all the missing data and export everything | recomputed |
