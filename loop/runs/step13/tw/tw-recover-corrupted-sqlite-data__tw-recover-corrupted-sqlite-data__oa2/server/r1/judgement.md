# Judgement

## Ground truth (recomputed)
Opening /data/database.db together with its WAL (/data/database.db-wal, one frame for page 2, the `records`
table leaf) using python3's sqlite3, `SELECT * FROM records` gives 15 rows: id 1..15, name 'record<id>',
value id*10. The task says to "recover all the missing data" (the rows held in the WAL) and gives the
format `[{"id": 1, "name": "record1", "value": X}, ...] sorted by id`. Getting the expected values only means
reading the task's own database with SQLite, so they are **recomputed**, not contested. The original
verifier's `value == id*10` / sum == 1200 checks agree with this.

## Candidates
The original verifier rejected none of them. A, B and C each contain exactly the 15 recomputed rows, as
integers, sorted by id, with only the keys id/name/value. Their database.db files are intact (integrity_check
ok, 15 rows). **A, B, C: correct.** (Only A to C exist in /candidates.)

## Requirement -> check
- "export everything to /data/recovered.json": file exists and parses as JSON (check.py: missing/invalid JSON).
- "The output format should be: [{"id": 1, "name": "record1", "value": X}, ...]": top level is a list; every
  element is an object with exactly the keys id, name, value; id and value are integers (not bool) and name is a string.
- "sorted by id": the list of ids is ascending.
- "recover all the missing data ... maybe 15 records total": exactly 15 records, and they equal the recomputed
  rows (id 1..15, name record<id>, value id*10). This rejects partial (8 committed rows only), fabricated or
  extra rows, and altered values.

## Check -> requirement (old verifier)
- The old checks for exists/valid/list/15 records/keys/types/sorted/ids 1-15/name/value/sum/first/last/8/9/
  no-duplicates are all covered by the same requirements and merged into check.py's single exact comparison.
- test_json_file_readable_multiple_times, test_recovered_json_in_correct_directory (a tautology about a
  constant path), test_json_file_permissions_readable: no task sentence requires these beyond "the file exists
  and is readable"; they are removed (reading the file already covers them).
- test.sh: it ran `apt-get`/curl to install uv, then `uv init`/`uv add pytest`. The grading container has no
  network, and this only worked because uv happened to have pytest available. That is fragile and no
  requirement asks for it. It is replaced by a self-contained `python3 check.py` that needs no network, pip or pytest.
- New: bool is rejected as an integer (Python's `isinstance(True, int)` loophole); the task's format has
  integer ids and values.

## Near misses (all rejected by the repaired verifier)
nm_unsorted (ids 10/11 swapped), nm_one_wrong_value (id 12 value 121), nm_wal_values_as_strings
(the WAL rows' values written as strings), nm_only_8_committed_rows (shortcut: only the visible rows),
nm_missing_record14, nm_fabricated_extra_record (a 16th row), nm_name_case ('Record4'),
nm_wrapped_in_object ({"records": [...]}).
The empty attempt is rejected (missing file).

## Contested / unverified
None. The expected rows come straight from the task's database.
A note on the environment: on this machine the main database file already holds page 2 with 15 rows (the
same bytes as the WAL frame). So the rows are recoverable either way, and the expected values do not
depend on how they are recovered.
