# Judgement

## Ground truth (recomputed)
Read the task's own database on this machine with sqlite3 (`SELECT * FROM records`), both through the WAL
(db + -wal + -shm) and from the main file alone (immutable, no WAL). Both give exactly 15 rows:
id 1..15, name "record<id>", value id*10. Parsing the WAL by hand: one valid frame (salts match) for page 2,
commit size 2, and its contents are byte-identical to page 2 of the main file. So the expected output is
fixed by reading the data the task points at - no interpretation needed (basis: recomputed).

## Candidates
Only A, B, C exist in /candidates (no D/E). All three are byte-equivalent JSON of the 15 rows above,
sorted by id, exactly keys id/name/value with int/str/int types. Verifier accepted all three; I judge all
correct. No candidate was rejected, so no rejection had to be adjudicated. The empty attempt is rejected.

## Trace: task requirement -> check
- "export everything to /data/recovered.json" -> test_recovered_json_exists, _in_correct_directory,
  _permissions_readable, _is_valid_json, _not_empty, _readable_multiple_times (format)
- "format should be: [{...}, ...]" (a list of objects) -> test_recovered_json_is_list, record-is-dict guard (format)
- keys "id","name","value" -> test_each_record_has_id/name/value, test_all_records_have_exactly_three_fields (format)
- example `"id": 1` / `"name": "record1"` -> test_id_is_integer (bool excluded now), test_name_is_string,
  test_value_is_integer (bool excluded now), test_name_field_not_empty (format)
- "sorted by id" -> test_records_sorted_by_id (task-text)
- "recover all the missing data" / "maybe 15 records" -> test_has_15_records, test_ids_are_1_to_15,
  test_no_duplicate_ids, test_first_8_records_committed, test_last_7_records_recovered (recomputed from DB)
- correct contents -> test_name_format_matches_id, test_value_is_id_times_10, test_first/last_record_correct,
  test_boundary_record_8/9_correct, test_total_value_sum (recomputed from DB)
Every check follows from one of these sentences; none was found that the task does not require.
No requirement about the state of /data/database.db is stated, so none is checked.

## Changes
1. test.sh no longer depends on apt/curl/uv download (network is absent in grading; it only worked
   because a cached venv happened to exist). It now runs a dependency-free python3 runner
   (run_checks.py) that executes every test_* function in test_outputs.py and writes 1/0.
2. id/value integer checks now exclude JSON booleans; per-record dict guard added. Format only.
No expected value was changed.

## Contested
None. No data-fitting readings were needed: the values are read directly from the database.

## Near misses (all rejected by repaired verifier; A/B/C accepted; empty rejected - run here with
/data and /tests/logs set up as the grader does)
nm1 missing record 15; nm2 record 9 value 9; nm3 records 10/11 swapped; nm4 ids as strings;
nm5 shortcut: only the 8 visible records; nm6 list wrapped in {"records": ...}; nm7 values as strings;
nm8 name "record_12"; nm9 duplicated last record (16 entries).
