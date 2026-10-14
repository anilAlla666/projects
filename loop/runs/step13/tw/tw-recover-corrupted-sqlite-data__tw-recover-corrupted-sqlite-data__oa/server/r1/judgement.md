# Judgement

## Evidence
evidence.md has no per-attempt results: the probe crashed (missing grader config.json), so the verifier was never run on anything.
Reading the original verifier: its `test.sh` ran `apt-get install curl`, then downloaded `uv` and installed pytest from the network.
The grading container has **no network**, so this setup fails. Once it fails, pytest can't run, `$?` is non-zero, and the reward is 0
for every attempt, correct ones included. That is a verifier bug: it rejects every correct candidate (A, B, C) and every other attempt too.
Only candidates A, B and C exist (/candidates/D and /candidates/E are absent).

## Recomputed expected value (basis: recomputed)
I opened copies of /data/database.db with sqlite3, both with the -wal/-shm files and with the main file alone, and ran `SELECT * FROM records`.
Both give 15 rows: id 1..15, name `record<id>`, value id*10. I also parsed the WAL by hand: it holds one committed frame (page 2) with
the same 15 cells. No other data exists, so the expected content is fixed by reading the task's own database. No choice is involved.

## Candidates
- A, B, C: the 15 records exactly as recomputed, sorted, keys id/name/value, all integer types. **Correct.** The original verifier would have
  rejected them only because of the network-dependent test.sh, not because of any content check. The check is at fault.

## Trace: requirement -> check
- "export everything to /data/recovered.json": file exists and is valid JSON (check.py read/parse).
- "recover all the missing data ... 15 records total": exactly 15 records, equal to the recomputed rows (count + full equality).
- "format should be: [{"id": 1, "name": "record1", "value": X}, ...]": top level is a list, each element an object with exactly the keys
  id/name/value, id and value are integers (bool excluded), name is a string. The source column types are INTEGER/TEXT/INTEGER.
- "sorted by id": ids ascending (and full equality with the ordered list).

## Trace: check -> requirement
The old pytest checks (counts, names, values = id*10, sum = 1200, first/last/boundary records, no duplicates, sorted, exactly three keys) are
all implied by "equals the recomputed 15 rows in id order" plus the format. I merged them into check.py. I dropped the checks that test
nothing the task asks for, or nothing at all: "readable multiple times", "permissions readable", "in correct directory" (always true).
test.sh now uses python3 only and always writes 0 or 1.

## Near misses (all rejected): nm_wrong_value (record 12 has value 121), nm_only_14_records, nm_only_8_records (a partial result), nm_unsorted (9 and 10 swapped),
nm_name_case (Record3), nm_value_strings (values written as strings), nm_wrapped_object ({"records": [...]}), nm_fabricated_extra (made-up record 16),
nm_extra_key (an extra field).
The empty attempt is rejected. A, B and C are accepted.

## Contested
None.
