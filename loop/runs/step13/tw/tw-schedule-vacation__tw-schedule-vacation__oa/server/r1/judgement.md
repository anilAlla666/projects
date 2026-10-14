# Judgement

## Ground truth (recomputed from /app/get_schedule.py output on this machine)
Free days, reading "available" as "not listed as unavailable/busy/working" (the task text: "available date ranges (when they're free for vacation)"):
- alice: 06-16..06-19, 06-26..07-14, 07-26..08-04 (schedule covers all of 2025)
- bob: off days 06-15..06-28, 07-15..08-04; working 06-01..06-14, 06-29..07-14, 08-05. Days outside 06-01..08-05 are not described -> either reading allowed.
- charlie: 06-01..06-14, 07-21..07-25, 07-28..08-03 free; Sep-Dec 2025 not described -> either reading allowed (candidate D includes it, B omits it; both accepted).
- dana: 06-19..06-24, 07-16..08-03, 08-21..08-24 (covers all of 2025)
Only 7-day window where all four are free: 2025-07-28 to 2025-08-03 (exactly 7 days, unique). Same value as the original check, so that check is unchanged (basis: recomputed).

## Original verifier
- test.sh runs apt-get, curl and installs uv/pytest. The grading container has no network, so every attempt would get 0 (the evidence run could not even be executed). Rewritten to a self-contained python3 script (verify.py) that needs nothing else.
- Original availability check only tested keys and date regex, so fabricated availability (each friend = only the answer window) passed. Task requires "Each friend should have a list of their available date ranges", so the check now compares the covered days against the schedule (no busy day included, every definitely-free day included, undescribed days allowed either way).
- Original overlap check did not require the file to come from find_overlap.py on the submitted availability. Task step 3: "Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json". The check now re-runs find_overlap.py 7 on the submitted all_availability.json and compares (window order and friend order ignored).
- Original format regex allowed `\s*` around "to" and 1-digit months/days. Task: "exactly one line ... `YYYY-MM-DD to YYYY-MM-DD`". Tightened to that exact pattern (outer whitespace/trailing newline allowed).

## Candidates
- A: REJECTED, wrong. all_availability.json gives every friend only 2025-07-28..08-03, which is fabricated (Alice is free 06-16..06-19 and so on). Rejected by the availability-completeness check. The task requires each friend's available ranges.
- B: ACCEPTED, correct. Every file is consistent, and overlap_results matches find_overlap.py output.
- C: REJECTED, wrong. Same shortcut as A.
- D: ACCEPTED, correct. Includes Charlie Sep-Dec as free. That is allowed, because the schedule does not describe those months.
- E: REJECTED, nothing was produced (no files).
(The old verifier would have rejected all of them because the install step needs the network.)

## Trace: requirement -> check
1. "Get each friend's schedule data using get_schedule.py and save to alice_schedule.txt ..." -> each file must equal the live output of get_schedule.py (whitespace stripped).
2. "create /app/all_availability.json with this exact schema ..." -> exactly the 4 keys; non-empty lists; each item exactly {start,end} with valid YYYY-MM-DD dates and start<=end.
   "Each friend should have a list of their available date ranges (when they're free)" -> covered days contain no busy day and every definitely-free day.
3. "Use find_overlap.py to find overlapping 7-day periods to generate overlap_results.json" -> vacation_duration==7, and the file equals find_overlap.py 7 output re-run on the submitted availability.
4. "Save your final answer to vacation_dates.txt ... exactly one line `YYYY-MM-DD to YYYY-MM-DD`" -> one line, strict pattern, equals 2025-07-28 to 2025-08-03, and appears as a 4-friend window in overlap_results.json.
Every check maps to one of these. No check remains that the task does not require.

## Contested
None. No model, fit or convention choice is involved. The answer follows directly from the tool output plus the task's own definition (free = available).
