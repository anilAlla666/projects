# Judgement

## Ground truth (recomputed from /app/get_schedule.py output, no free choices)
Free days per friend (2025): alice Jun16-19, Jun26-Jul14, Jul26-Aug4; bob (off days) Jun15-28, Jul15-Aug4;
charlie (complement of listed unavailability, Jun-Aug) Jun1-14, Jul21-25, Jul28-Aug3; dana Jun19-24, Jul16-Aug3, Aug21-24.
Intersection of all four = 2025-07-28..2025-08-03, exactly 7 days, the only 7-day window with all four free.
So the final answer "2025-07-28 to 2025-08-03" is settled (recomputed); the original verifier already expected it.
Only genuinely undefined days are left open: days outside the span a schedule covers (e.g. bob before Jun 1 / after Aug 5,
charlie Sep-Dec, anything outside 2025). Candidate D lists charlie Sep1-Dec31, B does not; both are accepted.

## Candidates (only A-D exist)
The old verifier accepted all four. With the repaired verifier:
- A: WRONG. all_availability.json lists only 2025-07-28..08-03 for every friend, i.e. the final overlap, not
  "a list of their available date ranges (when they're free for vacation)". Alice e.g. is free Jun16-19, Jun26-Jul14, Jul26-Aug4.
  This is the shortcut of writing the answer back into the input, so find_overlap.py has nothing to compute. Rejected by availability_content.
- C: WRONG, for the same reason as A.
- B: correct. Schedules match the tool, availability matches the schedules, overlap_results is exactly find_overlap.py output, answer correct.
- D: correct. The same as B plus charlie Sep-Dec, which the schedule leaves undefined, so it is allowed.

## Trace: requirement -> check
- "Get each friend's schedule data using get_schedule.py and save to <friend>_schedule.txt": schedule_files (content == tool stdout, after stripping whitespace; uses a pristine copy of the tool in /tests).
- "create /app/all_availability.json with this exact schema {alice:[{start,end}],...}": availability_schema (exactly the 4 keys, non-empty lists, YYYY-MM-DD strings, start<=end).
- "Parse the different schedule formats ... Each friend should have a list of their available date ranges": availability_content (no day the schedule marks busy/working is listed; every day the schedule clearly leaves free is listed).
- "Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json": overlap_structure (fields, duration 7, count consistent, the all-four window 07-28..08-03 present) and overlap_generated_by_tool (file == find_overlap.py 7 run on the submitted availability).
- "Save your final answer to /app/vacation_dates.txt ... exactly one line `YYYY-MM-DD to YYYY-MM-DD`": vacation_dates (one line, strict zero-padded format, value 2025-07-28 to 2025-08-03).
## Trace: check -> requirement
Each check above follows from the sentence quoted next to it. Removed or changed from the old verifier:
- The format regex allowed unpadded dates, any spacing and any case. It is now strict, as the task fixes the format.
- test.sh tried to install curl/uv/pytest from the network. It now runs plain python3, which is all the grading image has.
- The value check (07-28..08-03) is kept. It is settled by recomputation.

## Contested
None. Every expected value follows from the tool's output with no reading of my own.

## Near misses (all rejected; B and D accepted; the empty attempt rejected)
nm1 availability only overlap window (shortcut, like A/C); nm2 end date off by one; nm3 charlie's board meeting Jul 26-27 marked free;
nm4 overlap_results hand-trimmed; nm5 bob_schedule.txt missing last line; nm6 unpadded dates; nm7 two lines in vacation_dates.txt.
