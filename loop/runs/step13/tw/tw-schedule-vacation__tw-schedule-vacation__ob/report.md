# Grader report: tw-schedule-vacation

Rounds: 1 · cost $0.57 · candidates judged correct: B, D · judged wrong: A, C

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 5 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| each <friend>_schedule.txt equals get_schedule.py <friend> stdout (stripped) | Get each friend's schedule data using /app/get_schedule.py and save to alice_schedule.txt, ... | recomputed |
| all_availability.json has exactly keys alice,bob,charlie,dana; non-empty lists of {start,end} YYYY-MM-DD, start<=end | create a file called /app/all_availability.json with this exact schema | format |
| availability contains every day the schedule clearly leaves free and no day it marks busy/working | Parse the different schedule formats ... Each friend should have a list of their available date ranges (when they're free for vacation). | recomputed |
| overlap_results.json has vacation_duration 7, total_windows_found == len(windows), includes the all-four window 2025-07-28..2025-08-03 | Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json | recomputed |
| overlap_results.json equals find_overlap.py 7 output on the submitted all_availability.json | Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json | recomputed |
| vacation_dates.txt is exactly one line 'YYYY-MM-DD to YYYY-MM-DD' | The file must contain exactly one line (dates inclusive): `YYYY-MM-DD to YYYY-MM-DD` | format |
| vacation window is 2025-07-28 to 2025-08-03 | plan a 7-day vacation with my four friends ... Save your final answer | recomputed |
