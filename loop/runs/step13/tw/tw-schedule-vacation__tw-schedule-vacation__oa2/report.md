# Grader report: tw-schedule-vacation

Rounds: 1 · cost $0.59 · candidates judged correct: B, D · judged wrong: A, C

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 4 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** availability of Charlie Sep-Dec 2025 and Bob outside 2025-06-01..2025-08-05: the verifier expects not checked, the candidates give B omits, D lists charlie 2025-09-01..2025-12-31; the schedules do not mention these days; whether unmentioned = free is not fixed by the task

## Every check

| check | requirement | basis |
|---|---|---|
| vacation_dates.txt exists | Save your final answer to /app/vacation_dates.txt. | format |
| vacation_dates.txt is exactly one line 'YYYY-MM-DD to YYYY-MM-DD' equal to 2025-07-28 to 2025-08-03 | The file must contain exactly one line (dates inclusive): YYYY-MM-DD to YYYY-MM-DD | recomputed |
| overlap_results.json exists with vacation_duration 7, total_windows_found == len(windows) > 0 | Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json | format |
| overlap_results.json equals find_overlap.py 7 output on submitted all_availability.json and contains the 4-friend 07-28..08-03 window | Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json | recomputed |
| each *_schedule.txt equals get_schedule.py <friend> stdout (stripped) | Get each friend's schedule data using /app/get_schedule.py and save to alice_schedule.txt, ... | recomputed |
| all_availability.json has exactly alice/bob/charlie/dana, non-empty lists of {start,end} YYYY-MM-DD | create a file called /app/all_availability.json with this exact schema | format |
| each friend's listed days, within the span the schedule covers, equal the free days parsed from the schedule | Each friend should have a list of their available date ranges (when they're free for vacation). | recomputed |
