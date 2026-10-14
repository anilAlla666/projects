# Grader report: tw-schedule-vacation

Rounds: 1 · cost $0.51 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 4 |
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
| each <friend>_schedule.txt equals live get_schedule.py <friend> output | Get each friend's schedule data using /app/get_schedule.py and save to alice_schedule.txt, ... | recomputed |
| all_availability.json has exactly alice/bob/charlie/dana, non-empty lists of {start,end} valid ISO dates, start<=end | create a file called /app/all_availability.json with this exact schema | format |
| availability days per friend include no busy day and all definitely-free days; undescribed days (bob outside Jun1-Aug5, charlie Sep-Dec) are free either way | Each friend should have a list of their available date ranges (when they're free for vacation). | recomputed |
| overlap_results.json vacation_duration==7 and equals find_overlap.py 7 re-run on submitted all_availability.json (order-insensitive) | Use /app/find_overlap.py to find overlapping 7-day periods to generate /app/overlap_results.json | recomputed |
| vacation_dates.txt is exactly one line matching 'YYYY-MM-DD to YYYY-MM-DD' | The file must contain exactly one line (dates inclusive): YYYY-MM-DD to YYYY-MM-DD | format |
| answer equals 2025-07-28 to 2025-08-03 and is a 4-friend window in overlap_results.json | plan a 7-day vacation with my four friends ... Save your final answer to /app/vacation_dates.txt | recomputed |
