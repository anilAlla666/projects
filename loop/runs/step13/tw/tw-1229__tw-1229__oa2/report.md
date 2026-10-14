# Grader report: tw-1229

Rounds: 1 · cost $0.74 · candidates judged correct: A, B, C, D · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 5 |
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
| report exists, JSON object, keys valid_files/invalid_files/summary, summary numeric fields | Generate a JSON report at /workspace/media_report.json with this structure | format |
| exact absolute path sets: 5 valid, 6 invalid, no duplicates | Scan the directory recursively ... Validate files against these criteria | recomputed |
| reasons non-empty; readable invalid files name exactly their failing criterion; resolution_too_low / duration_too_short verbatim; corrupted has non-criterion reason | Validate files against these criteria; example reasons ["resolution_too_low", "duration_too_short"] | recomputed |
| valid entries: type, format, duration +-0.1, width/height, video_codec h264, frame_rate +-0.01, audio codec, sample_rate, channels | Extract metadata for each file including ... | recomputed |
| bitrate positive kb/s, >=500 for valid video, within 30% of ffprobe format or stream bitrate | Bitrate (in kb/s) | recomputed |
| summary counts 11/5/6 consistent with lists; total_valid_duration 52.042449 +-0.1 and = sum of listed | summary structure in example | recomputed |
| summary.txt exists, non-empty, contains 11, 5, 6 | Create a summary file at /workspace/summary.txt containing human-readable statistics | format |
| test.sh runs plain python3, no network | grader environment | format |
