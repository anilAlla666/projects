# Grader report: tw-1229

Rounds: 1 · cost $0.71 · candidates judged correct: A, B, C, D · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 7 |
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
| report exists, valid JSON, top-level keys and types | Generate a JSON report at /workspace/media_report.json with this structure | format |
| valid_files and invalid_files paths equal the expected absolute path sets exactly, with no duplicates | Scan the directory recursively ... Validate files against these criteria | recomputed |
| valid entries: format matches the extension or the ffprobe format_name; type is video/audio | Extract metadata ... File format/container type | task-text |
| duration within 0.1 s of ffprobe | Duration (in seconds) | recomputed |
| bitrate within ±35% of the envelope of ffprobe format and stream bit_rate in kb/s | Bitrate (in kb/s) | recomputed |
| video width/height exact, video_codec h264, frame_rate numeric within 0.05 | For video files: resolution (width x height), codec, frame rate | recomputed |
| audio codec (audio_codec or codec), sample_rate exact, channels exact | For audio files: codec, sample rate, channels | recomputed |
| reasons non-empty; name the failed criterion only; resolution_too_low and duration_too_short exact | invalid_files ... reasons [example] | task-text |
| summary counts 11/5/6 and consistent with the lists | summary: total_files, valid_count, invalid_count | recomputed |
| total_valid_duration = sum of valid durations within 0.15 (~52.04) | total_valid_duration | recomputed |
| summary.txt non-empty and contains 11, 5 and 6 | Create a summary file at /workspace/summary.txt containing human-readable statistics | format |
