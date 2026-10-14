# Grader report: tw-1229

Rounds: 1 · cost $0.62 · candidates judged correct: A, B, C, D · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 5 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 1 |
| unverified | 0 |

## Needs a human

- **contested:** exact bitrate value: the verifier expects band around ffprobe format/stream bitrate, the candidates give format bitrate int (A,D), video stream bitrate (B,C); task does not say container vs stream bitrate, rounding, or 1000 vs 1024; encodes also vary by build
- **contested:** bitrate within [0.75*min(format,stream), 1.3*max(format,stream)] kb/s (only a band holding under all readings)

## Every check

| check | requirement | basis |
|---|---|---|
| test.sh runs plain python3 without network installs | grader runs bash /tests/test.sh offline | format |
| report exists, valid JSON, keys valid_files/invalid_files/summary with right types | Generate a JSON report at /workspace/media_report.json with this structure | format |
| reported absolute path set equals media files found recursively in grader | Scan the directory recursively for all media files with these extensions | recomputed |
| valid set equals files passing all criteria per ffprobe | Validate files against these criteria | recomputed |
| valid entry type/format/duration/width/height/video_codec/frame_rate/audio codec/sample_rate/channels match ffprobe | Extract metadata for each file including ... | recomputed |
| bitrate within [0.75*min(format,stream), 1.3*max(format,stream)] kb/s | Bitrate (in kb/s) | contested |
| invalid reasons non-empty strings, keyword covering each failed criterion, no reason for passed criterion | invalid_files ... reasons | task-text |
| summary counts consistent and total_files == number of media files | summary: total_files, valid_count, invalid_count | recomputed |
| total_valid_duration == sum of ffprobe durations of valid files +-0.1 | total_valid_duration | recomputed |
| summary.txt exists, non-empty, states total/valid/invalid counts | Create a summary file at /workspace/summary.txt containing human-readable statistics | task-text |
