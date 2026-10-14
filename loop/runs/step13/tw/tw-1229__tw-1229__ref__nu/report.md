# Grader report: tw-1229

Rounds: 1 · cost $0.00 · candidates judged correct: A, B, C, D · judged wrong: none

| basis | checks |
|---|---:|
| task-text | 9 |
| recomputed | 3 |
| data-settled | 1 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 0 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| media_report.json exists and is valid JSON with required top-level keys (valid_files, invalid_files, summary) | Generate a JSON report at /workspace/media_report.json with this structure | task-text |
| summary.txt exists, non-empty, and mentions total files (11) | Create a summary file at /workspace/summary.txt containing human-readable statistics | task-text |
| All 5 expected valid files present in valid_files list with correct type-specific fields | Scan the directory recursively for all media files... Extract metadata for each file | task-text |
| All 6 expected invalid files present in invalid_files list with appropriate reasons | Validate files against these criteria... Files must not be corrupted | task-text |
| No file appears in both valid_files and invalid_files | Each file must be classified as either valid or invalid | task-text |
| Invalid files have non-empty reasons array with semantically appropriate reason strings | Invalid files must have reasons array as shown in example: "reasons": ["resolution_too_low", "duration_too_short"] | task-text |
| Valid video files meet minimum resolution (width >= 1280, height >= 720) and bitrate >= 500 kbps | Video files must have a minimum resolution of 720p (width >= 1280 AND height >= 720)... Video files must have a bitrate of at least 500 kb/s | task-text |
| Valid audio files meet minimum sample rate (>= 44100 Hz) | Audio files must have a sample rate of at least 44100 Hz | task-text |
| All valid files have duration >= 5 seconds | All files must have a duration of at least 5 seconds | task-text |
| Summary counts are mathematically consistent (total_files = valid_count + invalid_count, counts match actual array lengths) | Summary object with total_files, valid_count, invalid_count, total_valid_duration | recomputed |
| Total files = 11, valid_count = 5, invalid_count = 6 | Implied by the test environment created by the grading Dockerfile | data-settled |
| total_valid_duration matches sum of valid file durations (within 0.1s) and is approximately 52 seconds (within 2s) | Summary includes total_valid_duration | recomputed |
| Valid file metadata accuracy: duration, resolution, codec, frame rate, sample rate, channels, bitrate within expected ranges from ffprobe | Extract metadata for each file including: file format/container type, duration, bitrate, resolution, codec, frame rate, sample rate, channels | recomputed |
