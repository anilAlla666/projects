# Verifier Judgement

## Executive Summary

The original verifier had a critical flaw: it used a hardcoded SHA256 hash check that only accepted one exact byte-for-byte JSON output. This rejected 3 out of 4 correct candidate attempts (A, C, D) while accepting only candidate B whose output happened to match the hash.

The repaired verifier removes the hash check and replaces it with logical validation tests that verify:
1. Report structure and required fields
2. Correct classification of valid/invalid files per task criteria
3. Appropriate reason strings for invalid files
4. Mathematically consistent summary statistics
5. Accurate metadata values within reasonable tolerance
6. Existence and basic content of summary.txt

## Candidate Analysis

### Candidate A - CORRECT ✓
- All 5 valid files identified with accurate metadata
- All 6 invalid files identified with appropriate reasons
- Summary statistics correct (11 total, 5 valid, 6 invalid, ~52s duration)
- summary.txt contains required statistics in human-readable format
- **Verdict: ACCEPTED by repaired verifier**

### Candidate B - CORRECT ✓
- All 5 valid files identified with accurate metadata (slightly different float precision)
- All 6 invalid files identified with appropriate reasons ("corrupted_file" vs "corrupted")
- Summary statistics correct
- summary.txt contains required statistics
- **Verdict: ACCEPTED by repaired verifier (was accepted by original due to hash match)**

### Candidate C - CORRECT ✓
- All 5 valid files identified with accurate metadata
- All 6 invalid files identified with appropriate reasons ("corrupted_or_unreadable")
- Summary statistics correct
- summary.txt contains required statistics plus validation criteria
- **Verdict: ACCEPTED by repaired verifier**

### Candidate D - CORRECT ✓
- All 5 valid files identified with accurate metadata (includes redundant "codec" field)
- All 6 invalid files identified with appropriate reasons
- Summary statistics correct
- summary.txt contains required statistics
- Extra "codec" field is harmless and doesn't violate task requirements
- **Verdict: ACCEPTED by repaired verifier**

### Empty Attempt - INCORRECT ✗
- No media_report.json or summary.txt produced
- **Verdict: REJECTED by repaired verifier**

## Near Miss Analysis (All Correctly Rejected)

### NM1: Missing Valid File (nested_valid.mp4)
- Breaks requirement: "Scan the directory recursively for all media files"
- nested_valid.mp4 in subdir/ is valid but placed in invalid_files
- **Correctly rejected**

### NM2: Wrong Validation (invalid_low_resolution.mp4 marked valid)
- Breaks requirement: "Video files must have a minimum resolution of 720p"
- 640x480 file incorrectly classified as valid
- **Correctly rejected**

### NM3: Missing Reason for Invalid File
- Breaks requirement: invalid_files must have "reasons" array with at least one reason
- corrupted_file.mp4 has empty reasons array
- **Correctly rejected**

### NM4: Wrong Summary Statistics
- Breaks requirement: summary.total_valid_duration must match sum of valid file durations
- Reports 100s instead of ~52s
- **Correctly rejected**

### NM5: Hardcoded Shortcut (nonexistent file)
- Breaks requirement: must scan actual directory, not fabricate results
- Includes nonexistent_file.mp4, total_files=12 instead of 11
- **Correctly rejected**

## Check Traceability

| Task Requirement | Verifier Check | Basis |
|------------------|----------------|-------|
| Scan recursively for media extensions | Implicit: all 11 expected files must be accounted for | task-text |
| Extract metadata (format, duration, bitrate, resolution, codec, frame rate, sample rate, channels) | test_valid_file_metadata_accuracy validates all fields | task-text |
| Video: resolution >= 1280x720 | test_validation_logic checks width>=1280, height>=720 | task-text |
| Video: bitrate >= 500 kb/s | test_validation_logic checks bitrate>=500 | task-text |
| Audio: sample rate >= 44100 Hz | test_validation_logic checks sample_rate>=44100 | task-text |
| All: duration >= 5 seconds | test_validation_logic checks duration>=5.0 | task-text |
| Files not corrupted (readable by ffprobe) | test_validation_logic checks corrupted file has corruption reason | task-text |
| JSON report structure | test_report_exists_and_valid_json checks all required keys | task-text |
| valid_files array with specified fields | test_validation_logic validates each valid file's fields | task-text |
| invalid_files array with path and reasons | test_validation_logic validates each invalid file's reasons | task-text |
| summary with total_files, valid_count, invalid_count, total_valid_duration | test_summary_statistics validates all summary fields | task-text |
| summary.txt human-readable statistics | test_report_exists_and_valid_json checks file exists, non-empty, contains "11" | task-text |
| Summary counts mathematically consistent | test_summary_statistics cross-checks counts | recomputed |
| Total valid duration matches sum of valid file durations | test_summary_statistics checks with 0.1s tolerance | recomputed |
| Correct number of files (11 total, 5 valid, 6 invalid) | test_summary_statistics checks exact counts | data-settled (from Dockerfile) |
| Valid file metadata accuracy | test_valid_file_metadata_accuracy checks against ffprobe ground truth | recomputed (ffprobe on test machine) |

## Contested Values

None. All validation thresholds are explicitly stated in the task:
- 720p = width >= 1280 AND height >= 720
- Video bitrate >= 500 kb/s
- Audio sample rate >= 44100 Hz
- Duration >= 5 seconds
- Corrupted = not readable by ffprobe

The reason strings for invalid files are not explicitly specified (task only gives examples: "resolution_too_low", "duration_too_short"). The verifier accepts any semantically appropriate reason string (e.g., "corrupted", "corrupted_file", "corrupted_or_unreadable" all accepted for corrupted file).

## Unverified Checks

- The exact total file count (11) and valid/invalid counts (5/6) are derived from the Dockerfile's test file creation, not from the task text itself. However, since the grading container uses this exact Dockerfile, these values are "data-settled" from the task's own test environment specification.

- The metadata accuracy tolerances (bitrate ranges, duration tolerances) are based on ffprobe output from the actual test files created by the Dockerfile. These are "recomputed" from the task's own data.

## Changes Made

1. **Removed hardcoded SHA256 hash check** from test.sh - this was the primary defect
2. **Rewrote test_outputs.py** as standalone Python script (no pytest dependency) with:
   - Robust file existence and JSON structure validation
   - Logical validation of file classifications against task criteria
   - Semantic reason string checking (not exact string matching)
   - Mathematical consistency checks for summary statistics
   - Metadata accuracy validation against ground truth ffprobe values with reasonable tolerances
   - summary.txt existence and basic content validation
3. **Fixed test.sh** to properly capture exit code without `set -e` interference

## Verification Results

| Attempt | Original Verifier | Repaired Verifier |
|---------|-------------------|-------------------|
| Candidate A | REJECT (hash mismatch) | **ACCEPT** ✓ |
| Candidate B | ACCEPT (hash match) | **ACCEPT** ✓ |
| Candidate C | REJECT (hash mismatch) | **ACCEPT** ✓ |
| Candidate D | REJECT (hash mismatch) | **ACCEPT** ✓ |
| Empty | REJECT | **REJECT** ✓ |
| NM1 (missing valid) | N/A | **REJECT** ✓ |
| NM2 (wrong validation) | N/A | **REJECT** ✓ |
| NM3 (missing reason) | N/A | **REJECT** ✓ |
| NM4 (wrong summary) | N/A | **REJECT** ✓ |
| NM5 (hardcoded shortcut) | N/A | **REJECT** ✓ |

All correct candidates accepted, all incorrect/near-miss attempts rejected.
