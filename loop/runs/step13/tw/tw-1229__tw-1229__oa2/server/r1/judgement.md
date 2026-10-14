# Judgement

## Candidates
The original verifier accepted A, B, C, D and rejected the empty attempt; it rejected no candidate.
I re-checked every candidate against ffprobe on this machine (the task's own specified tool):
all four classify the 11 files identically and correctly (5 valid / 6 invalid), give the correct single
failing criterion for each invalid file, correct width/height/codec/frame rate/sample rate/channels,
durations within rounding of ffprobe's, total_valid_duration = 52.04 (sum of ffprobe durations 10+15+7+12.042449+8),
and a summary.txt with the statistics. **A, B, C, D: correct.**

Bitrates differ between candidates (A/D 2593 vs B/C 2578 for the 1080p file; this machine's file gives 2705 format /
2566 stream). Two causes: (1) format vs. video-stream bitrate, both plausible readings of "Bitrate (in kb/s)"; (2) the
x264 encodes are not bit-identical between docker builds (A's nested file 2173 format < B's 2186 stream, impossible
within one build). So exact bitrate values are not settled. The new check only requires a positive kb/s number, >=500
for valid videos (task's own criterion), and within +-30% of either the format or any stream bitrate that ffprobe
reports for the grader's copy of the file. Every reading (format/stream, 1000/1024) passes.

## Problems in the original verifier
- test.sh did `apt-get`, `curl astral.sh`, `uv init/add pytest`: depends on network/uv. Replaced by a plain python3
  runner (grader image has python3; no pytest needed).
- File matching used substring (`filename in path`): relative paths or files listed in both lists passed. Now exact
  absolute path sets (task example uses `/workspace/media_files/...`), no duplicates, valid/invalid sets exact.
- No check of per-file metadata at all (requirement 2). Added.
- Reasons: only "non-empty". Now each readable invalid file's reasons must name exactly the failing criterion
  (recomputed with ffprobe: each invalid file fails exactly one criterion), and the two names the task's example gives
  (`resolution_too_low`, `duration_too_short`) are required verbatim. Corrupted file must have a non-criterion reason.
- total_valid_duration tolerance tightened from +-5 to +-0.1 around the recomputed 52.042449 (recomputed: sum of
  ffprobe durations; durations are deterministic, set by -t / sine duration) and must equal the sum of listed durations.
- summary.txt: only existence; now must be non-empty and contain the counts 11, 5, 6.

## Trace: requirement -> check
1. Recursive scan, listed extensions -> test_classification (exact sets incl. subdir/nested_valid.mp4; total 11).
2. Metadata (format, duration, bitrate, video: width/height/video_codec/frame_rate, audio: codec/sample_rate/channels,
   type) -> test_valid_metadata. Audio key names not given by the task: codec accepted as `audio_codec` or `codec`;
   `sample_rate`, `channels` (only names in the spirit of the example's snake_case; all candidates use them).
   format accepted as extension or any ffprobe format_name component (task: "File format/container type").
3. Validation criteria -> test_classification + test_reasons (+ valid video bitrate >=500).
4. JSON structure (keys, summary fields, counts) -> test_structure, test_summary.
5. summary.txt -> test_summary_txt.
Every check above follows from one of these sentences; the original's "exactly 11 files" and "~52 s" are kept
(tightened, recomputed).

## Data-settling
No contested values: everything asserted is either fixed by the task text/Dockerfile or recomputed with ffprobe;
bitrate exact value is not asserted (weak range check that holds under every reading).

## Near misses (all rejected; A-D accepted; empty rejected), tested by running /tests/test.sh with files at /workspace
nm_extra_wrong_reason, nm_not_recursive, nm_wrong_frame_rate, nm_relative_paths, nm_total_duration_video_only,
nm_missing_sample_rate, nm_no_summary_txt, nm_wrong_reason_short_audio, shortcut_fabricated_bitrate.
Note: bwrap is not on this machine; I simulated the grader by copying files into /workspace and symlinking /tests.
