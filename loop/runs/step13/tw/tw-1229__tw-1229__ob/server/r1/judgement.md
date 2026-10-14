# Judgement

## Candidates
The original verifier accepted A, B, C, D (no E exists) and rejected the empty attempt. It rejected no candidate,
so there was no rejection to investigate. I checked each candidate by hand against ffprobe output on this machine
and against the Dockerfile that builds the media files:
- All four list the same 5 valid / 6 invalid absolute paths, the subdirectory file included, with correct metadata:
  durations 10/15/7/12.042449/8, resolutions, h264 at 30/24 fps, mp3/flac codecs, sample rates 44100/48000,
  1 channel, bitrates in kb/s, and a reason that matches the failed criterion. Their summary counts are 11/5/6,
  total_valid_duration is about 52.04, and summary.txt is non-empty and gives the counts. **All four are correct.**
- Bitrates differ between candidates: A and D report the container bit_rate (2593), B and C the video-stream
  bit_rate (2578). The task only says "Bitrate (in kb/s)", so both readings are allowed and the verifier accepts both.
  This machine's own files give different bitrates (1080p: format 2705, stream 2566), so x264 output is not
  byte-identical between builds. Bitrate is therefore checked as a band (envelope of the format and stream values in
  the grading container, ±35%), not as an exact value. The band still catches unit errors (b/s, Mb/s).

## Weaknesses in the original verifier (all accepted wrong results)
- Valid/invalid membership was checked by filename substring, so extra paths, duplicates, or a file listed in both
  lists all passed. Now the check is exact set equality on normalised absolute paths.
- No metadata was checked at all, so a report holding only paths passed (see shortcut_paths_only).
- total_valid_duration had a tolerance of ±5 s, so 49.0 passed. The expected value is recomputed as the sum of the
  ffprobe durations of the 5 valid files (52.042449), which is fixed by the Dockerfile's `-t`/`duration` arguments.
  The tolerance is now 0.15 s, which allows rounding to 2 decimals (B gives 52.04).
- Reasons were only required to be non-empty. Now each single-criterion invalid file must name its own criterion
  and no other one. The two reason strings the task's example names must be exact: "resolution_too_low" and
  "duration_too_short". For bitrate and sample rate the check is a keyword match, because the task gives no exact
  string for those. The corrupted file needs a reason other than a criterion reason, in any wording.
- summary.txt was only checked for existence. It must now be non-empty and contain the counts 11, 5 and 6.
- test.sh depended on network (curl, uv) and pytest. It now runs plain python3, which the grading image is
  documented to have.

## Trace: requirement -> check
- Recursive scan over the listed extensions -> exactly 11 files; nested_valid.mp4 must appear (total_files==11, path sets).
- Extract format/duration/bitrate -> each valid entry has format (the extension or the ffprobe format_name),
  duration within 0.1 s, and bitrate in the kb/s band.
- Video: resolution/codec/frame_rate -> width and height exact, video_codec=="h264" (key named in the example),
  frame_rate within 0.05 and numeric.
- Audio: codec/sample_rate/channels -> key audio_codec or codec equal to mp3/flac, sample_rate exact, channels==1.
  The task gives no key name for the audio codec, so both are accepted.
- Validation criteria -> the 5 valid and 6 invalid sets (fixed by the Dockerfile's encode parameters) and the
  reasons check.
- Report structure -> top-level keys, list and dict types, summary keys numeric, counts consistent with the lists.
- summary.txt with human-readable statistics -> non-empty, contains 11, 5 and 6.
- "type" field (in the example) -> "video" for the mp4 files, "audio" for mp3 and flac.

Every check above follows from one of these sentences. No check was kept that the task does not require.

## Contested
None. No expected value needed a choice the task leaves open. Where the task allows more than one reading (which
bitrate source, the format string, the audio codec key), the verifier accepts every reading.

## Near misses (all copies of A with one change; all rejected)
nm_bitrate_bps, nm_not_recursive, nm_wrong_reason, nm_total_duration_off, nm_missing_video_codec,
nm_frame_rate_wrong, nm_empty_summary_txt, nm_file_in_both_lists, nm_sample_rate_str_wrong, shortcut_paths_only.
