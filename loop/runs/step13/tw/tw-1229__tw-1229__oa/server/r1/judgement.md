# Judgement

## Evidence
`evidence.md` contains no runs (the probe crashed: missing grader config.json). I re-ran everything myself by
copying each candidate's two result files to `/workspace/` and running `bash /tests/test.sh`.

## Why the original verifier rejected everything
The original `test.sh` ran `apt-get install curl`, downloaded `uv` and `pytest` from the network. The grading
container has no network and no pytest, so the script fails for every attempt -> reward 0 for all four
candidates (A-D), including correct ones. **Check wrong, not the attempts.** Rewritten as plain `python3`
(only stdlib + ffprobe, both in the grading image).

## Candidate decisions (with the repaired verifier)
| cand | result | reason |
|---|---|---|
| A | correct (1) | all 11 files, right valid/invalid split, metadata matches ffprobe, consistent summary, summary.txt with counts |
| B | correct (1) | same; bitrates as floats (video-stream bitrate), durations rounded to 2dp - allowed by the task |
| C | correct (1) | same; video-stream bitrate in integer kb/s |
| D | correct (1) | same; extra `codec` key alongside `video_codec`/`audio_codec` - extra keys not forbidden |
| empty | wrong (0) | no report |

No candidate E exists in /candidates.

## Values: settled or contested
- Valid/invalid classification: recomputed in the grader with ffprobe against the thresholds the task states
  ("width >= 1280 AND height >= 720", "at least 500 kb/s", "at least 44100 Hz", "at least 5 seconds", "readable by
  ffprobe"). Basis recomputed. Expected: 5 valid, 6 invalid, 11 total (same as original verifier).
- Duration, width, height, codec, frame rate, sample rate, channels: read from ffprobe (task: "Use ffprobe").
  Recomputed; tolerance 0.1 s duration, 0.02 fps.
- Bitrate: the task does not say whether the container (format) bit_rate or the stream bit_rate is meant, nor
  1000 vs 1024, nor rounding; also mandelbrot/x264 encodes differ between builds (candidates' files had format
  bitrates 2593/1827/2173 vs 2705/1931/2367 on this machine). Contested in exact value; the verifier only checks a
  band [0.75*min(format,stream), 1.3*max(format,stream)] computed in the grader - holds under every reading, and
  rejects unit errors (b/s, Mb/s).
- total_valid_duration: original check was |x-52|<5 (task-text-free hard-coded). Settled by the task's own
  definition (sum of durations of valid files) recomputed via ffprobe: 52.042449; tightened to +-0.1. Old check
  would accept 48.0 (near miss nm_total_duration).
- Reason labels: task gives only the example labels "resolution_too_low", "duration_too_short"; other labels are
  not fixed. Checked by keyword (resolution / bitrate / sample / duration|short / corrupt|read|probe|error) that each
  failed criterion has a reason and no reason names a criterion the file passes (corrupt file exempt).
- format: example gives "mp4" for an .mp4 file; ffprobe gives "mov,mp4,m4a,3gp,3g2,mj2". Accept extension, any
  component of format_name, or the full format_name.
- audio key names are not given in the example: accept `audio_codec` or `codec`; `sample_rate`; `channels`.

No data-fitting readings were needed (no models involved).

## Trace: requirement -> check
1. Recursive scan of listed extensions -> reported path set == set of media files found by os.walk in grader
   (catches non-recursive scan, missing/extra files, duplicates); paths absolute (example uses absolute paths).
2. Metadata (format, duration, bitrate, type; video width/height/video_codec/frame_rate; audio codec/sample
   rate/channels) -> per-valid-file comparisons with ffprobe.
3. Validation criteria -> valid set == recomputed valid set; reasons non-empty strings and keyword-consistent.
4. JSON report structure -> top-level keys, list/object types, summary keys numeric, counts consistent,
   total_files == number of media files, total_valid_duration == recomputed sum.
5. summary.txt -> exists, non-empty, states total/valid/invalid counts ("human-readable statistics").

## Trace: check -> requirement
All checks above follow from the listed sentences. Removed: network install (not required, breaks grading);
hard-coded "11 total" and "52 +- 5" replaced by recomputed values (same expectation on this image, tighter).
Substring path matching replaced by exact absolute paths (example structure).

## Near misses (all rejected; A-D accepted; empty rejected)
nm_lowbitrate_valid, nm_nonrecursive, nm_wrong_framerate, nm_samplerate_khz, nm_bitrate_bps,
nm_total_duration, nm_wrong_reason, nm_empty_summary (shortcut), nm_basename_paths.
