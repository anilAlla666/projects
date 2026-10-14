#!/usr/bin/env python3
"""Standalone verifier (no pytest needed) for the media validation task."""
import json, os, re, shutil, subprocess, sys

ROOT = "/workspace/media_files"
REPORT = "/workspace/media_report.json"
SUMMARY = "/workspace/summary.txt"

# Facts fixed by the Dockerfile commands that create the media files.
VALID = {
    "valid_video_720p.mp4":  dict(type="video", duration=10.0, width=1280, height=720, codec="h264", fps=30.0, vbr=(500, 6000)),
    "valid_video_1080p.mp4": dict(type="video", duration=15.0, width=1920, height=1080, codec="h264", fps=24.0, vbr=(500, 6000)),
    "subdir/nested_valid.mp4": dict(type="video", duration=7.0, width=1920, height=1080, codec="h264", fps=30.0, vbr=(500, 6000)),
    "valid_audio.mp3":       dict(type="audio", duration=12.042449, codec="mp3", sr=44100, ch=1, vbr=(150, 260)),
    "valid_audio_48k.flac":  dict(type="audio", duration=8.0, codec="flac", sr=48000, ch=1, vbr=(40, 400)),
}
# invalid file -> required reason keyword (None = corruption, free wording)
INVALID = {
    "invalid_low_resolution.mp4": "resolution",
    "invalid_low_bitrate.mp4": "bitrate",
    "invalid_low_sample_rate.mp3": "sample",
    "invalid_short_duration.mp4": "duration",
    "invalid_short_audio.mp3": "duration",
    "corrupted_file.mp4": None,
}
CRIT = {
    "resolution": ("resolution",),
    "bitrate": ("bitrate", "bit_rate"),
    "sample": ("sample",),
    "duration": ("duration",),
}
FMT_NAMES = {"mp4": "mov,mp4,m4a,3gp,3g2,mj2", "mp3": "mp3", "flac": "flac"}

errors = []
def fail(msg):
    errors.append(msg)

def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def probe(path):
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                              "-show_format", "-show_streams", path],
                             capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}")
    except Exception:
        return None

def bitrate_band(rel, info):
    """Acceptable kb/s band: envelope of ffprobe format/stream bit_rate in this
    container, +-35% (files are re-encoded per build, so values drift a bit).
    Falls back to the static band if ffprobe is unavailable."""
    p = probe(os.path.join(ROOT, rel))
    vals = []
    if p:
        fb = p.get("format", {}).get("bit_rate")
        if fb:
            vals.append(float(fb) / 1000)
        for s in p.get("streams", []):
            if s.get("codec_type") == info["type"] and s.get("bit_rate"):
                vals.append(float(s["bit_rate"]) / 1000)
    if not vals:
        return info["vbr"]
    return (min(vals) * 0.65, max(vals) * 1.35)

def main():
    if not os.path.isfile(REPORT):
        print("FAIL: media_report.json missing"); return False
    try:
        data = json.load(open(REPORT))
    except Exception as e:
        print(f"FAIL: report is not valid JSON: {e}"); return False
    if not isinstance(data, dict):
        print("FAIL: report root is not an object"); return False
    for k in ("valid_files", "invalid_files", "summary"):
        if k not in data:
            fail(f"missing top-level key {k}")
    if errors:
        return False
    vf, inf, summ = data["valid_files"], data["invalid_files"], data["summary"]
    if not isinstance(vf, list) or not isinstance(inf, list) or not isinstance(summ, dict):
        print("FAIL: wrong types for valid_files/invalid_files/summary"); return False

    def norm(entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None
        p = entry["path"]
        if not p.startswith("/"):
            return None
        return os.path.normpath(p)

    vpaths = [norm(e) for e in vf]
    ipaths = [norm(e) for e in inf]
    if None in vpaths or None in ipaths:
        fail("some entry lacks an absolute string 'path'")
    exp_v = {os.path.join(ROOT, r) for r in VALID}
    exp_i = {os.path.join(ROOT, r) for r in INVALID}
    if len(vpaths) != len(set(vpaths)) or len(ipaths) != len(set(ipaths)):
        fail("duplicate paths in report")
    if set(vpaths) != exp_v:
        fail(f"valid_files paths {sorted(map(str, vpaths))} != expected {sorted(exp_v)}")
    if set(ipaths) != exp_i:
        fail(f"invalid_files paths {sorted(map(str, ipaths))} != expected {sorted(exp_i)}")

    # metadata of valid files
    total = 0.0
    for e in vf:
        p = norm(e)
        if p not in exp_v:
            continue
        rel = os.path.relpath(p, ROOT)
        info = VALID[rel]
        ext = rel.rsplit(".", 1)[1]
        for k in ("format", "duration", "bitrate", "type"):
            if k not in e:
                fail(f"{rel}: missing '{k}'")
        if errors:
            continue
        fmt = e["format"]
        if not (isinstance(fmt, str) and (fmt.lower() == ext or ext in fmt.lower().split(",")
                                          or fmt.lower() == FMT_NAMES[ext])):
            fail(f"{rel}: format {fmt!r} not {ext}")
        if e["type"] != info["type"]:
            fail(f"{rel}: type {e['type']!r} != {info['type']}")
        d = e["duration"]
        if not isnum(d) or abs(d - info["duration"]) > 0.1:
            fail(f"{rel}: duration {d!r} != ~{info['duration']}")
        else:
            total += d
        b = e["bitrate"]
        lo, hi = bitrate_band(rel, info)
        if not isnum(b) or not (lo <= b <= hi):
            fail(f"{rel}: bitrate {b!r} not in kb/s band [{lo:.0f},{hi:.0f}]")
        if info["type"] == "video":
            for k in ("width", "height", "video_codec", "frame_rate"):
                if k not in e:
                    fail(f"{rel}: missing '{k}'")
            if e.get("width") != info["width"] or e.get("height") != info["height"]:
                fail(f"{rel}: resolution {e.get('width')}x{e.get('height')} wrong")
            if str(e.get("video_codec", "")).lower() != info["codec"]:
                fail(f"{rel}: video_codec {e.get('video_codec')!r} wrong")
            fr = e.get("frame_rate")
            if not isnum(fr) or abs(fr - info["fps"]) > 0.05:
                fail(f"{rel}: frame_rate {fr!r} != {info['fps']}")
        else:
            codec = e.get("audio_codec", e.get("codec"))
            if str(codec).lower() != info["codec"]:
                fail(f"{rel}: audio codec {codec!r} wrong (key audio_codec or codec)")
            sr = e.get("sample_rate")
            try:
                ok = float(sr) == info["sr"]
            except Exception:
                ok = False
            if not ok:
                fail(f"{rel}: sample_rate {sr!r} != {info['sr']}")
            ch = e.get("channels")
            if not isnum(ch) or ch != info["ch"]:
                fail(f"{rel}: channels {ch!r} != {info['ch']}")

    # reasons
    for e in inf:
        p = norm(e)
        if p not in exp_i:
            continue
        rel = os.path.relpath(p, ROOT)
        reasons = e.get("reasons")
        if not isinstance(reasons, list) or not reasons or not all(isinstance(r, str) and r.strip() for r in reasons):
            fail(f"{rel}: reasons must be a non-empty list of strings"); continue
        low = [r.lower() for r in reasons]
        def has(crit, r):
            return any(w in r for w in CRIT[crit])
        need = INVALID[rel]
        if need is None:
            if not any(not any(has(c, r) for c in CRIT) for r in low):
                fail(f"{rel}: no corruption/unreadable reason in {reasons}")
        else:
            if not any(has(need, r) for r in low):
                fail(f"{rel}: reasons {reasons} do not name failed criterion '{need}'")
            for c in CRIT:
                if c != need and any(has(c, r) for r in low):
                    fail(f"{rel}: reasons {reasons} wrongly cite '{c}'")
        if rel == "invalid_low_resolution.mp4" and "resolution_too_low" not in low:
            fail(f"{rel}: expected reason 'resolution_too_low' (named in task)")
        if rel in ("invalid_short_duration.mp4", "invalid_short_audio.mp3") and "duration_too_short" not in low:
            fail(f"{rel}: expected reason 'duration_too_short' (named in task)")

    # summary
    for k in ("total_files", "valid_count", "invalid_count", "total_valid_duration"):
        if not isnum(summ.get(k)):
            fail(f"summary.{k} missing or not numeric")
    if not errors:
        if summ["total_files"] != 11 or summ["valid_count"] != 5 or summ["invalid_count"] != 6:
            fail(f"summary counts wrong: {summ}")
        if summ["valid_count"] != len(vf) or summ["invalid_count"] != len(inf):
            fail("summary counts do not match lists")
        exp_total = sum(v["duration"] for v in VALID.values())
        tvd = summ["total_valid_duration"]
        if abs(tvd - exp_total) > 0.15 or abs(tvd - total) > 0.15:
            fail(f"total_valid_duration {tvd} != sum of valid durations (~{exp_total:.3f})")

    # summary.txt
    if not os.path.isfile(SUMMARY):
        fail("summary.txt missing")
    else:
        txt = open(SUMMARY, errors="replace").read()
        nums = set(re.findall(r"(?<![\d.])\d+(?![\d])", txt))
        if not txt.strip():
            fail("summary.txt empty")
        for n in ("11", "5", "6"):
            if n not in nums:
                fail(f"summary.txt lacks statistic {n} (total/valid/invalid counts)")
    return not errors

if __name__ == "__main__":
    ok = main()
    for m in errors:
        print("FAIL:", m)
    print("PASS" if ok else "RESULT: FAIL")
    sys.exit(0 if ok else 1)
