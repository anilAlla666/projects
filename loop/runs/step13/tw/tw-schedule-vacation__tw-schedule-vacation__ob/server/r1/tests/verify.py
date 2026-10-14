#!/usr/bin/env python3
"""Verifier for the 7-day vacation planning task. Exit 0 iff every check passes."""
import json, os, re, subprocess, sys, tempfile, shutil
from datetime import date, timedelta
from pathlib import Path

APP = Path(os.environ.get("APP_DIR", "/app"))
HERE = Path(__file__).resolve().parent
FRIENDS = ["alice", "bob", "charlie", "dana"]
EXPECTED_WINDOW = (date(2025, 7, 28), date(2025, 8, 3))

failures = []

def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:  # noqa
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        failures.append(name)

def drange(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)

def D(s):
    return date.fromisoformat(s)

def days(*ranges):
    out = set()
    for a, b in ranges:
        out |= set(drange(D(a), D(b)))
    return out

# Ground truth derived directly from get_schedule.py output.
# must_free: days the schedule clearly leaves free; must_busy: days it marks unavailable/working.
TRUTH = {
    "alice": dict(
        busy=days(("2025-01-01", "2025-06-15"), ("2025-06-20", "2025-06-25"),
                  ("2025-07-15", "2025-07-25"), ("2025-08-05", "2025-12-31")),
        free=days(("2025-06-16", "2025-06-19"), ("2025-06-26", "2025-07-14"),
                  ("2025-07-26", "2025-08-04"))),
    "bob": dict(
        busy=days(("2025-06-01", "2025-06-14"), ("2025-06-29", "2025-07-14"),
                  ("2025-08-05", "2025-08-05")),
        free=days(("2025-06-15", "2025-06-28"), ("2025-07-15", "2025-08-04"))),
    "charlie": dict(
        busy=days(("2025-01-01", "2025-05-31"), ("2025-06-15", "2025-06-30"),
                  ("2025-07-01", "2025-07-20"), ("2025-07-26", "2025-07-27"),
                  ("2025-08-04", "2025-08-31")),
        free=days(("2025-06-01", "2025-06-14"), ("2025-07-21", "2025-07-25"),
                  ("2025-07-28", "2025-08-03"))),
    "dana": dict(
        busy=days(("2025-01-01", "2025-06-10"), ("2025-06-11", "2025-06-18"),
                  ("2025-06-25", "2025-07-15"), ("2025-08-04", "2025-08-20"),
                  ("2025-08-25", "2025-12-31")),
        free=days(("2025-06-19", "2025-06-24"), ("2025-07-16", "2025-08-03"),
                  ("2025-08-21", "2025-08-24"))),
}

def tool(name):
    p = HERE / name
    return p if p.exists() else APP / name

# 1. schedule files
def t_schedules():
    for f in FRIENDS:
        p = APP / f"{f}_schedule.txt"
        assert p.is_file(), f"{p} missing"
        r = subprocess.run([sys.executable, str(tool("get_schedule.py")), f],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        assert p.read_text().strip() == r.stdout.strip(), f"{p.name} differs from get_schedule.py output"

# 2. availability json
def load_avail():
    data = json.loads((APP / "all_availability.json").read_text())
    assert isinstance(data, dict), "top level must be an object"
    assert set(data.keys()) == set(FRIENDS), f"keys must be exactly {FRIENDS}, got {list(data)}"
    return data

def t_avail_schema():
    data = load_avail()
    pat = re.compile(r"\d{4}-\d{2}-\d{2}")
    for f, periods in data.items():
        assert isinstance(periods, list) and periods, f"{f}: must be a non-empty list"
        for p in periods:
            assert isinstance(p, dict) and set(p.keys()) >= {"start", "end"}, f"{f}: bad period {p}"
            for k in ("start", "end"):
                assert isinstance(p[k], str) and pat.fullmatch(p[k]), f"{f}: bad date {p[k]!r}"
                D(p[k])
            assert D(p["start"]) <= D(p["end"]), f"{f}: start after end in {p}"

def t_avail_content():
    data = load_avail()
    for f, periods in data.items():
        have = set()
        for p in periods:
            have |= set(drange(D(p["start"]), D(p["end"])))
        bad = sorted(have & TRUTH[f]["busy"])
        assert not bad, f"{f}: listed as available on busy days, e.g. {bad[:3]}"
        missing = sorted(TRUTH[f]["free"] - have)
        assert not missing, f"{f}: free days missing from availability, e.g. {missing[:3]} ({len(missing)} total)"

# 3. overlap results
def t_overlap():
    ov = json.loads((APP / "overlap_results.json").read_text())
    for k in ("vacation_duration", "overlapping_windows", "total_windows_found"):
        assert k in ov, f"missing {k}"
    assert type(ov["vacation_duration"]) is int and ov["vacation_duration"] == 7, "vacation_duration must be 7"
    assert isinstance(ov["overlapping_windows"], list)
    assert type(ov["total_windows_found"]) is int and ov["total_windows_found"] == len(ov["overlapping_windows"])
    full = [w for w in ov["overlapping_windows"]
            if isinstance(w, dict) and set(w.get("available_friends", [])) == set(FRIENDS)]
    assert full, "no window with all four friends"
    assert any((w.get("start"), w.get("end")) == ("2025-07-28", "2025-08-03") for w in full), \
        "all-four window 2025-07-28..2025-08-03 not present"

def t_overlap_from_tool():
    ov = json.loads((APP / "overlap_results.json").read_text())
    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(APP / "all_availability.json", Path(tmp) / "avail.json")
        r = subprocess.run([sys.executable, str(tool("find_overlap.py")), "7", "avail.json"],
                           cwd=tmp, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        exp = json.loads((Path(tmp) / "overlap_results.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert ov == exp, "overlap_results.json is not find_overlap.py's output for all_availability.json with 7 days"

# 4. final answer
def t_answer():
    raw = (APP / "vacation_dates.txt").read_text()
    lines = [l for l in raw.splitlines() if l.strip()]
    assert len(lines) == 1 and len(raw.strip().splitlines()) == 1, "must contain exactly one line"
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", lines[0].strip())
    assert m, f"line must be 'YYYY-MM-DD to YYYY-MM-DD', got {lines[0]!r}"
    assert (D(m.group(1)), D(m.group(2))) == EXPECTED_WINDOW, f"wrong window {lines[0]!r}"

for n, fn in [("schedule_files", t_schedules), ("availability_schema", t_avail_schema),
              ("availability_content", t_avail_content), ("overlap_structure", t_overlap),
              ("overlap_generated_by_tool", t_overlap_from_tool), ("vacation_dates", t_answer)]:
    check(n, fn)

print("FAILED:" if failures else "ALL PASSED", failures)
sys.exit(1 if failures else 0)
