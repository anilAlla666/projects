"""Phase 4 baseline post-process.

For each completed WL, reads:
  - expected/<wl>_p41_baseline.json  (already has mfu_pct_median etc.)
  - expected/<wl>_baseline.csv       (per-second CSV, now has device_power_w)
  - /tmp/cipher_tenant_<tenant>_progress  (run_for_duration progress file)
  - verify_run.py output (stored in JSON's "verify" field)

Writes back the JSON with extra fields:
  - tokens_per_sec        (total_tokens / end_t from progress file)
  - ops_per_sec           (total_units / end_t from progress file)
  - watts_avg             (median device_power_w in steady-state window)
  - tokens_per_watt       (tokens_per_sec / watts_avg, if both > 0)
  - ops_per_watt          (ops_per_sec / watts_avg, if both > 0)
  - baseline_valid        (bool)
  - baseline_invalid_reason (str, if any)
  - unit_label            ("tokens" or otherwise from progress file)

Idempotent: safe to re-run on an already-augmented JSON.

Usage:
    python3 phase4_baseline_postprocess.py WL01
    python3 phase4_baseline_postprocess.py WL05    # multi-tenant aggregate
"""
import csv
import json
import os
import re
import statistics
import sys


EXPECTED_DIR = "/home/ubuntu/cipher_workloads/expected"


def parse_progress_file(path):
    """Return (start_t, end_t, iters, units, tokens, unit_label, end_line)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = f.read().strip().splitlines()
    except Exception:
        return None
    if not lines:
        return None

    start_t = 0.0
    end_t = 0.0
    iters = 0
    units = 0
    tokens = 0
    unit_label = "ops"
    end_line = ""

    for line in lines:
        # Format: "<wl> <tenant> start t=<float>"
        m = re.match(r"\S+\s+\S+\s+start\s+t=(\S+)", line)
        if m:
            try:
                start_t = float(m.group(1))
            except ValueError:
                pass
            continue
        # Format: "<wl> <tenant> end t=<float>s iters=<int> <unit>=<int> tokens=<int>"
        m = re.match(
            r"\S+\s+\S+\s+end\s+t=(\S+?)s?\s+iters=(\d+)\s+(\w+)=(\d+)\s+tokens=(\d+)",
            line,
        )
        if m:
            try:
                end_t = float(m.group(1))
                iters = int(m.group(2))
                unit_label = m.group(3)
                units = int(m.group(4))
                tokens = int(m.group(5))
                end_line = line
            except ValueError:
                pass
            continue
        # Deferred multi-GPU short-circuit
        if "deferred-multi-gpu" in line:
            end_line = line

    return {
        "start_t": start_t,
        "end_t": end_t,
        "iters": iters,
        "units": units,
        "tokens": tokens,
        "unit_label": unit_label,
        "end_line": end_line,
    }


def median_watts(csv_path, t_start, t_end):
    if not os.path.exists(csv_path):
        return None, 0
    watts = []
    try:
        with open(csv_path) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                # tolerate missing column on legacy CSVs
                try:
                    ts = int(row["timestamp_s"])
                except (KeyError, ValueError):
                    continue
                if ts < t_start or ts > t_end:
                    continue
                v = row.get("device_power_w")
                if v is None or v == "" or v == "0":
                    continue
                try:
                    watts.append(float(v))
                except ValueError:
                    pass
    except Exception:
        return None, 0
    if not watts:
        return None, 0
    return round(statistics.median(watts), 2), len(watts)


def median_watts_wl05(t_start, t_end):
    """For WL05, the CSVs are per-child; gpu power is global, so grab one child."""
    for tenant in [f"wl05_t{i}" for i in range(1, 9)]:
        csv_path = os.path.join(EXPECTED_DIR, f"wl05_{tenant}_baseline.csv")
        w, n = median_watts(csv_path, t_start, t_end)
        if w is not None:
            return w, n
    # fall back to global wl05 csv if present
    csv_path = os.path.join(EXPECTED_DIR, "wl05_baseline.csv")
    return median_watts(csv_path, t_start, t_end)


def aggregate_progress_wl05():
    """WL05 has 8 child tenants. Sum tokens & report aggregate tokens_per_sec
    using max end_t (children should be within ~seconds of each other)."""
    sum_tokens = 0
    sum_units = 0
    max_end = 0.0
    children_complete = 0
    children_total = 0
    unit_label = "ops"
    for i in range(1, 9):
        tname = f"wl05_t{i}"
        path = f"/tmp/cipher_tenant_{tname}_progress"
        children_total += 1
        p = parse_progress_file(path)
        if not p:
            continue
        if p["end_t"] > 0 and p["units"] > 0:
            children_complete += 1
            sum_tokens += p["tokens"]
            sum_units += p["units"]
            max_end = max(max_end, p["end_t"])
            unit_label = p["unit_label"]
    return {
        "tokens": sum_tokens,
        "units": sum_units,
        "end_t": max_end,
        "unit_label": unit_label,
        "children_complete": children_complete,
        "children_total": children_total,
    }


def postprocess(wl_id):
    wl_low = wl_id.lower()
    json_path = os.path.join(EXPECTED_DIR, f"{wl_low}_p41_baseline.json")
    csv_path = os.path.join(EXPECTED_DIR, f"{wl_low}_baseline.csv")

    if not os.path.exists(json_path):
        print(f"[postprocess] {wl_id}: no JSON at {json_path}", file=sys.stderr)
        return 1
    with open(json_path) as f:
        j = json.load(f)

    duration = float(j.get("duration_s", 600))
    t_start = 180.0
    t_end = min(duration, 600.0)

    # ---- tokens / ops / unit_label / end_t ----
    if wl_id.upper() == "WL05":
        agg = aggregate_progress_wl05()
        end_t = agg["end_t"] or duration
        tokens = agg["tokens"]
        units = agg["units"]
        unit_label = agg["unit_label"]
        j["wl05_children_complete"] = agg["children_complete"]
        j["wl05_children_total"] = agg["children_total"]
        watts_avg, watts_samples = median_watts_wl05(int(t_start), int(t_end))
    else:
        tenant = j.get("tenant_id") or f"{wl_low}_baseline"
        progress_path = f"/tmp/cipher_tenant_{tenant}_progress"
        p = parse_progress_file(progress_path)
        if p is None:
            print(f"[postprocess] {wl_id}: no progress file at {progress_path}", file=sys.stderr)
            tokens = 0
            units = 0
            end_t = duration
            unit_label = "ops"
        else:
            tokens = p["tokens"]
            units = p["units"]
            end_t = p["end_t"] or duration
            unit_label = p["unit_label"]
        watts_avg, watts_samples = median_watts(csv_path, int(t_start), int(t_end))

    tokens_per_sec = round(tokens / max(end_t, 1e-6), 2) if end_t > 0 else 0.0
    ops_per_sec = round(units / max(end_t, 1e-6), 2) if end_t > 0 else 0.0

    j["unit_label"] = unit_label
    j["tokens"] = tokens
    j["ops"] = units
    j["tokens_per_sec"] = tokens_per_sec
    j["ops_per_sec"] = ops_per_sec
    j["watts_avg"] = watts_avg if watts_avg is not None else 0.0
    j["watts_samples_in_window"] = watts_samples

    if watts_avg and watts_avg > 0:
        if tokens_per_sec > 0:
            j["tokens_per_watt"] = round(tokens_per_sec / watts_avg, 4)
        else:
            j["tokens_per_watt"] = 0.0
        if ops_per_sec > 0:
            j["ops_per_watt"] = round(ops_per_sec / watts_avg, 4)
        else:
            j["ops_per_watt"] = 0.0
    else:
        j["tokens_per_watt"] = None
        j["ops_per_watt"] = None

    # ---- baseline-valid / invalid markers ----
    verify = j.get("verify", "") or ""
    invalid_reason = ""
    valid = True
    if verify.startswith("invalid"):
        valid = False
        invalid_reason = verify
    elif "deferred-multi-gpu" in verify:
        valid = True
        invalid_reason = "deferred-multi-gpu (single-GPU pod)"
    # Additional sanity gates
    if watts_avg is None or watts_avg == 0:
        invalid_reason = (invalid_reason + "; " if invalid_reason else "") + "no watts samples in window"
    if j.get("samples_in_window", 0) < 30 and "deferred-multi-gpu" not in verify:
        invalid_reason = (invalid_reason + "; " if invalid_reason else "") + f"only {j.get('samples_in_window')} samples in window"
        valid = False
    if wl_id.upper() != "WL05" and tokens_per_sec == 0 and unit_label in ("tokens",) and "deferred-multi-gpu" not in verify:
        invalid_reason = (invalid_reason + "; " if invalid_reason else "") + "tokens_per_sec=0"
    j["baseline_valid"] = valid
    j["baseline_invalid_reason"] = invalid_reason

    with open(json_path, "w") as f:
        json.dump(j, f, indent=2)
    print(json.dumps({
        "wl_id": j.get("wl_id", wl_id),
        "steady_state_mfu_pct": j.get("mfu_pct_median"),
        "tokens_per_sec": tokens_per_sec,
        "ops_per_sec": ops_per_sec,
        "watts_avg": j.get("watts_avg"),
        "tokens_per_watt": j.get("tokens_per_watt"),
        "ops_per_watt": j.get("ops_per_watt"),
        "baseline_valid": j["baseline_valid"],
        "baseline_invalid_reason": j["baseline_invalid_reason"],
        "kmod_version": j.get("kmod_version"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: phase4_baseline_postprocess.py WL01 [WL02 ...]", file=sys.stderr)
        sys.exit(2)
    rc = 0
    for wl in sys.argv[1:]:
        r = postprocess(wl)
        if r != 0:
            rc = r
    sys.exit(rc)
