#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""
cipher-exporter -- Prometheus exporter for cipher_kmod.

Reads /proc/cipher/gpu_state and /proc/cipher/stats on each GET /metrics,
emits the Prometheus text exposition format. Pure Python 3 stdlib --
no flask, no prometheus_client. Single-threaded http.server is plenty
for the typical 15-second Prometheus scrape cadence.

Phase 3 Layer D. Pairs with the daemon (Task 4) and libcipher_v2's
CUPTI callback (Task 5).
"""

import argparse
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

CIPHER_PROC_DIR        = "/proc/cipher"
CIPHER_PROC_GPU_STATE  = "/proc/cipher/gpu_state"
CIPHER_PROC_STATS      = "/proc/cipher/stats"


# ---------------------------------------------------------------- helpers

def _label_escape(s):
    """Prometheus label-value escaping: backslash, newline, double-quote."""
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return None


# ---------------------------------------------------------------- gpu_state parser

# Match lines like:
#   power=350000 mW (350.000 W)
#   temp=65 C
#   sm_clock=1980 MHz, mem_clock=2619 MHz
#   sm_util=75%, mem_util=40%
#   fb_used=8192 MB
#   age=0.24 s since last update
_RE_POWER     = re.compile(r"^power=(\d+) mW")
_RE_TEMP      = re.compile(r"^temp=(\d+) C")
_RE_CLOCKS    = re.compile(r"^sm_clock=(\d+) MHz, mem_clock=(\d+) MHz")
_RE_UTILS     = re.compile(r"^sm_util=(\d+)%, mem_util=(\d+)%")
_RE_FB_USED   = re.compile(r"^fb_used=(\d+) MB")
_RE_AGE       = re.compile(r"^age=([\d.]+) s")
_NO_SAMPLE    = "no sample yet"

def parse_gpu_state(text):
    """Parse gpu_state text. Returns dict of metric_name -> float.
    If 'no sample yet', returns all zeros."""
    out = {
        "power_watts":      0.0,
        "temp_celsius":     0.0,
        "sm_clock_mhz":     0.0,
        "mem_clock_mhz":    0.0,
        "sm_util_pct":      0.0,
        "mem_util_pct":     0.0,
        "fb_used_mb":       0.0,
        "state_age_seconds":  0.0,
        "has_sample":       0,
    }
    if text is None or _NO_SAMPLE in text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if m := _RE_POWER.match(line):
            out["power_watts"] = int(m.group(1)) / 1000.0
        elif m := _RE_TEMP.match(line):
            out["temp_celsius"] = float(m.group(1))
        elif m := _RE_CLOCKS.match(line):
            out["sm_clock_mhz"]  = float(m.group(1))
            out["mem_clock_mhz"] = float(m.group(2))
        elif m := _RE_UTILS.match(line):
            out["sm_util_pct"]  = float(m.group(1))
            out["mem_util_pct"] = float(m.group(2))
        elif m := _RE_FB_USED.match(line):
            out["fb_used_mb"] = float(m.group(1))
        elif m := _RE_AGE.match(line):
            out["state_age_seconds"] = float(m.group(1))
    out["has_sample"] = 1
    return out


# ---------------------------------------------------------------- stats parser

# Header anchor for the per-PID block:
_PER_PID_HEADER   = "Per-PID summary"
_PER_TGID_HEADER  = "Per-TGID summary"
_RE_UPTIME        = re.compile(r"uptime=\d+ jiffies \(([\d.]+) s\)")

def parse_stats(text):
    """Returns (uptime_seconds_or_None, list of per-pid row dicts).
    Each row dict has fields: pid, tgid, comm, tenant, total, age_s,
    optional sm_util_pct, mem_util_pct, launches, age_tm (numeric or None)."""
    uptime = None
    rows = []
    if text is None:
        return uptime, rows

    # Uptime from the banner line
    for line in text.splitlines()[:5]:
        if m := _RE_UPTIME.search(line):
            uptime = float(m.group(1))
            break

    # Locate the Per-PID block
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(_PER_PID_HEADER))
    except StopIteration:
        return uptime, rows

    try:
        end = next(i for i in range(start + 1, len(lines))
                   if lines[i].startswith(_PER_TGID_HEADER))
    except StopIteration:
        end = len(lines)

    # Skip the header line (line right after start) and the column-name line
    # The block is:
    #   Per-PID summary (top 16 by total, N PIDs total):
    #     PID      TGID     ...header...
    #     <rows>
    for raw in lines[start + 2 : end]:
        line = raw.rstrip()
        if not line.strip():
            continue
        fields = line.split()
        # Expect 16 fields: pid tgid comm tenant total rm_ctrl rm_alloc rm_free
        #                   map_mem other err age_s sm mem launches age_tm
        if len(fields) < 16:
            continue
        try:
            pid     = fields[0]
            tgid    = fields[1]
            comm    = fields[2]
            tenant  = fields[3]
            total   = int(fields[4])
            age_s   = float(fields[11])
            sm_str, mem_str, launches_str, age_tm_str = fields[12:16]
            row = {
                "pid":    pid,
                "tgid":   tgid,
                "comm":   comm,
                "tenant": tenant,
                "total":  total,
                "age_s":  age_s,
                "sm_util_pct":  None,
                "mem_util_pct": None,
                "launches":     None,
                "age_tm":       None,
            }
            if sm_str       != "-":  row["sm_util_pct"]  = float(sm_str)
            if mem_str      != "-":  row["mem_util_pct"] = float(mem_str)
            if launches_str != "-":  row["launches"]     = int(launches_str)
            if age_tm_str   != "-":  row["age_tm"]       = float(age_tm_str)
            rows.append(row)
        except (ValueError, IndexError):
            continue

    return uptime, rows


# ---------------------------------------------------------------- exposition

def render_metrics():
    """Render the full Prometheus exposition body as a string."""
    out = []
    e = out.append

    # ---- module presence
    if not os.path.isdir(CIPHER_PROC_DIR):
        e("# HELP cipher_module_loaded 1 if /proc/cipher exists, 0 otherwise.")
        e("# TYPE cipher_module_loaded gauge")
        e("cipher_module_loaded 0")
        return "\n".join(out) + "\n"

    e("# HELP cipher_module_loaded 1 if /proc/cipher exists, 0 otherwise.")
    e("# TYPE cipher_module_loaded gauge")
    e("cipher_module_loaded 1")

    # ---- device-wide (gpu_state)
    gs_text = _read_text(CIPHER_PROC_GPU_STATE)
    gs = parse_gpu_state(gs_text)

    e("# HELP cipher_gpu_state_has_sample 1 if cipher-gpustate has submitted at least once.")
    e("# TYPE cipher_gpu_state_has_sample gauge")
    e(f'cipher_gpu_state_has_sample{{device="0"}} {gs["has_sample"]}')

    for metric, value in [
        ("cipher_gpu_power_watts",        gs["power_watts"]),
        ("cipher_gpu_temp_celsius",       gs["temp_celsius"]),
        ("cipher_gpu_sm_clock_mhz",       gs["sm_clock_mhz"]),
        ("cipher_gpu_mem_clock_mhz",      gs["mem_clock_mhz"]),
        ("cipher_gpu_sm_util_pct",        gs["sm_util_pct"]),
        ("cipher_gpu_mem_util_pct",       gs["mem_util_pct"]),
        ("cipher_gpu_fb_used_mb",         gs["fb_used_mb"]),
        ("cipher_gpu_state_age_seconds",  gs["state_age_seconds"]),
    ]:
        e(f"# TYPE {metric} gauge")
        e(f'{metric}{{device="0"}} {value:g}')

    # ---- stats (uptime + per-PID/tenant)
    stats_text = _read_text(CIPHER_PROC_STATS)
    uptime, rows = parse_stats(stats_text)

    e("# HELP cipher_module_uptime_seconds Seconds since cipher_kmod load (jiffies/HZ).")
    e("# TYPE cipher_module_uptime_seconds gauge")
    e(f"cipher_module_uptime_seconds {uptime if uptime is not None else 0:g}")

    # Per-tenant: only rows with TENANT != "-" (system/daemon rows excluded).
    tenant_rows = [r for r in rows if r["tenant"] != "-"]

    if tenant_rows:
        for metric, key, mtype in [
            ("cipher_tenant_ioctl_total",          "total",        "counter"),
            ("cipher_tenant_sm_util_pct",          "sm_util_pct",  "gauge"),
            ("cipher_tenant_mem_util_pct",         "mem_util_pct", "gauge"),
            ("cipher_tenant_launches_total",       "launches",     "counter"),
            ("cipher_tenant_age_seconds",          "age_s",        "gauge"),
            ("cipher_tenant_telemetry_age_seconds","age_tm",       "gauge"),
        ]:
            e(f"# TYPE {metric} {mtype}")
            for r in tenant_rows:
                v = r[key]
                if v is None:
                    continue
                labels = (
                    f'tenant="{_label_escape(r["tenant"])}",'
                    f'pid="{r["pid"]}",'
                    f'comm="{_label_escape(r["comm"])}"'
                )
                e(f'{metric}{{{labels}}} {v:g}')

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- HTTP server

class Handler(BaseHTTPRequestHandler):
    server_version  = "cipher-exporter/0.1"
    sys_version     = ""
    cipher_debug    = False  # set via subclass per-instance below

    def log_message(self, fmt, *args):
        if self.cipher_debug:
            sys.stderr.write("cipher-exporter: %s - %s\n" %
                             (self.address_string(), fmt % args))

    def _send_text(self, code, body, ctype="text/plain; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            try:
                body = render_metrics()
                self._send_text(200, body, "text/plain; version=0.0.4")
            except Exception as ex:
                self._send_text(500, f"render error: {ex}\n")
        elif self.path == "/health":
            if os.path.isdir(CIPHER_PROC_DIR):
                self._send_text(200, "ok\n")
            else:
                self._send_text(503, "no cipher_kmod\n")
        else:
            self._send_text(404, "not found\n")


def main():
    p = argparse.ArgumentParser(description="cipher_kmod Prometheus exporter")
    p.add_argument("--port", type=int, default=9402)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--debug", action="store_true",
                   help="log each request to stderr")
    args = p.parse_args()

    Handler.cipher_debug = args.debug

    httpd = HTTPServer((args.bind, args.port), Handler)
    sys.stderr.write(
        f"cipher-exporter: listening on {args.bind}:{args.port} "
        f"(debug={args.debug})\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("cipher-exporter: shutting down\n")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
