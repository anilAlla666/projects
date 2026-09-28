#!/usr/bin/env python3
"""CP 3.4 ingest agent.

Reads /proc/cipher/flops (CP 3.3 telemetry surface, kmod 0.4.7) once per
second and INSERTs one cipher.device_state row plus one cipher.tenant_state
row per active tenant into ClickHouse over the HTTP interface (:8123).

Pure stdlib. No daemon is modified; this only reads an existing proc file.
"""
import sys, time, signal, urllib.request, urllib.parse
from datetime import datetime

PROC = "/proc/cipher/flops"
CH_URL = "http://127.0.0.1:8123/"

_run = True
def _stop(*_):
    global _run
    _run = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def ch(query, body=b""):
    """POST a query to ClickHouse HTTP; return response text."""
    url = CH_URL + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode().strip()


def kv(token):
    """'sm_util_pct=100' -> ('sm_util_pct', '100')."""
    k, _, v = token.partition("=")
    return k, v


def parse(text):
    """Return (device_dict_or_None, [tenant_dict, ...])."""
    dev, tenants, in_tenants = None, [], False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("device") and "status=" in line:
            d = dict(kv(t) for t in line.split()[1:] if "=" in t)
            dev = {
                "tensor_tflops": float(d.get("tensor_tflops", 0)),
                "mfu_pct":       float(d.get("mfu_pct", 0)),
                "sm_clock_mhz":  int(float(d.get("sm_clock_mhz", 0))),
                "sm_util_pct":   int(float(d.get("sm_util_pct", 0))),
                "power_w":       float(d.get("power_w", 0)),
                "status":        d.get("status", "unknown"),
            }
            continue
        if line.startswith("device"):          # ring-metadata line — skip
            continue
        if line.startswith("PID"):             # tenant table header
            in_tenants = True
            continue
        if in_tenants:
            f = line.split()
            if len(f) < 6:
                continue
            pid, tgid, name = f[0], f[1], f[2]
            tenant = name if name != "-" else "pid:%s" % pid
            tenants.append({
                "tenant": tenant, "pid": int(pid), "tgid": int(tgid),
                "launch_delta": int(f[3]),
                "flops_per_s":  float(f[4]),
                "mfu_pct":      float(f[5]),
            })
    return dev, tenants


def main():
    print("[ingest] CP 3.4 agent — %s -> ClickHouse %s" % (PROC, CH_URL),
          flush=True)
    ticks = dev_rows = tenant_rows = skipped = 0
    while _run:
        t = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.") + \
             "%03d" % (int((t % 1) * 1000))
        try:
            with open(PROC) as fh:
                dev, tenants = parse(fh.read())
        except Exception as e:
            print("[ingest] proc read failed: %s" % e, flush=True)
            time.sleep(1.0); continue

        if dev is None or dev["status"] != "live":
            skipped += 1
        else:
            ch("INSERT INTO cipher.device_state VALUES "
               "('%s',%g,%g,%d,%d,%g)" % (ts, dev["tensor_tflops"],
               dev["mfu_pct"], dev["sm_clock_mhz"], dev["sm_util_pct"],
               dev["power_w"]))
            dev_rows += 1
            if tenants:
                vals = ",".join(
                    "('%s','%s',%d,%d,%d,%g,%g)" % (ts, x["tenant"],
                     x["pid"], x["tgid"], x["launch_delta"],
                     x["flops_per_s"], x["mfu_pct"]) for x in tenants)
                ch("INSERT INTO cipher.tenant_state VALUES " + vals)
                tenant_rows += len(tenants)
        ticks += 1
        if ticks % 15 == 0:
            print("[ingest] ticks=%d device_rows=%d tenant_rows=%d "
                  "stale_skipped=%d" % (ticks, dev_rows, tenant_rows,
                  skipped), flush=True)
        dt = 1.0 - (time.time() - t)
        if dt > 0:
            time.sleep(dt)
    print("[ingest] stopped — ticks=%d device_rows=%d tenant_rows=%d "
          "stale_skipped=%d" % (ticks, dev_rows, tenant_rows, skipped),
          flush=True)


if __name__ == "__main__":
    main()
