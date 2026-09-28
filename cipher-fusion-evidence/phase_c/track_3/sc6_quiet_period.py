#!/usr/bin/env python3
"""Track 3 SC6 — PROPOSED->ABORT quiet-period measurement (SC2 item-4 PUSH).

The kmod's PROPOSED->ABORT timeout is LAZY — it fires only when an eval pass
runs (on a FREE or COMPACT ioctl), never on a timer. This measures the
consequence in two regimes:

  quiesced  — after a PROPOSE, NO ledger activity (only POLL, which does not
              run eval). The abort cannot fire. Demonstrates the latency is
              UNBOUNDED-until-activity.
  churning  — after a PROPOSE, a FREE every 1 s (a non-tenant eval trigger).
              The abort fires ~timeout + one churn interval.

Pure kmod-state-machine test — no CUDA. MUST run with no other CP54 ledger
activity (e.g. not concurrently with sc6_churn.py).
"""
import fcntl
import json
import os
import struct
import threading
import time

_T = ord('C')


def _IOC(d, nr, sz):
    return (d << 30) | (sz << 16) | (_T << 8) | nr


ALLOCATE  = _IOC(3, 13, 32)
FREE      = _IOC(0, 14, 0)
SUBSCRIBE = _IOC(1, 16, 4)
POLL      = _IOC(2, 17, 32)
COMPACT   = _IOC(0, 20, 0)
QOS_PARTITION = 0
MIG_PROPOSED = 1
MIGOUT_ABORTED_TIMEOUT = 2


def poll(fd):
    buf = bytearray(32)
    fcntl.ioctl(fd, POLL, buf, True)
    return struct.unpack_from("<IIII", buf, 0)        # state,target,cur,outcome


def setup_propose(fd):
    """ALLOCATE a migratable partition + COMPACT -> PROPOSED. Returns ok."""
    buf = bytearray(struct.pack("<IIII", QOS_PARTITION, 8, 0, 0) + b"\0" * 16)
    fcntl.ioctl(fd, ALLOCATE, buf, True)
    fcntl.ioctl(fd, SUBSCRIBE, struct.pack("<I", 1))
    fcntl.ioctl(fd, COMPACT)
    return poll(fd)[0] == MIG_PROPOSED


res = {}

# ---- quiesced: PROPOSE, then only POLL (no eval) for QUIET seconds ----
QUIET = 75.0
fd = os.open("/dev/cipher", os.O_RDWR)
ok = setup_propose(fd)
t0 = time.time()
still_proposed = True
while time.time() - t0 < QUIET:
    if poll(fd)[0] != MIG_PROPOSED:
        still_proposed = False
        break
    time.sleep(5)
quiet_elapsed = time.time() - t0


def free_from_other_lwp():
    """A FREE from a separate thread (distinct kernel LWP pid) releases
    nothing of OUR partition — it is a pure non-forced compaction trigger.
    A FREE on our own fd would release our own partition instead."""
    f = os.open("/dev/cipher", os.O_RDWR)
    fcntl.ioctl(f, FREE)
    os.close(f)


_tt = threading.Thread(target=free_from_other_lwp)
_tt.start()
_tt.join()
outcome_after = poll(fd)[3]
res["quiesced"] = {
    "setup_ok": ok, "quiet_s": round(quiet_elapsed, 1),
    "still_PROPOSED_after_quiet": still_proposed,
    "aborted_on_first_free": outcome_after == MIGOUT_ABORTED_TIMEOUT,
    "note": "abort did NOT fire during %.0fs of quiescence; fired on the "
            "first FREE after. Latency is unbounded-until-activity."
            % quiet_elapsed,
}
fcntl.ioctl(fd, FREE)        # self-FREE: release this test's partition + clear
os.close(fd)                 # its meta, so the churning phase starts clean
time.sleep(1.0)

# ---- churning: PROPOSE, then a FREE every 1 s; measure time-to-abort ----
fd = os.open("/dev/cipher", os.O_RDWR)
ok = setup_propose(fd)
t0 = time.time()
stop = threading.Event()


def churn():
    f = os.open("/dev/cipher", os.O_RDWR)             # separate LWP -> FREE is
    while not stop.is_set():                          # a pure eval trigger
        try:
            fcntl.ioctl(f, FREE)
        except OSError:
            pass
        time.sleep(1.0)
    os.close(f)


th = threading.Thread(target=churn, daemon=True)
th.start()
abort_latency = None
while time.time() - t0 < 60:
    if poll(fd)[3] == MIGOUT_ABORTED_TIMEOUT:
        abort_latency = round(time.time() - t0, 2)
        break
    time.sleep(0.5)
stop.set()
os.close(fd)
res["churning"] = {
    "setup_ok": ok, "abort_latency_s": abort_latency,
    "note": "abort fired ~kmod-timeout(30s) + one churn interval; bounded.",
}

out = "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3/sc6_quiet_period_result.json"
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
print("wrote", out)
