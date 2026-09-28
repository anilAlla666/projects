#!/usr/bin/env python3
"""Track 3 SC6 — minimal POOL process.

ALLOCATEs qos=pool and re-ALLOCATEs periodically (the POOL.check_resize
behaviour — re-claim freed low groups into the contiguous low prefix). Holds
the low prefix in the kmod ledger so the SC6 stranding metric measures
churn-induced gaps, not a static free region (the F2 artifact fix). No CUDA,
no green context — a pure kmod-ledger participant. Runs for SC6_POOL_SECONDS.
"""
import fcntl
import os
import struct
import time

_TYPE = ord('C')


def _IOC(d, nr, sz):
    return (d << 30) | (sz << 16) | (_TYPE << 8) | nr


CP54_ALLOCATE = _IOC(3, 13, 32)        # _IOWR(C,13,struct cipher_cp54_allocate)
QOS_POOL = 2

dur = float(os.environ.get("SC6_POOL_SECONDS", "120"))
fd = os.open("/dev/cipher", os.O_RDWR)
t0 = time.time()
n = 0
while time.time() - t0 < dur:
    buf = bytearray(struct.pack("<IIII", QOS_POOL, 0, 0, 0) + b"\0" * 16)
    try:
        fcntl.ioctl(fd, CP54_ALLOCATE, buf, True)
        n += 1
    except OSError:
        pass
    time.sleep(0.4)
os.close(fd)
print("[sc6-pool] %d ALLOCATE(pool) calls over %.0fs" % (n, dur), flush=True)
