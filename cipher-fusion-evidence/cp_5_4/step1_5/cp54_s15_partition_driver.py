#!/usr/bin/env python3
"""CP 5.4 Step 1.5B — partition-driver subprocess for the real-resize bench.

ALLOCATEs a PARTITION via the CP 5.4 ioctl (which shrinks the POOL in the kmod
ledger), prints ALLOCATED, then holds the allocation until it reads 'FREE' on
stdin, then FREEs and exits. Raw ioctl, no libcipher_rt — the kmod ledger does
not care which process calls; a raw ALLOCATE(PARTITION) drives the exact same
pool-shrink path a real libcipher_rt tenant triggers.

Throwaway Step 1.5 measurement harness — not a campaign anchor, not substrate.
"""
import fcntl
import os
import struct
import sys

_TYPE = ord('C')


def _IOC(direction, nr, size):
    return (direction << 30) | (size << 16) | (_TYPE << 8) | nr


_CP54_STRUCT = 32
CP54_ALLOCATE = _IOC(3, 13, _CP54_STRUCT)   # _IOWR
CP54_FREE = _IOC(0, 14, 0)                  # _IO
QOS_PARTITION = 0


def main():
    sm_count = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    fd = os.open("/dev/cipher", os.O_RDWR)

    buf = bytearray(_CP54_STRUCT)
    struct.pack_into("<II", buf, 0, QOS_PARTITION, sm_count)  # qos, sm_count
    fcntl.ioctl(fd, CP54_ALLOCATE, buf, True)
    _, _, grp_mask, grp_count = struct.unpack_from("<IIII", buf, 0)
    sys.stdout.write("ALLOCATED mask=0x%04x count=%d\n" % (grp_mask, grp_count))
    sys.stdout.flush()

    for line in sys.stdin:                       # hold until told to free
        if line.strip() == "FREE":
            break

    fcntl.ioctl(fd, CP54_FREE)
    os.close(fd)
    sys.stdout.write("FREED\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
