#!/usr/bin/env python3
"""Track 2 SC5 — kmod weight-arena registry unit tests.

Exercises the 4 ioctls (REGISTER/IMPORT/LEAVE/QUERY) at the kmod boundary,
with no CUDA and no model — the registry is pure kernel fd-table work, so a
plain temp-file fd stands in for the exported VMM POSIX fd (the kmod fget's
whatever fd it is handed; it never interprets it as a CUDA handle).

Run while cipher_kmod (SC5) is loaded. Registered arenas keep this process
as their producer pid, so they survive until the test LEAVEs them or the
process exits (the 5 s liveness reaper then collects them).
"""
import errno
import json
import os
import sys

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
import sc5_arena_ioctl as aioctl                              # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": str(detail)})
    print("%-44s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def dummy_fd():
    """A real fd for REGISTER to fget — its contents are irrelevant."""
    return os.open("/dev/null", os.O_RDONLY)


def main():
    # T1 — QUERY succeeds on a (possibly empty) registry
    base_arenas = aioctl.query()
    check("T1_query_ok", isinstance(base_arenas, list),
          "%d pre-existing arenas" % len(base_arenas))

    # T2 — REGISTER returns a fresh id; QUERY reflects it
    fd = dummy_fd()
    blob = b'{"sc5":"unit","n":1}'
    aid = aioctl.register(fd, 0xdead0000, 4096, blob)
    os.close(fd)
    q = {a["arena_id"]: a for a in aioctl.query()}
    check("T2_register_returns_id", aid > 0, "arena_id=%d" % aid)
    check("T2_query_shows_arena", aid in q, q.get(aid))
    check("T2_producer_is_us",
          aid in q and q[aid]["producer_pid"] == os.getpid(),
          "producer_pid=%s pid=%d" % (q.get(aid, {}).get("producer_pid"),
                                      os.getpid()))
    check("T2_zero_consumers_initially",
          aid in q and q[aid]["n_consumers"] == 0)

    # T3 — IMPORT returns matching base/size and the exact blob
    ifd, ibase, isize, iblob = aioctl.import_arena(aid)
    check("T3_import_base_matches", ibase == 0xdead0000, hex(ibase))
    check("T3_import_size_matches", isize == 4096, isize)
    check("T3_import_blob_roundtrip", iblob == blob, iblob)
    check("T3_import_fd_valid", ifd >= 0, "fd=%d" % ifd)
    os.close(ifd)
    q = {a["arena_id"]: a for a in aioctl.query()}
    check("T3_consumer_count_incremented",
          q.get(aid, {}).get("n_consumers") == 1,
          q.get(aid, {}).get("n_consumers"))

    # T4 — IMPORT of a non-existent arena id => ENOENT
    bad_id = max([a["arena_id"] for a in aioctl.query()] + [0]) + 9999
    try:
        aioctl.import_arena(bad_id)
        check("T4_import_missing_enoent", False, "no error raised")
    except OSError as e:
        check("T4_import_missing_enoent", e.errno == errno.ENOENT,
              "errno=%d (%s)" % (e.errno, errno.errorcode.get(e.errno)))

    # T5 — multi-consumer: three IMPORTs => n_consumers == 3
    fds = [aioctl.import_arena(aid)[0] for _ in range(3)]
    q = {a["arena_id"]: a for a in aioctl.query()}
    # 1 from T3 + 3 here = 4 total consumer slots claimed by this pid
    check("T5_multi_consumer_count",
          q.get(aid, {}).get("n_consumers") == 4,
          "n_consumers=%s" % q.get(aid, {}).get("n_consumers"))
    for f in fds:
        os.close(f)

    # T6 — IMPORT with blob_cap smaller than the stored blob => ENOSPC
    try:
        aioctl.import_arena(aid, blob_cap=4)
        check("T6_import_small_buf_enospc", False, "no error raised")
    except OSError as e:
        check("T6_import_small_buf_enospc", e.errno == errno.ENOSPC,
              "errno=%d (%s)" % (e.errno, errno.errorcode.get(e.errno)))

    # T7 — oversized blob rejected client-side before the ioctl
    try:
        fd2 = dummy_fd()
        aioctl.register(fd2, 0, 0, b"x" * (aioctl.CIPHER_WA_BLOB_MAX + 1))
        os.close(fd2)
        check("T7_oversized_blob_rejected", False, "no error raised")
    except ValueError:
        os.close(fd2)
        check("T7_oversized_blob_rejected", True, "ValueError as expected")

    # T8 — LEAVE: a fresh arena, sole producer leaves => participant count 0
    #      => the kmod releases the slot => QUERY no longer shows it
    fd3 = dummy_fd()
    aid2 = aioctl.register(fd3, 0x1000, 8192, b"leavetest")
    os.close(fd3)
    present = any(a["arena_id"] == aid2 for a in aioctl.query())
    aioctl.leave(aid2)
    gone = not any(a["arena_id"] == aid2 for a in aioctl.query())
    check("T8_leave_releases_sole_arena", present and gone,
          "present_before=%s gone_after=%s" % (present, gone))

    # cleanup: leave the T2 arena so we exit without depending on the reaper
    aioctl.leave(aid)

    npass = sum(r["pass"] for r in results)
    n = len(results)
    summary = {"passed": npass, "total": n, "PASS": npass == n,
               "results": results}
    json.dump(summary, open(HERE + "/sc5_unit_result.json", "w"), indent=2)
    print("\nSC5 UNIT: %d/%d  %s"
          % (npass, n, "PASS" if npass == n else "FAIL"))
    sys.exit(0 if npass == n else 1)


if __name__ == "__main__":
    main()
