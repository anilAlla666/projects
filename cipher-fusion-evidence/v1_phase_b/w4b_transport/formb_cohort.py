"""W.4b.2 Form B' prototype — cohort read + eligibility gate (§5 item 4).

Two responsibilities, kept honest to the W.4a substrate-line:

  1. Read the live W.6 sub-C co-residence cohort straight from the kmod via the
     raw NR 31 QUERY ioctl with caller_fingerprint=0 ("query-only", confirmed at
     cipher_coresidence_registry.c:219 — fp==0 ⇒ NO self-register). The vanilla
     executor must NOT appear in its own cohort, so it cannot use the substrate's
     cipher_rt_coresidence_update wrapper (that wrapper skips the ioctl when
     fp==0 and would otherwise heartbeat-insert the caller).

  2. Run the eligibility partition through the AUDITED W.4a guard
     cipher_rt_pool_partition() loaded from libcipher_rt.so via ctypes. Per memo
     §2 "the W.4a substrate is reused unchanged ... No second copy of the
     catastrophic guard is written." We do NOT reimplement the same-fingerprint
     filter in Python.
"""
import os
import ctypes
import fcntl
import struct
import numpy as np

LIBCIPHER_RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
CIPHER_DEV = "/dev/cipher"
COHORT_MAX = 128

# --- _IOWR('C', 31, struct cipher_cohort_query) ----------------------------
_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS = 8, 8, 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE, _IOC_READ = 1, 2

# sizeof(struct cipher_cohort_query) = u64 + u32 + u32 + 128*{u32,u32,u64}
_QUERY_HDR = struct.Struct("<QII")          # caller_fp, max_entries, n_resident
_ENTRY = struct.Struct("<IIQ")              # tgid, _pad, model_fingerprint
_QUERY_SIZE = _QUERY_HDR.size + COHORT_MAX * _ENTRY.size   # 2064

_COHORT_QUERY = ((_IOC_READ | _IOC_WRITE) << _IOC_DIRSHIFT) | \
                (ord('C') << _IOC_TYPESHIFT) | \
                (31 << _IOC_NRSHIFT) | \
                (_QUERY_SIZE << _IOC_SIZESHIFT)


def query_cohort():
    """Return [(tgid, fingerprint), ...] of LIVE co-resident tenants.
    query-only (caller_fingerprint=0) — does not register the caller."""
    buf = bytearray(_QUERY_SIZE)
    _QUERY_HDR.pack_into(buf, 0, 0, COHORT_MAX, 0)   # caller_fp=0, max=128, n=0
    fd = os.open(CIPHER_DEV, os.O_RDWR)
    try:
        fcntl.ioctl(fd, _COHORT_QUERY, buf, True)
    finally:
        os.close(fd)
    _, _, n = _QUERY_HDR.unpack_from(buf, 0)
    out = []
    off = _QUERY_HDR.size
    for i in range(min(n, COHORT_MAX)):
        tgid, _pad, fp = _ENTRY.unpack_from(buf, off + i * _ENTRY.size)
        out.append((tgid, fp))
    return out


# --- ctypes mirror of struct cipher_rt_pool_group --------------------------
class CohortPeer(ctypes.Structure):
    _fields_ = [("tgid", ctypes.c_uint32), ("_pad", ctypes.c_uint32),
                ("model_fingerprint", ctypes.c_uint64)]


class PoolGroup(ctypes.Structure):
    _fields_ = [("group_id", ctypes.c_uint64), ("fingerprint", ctypes.c_uint64),
                ("self_tgid", ctypes.c_uint32), ("n_in_group", ctypes.c_uint32),
                ("distinct_rejected", ctypes.c_uint32), ("eligible", ctypes.c_uint32),
                ("peer_tgids", ctypes.c_uint32 * COHORT_MAX),
                ("K", ctypes.c_int32), ("N", ctypes.c_int32), ("dtype", ctypes.c_int32)]


_lib = None


def _load_guard():
    global _lib
    if _lib is not None:
        return _lib
    # RTLD_LAZY: bind only what we call (the pure partition fn); never resolve
    # the libcipher_rt CUDA symbols on the vanilla executor.
    _lib = ctypes.CDLL(LIBCIPHER_RT, mode=os.RTLD_LOCAL | 0x0001)
    _lib.cipher_rt_pool_partition.restype = None
    _lib.cipher_rt_pool_partition.argtypes = [
        ctypes.c_uint32, ctypes.c_uint64,
        ctypes.POINTER(CohortPeer), ctypes.c_uint32, ctypes.POINTER(PoolGroup)]
    return _lib


def partition(self_tgid, my_fingerprint, peers):
    """Call the audited W.4a guard. `peers` is [(tgid, fp), ...].
    Returns a dict {eligible, n_in_group, distinct_rejected, peer_tgids, group_id}."""
    lib = _load_guard()
    n = len(peers)
    arr = (CohortPeer * max(n, 1))()
    for i, (tgid, fp) in enumerate(peers):
        arr[i].tgid = tgid
        arr[i]._pad = 0
        arr[i].model_fingerprint = fp
    out = PoolGroup()
    lib.cipher_rt_pool_partition(self_tgid, my_fingerprint, arr, n, ctypes.byref(out))
    return {
        "eligible": int(out.eligible),
        "n_in_group": int(out.n_in_group),
        "distinct_rejected": int(out.distinct_rejected),
        "group_id": int(out.group_id),
        "peer_tgids": [int(out.peer_tgids[i]) for i in range(int(out.n_in_group))],
    }


_FPTR = ctypes.POINTER(ctypes.c_float)


def correctness_check(rows_out, rows_ref, tol):
    """Audited W.4a per-tenant correctness backstop (cipher_rt_pool_correctness_check,
    reused verbatim). rows_out/rows_ref: lists of 1D float arrays (the batched-path
    logit row vs the single-tenant solo reference row). Returns (n_fail, per_pass)."""
    lib = _load_guard()
    lib.cipher_rt_pool_correctness_check.restype = ctypes.c_uint32
    lib.cipher_rt_pool_correctness_check.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_FPTR), ctypes.POINTER(_FPTR),
        ctypes.c_size_t, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
    R = len(rows_out)
    vocab = int(rows_out[0].shape[0])
    out_arr = (_FPTR * R)()
    ref_arr = (_FPTR * R)()
    keep = []
    for i in range(R):
        o = np.ascontiguousarray(rows_out[i], dtype=np.float32)
        r = np.ascontiguousarray(rows_ref[i], dtype=np.float32)
        keep += [o, r]
        out_arr[i] = o.ctypes.data_as(_FPTR)
        ref_arr[i] = r.ctypes.data_as(_FPTR)
    pa = (ctypes.c_int * R)()
    fails = lib.cipher_rt_pool_correctness_check(
        R, out_arr, ref_arr, vocab, ctypes.c_double(tol), pa)
    return int(fails), [int(pa[i]) for i in range(R)]


def mark_blocked(fp, K, N, dtype):
    lib = _load_guard()
    lib.cipher_rt_pool_mark_blocked.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.cipher_rt_pool_mark_blocked(fp, K, N, dtype)


def is_blocked(fp, K, N, dtype):
    lib = _load_guard()
    lib.cipher_rt_pool_is_blocked.restype = ctypes.c_int
    lib.cipher_rt_pool_is_blocked.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return int(lib.cipher_rt_pool_is_blocked(fp, K, N, dtype))


def counters():
    lib = _load_guard()
    out = {}
    for fn in ("cipher_rt_pool_blocked_by_correctness",
               "cipher_rt_pool_distinct_fp_rejected",
               "cipher_rt_pool_eligible_groups", "cipher_rt_pool_solo"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_uint64
        out[fn.replace("cipher_rt_pool_", "")] = int(f())
    return out
