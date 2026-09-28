#!/usr/bin/env python3
"""Track 3 SC3 — poll-and-migrate handler (design memo §3).

TWO-LAYER migration handler the tenant runs inline at each decode-round
boundary (NOT a background thread):
  - this Python layer: POLL_MIGRATE / START_MIGRATE / ACK_MIGRATE ioctls on
    /dev/cipher, and the L2 %smid post-swap verify;
  - the libcipher_rt C primitive `cipher_rt_green_ctx_migrate` (drain / build /
    L1 structural verify / swap / release).

Reaches the INJECTED libcipher_rt instance via `ctypes.CDLL(path, RTLD_NOLOAD)`
— SC3-2 reachability gate confirmed this resolves to the injected instance
(CDLL(None) does NOT; the injection lib is RTLD_LOCAL). See
sc3_reachability_result.json.

Tenant usage:
    h = MigrateHandler()
    for each decode round:
        decode()
        act = h.step()        # polls; migrates if the kmod proposed one
"""
import ctypes
import fcntl
import os
import struct
import time

RT_PATH = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
_TYPE = ord('C')


def _IOC(direction, nr, size):
    return (direction << 30) | (size << 16) | (_TYPE << 8) | nr


_POLL_STRUCT = 32                                  # cipher_cp54_migrate_poll
SUBSCRIBE_MIGRATE = _IOC(1, 16, 4)                 # _IOW(__u32)
POLL_MIGRATE      = _IOC(2, 17, _POLL_STRUCT)      # _IOR(struct)
START_MIGRATE     = _IOC(0, 18, 0)                 # _IO
ACK_MIGRATE       = _IOC(1, 19, 4)                 # _IOW(__u32)
COMPACT_MIGRATE   = _IOC(0, 20, 0)                 # _IO

MIG_IDLE, MIG_PROPOSED, MIG_MIGRATING = 0, 1, 2
MIGOUT = {0: "NONE", 1: "COMMITTED", 2: "ABORTED_TIMEOUT",
          3: "ABORTED_KMOD_REFUSED", 4: "ABORTED_TENANT_NACK"}

# group g -> physical SM set (PHASE_1_3A_PROBE.md; same map as cp54_pool.py /
# cp54_s16_partition_tenant.py).
GROUP_SMS = {
    0:  {0, 1, 16, 17, 32, 33, 48, 49},   1:  {2, 3, 18, 19, 34, 35, 50, 51},
    2:  {4, 5, 20, 21, 36, 37, 52, 53},   3:  {6, 7, 22, 23, 38, 39, 54, 55},
    4:  {8, 9, 24, 25, 40, 41, 56, 57},   5:  {10, 11, 26, 27, 42, 43, 58, 59},
    6:  {12, 13, 28, 29, 44, 45, 60, 61}, 7:  {14, 15, 30, 31, 46, 47, 62, 63},
    8:  {64, 65, 78, 79, 92, 93, 106, 107},  9:  {66, 67, 80, 81, 94, 95, 108, 109},
    10: {68, 69, 82, 83, 96, 97, 110, 111},  11: {70, 71, 84, 85, 98, 99, 112, 113},
    12: {72, 73, 86, 87, 100, 101, 114, 115}, 13: {74, 75, 88, 89, 102, 103, 116, 117},
    14: {76, 77, 90, 91, 104, 105, 118, 119},
}


def mask_to_sms(grp_mask):
    s = set()
    for g in range(15):
        if grp_mask & (1 << g):
            s |= GROUP_SMS[g]
    return s


_SMID = None


def _smid_ext():
    """The L2 %smid probe — same load_inline kernel as cp54_s16_partition_tenant."""
    global _SMID
    if _SMID is None:
        from torch.utils.cpp_extension import load_inline
        cpp = "void probe_smid(torch::Tensor hit);"
        cu = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void probe_smid_k(int* hit){
    unsigned s; asm volatile("mov.u32 %0, %%smid;":"=r"(s));
    if (s < 256) hit[s] = 1;
}
void probe_smid(torch::Tensor hit){
    probe_smid_k<<<4096, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        hit.data_ptr<int>());
}
'''
        _SMID = load_inline(name="cipher_migrate_smid", cpp_sources=[cpp],
                            cuda_sources=[cu], functions=["probe_smid"],
                            verbose=False)
    return _SMID


def probe_sms():
    """L2: run %smid on the current (green-ctx) stream; sorted observed SM ids."""
    import torch
    hit = torch.zeros(256, dtype=torch.int32, device="cuda")
    _smid_ext().probe_smid(hit)
    torch.cuda.synchronize()
    return sorted((hit == 1).nonzero().flatten().tolist())


class MigrateHandler:
    """Per-tenant migration handler. One /dev/cipher fd + the RTLD_NOLOAD
    binding to the injected libcipher_rt."""

    def __init__(self, rt_path=RT_PATH):
        self.fd = os.open("/dev/cipher", os.O_RDWR)
        self.lib = None
        self.lib_error = None
        try:
            self.lib = ctypes.CDLL(rt_path, mode=os.RTLD_NOLOAD)
            self.lib.cipher_rt_green_ctx_migrate.argtypes = [ctypes.c_uint]
            self.lib.cipher_rt_green_ctx_migrate.restype = ctypes.c_int
            self.lib.cipher_rt_green_ctx_cur_mask.restype = ctypes.c_uint
            self.lib.cipher_rt_green_ctx_sm_count.restype = ctypes.c_uint
        except OSError as e:
            self.lib_error = str(e)        # libcipher_rt not injected -> no-op

    def subscribe(self, migratable=1):
        fcntl.ioctl(self.fd, SUBSCRIBE_MIGRATE, struct.pack("<I", migratable))

    def compact(self):
        fcntl.ioctl(self.fd, COMPACT_MIGRATE)

    def poll(self):
        buf = bytearray(_POLL_STRUCT)
        fcntl.ioctl(self.fd, POLL_MIGRATE, buf, True)
        state, target, cur, outcome = struct.unpack_from("<IIII", buf, 0)
        return {"state": state, "target_mask": target, "cur_mask": cur,
                "last_outcome": outcome,
                "last_outcome_str": MIGOUT.get(outcome, "?")}

    def cur_mask(self):
        return int(self.lib.cipher_rt_green_ctx_cur_mask()) if self.lib else 0

    def sm_count(self):
        return int(self.lib.cipher_rt_green_ctx_sm_count()) if self.lib else 0

    def migrate_primitive(self, new_mask):
        """Direct call to the libcipher_rt C primitive (drain/build/L1/swap/
        release). Returns 0 on success, <0 on a pre-swap failure."""
        if not self.lib:
            return -1
        return int(self.lib.cipher_rt_green_ctx_migrate(new_mask))

    def step(self, verify_l2=True):
        """One decode-round handler call: poll, and if the kmod has PROPOSED a
        migration, run it (START -> primitive -> L2 -> ACK). Returns a dict
        describing the action."""
        p = self.poll()
        if p["state"] != MIG_PROPOSED:
            return {"action": "none", **p}
        target = p["target_mask"]
        fcntl.ioctl(self.fd, START_MIGRATE, 0)              # PROPOSED->MIGRATING
        t0 = time.perf_counter()
        rc = self.migrate_primitive(target)                 # drain/build/L1/swap/release
        primitive_ms = (time.perf_counter() - t0) * 1e3
        l2_ok, outside, l2_ms = True, [], 0.0
        if rc == 0 and verify_l2:
            t1 = time.perf_counter()
            sms = probe_sms()                               # L2 %smid probe
            l2_ms = (time.perf_counter() - t1) * 1e3
            outside = sorted(set(sms) - mask_to_sms(target))
            l2_ok = (len(outside) == 0)
        ok = (rc == 0) and l2_ok
        fcntl.ioctl(self.fd, ACK_MIGRATE, struct.pack("<I", 1 if ok else 0))
        return {"action": "migrate", "target_mask": target, "primitive_rc": rc,
                "l2_ok": l2_ok, "l2_outside": outside, "committed": ok,
                "primitive_ms": primitive_ms, "l2_ms": l2_ms}

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass
