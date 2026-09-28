"""CP 5.4 Step 1.3b' — POOL client for the Phase B batch executor.

Option 2 (adjudicated 2026-05-19). The executor stays on libcipher_v2.so; this
thin client gives it a CP 5.4 POOL allocation and a kmod-aligned green context:

  - ALLOCATE(POOL) via /dev/cipher  -> the residual 8-SM-group mask.
  - torch.cuda.GreenContext.create(grp_count*8) -> a green context. Per the
    Step 1.3b' Appendix-A probe, create(8N) picks groups 0..N-1 (the low
    prefix), deterministically; the kmod places the POOL on that same low
    prefix (it claims all groups; PARTITIONs shrink it from the high end), so
    torch's pick aligns with the kmod's pool ownership -> physically disjoint
    from partition tenants.
  - SELF-VERIFY: a %smid kernel confirms the green context's physical SM set
    is exactly the expected low-prefix groups, and raises if not -- the safety
    net for the (beta, undocumented) PyTorch low-prefix-pick dependency.
  - check_resize(): re-ALLOCATE(POOL) (idempotent) each round; if the pool
    group count changed (a PARTITION arrived / left), rebuild + re-verify.
"""
import os
import struct
import fcntl

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
import torch
from torch.utils.cpp_extension import load_inline

# ---- CP 5.4 ioctl ABI (cipher_kmod/cipher_ioctl.h, nrs 13/14/15) ----------
_TYPE = ord('C')


def _IOC(direction, nr, size):
    return (direction << 30) | (size << 16) | (_TYPE << 8) | nr


_CP54_STRUCT = 32                       # 8 x u32, both allocate and query
CP54_ALLOCATE = _IOC(3, 13, _CP54_STRUCT)   # _IOWR
CP54_FREE     = _IOC(0, 14, 0)              # _IO
CP54_QUERY    = _IOC(2, 15, _CP54_STRUCT)   # _IOR
QOS_PARTITION, QOS_SHARED, QOS_POOL = 0, 1, 2

# group g -> physical SM set (PHASE_1_3A_PROBE.md; cuDevSmResourceSplitByCount)
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


def _log(msg):
    print("[cp54_pool] " + msg, flush=True)


# ---- %smid self-verification kernel (load_inline; compiled once, cached) ---
_SMID_EXT = None


def _smid_ext():
    global _SMID_EXT
    if _SMID_EXT is None:
        cpp_src = "void probe_smid(torch::Tensor hit);"
        cuda_src = r'''
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
        _SMID_EXT = load_inline(name="cp54_pool_smid", cpp_sources=[cpp_src],
                                cuda_sources=[cuda_src],
                                functions=["probe_smid"], verbose=False)
    return _SMID_EXT


def _green_ctx_sms(gc_stream):
    """Run the %smid kernel on the green context's stream; return the sorted
    set of physical SM ids the green context actually used."""
    hit = torch.zeros(256, dtype=torch.int32, device="cuda")
    with torch.cuda.stream(gc_stream):
        _smid_ext().probe_smid(hit)
    torch.cuda.synchronize()
    return sorted((hit == 1).nonzero().flatten().tolist())


class PoolBinding:
    """A CP 5.4 POOL allocation + its torch green context, for the executor."""

    def __init__(self, dev="/dev/cipher"):
        self.fd = os.open(dev, os.O_RDWR)
        self.grp_mask, self.grp_count = self._allocate()
        self.gc = None
        self.gc_stream = None
        self._gc_count = None
        # CP 5.4 Step 1.4 MEASUREMENT INSTRUMENT — not a production feature.
        # CIPHER_POOL_MAX_GROUPS caps the green context to the low K groups so
        # the confined-pool lift curve can be swept in isolation (pure
        # confinement cost, no partitions present). Production POOL sizing is
        # set by the kmod ledger from real PARTITION allocations, never by
        # operator config. Unset => no cap (normal behaviour).
        cap = os.environ.get("CIPHER_POOL_MAX_GROUPS")
        self.cap = int(cap) if cap else None
        _log("ALLOCATE(POOL) -> grp_mask=0x%04x grp_count=%d (%d SMs)%s"
             % (self.grp_mask, self.grp_count, self.grp_count * 8,
                "" if self.cap is None
                else "  [CIPHER_POOL_MAX_GROUPS cap=%d]" % self.cap))

    def _effective_count(self):
        """Group count the green context is actually built at: the pool's
        grp_count, capped by CIPHER_POOL_MAX_GROUPS (Step 1.4 instrument)."""
        if self.cap is not None:
            return min(self.grp_count, self.cap)
        return self.grp_count

    def _allocate(self):
        """Idempotent ALLOCATE(POOL): registers/grows the pool, returns
        (grp_mask, grp_count)."""
        buf = bytearray(_CP54_STRUCT)
        struct.pack_into("<II", buf, 0, QOS_POOL, 0)        # qos_class, sm_count
        fcntl.ioctl(self.fd, CP54_ALLOCATE, buf, True)
        _, _, grp_mask, grp_count = struct.unpack_from("<IIII", buf, 0)
        return grp_mask, grp_count

    def query(self):
        buf = bytearray(_CP54_STRUCT)
        fcntl.ioctl(self.fd, CP54_QUERY, buf, True)
        n_part, pool_cnt, free_cnt, my_mask, my_qos = struct.unpack_from(
            "<IIIII", buf, 0)
        return {"n_partitions": n_part, "pool_grp_count": pool_cnt,
                "free_grp_count": free_cnt, "my_grp_mask": my_mask,
                "my_qos_class": my_qos}

    def _expected_sms(self):
        """The low-prefix group set the green context must cover (Option 2:
        groups 0..effective_count-1)."""
        return set().union(*(GROUP_SMS[g]
                             for g in range(self._effective_count())))

    def build_green_ctx(self):
        """Create (or recreate) the torch green context over grp_count*8 SMs,
        bind it, and self-verify its physical SM set. Raises on mismatch."""
        if self.gc is not None:
            try:
                self.gc.pop_context()
            except Exception as e:                       # noqa: BLE001
                _log("warn: pop_context on rebuild: %r" % e)
            self.gc = None
            self.gc_stream = None

        eff = self._effective_count()
        num_sms = eff * 8
        self.gc = torch.cuda.GreenContext.create(num_sms, 0)
        self.gc_stream = self.gc.Stream()
        self.gc.set_context()

        actual = set(_green_ctx_sms(self.gc_stream))
        expected = self._expected_sms()
        if actual != expected:
            raise RuntimeError(
                "CP 5.4 POOL self-verify FAILED — green ctx SM set does not "
                "match the kmod low-prefix groups 0..%d. expected=%s actual=%s "
                "(torch.cuda.GreenContext.create low-prefix-pick assumption "
                "violated — Option 2 dependency broken)"
                % (eff - 1, sorted(expected), sorted(actual)))
        self._gc_count = eff
        capnote = ("  [CIPHER_POOL_MAX_GROUPS cap: %d of %d pool groups]"
                   % (eff, self.grp_count)) if self.cap is not None else ""
        _log("green ctx ready — %d groups / %d SMs, self-verify PASS "
             "(SMs = kmod groups 0..%d)%s" % (eff, num_sms, eff - 1, capnote))

    def check_resize(self):
        """Re-ALLOCATE(POOL) (idempotent — reclaims groups a PARTITION freed,
        reflects a shrink). If the group count changed, rebuild the green
        context. Returns True if the pool resized."""
        new_mask, new_count = self._allocate()
        self.grp_mask, self.grp_count = new_mask, new_count
        if self._effective_count() != self._gc_count:
            _log("POOL effective size %d -> %d groups — rebuilding green ctx"
                 % (self._gc_count, self._effective_count()))
            self.build_green_ctx()
            return True
        return False

    def probe_observed_sms(self):
        """CP 5.4 Step 1.6 — run the %smid kernel on the POOL's green-ctx
        stream; return the sorted physical SM ids actually used. The
        runtime-disjointness probe input for the mixed-deployment orchestrator
        (continuous, per-round — not just at green-ctx build)."""
        return _green_ctx_sms(self.gc_stream)

    def free(self):
        """Defensive explicit FREE on clean exit (the do_exit reaper also
        reclaims on crash)."""
        try:
            if self.gc is not None:
                try:
                    self.gc.pop_context()
                except Exception:                        # noqa: BLE001
                    pass
            fcntl.ioctl(self.fd, CP54_FREE)
            _log("FREE(POOL) issued")
        except Exception as e:                           # noqa: BLE001
            _log("warn: FREE: %r" % e)
        finally:
            os.close(self.fd)
