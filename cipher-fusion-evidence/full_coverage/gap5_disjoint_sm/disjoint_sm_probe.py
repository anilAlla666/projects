#!/usr/bin/env python3
# GAP5 — DISJOINT-SM placement via GREEN CONTEXTS.
# Pure CUDA driver-API ctypes probe (NO torch runtime context, NO production .so,
# NO kmod, NO vLLM). Compiles kernels.cu -> kernels.cubin via nvcc -cubin -arch=sm_90a
# (done out-of-band; this script loads the cubin with cuModuleLoadData).
#
# Proves, EMPIRICALLY:
#   (b) two green contexts hold DISJOINT SM-id sets S_A, S_B with S_A ∩ S_B = ∅;
#   (3) a deterministic SM-localized fault (fp16 bit14 flip on a chosen %smid) is
#       CAUGHT by recompute on the disjoint partition (FAULTY_SM ∉ S_B by construction),
#       and the SAME-SM recompute "reproduce-and-miss" is reported CONDITIONALLY on
#       measured block->SM stability (advisor #2: never assert it — measure it).
#
# H100 = compute capability 9.0 => green-ctx SM split minimum 8 SMs, multiple of 8
# (cuda.h line 25179). 132 SMs total; minCount=64 => two 64-SM groups, 4 stranded.
import ctypes as C
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
CUBIN = os.path.join(HERE, "kernels.cubin")
OUT = os.path.join(HERE, "disjoint_sm_result.json")

# ---- driver handle (real driver, not the stub) ----
cu = C.CDLL("libcuda.so.1")

# ---- CUresult check ----
def ck(r, what):
    if r != 0:
        s = C.c_char_p()
        cu.cuGetErrorString(r, C.byref(s))
        raise RuntimeError(f"{what} failed: CUresult={r} {s.value.decode() if s.value else ''}")

# ---- struct mirror (cuda.h 25253): type@0(uint), pad[92], union@96 (smCount uint first) ----
# total 4 + 92 + 48 = 144 bytes.  smCount lives at offset 96.
class CUdevResource(C.Structure):
    _fields_ = [
        ("type", C.c_uint),
        ("_pad", C.c_ubyte * 92),
        ("smCount", C.c_uint),               # union start (CUdevSmResource.smCount)
        ("_oversize", C.c_ubyte * (48 - 4)),
    ]
assert C.sizeof(CUdevResource) == 144, C.sizeof(CUdevResource)
assert CUdevResource.smCount.offset == 96, CUdevResource.smCount.offset

CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 0x1

# argtype hygiene (avoid 32-bit truncation of pointers)
cu.cuDeviceGetDevResource.argtypes = [C.c_int, C.POINTER(CUdevResource), C.c_uint]
cu.cuDevSmResourceSplitByCount.argtypes = [
    C.POINTER(CUdevResource), C.POINTER(C.c_uint), C.POINTER(CUdevResource),
    C.POINTER(CUdevResource), C.c_uint, C.c_uint]
cu.cuDevResourceGenerateDesc.argtypes = [C.POINTER(C.c_void_p), C.POINTER(CUdevResource), C.c_uint]
cu.cuGreenCtxCreate.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_int, C.c_uint]
cu.cuCtxFromGreenCtx.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
cu.cuGreenCtxStreamCreate.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint, C.c_int]
cu.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
cu.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
cu.cuLaunchKernel.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                              C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                              C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
cu.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
cu.cuMemcpyHtoD_v2.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
cu.cuMemcpyDtoH_v2.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
cu.cuMemsetD32_v2.argtypes = [C.c_void_p, C.c_uint, C.c_size_t]
cu.cuStreamSynchronize.argtypes = [C.c_void_p]
cu.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
cu.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
cu.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]

# ---------------------------------------------------------------------------
ck(cu.cuInit(0), "cuInit")
dev = C.c_int(0)
ck(cu.cuDeviceGet(C.byref(dev), 0), "cuDeviceGet")

# advisor #1: primary ctx active before cuGreenCtxCreate (header advises this)
pctx = C.c_void_p()
ck(cu.cuDevicePrimaryCtxRetain(C.byref(pctx), dev), "primaryCtxRetain")
ck(cu.cuCtxPushCurrent_v2(pctx), "push primary")

# ---- advisor #3: read whole-device SM resource, ASSERT smCount==132 ----
whole = CUdevResource()
ck(cu.cuDeviceGetDevResource(dev, C.byref(whole), CU_DEV_RESOURCE_TYPE_SM), "getDevResource")
total_sm = whole.smCount
print(f"[layout-check] whole-device smCount = {total_sm}  (type={whole.type})")
assert whole.type == CU_DEV_RESOURCE_TYPE_SM
assert total_sm == 132, f"expected 132 SMs on H100, got {total_sm} -> struct offset wrong"

# ---- advisor #4: two-call split.  query nbGroups first (result=NULL) ----
MINCOUNT = 64
nb = C.c_uint(0)
ck(cu.cuDevSmResourceSplitByCount(None, C.byref(nb), C.byref(whole), None, 0, MINCOUNT),
   "split-query")
print(f"[split-query] minCount={MINCOUNT} -> nbGroups would be {nb.value}")
NG = nb.value
assert NG >= 2, f"expected >=2 groups at minCount={MINCOUNT}, got {NG}"

results = (CUdevResource * NG)()
remaining = CUdevResource()
nb2 = C.c_uint(NG)
ck(cu.cuDevSmResourceSplitByCount(results, C.byref(nb2), C.byref(whole),
                                  C.byref(remaining), 0, MINCOUNT), "split-do")
print(f"[split-do] created {nb2.value} groups; remaining smCount={remaining.smCount}")
grp_sm = [results[i].smCount for i in range(nb2.value)]
print(f"[split-do] group SM counts = {grp_sm}")

# ---- build a green ctx + stream + module-function for each of two partitions ----
def make_partition(res):
    desc = C.c_void_p()
    one = (CUdevResource * 1)(res)
    ck(cu.cuDevResourceGenerateDesc(C.byref(desc), one, 1), "genDesc")
    gctx = C.c_void_p()
    ck(cu.cuGreenCtxCreate(C.byref(gctx), desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "greenCtxCreate")
    cctx = C.c_void_p()
    ck(cu.cuCtxFromGreenCtx(C.byref(cctx), gctx), "ctxFromGreen")
    # advisor #1: load module UNDER this green CUcontext (CUfunction is per-context)
    ck(cu.cuCtxPushCurrent_v2(cctx), "push green")
    with open(CUBIN, "rb") as f:
        img = f.read()
    mod = C.c_void_p()
    ck(cu.cuModuleLoadData(C.byref(mod), img), "moduleLoad")
    f_rec = C.c_void_p(); f_flt = C.c_void_p()
    ck(cu.cuModuleGetFunction(C.byref(f_rec), mod, b"record_smid"), "getFn record")
    ck(cu.cuModuleGetFunction(C.byref(f_flt), mod, b"fma_smlocal_fault"), "getFn fault")
    strm = C.c_void_p()
    ck(cu.cuGreenCtxStreamCreate(C.byref(strm), gctx, 0x1, 0), "greenStream")  # NON_BLOCKING
    ck(cu.cuCtxPopCurrent_v2(C.byref(C.c_void_p())), "pop green")
    return dict(gctx=gctx, cctx=cctx, strm=strm, f_rec=f_rec, f_flt=f_flt, smCount=res.smCount)

P = [make_partition(results[0]), make_partition(results[1])]

SPIN = 2_000_000  # cycles -> ~1ms @1.8GHz so blocks spread across the whole partition

def launch_record(part, nblocks):
    """(b) launch record_smid; return sorted set of distinct smids touched."""
    ck(cu.cuCtxPushCurrent_v2(part["cctx"]), "push")
    dptr = C.c_void_p()
    ck(cu.cuMemAlloc_v2(C.byref(dptr), nblocks * 4), "alloc smid")
    ck(cu.cuMemsetD32_v2(dptr, 0xFFFFFFFF, nblocks), "memset")
    cyc = C.c_ulonglong(SPIN)
    args = (C.c_void_p * 2)(C.cast(C.byref(dptr), C.c_void_p), C.cast(C.byref(cyc), C.c_void_p))
    ck(cu.cuLaunchKernel(part["f_rec"], nblocks,1,1, 32,1,1, 0, part["strm"],
                         args, None), "launch record")
    ck(cu.cuStreamSynchronize(part["strm"]), "sync")
    host = (C.c_uint * nblocks)()
    ck(cu.cuMemcpyDtoH_v2(host, dptr, nblocks * 4), "dtoh")
    cu.cuMemFree_v2(dptr)
    ck(cu.cuCtxPopCurrent_v2(C.byref(C.c_void_p())), "pop")
    return sorted(set(host[i] for i in range(nblocks)))

# nblocks >> SM count to force spread over the whole partition
NB = 4 * 132
S_A = launch_record(P[0], NB)
S_B = launch_record(P[1], NB)
inter = sorted(set(S_A) & set(S_B))
print(f"\n[disjointness] |S_A|={len(S_A)} smids={S_A}")
print(f"[disjointness] |S_B|={len(S_B)} smids={S_B}")
print(f"[disjointness] S_A ∩ S_B = {inter}  ->  {'DISJOINT ✅' if not inter else 'OVERLAP ❌'}")

# --------------------------------------------------------------------------
# (3) A/B fault test.
#   N output elements, one block each.  Reference run = clean (faulty_sm=-1).
#   Faulted run on partition A with a chosen faulty SM that IS in S_A.
#   Detector = recompute; we compare three placements of the recompute:
#     (i)  SAME-SM (partition A again):  records smid-per-element on BOTH A-runs,
#          reports whether the corrupted-element set is identical (block->SM stable?).
#     (ii) DISJOINT (partition B):  FAULTY_SM ∉ S_B by construction -> no corruption ->
#          recompute mismatches original-faulted output -> CATCH (guaranteed).
N = 8192

def make_inputs():
    # a = b = 4.0 -> y = 8.0 (fp16 0x4800).  bit14 flip -> 0x0800 (tiny) => |delta|~8.0,
    # comfortably in the Step-A harmful class (|delta|>=2.76).
    half_four = 0x4400  # fp16 4.0
    a = (C.c_uint16 * N)(*([half_four] * N))
    b = (C.c_uint16 * N)(*([half_four] * N))
    return a, b

def to_dev(host_arr, nbytes):
    d = C.c_void_p()
    ck(cu.cuMemAlloc_v2(C.byref(d), nbytes), "alloc")
    ck(cu.cuMemcpyHtoD_v2(d, host_arr, nbytes), "htod")
    return d

def run_fault(part, da, db, faulty_sm):
    """Launch fault kernel on `part`; return (y_host[uint16], smid_host[uint32])."""
    ck(cu.cuCtxPushCurrent_v2(part["cctx"]), "push")
    dy = C.c_void_p(); dsm = C.c_void_p()
    ck(cu.cuMemAlloc_v2(C.byref(dy), N*2), "alloc y")
    ck(cu.cuMemAlloc_v2(C.byref(dsm), N*4), "alloc sm")
    cyc = C.c_ulonglong(SPIN); fsm = C.c_int(faulty_sm); n = C.c_int(N)
    # kernel sig: (half* a, half* b, half* y, uint* smid, int faulty_sm, ull cyc, int n)
    args = (C.c_void_p*7)(
        C.cast(C.byref(da),  C.c_void_p), C.cast(C.byref(db), C.c_void_p),
        C.cast(C.byref(dy),  C.c_void_p), C.cast(C.byref(dsm),C.c_void_p),
        C.cast(C.byref(fsm), C.c_void_p), C.cast(C.byref(cyc),C.c_void_p),
        C.cast(C.byref(n),   C.c_void_p))
    ck(cu.cuLaunchKernel(part["f_flt"], N,1,1, 32,1,1, 0, part["strm"], args, None), "launch fault")
    ck(cu.cuStreamSynchronize(part["strm"]), "sync")
    yh = (C.c_uint16 * N)(); smh = (C.c_uint * N)()
    ck(cu.cuMemcpyDtoH_v2(yh, dy, N*2), "dtoh y")
    ck(cu.cuMemcpyDtoH_v2(smh, dsm, N*4), "dtoh sm")
    cu.cuMemFree_v2(dy); cu.cuMemFree_v2(dsm)
    ck(cu.cuCtxPopCurrent_v2(C.byref(C.c_void_p())), "pop")
    return [yh[i] for i in range(N)], [smh[i] for i in range(N)]

ah, bh = make_inputs()
# inputs must live in each partition's context address space; primary ctx is shared
# memory across green ctxs of the same device/primary ctx, so alloc once under primary.
da = to_dev(ah, N*2); db = to_dev(bh, N*2)

# choose a faulty SM that is provably in S_A and provably NOT in S_B
FAULTY_SM = S_A[len(S_A)//2]
assert FAULTY_SM in set(S_A) and FAULTY_SM not in set(S_B), "faulty SM not cleanly in A-only"

# ORIGINAL (faulted) run on A
y_orig, sm_orig = run_fault(P[0], da, db, FAULTY_SM)
# Detector compares original output vs a recompute (any mismatch = detected).
# Reference run (faulty_sm=-1) gives ground-truth clean output for corrupted-set bookkeeping.
def fp16_to_f(u):
    import struct
    return struct.unpack('e', struct.pack('H', u))[0]

# recompute placements
y_same, sm_same = run_fault(P[0], da, db, FAULTY_SM)     # SAME-SM, fault still active
y_disj, sm_disj = run_fault(P[1], da, db, FAULTY_SM)     # DISJOINT, fault SM absent -> clean
y_ref,  sm_ref  = run_fault(P[0], da, db, -1)            # ground-truth clean reference

def corrupted_set(y):
    return set(i for i in range(N) if y[i] != y_ref[i])

corr_orig = corrupted_set(y_orig)
corr_same = corrupted_set(y_same)
# detector = (original output) != (recompute output)
catch_disjoint = any(y_orig[i] != y_disj[i] for i in range(N))
catch_same     = any(y_orig[i] != y_same[i] for i in range(N))

# advisor #2: is block->SM mapping stable across the two A-runs? (the honesty crux)
same_sm_stable = (sm_orig == sm_same)
# fraction of corrupted elements that recur on the SAME exact element on the same-SM rerun:
recur = len(corr_orig & corr_same) / max(1, len(corr_orig))

print(f"\n[fault] FAULTY_SM={FAULTY_SM}  in S_A={FAULTY_SM in set(S_A)}  in S_B={FAULTY_SM in set(S_B)}")
print(f"[fault] #corrupted in ORIGINAL = {len(corr_orig)}  (elements produced on FAULTY_SM)")
print(f"[fault] DISJOINT recompute (partition B) catches mismatch? {catch_disjoint}  <- GUARANTEED")
print(f"[fault] SAME-SM   recompute (partition A) catches mismatch? {catch_same}")
print(f"[fault] block->SM stable across two A-runs? {same_sm_stable}")
print(f"[fault] corrupted-element recurrence fraction (same-SM) = {recur:.4f}")
print(f"[fault] INTERPRETATION: same-SM 'reproduce-and-MISS' holds iff the faulted")
print(f"        element lands on FAULTY_SM both runs.  recur=1.0 => deterministic miss;")
print(f"        recur<1.0 => block roves => same-SM catches BY ACCIDENT (wrong reason).")

result = dict(
    total_sm=total_sm, minCount=MINCOUNT, nbGroups=nb2.value, group_sm=grp_sm,
    remaining_sm=remaining.smCount, S_A=S_A, S_B=S_B, intersection=inter,
    disjoint=(len(inter)==0), faulty_sm=FAULTY_SM,
    n_corrupted_original=len(corr_orig),
    catch_disjoint=catch_disjoint, catch_same=catch_same,
    same_sm_blockmap_stable=same_sm_stable,
    corrupted_recurrence_fraction=recur,
)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] wrote {OUT}")
