#!/usr/bin/env python3
# FORK-2 Mech-2 decisive sub-measurement: the green-context minimum SM partition granularity on this H100.
# Queries cuDevSmResourceSplitByCount at a sweep of minCount -> the smallest achievable partition (in SMs).
# If the floor is 8 SMs (6.06% of 132), the smallest possible DMR shadow already costs >3% before HBM contention.
# Pure driver-API ctypes (no torch, no .so). Read-only.
import ctypes as C, json
cu = C.CDLL("libcuda.so.1")
def ck(r, what):
    if r != 0:
        s=C.c_char_p(); cu.cuGetErrorString(r, C.byref(s))
        raise RuntimeError(f"{what}: CUresult={r} {s.value.decode() if s.value else ''}")
class CUdevResource(C.Structure):
    _fields_=[("type",C.c_uint),("_pad",C.c_ubyte*92),("smCount",C.c_uint),("_oversize",C.c_ubyte*(48-4))]
assert C.sizeof(CUdevResource)==144
CU_DEV_RESOURCE_TYPE_SM=1
cu.cuDeviceGetDevResource.argtypes=[C.c_int,C.POINTER(CUdevResource),C.c_uint]
cu.cuDevSmResourceSplitByCount.argtypes=[C.POINTER(CUdevResource),C.POINTER(C.c_uint),C.POINTER(CUdevResource),C.POINTER(CUdevResource),C.c_uint,C.c_uint]
cu.cuDevicePrimaryCtxRetain.argtypes=[C.POINTER(C.c_void_p),C.c_int]
cu.cuCtxPushCurrent_v2.argtypes=[C.c_void_p]
ck(cu.cuInit(0),"init"); dev=C.c_int(0); ck(cu.cuDeviceGet(C.byref(dev),0),"devget")
pctx=C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(pctx),dev),"retain"); ck(cu.cuCtxPushCurrent_v2(pctx),"push")
whole=CUdevResource(); ck(cu.cuDeviceGetDevResource(dev,C.byref(whole),CU_DEV_RESOURCE_TYPE_SM),"whole")
total=whole.smCount
print(f"total SMs = {total}")
out={"total_sm":int(total),"sweep":[]}
for mc in [1,2,4,6,8,12,16,32,64]:
    nb=C.c_uint(0)
    r=cu.cuDevSmResourceSplitByCount(None,C.byref(nb),C.byref(whole),None,0,mc)
    if r!=0:
        out["sweep"].append({"minCount":mc,"error":r}); print(f"minCount={mc}: ERR {r}"); continue
    ng=nb.value
    # do the split to read the actual group sizes
    res=(CUdevResource*ng)() if ng>0 else None
    rem=CUdevResource(); nb2=C.c_uint(ng)
    grp=[]
    if ng>0:
        r2=cu.cuDevSmResourceSplitByCount(res,C.byref(nb2),C.byref(whole),C.byref(rem),0,mc)
        if r2==0: grp=[int(res[i].smCount) for i in range(nb2.value)]
    smallest = min(grp) if grp else None
    out["sweep"].append({"minCount":mc,"nbGroups":int(ng),"group_sm":grp,"remaining":int(rem.smCount),"smallest_group_sm":smallest})
    print(f"minCount={mc:3d} -> nbGroups={ng:3d}  groups={grp[:6]}{'...' if len(grp)>6 else ''}  remaining={rem.smCount}")
# smallest achievable partition = smallest group across the finest split that succeeds
floors=[s["smallest_group_sm"] for s in out["sweep"] if s.get("smallest_group_sm")]
out["min_partition_sm"]=min(floors) if floors else None
out["min_partition_pct_of_132"]=round(out["min_partition_sm"]/total*100,2) if out["min_partition_sm"] else None
print(f"\nMIN PARTITION = {out['min_partition_sm']} SMs = {out['min_partition_pct_of_132']}% of {total}")
json.dump(out,open("/home/ubuntu/cipher-fusion-evidence/ra_fork2/granularity_result.json","w"),indent=1)
print("wrote granularity_result.json")
