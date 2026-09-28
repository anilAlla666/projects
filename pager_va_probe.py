#!/usr/bin/env python3
# PROBE 1 (throwaway, no production source): does a real cuBLAS GEMM read its WEIGHT through a cuMemMap-backed
# VA I own, and does unmap->remap->recopy RESTORE it correctly? Identity matmul C=I@W=W (exact in fp16, no
# accumulation) -> C must equal W byte-for-byte iff cuBLAS read W through my VA. The load-bearing pager primitive.
import ctypes, torch
import warnings; warnings.filterwarnings("ignore")
cuda=ctypes.CDLL("libcuda.so.1"); rt=ctypes.CDLL("libcudart.so"); cublas=ctypes.CDLL("libcublas.so.13")
def CK(rc,what):
    if rc!=0: raise RuntimeError(f"{what} -> CUDA driver err {rc}")
class Loc(ctypes.Structure): _fields_=[("type",ctypes.c_int),("id",ctypes.c_int)]
class Prop(ctypes.Structure):
    _fields_=[("type",ctypes.c_int),("reqHandle",ctypes.c_int),("location",Loc),
              ("win32",ctypes.c_void_p),("compType",ctypes.c_ubyte),("rdma",ctypes.c_ubyte),
              ("usage",ctypes.c_ushort),("reserved",ctypes.c_ubyte*4)]
class Acc(ctypes.Structure): _fields_=[("location",Loc),("flags",ctypes.c_int)]
PINNED=1; DEV=1; RW=3
torch.cuda.init(); CK(cuda.cuInit(0),"cuInit")
prop=Prop(); prop.type=PINNED; prop.location.type=DEV; prop.location.id=0
gran=ctypes.c_size_t(); CK(cuda.cuMemGetAllocationGranularity(ctypes.byref(gran),ctypes.byref(prop),0),"gran")
G=gran.value
N=4096; raw=N*N*2; size=((raw+G-1)//G)*G
print(f"region: {N}x{N} fp16 = {raw/2**20:.0f} MiB, granularity {G/2**20:.1f} MiB, mapped {size/2**20:.0f} MiB")

va=ctypes.c_ulonglong()
CK(cuda.cuMemAddressReserve(ctypes.byref(va),ctypes.c_size_t(size),ctypes.c_size_t(0),ctypes.c_ulonglong(0),ctypes.c_ulonglong(0)),"reserve")
VA=va.value
def make_hbm():
    h=ctypes.c_ulonglong(); CK(cuda.cuMemCreate(ctypes.byref(h),ctypes.c_size_t(size),ctypes.byref(prop),ctypes.c_ulonglong(0)),"create"); return h
def map_va(h):
    CK(cuda.cuMemMap(ctypes.c_ulonglong(VA),ctypes.c_size_t(size),ctypes.c_size_t(0),h,ctypes.c_ulonglong(0)),"map")
    acc=Acc(); acc.location.type=DEV; acc.location.id=0; acc.flags=RW
    CK(cuda.cuMemSetAccess(ctypes.c_ulonglong(VA),ctypes.c_size_t(size),ctypes.byref(acc),ctypes.c_size_t(1)),"setaccess")

# known weight W + identity A + output C (torch tensors in normal mem; W also copied into my VA)
torch.manual_seed(0)
W=torch.randn(N,N,device="cuda",dtype=torch.float16)*0.1
A=torch.eye(N,device="cuda",dtype=torch.float16)
C=torch.empty(N,N,device="cuda",dtype=torch.float16)
def copy_W_into_VA():
    CK(rt.cudaMemcpy(ctypes.c_void_p(VA),ctypes.c_void_p(W.data_ptr()),ctypes.c_size_t(raw),3),"memcpy DtoD")  # 3=D2D
    torch.cuda.synchronize()
# cublas handle + GemmEx(C = A @ B), B at my VA. col-major identity copy: C bytes == W bytes.
h=ctypes.c_void_p(); cublas.cublasCreate_v2(ctypes.byref(h))
R16F=2; COMPUTE32F=68; ALGO_DEFAULT=-1; OP_N=0
alpha=ctypes.c_float(1.0); beta=ctypes.c_float(0.0)
def gemm():
    C.zero_(); torch.cuda.synchronize()
    st=cublas.cublasGemmEx(h, OP_N, OP_N, N, N, N, ctypes.byref(alpha),
        ctypes.c_void_p(A.data_ptr()), R16F, N, ctypes.c_void_p(VA), R16F, N,
        ctypes.byref(beta), ctypes.c_void_p(C.data_ptr()), R16F, N, COMPUTE32F, ALGO_DEFAULT)
    torch.cuda.synchronize();
    if st!=0: raise RuntimeError(f"cublasGemmEx -> {st}")

# --- round 1: map, copy W, GEMM, verify C==W (cuBLAS read W through my VA) ---
hbm1=make_hbm(); map_va(hbm1); copy_W_into_VA(); gemm()
ok1=torch.equal(C,W); rel1=(C.float()-W.float()).abs().max().item()
print(f"[round 1 mapped] cuBLAS GEMM(weight=my-VA) == W: {ok1}  (max_abs_diff={rel1:.2e})")

# --- EVICT: unmap + release the HBM (the VA now backs nothing) ---
CK(cuda.cuMemUnmap(ctypes.c_ulonglong(VA),ctypes.c_size_t(size)),"unmap")
CK(cuda.cuMemRelease(hbm1),"release")
print("[evict] unmapped + released HBM (VA reserved, unbacked)")

# --- PAGE-IN: fresh HBM, remap SAME VA, recopy W, GEMM, verify restored ---
hbm2=make_hbm(); map_va(hbm2); copy_W_into_VA(); gemm()
ok2=torch.equal(C,W); rel2=(C.float()-W.float()).abs().max().item()
print(f"[round 2 remapped] cuBLAS GEMM(weight=my-VA) == W after evict+pagein: {ok2}  (max_abs_diff={rel2:.2e})")
print(f"PRIMITIVE {'HOLDS' if ok1 and ok2 else 'FAILS'}: weight-behind-pageable-VA read by a non-CIPHER GEMM, restored across unmap/remap")
CK(cuda.cuMemUnmap(ctypes.c_ulonglong(VA),ctypes.c_size_t(size)),"unmap2"); CK(cuda.cuMemRelease(hbm2),"release2")
