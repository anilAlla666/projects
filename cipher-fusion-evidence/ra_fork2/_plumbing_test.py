import ctypes as C, time
cu=C.CDLL("libcuda.so.1"); cublas=C.CDLL("libcublas.so")
def ck(r,w):
    if r: s=C.c_char_p(); cu.cuGetErrorString(r,C.byref(s)); raise RuntimeError(f"{w}:{r} {s.value}")
def bk(r,w):
    if r: raise RuntimeError(f"{w}: cublas {r}")
class Res(C.Structure): _fields_=[("type",C.c_uint),("_p",C.c_ubyte*92),("smCount",C.c_uint),("_o",C.c_ubyte*44)]
for n,a in [("cuDeviceGetDevResource",[C.c_int,C.POINTER(Res),C.c_uint]),
 ("cuDevSmResourceSplitByCount",[C.POINTER(Res),C.POINTER(C.c_uint),C.POINTER(Res),C.POINTER(Res),C.c_uint,C.c_uint]),
 ("cuDevResourceGenerateDesc",[C.POINTER(C.c_void_p),C.POINTER(Res),C.c_uint]),
 ("cuGreenCtxCreate",[C.POINTER(C.c_void_p),C.c_void_p,C.c_int,C.c_uint]),
 ("cuCtxFromGreenCtx",[C.POINTER(C.c_void_p),C.c_void_p]),
 ("cuGreenCtxStreamCreate",[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint,C.c_int]),
 ("cuCtxSetCurrent",[C.c_void_p]),("cuDevicePrimaryCtxRetain",[C.POINTER(C.c_void_p),C.c_int]),
 ("cuMemAlloc_v2",[C.POINTER(C.c_void_p),C.c_size_t]),("cuStreamSynchronize",[C.c_void_p])]:
    getattr(cu,n).argtypes=a
cublas.cublasCreate_v2.argtypes=[C.POINTER(C.c_void_p)]; cublas.cublasSetStream_v2.argtypes=[C.c_void_p,C.c_void_p]
cublas.cublasGemmEx.argtypes=[C.c_void_p]+[C.c_int]*5+[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
ck(cu.cuInit(0),"init"); dev=C.c_int(0); cu.cuDeviceGet(C.byref(dev),0)
p=C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(p),dev),"retain"); ck(cu.cuCtxSetCurrent(p),"setP")
whole=Res(); ck(cu.cuDeviceGetDevResource(dev,C.byref(whole),1),"whole"); print("total SM",whole.smCount)
# split into 8s, take ONE 8-SM group
nb=C.c_uint(0); cu.cuDevSmResourceSplitByCount(None,C.byref(nb),C.byref(whole),None,0,8)
ng=nb.value; res=(Res*ng)(); rem=Res(); nb2=C.c_uint(ng)
cu.cuDevSmResourceSplitByCount(res,C.byref(nb2),C.byref(whole),C.byref(rem),0,8)
arr=(Res*1)(res[0]); desc=C.c_void_p(); ck(cu.cuDevResourceGenerateDesc(C.byref(desc),arr,1),"desc")
g=C.c_void_p(); ck(cu.cuGreenCtxCreate(C.byref(g),desc,dev,0x1),"green")
cc=C.c_void_p(); ck(cu.cuCtxFromGreenCtx(C.byref(cc),g),"fromg"); ck(cu.cuCtxSetCurrent(cc),"setG")
strm=C.c_void_p(); ck(cu.cuGreenCtxStreamCreate(C.byref(strm),g,0x1,0),"gstream")
h=C.c_void_p(); bk(cublas.cublasCreate_v2(C.byref(h)),"create"); bk(cublas.cublasSetStream_v2(h,strm),"setstream")
M,Kd,Nd=512,4096,14336
def alloc(nb): q=C.c_void_p(); ck(cu.cuMemAlloc_v2(C.byref(q),nb),"alloc"); return q
A=alloc(M*Kd*2); B=alloc(Kd*Nd*2); Cc=alloc(M*Nd*2)
al=C.c_float(1); be=C.c_float(0)
def gemm(): bk(cublas.cublasGemmEx(h,0,0,M,Nd,Kd,C.byref(al),A,2,M,B,2,Kd,C.byref(be),Cc,2,M,68,-1),"gemm")
for _ in range(3): gemm()
ck(cu.cuStreamSynchronize(strm),"sync")
t0=time.perf_counter(); n=0
while time.perf_counter()-t0<1.0: gemm(); ck(cu.cuStreamSynchronize(strm),"sync"); n+=1
dt=time.perf_counter()-t0
flop=2*M*Kd*Nd; print(f"8-SM partition gate GEMM: {n/dt:.1f}/s, {flop*n/dt/1e12:.1f} TFLOP/s  (8/132 of full)")
print("PLUMBING OK")
