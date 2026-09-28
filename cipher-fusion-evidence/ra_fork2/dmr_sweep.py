#!/usr/bin/env python3
# FORK-2 Mech 2 (DMR shadow, small S) + Mech 3 (lagged full recompute, large S): ONE green-context sweep.
# Main runs the Mistral-7B linear forward (weight-streamed, distinct per-layer weights -> real HBM reads) on a
# (120-S)-SM partition; a SHADOW runs the SAME forward on a disjoint S-SM partition CONCURRENTLY (advisor: the
# shadow must do real weight reads so global HBM contention -- which green-ctx SM-partitioning does NOT isolate --
# shows up). Per phase (decode B=128 / prefill M=512): full-132 baseline, main-alone(P), main-under-shadow,
# shadow tp -> total DMR cost vs full, SM-carve component, HBM-contention component, and shadow COVERAGE.
# Pure driver-API + cuBLAS ctypes; shared primary address space. No torch, no .so, no fork-1 edit.
import ctypes as C, time, json, threading, subprocess
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
 ("cuMemAlloc_v2",[C.POINTER(C.c_void_p),C.c_size_t]),("cuStreamSynchronize",[C.c_void_p]),
 ("cuStreamCreate",[C.POINTER(C.c_void_p),C.c_uint])]:
    getattr(cu,n).argtypes=a
cublas.cublasCreate_v2.argtypes=[C.POINTER(C.c_void_p)]; cublas.cublasSetStream_v2.argtypes=[C.c_void_p,C.c_void_p]
cublas.cublasGemmEx.argtypes=[C.c_void_p]+[C.c_int]*5+[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]

ck(cu.cuInit(0),"init"); dev=C.c_int(0); cu.cuDeviceGet(C.byref(dev),0)
pctx=C.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(C.byref(pctx),dev),"retain"); ck(cu.cuCtxSetCurrent(pctx),"setP")
whole=Res(); ck(cu.cuDeviceGetDevResource(dev,C.byref(whole),1),"whole"); TOTAL=whole.smCount
H,KV,FFN=4096,1024,14336; L=32
def linears(M): return [("q",M,H,H),("k",M,H,KV),("v",M,H,KV),("o",M,H,H),("gate",M,H,FFN),("up",M,H,FFN),("down",M,FFN,H)]
al=C.c_float(1); be=C.c_float(0)
def alloc(nb): q=C.c_void_p(); ck(cu.cuMemAlloc_v2(C.byref(q),nb),"alloc"); return q
print("[alloc] ~14GB Mistral-7B fp16 weight set (distinct per layer -> HBM streaming)...",flush=True)
W={}
for li in range(L):
    for (nm,_,Kd,Nd) in linears(1): W[(li,nm)]=alloc(Kd*Nd*2)
MMAX=512; actbuf=alloc(MMAX*FFN*2); outbuf=alloc(MMAX*FFN*2)
print("[alloc] done",flush=True)

def step(h,M):
    for li in range(L):
        for (nm,_,Kd,Nd) in linears(M):
            bk(cublas.cublasGemmEx(h,0,0,M,Nd,Kd,C.byref(al),actbuf,2,M,W[(li,nm)],2,Kd,C.byref(be),outbuf,2,M,68,-1),"gemm")

def split8():
    nb=C.c_uint(0); ck(cu.cuDevSmResourceSplitByCount(None,C.byref(nb),C.byref(whole),None,0,8),"q8")
    ng=nb.value; res=(Res*ng)(); rem=Res(); nb2=C.c_uint(ng)
    ck(cu.cuDevSmResourceSplitByCount(res,C.byref(nb2),C.byref(whole),C.byref(rem),0,8),"s8")
    return [res[i] for i in range(nb2.value)]
groups=split8()
def make_partition(reslist):
    n=len(reslist); arr=(Res*n)(*reslist)
    desc=C.c_void_p(); ck(cu.cuDevResourceGenerateDesc(C.byref(desc),arr,n),"desc")
    g=C.c_void_p(); ck(cu.cuGreenCtxCreate(C.byref(g),desc,dev,0x1),"green")
    cc=C.c_void_p(); ck(cu.cuCtxFromGreenCtx(C.byref(cc),g),"fromg"); ck(cu.cuCtxSetCurrent(cc),"setG")
    strm=C.c_void_p(); ck(cu.cuGreenCtxStreamCreate(C.byref(strm),g,0x1,0),"gstream")
    h=C.c_void_p(); bk(cublas.cublasCreate_v2(C.byref(h)),"create"); bk(cublas.cublasSetStream_v2(h,strm),"setstream")
    return dict(cctx=cc,strm=strm,h=h,sm=sum(r.smCount for r in reslist))

def tp_alone(part,M,secs=2.0):
    ck(cu.cuCtxSetCurrent(part['cctx']),"setcur")
    for _ in range(2): step(part['h'],M)
    ck(cu.cuStreamSynchronize(part['strm']),"sync")
    t0=time.perf_counter(); n=0
    while time.perf_counter()-t0<secs: step(part['h'],M); ck(cu.cuStreamSynchronize(part['strm']),"sync"); n+=1
    return n/(time.perf_counter()-t0)

def tp_full(M,secs=2.0):  # full 132 via primary ctx default stream
    ck(cu.cuCtxSetCurrent(pctx),"setP")
    s=C.c_void_p(); ck(cu.cuStreamCreate(C.byref(s),0),"sfull")
    h=C.c_void_p(); bk(cublas.cublasCreate_v2(C.byref(h)),"chf"); bk(cublas.cublasSetStream_v2(h,s),"shf")
    for _ in range(2): step(h,M)
    ck(cu.cuStreamSynchronize(s),"sync")
    t0=time.perf_counter(); n=0
    while time.perf_counter()-t0<secs: step(h,M); ck(cu.cuStreamSynchronize(s),"sync"); n+=1
    return n/(time.perf_counter()-t0)

def concurrent(main_p,shadow_p,M,secs=2.0):
    stop=[False]; sh_n=[0]
    def sh():
        ck(cu.cuCtxSetCurrent(shadow_p['cctx']),"shset")
        for _ in range(2): step(shadow_p['h'],M)
        ck(cu.cuStreamSynchronize(shadow_p['strm']),"shsync")
        while not stop[0]:
            step(shadow_p['h'],M); ck(cu.cuStreamSynchronize(shadow_p['strm']),"shsync"); sh_n[0]+=1
    ck(cu.cuCtxSetCurrent(main_p['cctx']),"mset")
    for _ in range(2): step(main_p['h'],M)
    ck(cu.cuStreamSynchronize(main_p['strm']),"msync")
    th=threading.Thread(target=sh); th.start(); time.sleep(0.3)
    ck(cu.cuCtxSetCurrent(main_p['cctx']),"mset2")
    t0=time.perf_counter(); n=0
    while time.perf_counter()-t0<secs: step(main_p['h'],M); ck(cu.cuStreamSynchronize(main_p['strm']),"msync"); n+=1
    dt=time.perf_counter()-t0; stop[0]=True; th.join()
    return n/dt, sh_n[0]/dt

base={}
for ph,M in [("decode",128),("prefill",512)]:
    base[ph]=tp_full(M); print(f"[full-132] {ph}: {base[ph]:.2f} steps/s",flush=True)

results=[]
for S in [8,16,32,64]:
    ng=S//8
    shadow_p=make_partition(groups[:ng]); main_p=make_partition(groups[ng:])
    row={"shadow_sm":shadow_p['sm'],"main_sm":main_p['sm']}
    for ph,M in [("decode",128),("prefill",512)]:
        m_alone=tp_alone(main_p,M)
        m_cc,s_cc=concurrent(main_p,shadow_p,M)
        cost_full=1-m_cc/base[ph]; carve=1-main_p['sm']/TOTAL
        contention=1-m_cc/m_alone if m_alone>0 else 0; cover=s_cc/m_cc if m_cc>0 else 0
        row[ph]=dict(main_alone_tp=round(m_alone,2),main_cc_tp=round(m_cc,2),shadow_tp=round(s_cc,2),
            cost_vs_full_pct=round(cost_full*100,2),sm_carve_pct=round(carve*100,2),
            hbm_contention_pct=round(contention*100,2),coverage_frac=round(cover,3))
        print(f"S={shadow_p['sm']} main={main_p['sm']} {ph}: full={base[ph]:.1f} alone={m_alone:.1f} cc={m_cc:.1f} sh={s_cc:.1f} "
              f"| DMRcost={cost_full*100:.1f}% (carve {carve*100:.1f}% + HBM-cont {contention*100:.1f}%) cover={cover:.2f}",flush=True)
    results.append(row)
clk=subprocess.run(["nvidia-smi","--query-gpu=clocks.gr,memory.used","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
out={"total_sm":int(TOTAL),"baseline_full132_steps_s":base,"clock":clk,"stranded_sm":12,"sweep":results}
json.dump(out,open("/home/ubuntu/cipher-fusion-evidence/ra_fork2/dmr_sweep_result.json","w"),indent=1)
print("wrote dmr_sweep_result.json")
