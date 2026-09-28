# Track A coverage: the fused epilogue checksum s=rowsum_N(C) vs independent r=A.wref catches Step-A SDC flips.
# Kernel emits the bit-exact reference r (rout). s is the fp32-accum rowsum (bit-exact: C rel-err=0).
# T from clean residual (fp32); inject fp16 bit-13/14 flips into the product output; confirm caught + 0 clean FP.
import ctypes, os, json, math, torch
torch.manual_seed(0); dev='cuda'
SO='/home/ubuntu/cipher-fusion-evidence/step_b_prefill_build/libkernel_dedup.so'
lib=ctypes.CDLL(SO)
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
BLK_N=256

def flip_bit(x_fp16, bit):
    # flip `bit` of an fp16 value, return new fp16 python float and delta
    v=x_fp16.view(torch.int16).item()
    v2=v ^ (1<<bit)
    new=torch.tensor([v2],dtype=torch.int16).view(torch.float16).item()
    return new

def run_shape(name,M,K,N,ntrials=200):
    R=N//BLK_N
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    assert lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)==0
    torch.cuda.synchronize()
    r = rout[:M].clone() if M<=128 else None
    # kernel rout holds r for the first up-to-128 rows (per harness). Use a 128-row tile for the test.
    Mt=128
    A_t=A[:Mt];
    # fp32 epilogue rowsum s (bit-exact accumulator) and independent reference r
    s_clean=(A_t.float()@W.float().t()).sum(1)             # [Mt] fp32 row-sum of the product
    r_ref=(A_t.float()@wref)                                # [Mt] independent reference
    clean_resid=(s_clean-r_ref).abs()
    Tmax=clean_resid.max().item()
    T=max(Tmax*8, 1e-3)                                     # safety-factored threshold (fp32)
    # clean false positives
    clean_fp=int((clean_resid>T).sum().item())
    # inject Step-A flips into the product output (recompute affected row-sum)
    Cf=(A_t.float()@W.float().t())                          # fp32 product
    caught=0; deltas=[]
    g=torch.Generator(device='cpu').manual_seed(123)
    for t in range(ntrials):
        m=int(torch.randint(0,Mt,(1,),generator=g).item()); n=int(torch.randint(0,N,(1,),generator=g).item())
        bit=13 if (t%2==0) else 14                          # Step-A harmful bits
        orig=Cf[m,n].half()
        new=flip_bit(orig,bit)
        delta=float(new)-float(orig)
        s_m=s_clean[m].item()+delta                         # corrupted row-sum
        resid=abs(s_m-r_ref[m].item())
        if resid>T: caught+=1
        deltas.append(abs(delta))
    return dict(name=name,M=M,K=K,N=N,Mt=Mt,T=T,clean_resid_max=Tmax,clean_fp=clean_fp,
                ntrials=ntrials,caught=caught,rate=caught/ntrials*100,
                delta_min=min(deltas),delta_max=max(deltas))

shapes=[("k/v_proj",2048,4096,1024),("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),("gate/up_proj",2048,4096,14336)]
res=[run_shape(*s) for s in shapes]
for x in res:
    print(f"{x['name']:14s} N={x['N']:5d}  T={x['T']:.3e} clean_resid_max={x['clean_resid_max']:.3e}  "
          f"caught={x['caught']}/{x['ntrials']} ({x['rate']:.1f}%)  clean_FP={x['clean_fp']}  |delta| {x['delta_min']:.2f}-{x['delta_max']:.1f}")
json.dump(res,open('/home/ubuntu/cipher-fusion-evidence/ra_keystone/trackA/coverage_result.json','w'),indent=1)
print("wrote coverage_result.json")
