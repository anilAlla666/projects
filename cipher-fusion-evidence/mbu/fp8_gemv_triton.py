# Custom GEMV-optimized FP8 decode kernel (Triton) vs fp16 GEMV — does a GEMV-specialized
# FP8 kernel reach fp16's ~74% MBU at batch=1 (=> ~2x, since half the bytes)?
import torch, time, triton, triton.language as tl
torch.manual_seed(0); DEV='cuda'; PEAK=3.35e12
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),
    'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32

@triton.jit
def gemv_fp8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid*BLOCK_N + tl.arange(0, BLOCK_N)          # output rows this block owns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        xk = tl.load(x_ptr + ks, mask=ks < K, other=0.0).to(tl.float32)      # [BLOCK_K] fp16 act
        w = tl.load(w_ptr + rows[:,None]*K + ks[None,:],
                    mask=(rows[:,None]<N)&(ks[None,:]<K), other=0.0)          # [BLOCK_N,BLOCK_K] fp8
        acc += tl.sum(w.to(tl.float32) * xk[None,:], axis=1)
    s = tl.load(s_ptr + rows, mask=rows<N, other=0.0).to(tl.float32)
    tl.store(y_ptr + rows, (acc*s).to(tl.float16), mask=rows<N)

def gemv_fp8(x, w8, s, N, K):
    y=torch.empty(N, device=DEV, dtype=torch.float16)
    grid=(triton.cdiv(N,64),)
    gemv_fp8_kernel[grid](x, w8, s, y, N, K, BLOCK_N=64, BLOCK_K=256)
    return y

def bytes_fp16(): return sum(o*i*2 for o,i in SH.values())*L
def bytes_fp8():  return sum(o*i*1 + o*4 for o,i in SH.values())*L
def bench(fn,iters=50,warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters

W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
W8={}; SC={}
for k,(o,i) in SH.items():
    w=W16[k].to(torch.float32); s=(w.abs().amax(1)/448.0)
    W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous(); SC[k]=s.to(torch.float16).contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}

# correctness vs fp16 reference (one shape)
k='down'; o,i=SH[k]
ref=torch.nn.functional.linear(x[k].unsqueeze(0), W16[k]).squeeze(0)
got=gemv_fp8(x[k], W8[k], SC[k], o, i)
rel=(got.float()-ref.float()).norm()/ref.float().norm()
print(f"correctness (down_proj) rel_err={rel:.4f}  (fp8 quant noise expected ~0.05-0.1)")

def run_fp16():
    for _ in range(L):
        for k,(o,i) in SH.items(): _=torch.nn.functional.linear(x[k].unsqueeze(0), W16[k])
def run_fp8_triton():
    for _ in range(L):
        for k,(o,i) in SH.items(): _=gemv_fp8(x[k], W8[k], SC[k], o, i)

t16=bench(run_fp16); t8=bench(run_fp8_triton)
b16=bytes_fp16(); b8=bytes_fp8()
print(f"fp16 GEMV:        {t16*1000:.3f} ms  MBU={b16/t16/PEAK:.3f}  tok/s={1/t16:.0f}")
print(f"fp8 GEMV(triton): {t8*1000:.3f} ms  MBU={b8/t8/PEAK:.3f}  tok/s={1/t8:.0f}  SPEEDUP={t16/t8:.2f}x")
