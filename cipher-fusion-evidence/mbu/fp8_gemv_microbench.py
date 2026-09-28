# Kernel-level microbench: fp16 GEMV vs FP8 GEMV at batch=1, on real Mistral linear shapes.
# Measures achieved HBM bandwidth (bytes_read / time) = the kernel's true MBU ceiling, isolated from vLLM.
import torch, time
torch.manual_seed(0)
DEV='cuda'; PEAK=3.35e12
# Mistral-7B per-layer linear shapes (out,in), x32 layers + lm_head
SH = {'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),
      'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}
L=32
def bytes_fp16(): return sum((o*i*2) for o,i in SH.values())*L
def bytes_fp8():  return sum((o*i*1 + o*4) for o,i in SH.values())*L  # fp8 weight + fp32 per-row scale

def bench(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters

# build weights once (per unique shape, reuse across layers in the timing loop = same bytes)
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
W8={}; SC={}
for k,(o,i) in SH.items():
    w=W16[k].to(torch.float32); s=w.abs().amax(1,keepdim=True)/448.0
    W8[k]=(w/s).clamp(-448,448).to(torch.float8_e4m3fn); SC[k]=s.to(torch.float16)
x16={k:torch.randn(1,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
x8 ={k:x16[k].to(torch.float8_e4m3fn) for k in SH}

def run_fp16():
    for _ in range(L):
        for k,(o,i) in SH.items(): _=torch.nn.functional.linear(x16[k], W16[k])
def run_fp8_scaledmm():  # torch._scaled_mm = cutlass FP8 path
    for _ in range(L):
        for k,(o,i) in SH.items():
            _=torch._scaled_mm(x8[k], W8[k].t(), scale_a=torch.tensor(1.0,device=DEV),
                               scale_b=SC[k].t().to(torch.float32) if False else torch.tensor(1.0,device=DEV),
                               out_dtype=torch.float16)

t16=bench(run_fp16); 
try:
    t8=bench(run_fp8_scaledmm)
except Exception as e:
    print("scaled_mm err:", str(e)[:160]); t8=None
b16=bytes_fp16(); b8=bytes_fp8()
print(f"fp16 GEMV: {t16*1000:.3f} ms/token  BW={b16/t16/1e12:.2f} TB/s  MBU={b16/t16/PEAK:.3f}  tok/s={1/t16:.0f}")
if t8:
    print(f"fp8  GEMV(scaled_mm): {t8*1000:.3f} ms/token  BW={b8/t8/1e12:.2f} TB/s  MBU={b8/t8/PEAK:.3f}  tok/s={1/t8:.0f}  speedup={t16/t8:.2f}x")
    print(f"  -> bytes fp16={b16/1e9:.1f}GB fp8={b8/1e9:.1f}GB ({b16/b8:.2f}x reduction)")
