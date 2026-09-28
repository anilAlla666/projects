#!/usr/bin/env python3
# V.0 Phase A -- MAX MFU on the compute-bound regime. Batch is the legitimate MFU lever (more compute-bound ->
# amortizes the non-GEMM fixed overhead). Run at the NATURAL sustained operating clock (GPU is 700W-power-limited to
# ~1380MHz under full tensor-core load; boost 1980 is unsustainable -- NOT an artificially-low clock to inflate MFU%).
# Measure the ACTUAL clock during each timed window, MFU = achieved / peak-at-measured-clock. Sweep batch, find where
# end-to-end MFU genuinely peaks; report MFU + (batch, clock). FP8 via env (CIPHER_FP8). Model loaded once, swept.
import os, sys, ctypes, time, json
import torch, pynvml
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL=os.environ.get("GO3_MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
SEQ=int(os.environ.get("GO3_SEQ","2048")); MODE=os.environ.get("V0_MODE","bf16")
BATCHES=[int(x) for x in os.environ.get("V0_BATCHES","1,2,4,8").split(",")]
ITERS=int(os.environ.get("GO3_ITERS","20")); WARM=int(os.environ.get("GO3_WARM","6"))
PEAK_BF16_BOOST=989.4  # H100 SXM bf16 dense TFLOP/s @1980 (FP8 peak = 2x)
pynvml.nvmlInit(); d=pynvml.nvmlDeviceGetHandleByIndex(0)
def smclk(): return pynvml.nvmlDeviceGetClockInfo(d,pynvml.NVML_CLOCK_SM)
def powW(): return pynvml.nvmlDeviceGetPowerUsage(d)/1000.0
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_fp8_calls_handled","cipher_rt_fp8_weights_quantized","cipher_rt_fp8_is_active"):
    getattr(lib,s).restype=ctypes.c_ulong
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
c=m.config; H=c.hidden_size; I=c.intermediate_size; L=c.num_hidden_layers; nh=c.num_attention_heads
nkv=getattr(c,"num_key_value_heads",nh); hd=H//nh; V=c.vocab_size
lin=2*H*H+2*2*H*(nkv*hd)+2*H*H+2*(2*H*I)+2*(I*H)
flop_tok=L*(lin+4*nh*hd*SEQ)+2*H*V
base=tok("The history of artificial intelligence and high performance computing spans decades. ",return_tensors="pt").input_ids
def make(bs):
    x=base.repeat(1,(SEQ//base.shape[1])+1)[:, :SEQ]; return x.repeat(bs,1).cuda()
results=[]
print(f"[v0-A] MODE={MODE} fp8_active={lib.cipher_rt_fp8_is_active()} model={os.path.basename(MODEL)} seq={SEQ} flop/tok={flop_tok/1e9:.2f}G",flush=True)
for bs in BATCHES:
    try:
        ids=make(bs); toks=ids.numel()
        with torch.no_grad():
            for _ in range(WARM): _=m(ids).logits
            torch.cuda.synchronize()
        clks=[]; pws=[]; t0=time.perf_counter()
        with torch.no_grad():
            for i in range(ITERS):
                _=m(ids).logits
                if i%3==0: clks.append(smclk()); pws.append(powW())
            torch.cuda.synchronize()
        dt=time.perf_counter()-t0
        tps=toks*ITERS/dt; ach=flop_tok*toks*ITERS/dt/1e12
        clk=sum(clks)/len(clks); peak=PEAK_BF16_BOOST*clk/1980.0; mfu=ach/peak*100
        hd_=lib.cipher_rt_fp8_calls_handled()
        r=dict(mode=MODE,batch=bs,toks=toks,tps=round(tps,1),tflops=round(ach,1),mfu=round(mfu,1),clk=round(clk),powW=round(sum(pws)/len(pws)),handled=hd_)
        results.append(r)
        print(f"[v0-A] batch={bs:>2} n={toks:>6} | {tps:>8,.0f} tok/s | {ach:>6.1f} TFLOP/s | MFU={mfu:>5.1f}% (peak {peak:.0f}@{clk:.0f}MHz) | {sum(pws)/len(pws):.0f}W | fp8_handled={hd_}",flush=True)
    except RuntimeError as e:
        print(f"[v0-A] batch={bs} OOM/err: {str(e)[:80]}",flush=True); torch.cuda.empty_cache()
best=max(results,key=lambda r:r['mfu']) if results else {}
print(f"[v0-A MAX] MODE={MODE} peak end-to-end MFU={best.get('mfu')}% at batch={best.get('batch')} clk={best.get('clk')}MHz ({best.get('tflops')} TFLOP/s, {best.get('tps')} tok/s)",flush=True)
print("V0A_JSON "+json.dumps(dict(mode=MODE,results=results,best=best)),flush=True)
sys.stdout.flush(); os._exit(0)
