#!/usr/bin/env python3
# G-O3 STEP 2 -- equal-clock MFU. Compute-bound prefill forward, FP8-substitution vs bf16 baseline, at a LOCKED SM
# clock (the prior trap: 1.32x was DVFS headroom; equal-clock was ~parity). Read back clocks.sm to CONFIRM the lock
# held (no boost). Lead metric = FP8/bf16 achieved-throughput RATIO at the locked clock (FLOP-formula & peak
# independent). Also report absolute MFU (forward FLOP, explicit peak/dtype) + bf16 baseline MFU (is 85% even on the
# table single-GPU? reason A vs the quant tax reason B). CIPHER_FP8 / CIPHER_FP8_SHARE_ACT via env.
import os, sys, ctypes, time
import torch, pynvml
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL=os.environ.get("GO3_MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
SEQ=int(os.environ.get("GO3_SEQ","2048")); BATCH=int(os.environ.get("GO3_BATCH","1"))
ITERS=int(os.environ.get("GO3_ITERS","30")); WARM=int(os.environ.get("GO3_WARM","8"))
TAG=os.environ.get("GO3_TAG","run")
pynvml.nvmlInit(); dev=pynvml.nvmlDeviceGetHandleByIndex(0)
def smclk(): return pynvml.nvmlDeviceGetClockInfo(dev,pynvml.NVML_CLOCK_SM)
def powW(): return pynvml.nvmlDeviceGetPowerUsage(dev)/1000.0
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_fp8_calls_handled","cipher_rt_fp8_weights_quantized","cipher_rt_fp8_max_n","cipher_rt_fp8_is_active"):
    getattr(lib,s).restype=ctypes.c_ulong
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
c=m.config
# forward FLOP/tok from real dims (2*sum(m*n*k) over matmuls + attention 4*n_heads*head_dim*S), forward-only.
H=c.hidden_size; I=c.intermediate_size; L=c.num_hidden_layers; nh=c.num_attention_heads
nkv=getattr(c,"num_key_value_heads",nh); hd=H//nh; V=c.vocab_size
lin_per_layer = 2*H*H + 2*2*H*(nkv*hd) + 2*H*H + 2*(2*H*I) + 2*(I*H)   # q,o + k,v + gate,up + down
attn_per_layer = 4*nh*hd*SEQ                                            # QK^T + AV, forward
flop_tok = L*(lin_per_layer+attn_per_layer) + 2*H*V                     # + lm_head
GPUNAME=pynvml.nvmlDeviceGetName(dev);
PEAK_BF16_BOOST=float(os.environ.get("GO3_PEAK_BF16","989.4"))          # H100 SXM bf16 dense TFLOP/s @1980; PCIe~756
ids=tok("The history of artificial intelligence and high performance computing spans decades of research. ",return_tensors="pt").input_ids
ids=ids.repeat(1,(SEQ//ids.shape[1])+1)[:,:SEQ].repeat(BATCH,1).cuda()
toks=ids.numel()
import subprocess
LOCK=os.environ.get("GO3_LOCK_MHZ","1200")
with torch.no_grad():
    for _ in range(WARM): _=m(ids).logits
    torch.cuda.synchronize()
# Lock the SM clock IN-PROCESS right before timing (a bash-set lock is lost when the GPU idles between processes).
subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{LOCK},{LOCK}"],capture_output=True)
time.sleep(0.5)
with torch.no_grad():
    for _ in range(3): _=m(ids).logits   # settle at locked clock
    torch.cuda.synchronize()
clk0=smclk(); h0=lib.cipher_rt_fp8_calls_handled()
clks=[]; pws=[]; t0=time.perf_counter()
with torch.no_grad():
    for i in range(ITERS):
        _=m(ids).logits
        if i%5==0: clks.append(smclk()); pws.append(powW())
    torch.cuda.synchronize()
dt=time.perf_counter()-t0
h1=lib.cipher_rt_fp8_calls_handled()
tps=toks*ITERS/dt
ach_tflops=flop_tok*toks*ITERS/dt/1e12
clk_avg=sum(clks)/len(clks); peak_at_clk=PEAK_BF16_BOOST*clk_avg/1980.0   # bf16 peak scaled to locked clock
mfu=ach_tflops/peak_at_clk*100
print(f"[go3-s2 {TAG}] fp8_active={lib.cipher_rt_fp8_is_active()} handled_delta={h1-h0} wq={lib.cipher_rt_fp8_weights_quantized()} max_n={lib.cipher_rt_fp8_max_n()}",flush=True)
print(f"[go3-s2 {TAG}] CLOCK locked? sm_clk avg={clk_avg:.0f}MHz (min {min(clks)} max {max(clks)}) [boost=1980 -> lock held if ~flat] avgP={sum(pws)/len(pws):.0f}W",flush=True)
print(f"[go3-s2 {TAG}] tokens/fwd={toks} iters={ITERS} wall={dt:.3f}s  THROUGHPUT={tps:,.0f} tok/s",flush=True)
print(f"[go3-s2 {TAG}] forward FLOP/tok={flop_tok/1e9:.2f}G  ACHIEVED={ach_tflops:.1f} TFLOP/s  MFU={mfu:.1f}% (vs bf16-peak {peak_at_clk:.0f} TFLOP/s @ {clk_avg:.0f}MHz; {GPUNAME})",flush=True)
print(f"GO3_RESULT {TAG} tps={tps:.1f} tflops={ach_tflops:.2f} mfu={mfu:.2f} clk={clk_avg:.0f} handled_delta={h1-h0}",flush=True)
sys.stdout.flush(); os._exit(0)
