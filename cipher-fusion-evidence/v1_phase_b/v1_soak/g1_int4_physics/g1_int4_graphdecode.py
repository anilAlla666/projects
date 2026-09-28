#!/usr/bin/env python3
# G-O1 increment-1: int4 (W4A16 Marlin, compressed-tensors) static-KV graph-decode.
# Extends today's fp16 KL=0 8B result (g1_capture_min/g1_graphdecode_latency_v2) to the INT4/Marlin regime.
# Answers TWO separated questions (advisor framing):
#   PHYSICS  : is int4 batch-1 decode bandwidth-floored (~4x fp16)? per-tok measured vs computed int4 weight floor
#              and vs today's fp16 8B (8.2ms/tok eager, ~bandwidth floor). The kill-shot for the engine latency premise.
#   FIDELITY : graph-int4 vs EAGER-INT4 (same model) must be exact 128/128 greedy. NOT vs fp16 (separate quality delta).
# Stock Marlin kernel via compressed-tensors -> no CIPHER .so. Same harness shape as the fp16 latency probe.
import os, sys, time, json, glob, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD","1")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)

path = sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/Llama-3.1-8B-Instruct-w4a16"
N    = int(sys.argv[2]) if len(sys.argv)>2 else 128
fp16_per_tok_ms = 8.2  # measured today on Llama-3.1-8B fp16 (memory: cipher-go1-graphcapture-probe)

tok = AutoTokenizer.from_pretrained(path); V = tok.vocab_size
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()

# Report what we actually loaded (quant method + on-disk weight bytes -> the int4 bandwidth floor).
qcfg = getattr(m.config, "quantization_config", None)
wbytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors")))
HBM_BW = 2.5e12  # TB/s effective, the divisor today's fp16 floor used (16GB/2.5TBs=6.4ms -> measured 8.2)
floor_ms = wbytes / HBM_BW * 1000.0
print(f"[load] {path}", flush=True)
print(f"[load] quant_config={qcfg if qcfg is None else getattr(qcfg,'quant_method',qcfg)} dtype={next(m.parameters()).dtype}", flush=True)
print(f"[load] on-disk weight bytes={wbytes/1e9:.2f}GB -> int4 weight-bandwidth floor ~{floor_ms:.2f}ms/tok (@2.5TB/s)", flush=True)

prompt = "The history of artificial intelligence began in the 1950s, when researchers first"
pids = tok(prompt, return_tensors="pt").input_ids.cuda(); P = pids.shape[1]; dev = pids.device; Lc = P+N+64
def clamp(t): return int(max(0, min(V-1, int(t))))

@torch.no_grad()
def prefill(c):
    return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()

# --- eager-int4 burst (advancing cache_position; the reference) ---
@torch.no_grad()
def eager_burst():
    c = StaticCache(config=m.config, max_cache_len=Lc); nt = prefill(c); out=[]
    torch.cuda.synchronize(); t0=time.time()
    for i in range(N):
        out.append(clamp(nt))
        nt = m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev),
               past_key_values=c, use_cache=True).logits[0,-1].argmax()
    torch.cuda.synchronize(); return out, time.time()-t0

es, te = eager_burst()

# --- graph-int4 burst (advancing replay; cache RETAINED = today's use-after-free fix) ---
cache = StaticCache(config=m.config, max_cache_len=Lc); t1 = prefill(cache)
sin = torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos = torch.tensor([P],device=dev)
g = torch.cuda.CUDAGraph()
try:
    with torch.no_grad(), torch.cuda.graph(g):
        o = m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog = o.logits
except Exception as e:
    print(f"\n[CAPTURE-FAIL] int4 Marlin kernel is NOT torch.cuda.graph capture-safe in this path:", flush=True)
    print(f"  {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"  -> FINDING: stock compressed-tensors Marlin needs a capture-safe path (cf CIPHER_MARLIN_CAPTURE_SAFE).", flush=True)
    sys.stdout.flush(); os._exit(0)

gs = [clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
torch.cuda.synchronize(); t0=time.time()
for _ in range(N-1):
    g.replay(); torch.cuda.synchronize()
    nt = clamp(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
tg = time.time()-t0

match = sum(a==b for a,b in zip(gs,es)); fd = next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y), -1)
res = {
  "model": path, "N": N, "P": P,
  "fidelity_graph_vs_eager_int4": f"{match}/{N}", "first_diff": fd, "fidelity_pass": match==N,
  "eager_ms": round(te*1000,1), "graph_ms": round(tg*1000,1), "speedup": round(te/tg,2),
  "eager_per_tok_ms": round(te/N*1000,2), "graph_per_tok_ms": round(tg/N*1000,2),
  "int4_weight_floor_ms": round(floor_ms,2), "fp16_per_tok_ms_measured": fp16_per_tok_ms,
}
print(f"\n=== INT4 (W4A16 Marlin) static-KV graph-decode, N={N} ===", flush=True)
print(f"  FIDELITY graph-int4 vs eager-int4: {match}/{N} (1stdiff@{fd})  {'PASS (KL=0)' if match==N else 'DIVERGES'}", flush=True)
print(f"  LATENCY  eager={res['eager_ms']:.0f}ms graph={res['graph_ms']:.0f}ms speedup={res['speedup']}x", flush=True)
print(f"           per-tok eager={res['eager_per_tok_ms']}ms graph={res['graph_per_tok_ms']}ms", flush=True)
print(f"  PHYSICS  int4 graph per-tok={res['graph_per_tok_ms']}ms  vs int4 weight floor ~{floor_ms:.2f}ms  vs fp16 measured {fp16_per_tok_ms}ms/tok", flush=True)
spd = fp16_per_tok_ms/res['graph_per_tok_ms'] if res['graph_per_tok_ms'] else 0
print(f"           int4 vs fp16 per-tok speedup = {spd:.2f}x (bandwidth-floored expectation ~3-4x; <2x => dequant-bound STOP signal)", flush=True)
json.dump(res, open("/home/ubuntu/g1_int4_graphdecode_result.json","w"), indent=2)
print(f"  [written] /home/ubuntu/g1_int4_graphdecode_result.json", flush=True)
sys.stdout.flush(); os._exit(0)
