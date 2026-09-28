#!/usr/bin/env python3
# G-O1 PROBE-FIRST (read-only): settle the multiplexing DISPATCH mechanism before the scheduler build.
# M1 GIL/dispatch wall: is concurrent<serial GIL, single-stream, or graph exclusivity? serial vs multi-stream
#    (single-thread issue) vs python-threaded -- which gives real GPU overlap.
# M2 same-model batch (G1∩G3): forward at B={1,2,4,8,15} same model -> does tok/s lift toward compute-bound?
# M3 distinct-model interleave (consolidation): per-burst GPU time -> agents-per-GPU = (burst+idle)/burst, at the
#    GIL-bound burst vs the bandwidth-ceiling burst (shows dispatch efficiency unlocks the agent count).
import os, sys, time, threading, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
INT4=["/home/ubuntu/models_int4/Mistral-7B-v0.1","/home/ubuntu/models_int4/Qwen2-7B","/home/ubuntu/models_int4/Llama-3.1-8B"]
from transformers import AutoModelForCausalLM, AutoTokenizer
m=[]; ids=[]
for p in INT4:
    mm=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, device_map="cuda")
    t=AutoTokenizer.from_pretrained("/home/ubuntu/models/"+os.path.basename(p))
    m.append(mm); ids.append(t("Write a detailed analysis of", return_tensors="pt").input_ids.cuda())
def sync(): torch.cuda.synchronize()
@torch.no_grad()
def fwd(i, B=1, S=8, stream=None):
    x=ids[i][:, :S]; x=x.repeat(B,1) if B>1 else x
    if stream is not None:
        with torch.cuda.stream(stream): return m[i](x)
    return m[i](x)
# warmup
for i in range(3): fwd(i);
sync()

print("=== M1: DISPATCH WALL (3 distinct models, 1 forward each) ===",flush=True)
R=10
sync(); t=time.time()
for _ in range(R):
    for i in range(3): fwd(i)
sync(); T_serial=(time.time()-t)/R
streams=[torch.cuda.Stream() for _ in range(3)]
sync(); t=time.time()
for _ in range(R):
    for i in range(3): fwd(i, stream=streams[i])
sync(); T_stream=(time.time()-t)/R
def threaded():
    ths=[threading.Thread(target=fwd,args=(i,)) for i in range(3)]
    [x.start() for x in ths]; [x.join() for x in ths]
sync(); t=time.time()
for _ in range(R): threaded()
sync(); T_thread=(time.time()-t)/R
print(f"  serial single-thread:   {T_serial*1000:.1f}ms/round -> {3/T_serial:.0f} fwd/s",flush=True)
print(f"  multi-stream 1-thread:  {T_stream*1000:.1f}ms/round -> {3/T_stream:.0f} fwd/s  (overlap vs serial: {T_serial/T_stream:.2f}x)",flush=True)
print(f"  python-threaded (3):    {T_thread*1000:.1f}ms/round -> {3/T_thread:.0f} fwd/s  (vs serial: {T_serial/T_thread:.2f}x <1 => GIL serializes)",flush=True)

print("=== M2: SAME-MODEL BATCH LIFT (model0, forward B={1,2,4,8,15}, S=8) ===",flush=True)
base=None
for B in [1,2,4,8,15]:
    sync(); t=time.time()
    for _ in range(R): fwd(0, B=B, S=8)
    sync(); lat=(time.time()-t)/R; toks=B*8; tps=toks/lat
    if base is None: base=tps
    print(f"  B={B:2d}: {lat*1000:.1f}ms/fwd  {tps:.0f} tok/s  ({tps/base:.2f}x vs B=1)  [{toks} tok/fwd]",flush=True)

print("=== M3: CONSOLIDATION (per-burst GPU time -> agents-per-GPU) ===",flush=True)
@torch.no_grad()
def burst(i,n=30): return m[i].generate(ids[i], max_new_tokens=n, do_sample=False, use_cache=True)
burst(0,4)  # warmup
sync(); t=time.time(); burst(0,30); sync(); bt=time.time()-t
# bandwidth-ceiling burst: 30 tok / 957 tok/s (4-bit 7B HBM roofline)
bt_ceiling=30/957.0
print(f"  measured 30-tok burst (generate, GIL-bound): {bt*1000:.0f}ms -> {30/bt:.0f} tok/s single-agent",flush=True)
print(f"  bandwidth-ceiling 30-tok burst: {bt_ceiling*1000:.0f}ms (at 957 tok/s roofline)",flush=True)
for idle in [2,3,5]:
    a_gil=(bt+idle)/bt; a_ceil=(bt_ceiling+idle)/bt_ceiling
    print(f"  idle={idle}s -> agents/GPU = (burst+idle)/burst: GIL-bound dispatch ~{a_gil:.0f}, bandwidth-ceiling dispatch ~{a_ceil:.0f}",flush=True)
print("  (these assume NON-colliding bursts served serially; true P99 under correlated arrivals needs the scheduler -- not this probe)",flush=True)
sys.stdout.flush(); os._exit(0)
