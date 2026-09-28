#!/usr/bin/env python3
"""
GAP 6 — vLLM incompatibility root-cause: minimal CUDA-graph + injection MECHANISM (no vLLM run).
Demonstrates the two structural reasons an injected in-context recompute detector cannot live inside
vLLM's CUDA-graphed decode:
  (1) FROZEN GRAPH: kernels launched AFTER capture do NOT appear in replay -> a detector cannot be
      added to vLLM's already-captured decode graph post-hoc.
  (2) CAPTURE-ILLEGAL host op: a detector that makes a host-visible decision (device->host sync, e.g.
      .item() to branch on "mismatch?") DURING capture trips cudaErrorStreamCaptureInvalidated — the
      exact class of failure injected CIPHER actuators hit inside vLLM's capture (cf. Koopman
      cudaHostAlloc err 900). And a raw external allocation during capture likewise invalidates.
READ-ONLY, no production .so, no injection of the real .so. fp32. Prints PASS/observed-error per point.
"""
import json, ctypes, torch
dev='cuda'; HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/gap6_vllm'
res={}

# ---- (1) FROZEN GRAPH — MEASURED: replay re-runs ONLY the captured op-set; a post-capture kernel is absent ----
xbuf=torch.ones(1024,1024,device=dev)        # static input buffer the captured op reads
ybuf=torch.zeros_like(xbuf)                  # captured op writes here (ybuf=xbuf*2)
zbuf=torch.zeros_like(xbuf)                  # a POST-capture kernel writes here (zbuf=xbuf*100); must NOT be replayed
def stepfn(): ybuf.copy_(xbuf*2.0)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): stepfn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): stepfn()
# AFTER capture: run a DIFFERENT kernel ONCE (eager) — a would-be injected detector op (zbuf=xbuf*100).
xbuf.fill_(3.0); zbuf.copy_(xbuf*100.0)      # zbuf now = 300, ran once, NOT part of graph g
# change the input and REPLAY: the captured op must track the new input; the post-capture kernel must NOT re-run.
xbuf.fill_(7.0); g.replay(); torch.cuda.synchronize()
captured_tracked_input = bool(torch.allclose(ybuf, torch.full_like(ybuf,14.0)))  # ybuf=7*2=14 -> captured op replayed w/ NEW input
postcap_absent          = bool(torch.allclose(zbuf, torch.full_like(zbuf,300.0))) # zbuf still 300 (not 700) -> post-capture kernel NOT replayed
res['frozen_graph']=dict(measured=True, captured_op_replayed_with_new_input=captured_tracked_input,
    post_capture_kernel_in_replay=(not postcap_absent), ybuf_val=float(ybuf.flatten()[0]), zbuf_val=float(zbuf.flatten()[0]),
    note='replay re-ran captured ybuf=xbuf*2 (tracks input -> 14) but did NOT re-run the post-capture zbuf=xbuf*100 (stuck at 300, not 700) => frozen captured op-set; an injected detector kernel launched after capture is never replayed')
print(f"[gap6.1] FROZEN GRAPH MEASURED: captured op replayed with NEW input (ybuf={float(ybuf.flatten()[0])}=14? {captured_tracked_input}); "
      f"post-capture kernel ABSENT from replay (zbuf={float(zbuf.flatten()[0])}=300 not 700? {postcap_absent})",flush=True)

# ---- (2a) host-sync (.item) during capture -> capture invalidated ----
g2=torch.cuda.CUDAGraph(); err_hostsync=None
try:
    with torch.cuda.graph(g2):
        z=(xbuf*2.0)
        flag=z.sum().item()          # device->host sync = the detector's host-visible 'mismatch?' decision
except Exception as e:
    err_hostsync=f"{type(e).__name__}: {str(e)[:160]}"
res['host_sync_during_capture']=dict(raised=err_hostsync is not None, error=err_hostsync)
print(f"[gap6.2a] HOST-SYNC (.item) during capture -> {'RAISED: '+err_hostsync if err_hostsync else 'did NOT raise (unexpected)'}",flush=True)

# ---- (2b) raw external cudaHostAlloc during capture -> stream-capture-invalidated (injected-lib class) ----
# torch's caching allocator uses the capture pool (legal); a RAW driver alloc (what an injected lib does)
# is outside the pool. We test cudaHostAlloc via cudart during an active torch capture stream.
err_rawalloc=None; raw_ret=None
try:
    cudart=ctypes.CDLL('libcudart.so')
    g3=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3):
        zz=(xbuf*2.0)
        p=ctypes.c_void_p()
        raw_ret=cudart.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(1<<20), ctypes.c_uint(0))  # 0=default
        # cudaHostAlloc returns cudaError_t; nonzero => error (716/cudaErrorStreamCaptureUnsupported or similar)
    res['raw_hostalloc_during_capture']=dict(cuda_ret=raw_ret, raised=False)
except Exception as e:
    err_rawalloc=f"{type(e).__name__}: {str(e)[:160]}"
    res['raw_hostalloc_during_capture']=dict(raised=True, error=err_rawalloc, cuda_ret=raw_ret)
print(f"[gap6.2b] RAW cudaHostAlloc during capture -> ret={raw_ret} raised={err_rawalloc}",flush=True)

# decode error name for the raw ret if available
try:
    cudart=ctypes.CDLL('libcudart.so'); cudart.cudaGetErrorString.restype=ctypes.c_char_p
    if raw_ret: res['raw_hostalloc_during_capture']['ret_name']=cudart.cudaGetErrorString(ctypes.c_int(raw_ret)).decode()
except Exception: pass

json.dump(res,open(f'{HERE}/gap6_mechanism_result.json','w'),indent=1,default=str)
print(f"[gap6] wrote gap6_mechanism_result.json",flush=True)
