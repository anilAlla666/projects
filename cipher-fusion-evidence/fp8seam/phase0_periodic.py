# Phase 0 micro: which periodic mechanism actually SAVES cost under vLLM-style manual cudagraph?
#  (A) DUAL GRAPHS: compile a CHECKED variant (recompute present) and an UNCHECKED variant (absent),
#      capture each as its own cudagraph, alternate host-side. Expect unchecked step ~= bare matmul.
#  (B) DEVICE-GATE: single graph, recompute * device 0/1 flag. Expect NO saving (kernels still launch).
import torch, vllm._custom_ops
dev="cuda"
cl=torch.library.Library("cipperp","FRAGMENT")
cl.define("sink(Tensor d, Tensor(a!) res, Tensor(b!) cnt) -> ()")
cl.impl("sink", lambda d,res,cnt:(torch.maximum(res,d.reshape(1),out=res),cnt.add_(1)) and None, "CompositeExplicitAutograd")
torch.library.register_fake("cipperp::sink", lambda d,res,cnt:None)
_real=torch.ops._C.cutlass_scaled_mm
M,K,N=1,4096,14336                       # decode shape (M=1), a heavy down_proj-ish linear
a=(torch.randn(M,K,device=dev)*0.1).to(torch.float8_e4m3fn)
b=(torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn).t()
sa=torch.tensor(1.0,device=dev); sb=torch.tensor(1.0,device=dev)
RES=torch.zeros(1,device=dev); CNT=torch.zeros(1,device=dev); INJ=torch.zeros(1,device=dev); GATE=torch.zeros(1,device=dev)
for t in (RES,CNT,INJ,GATE): torch._dynamo.mark_static_address(t)
OUT=torch.empty((M,N),dtype=torch.bfloat16,device=dev)

def checked():                            # recompute PRESENT
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
    o=o+INJ.to(o.dtype)*0.5
    o2=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o2,a,b,sa,sb,None)
    d=(o2.float()-o.float()).abs().amax(); torch.ops.cipperp.sink(d,RES,CNT); return o
def unchecked():                          # recompute ABSENT
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None); return o
def devgate():                            # recompute present but result * device 0/1 gate (kernels still in graph)
    o=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o,a,b,sa,sb,None)
    o2=torch.empty((M,N),dtype=torch.bfloat16,device=dev); _real(o2,a,b,sa,sb,None)
    d=((o2.float()-o.float()).abs().amax())*GATE.squeeze(); torch.ops.cipperp.sink(d,RES,CNT); return o

cch=torch.compile(checked,fullgraph=True); cun=torch.compile(unchecked,fullgraph=True); cdg=torch.compile(devgate,fullgraph=True)
for fn in (cch,cun,cdg):
    for _ in range(3): fn()
torch.cuda.synchronize()

def capture(fn):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
g_ch=capture(cch); g_un=capture(cun); g_dg=capture(cdg)

def timed(g,iters=2000):
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    for _ in range(50): g.replay()
    torch.cuda.synchronize(); s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/iters    # ms/step
t_un=timed(g_un); t_ch=timed(g_ch); t_dg=timed(g_dg)
print("P0PER-TIMING", {"unchecked_ms":round(t_un,4),"checked_ms":round(t_ch,4),"devgate_ms":round(t_dg,4),
    "checked/unchecked":round(t_ch/t_un,3),"devgate/unchecked":round(t_dg/t_un,3),
    "dualgraph_saves": t_un < 0.7*t_ch, "devgate_saves": t_dg < 0.7*t_ch})

# host-alternated dual graph, N=8: unchecked x7 + checked x1, with a fault on the checked step
RES.zero_();CNT.zero_();INJ.zero_(); Nperiod=8; caught_step=None
for step in range(40):
    if step==20: INJ.fill_(1.0)                 # persistent fault from step 20
    if step % Nperiod == (Nperiod-1):
        RES.zero_(); g_ch.replay(); torch.cuda.synchronize()
        if RES.item()>0 and caught_step is None: caught_step=step
    else:
        g_un.replay()
torch.cuda.synchronize()
# effective per-step cost of the N=8 schedule
sched_ms=(t_un*(Nperiod-1)+t_ch)/Nperiod
print("P0PER-SCHED", {"N":Nperiod,"eff_ms_per_step":round(sched_ms,4),"eff/unchecked":round(sched_ms/t_un,3),
    "caught_step":caught_step,"caught_within_N_of_fault": caught_step is not None and (caught_step-20)<Nperiod})
