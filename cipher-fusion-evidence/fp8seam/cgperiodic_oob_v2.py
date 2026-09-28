# OUT-OF-BAND FP8 SDC detection — v2 catch validation under DEFAULT cudagraph.
# Add.9/10/12 left the CATCH INCONCLUSIVE: the in-place fault op (cgperiodic_oob.py) modeled a
# TRANSIENT post-production memory flip, which inductor split into two readers — the live `out`
# (attention) saw the fault (generation diverged) but the in-graph obuf.copy_(out) read the
# PRE-fault SSA value (resid 0, caught=false). That is a COMPILER ARTIFACT, not a detector flaw.
#
# v2 models the fault class the detector is actually scoped to: a PERSISTENT GEMM fault corrupts
# `out` AS PRODUCED, so EVERY reader sees the same wrong value. We model that faithfully with a
# FUNCTIONAL corruption op whose output feeds BOTH the exposed copy AND the downstream return via a
# single real data dependency — exactly the property of one corrupted memory all consumers read.
# No reader split is possible. Clean path (flag=0) returns out unchanged -> 0 FP preserved.
#
# HONEST SCOPE: this validates catch for persistent/at-production faults (out wrong when written,
# copy samples it). A transient flip timed strictly between the copy and a later reader is out of
# scope (the copy is a sampling point) — matches the detector's documented persistent-within-N bound.
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING"); os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import torch
N=int(os.environ.get("CG_N","8")); MAXB=512
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8 as CW
_LAYERS=[]
_oc=CW.create_weights
def _cw(self,layer,*a,**k):
    r=_oc(self,layer,*a,**k)
    Wn=layer.weight.shape[0] if layer.weight.dim()==2 else layer.weight.shape[-1]
    Kk=layer.weight.shape[1] if layer.weight.dim()==2 else layer.weight.shape[0]
    layer.register_buffer("xbuf",torch.zeros(MAXB,Kk,dtype=torch.bfloat16,device="cuda"),persistent=False)
    layer.register_buffer("obuf",torch.zeros(MAXB,Wn,dtype=torch.bfloat16,device="cuda"),persistent=False)
    layer.register_buffer("inj",torch.zeros(1,device="cuda"),persistent=False)   # device fault flag (test only)
    for t in (layer.xbuf,layer.obuf,layer.inj):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    layer._cipher_sch=self
    _LAYERS.append(layer); return r
CW.create_weights=_cw
# FUNCTIONAL fault op: returns a corrupted COPY of out (no-op when flag=0). Feeding its output to
# both the copy and downstream forces a single data dependency -> all readers see the SAME value,
# which is what a persistent at-production GEMM SDC does. (Test scaffolding only; absent in deploy.)
_clf=torch.library.Library("cfault2","FRAGMENT"); _clf.define("faultf(Tensor out, Tensor flag)->Tensor")
def _fltf(out,flag):
    o=out.clone()
    v=o.view(-1)
    v[:1]=v[:1]+flag.to(out.dtype)*5.0
    return o
_clf.impl("faultf",_fltf,"CompositeExplicitAutograd")
torch.library.register_fake("cfault2::faultf",lambda out,flag:torch.empty_like(out))
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    outc=torch.ops.cfault2.faultf(out, layer.inj)   # functional corrupt (flag=0 -> outc==out)
    b=x.shape[0]
    if b <= layer.xbuf.shape[0]:                     # decode(small)->copy, profile(16384)->skip
        layer.xbuf.narrow(0,0,b).copy_(x)
        layer.obuf.narrow(0,0,b).copy_(outc)         # copy CONSUMES outc -> no reader split
    return outc                                      # downstream CONSUMES outc -> same corrupted value
CW.apply_weights=_aw

import vllm.compilation.cuda_graph as cg
_orig_call=cg.CUDAGraphWrapper.__call__
TOL=float(os.environ.get("CG_TOL","0.05"))
class C:
    step=0; checks=0; first_detect=None; det_count=0; clean_checks=0; max_resid=0.0
_MX=torch.zeros(1,device="cuda")
def checker(b):
    if b<=0 or b>MAXB: return False
    _MX.zero_()
    with torch.no_grad():
        for L in _LAYERS:
            out2=_oa(L._cipher_sch,L,L.xbuf[:b],None)        # clean exact recompute (no fault op)
            r=(out2.float()-L.obuf[:b].float()).abs().max()/(L.obuf[:b].abs().max()+1e-6)
            torch.maximum(_MX, r.reshape(1), out=_MX)
    m=_MX.item()
    if m>C.max_resid: C.max_resid=m
    return m>TOL
def patched_call(self,*args,**kwargs):
    out=_orig_call(self,*args,**kwargs)
    if not cg.is_forward_context_available(): return out
    fc=cg.get_forward_context()
    if fc.cudagraph_runtime_mode!=self.runtime_mode or self.runtime_mode!=cg.CUDAGraphMode.FULL:
        return out
    C.step+=1
    if C.step % N == 0:
        C.checks+=1
        if checker(fc.batch_descriptor.num_tokens):
            C.det_count+=1
            if C.first_detect is None: C.first_detect=C.step
        else: C.clean_checks+=1
    return out
cg.CUDAGraphWrapper.__call__=patched_call

from vllm import LLM, SamplingParams
def main():
    M=os.environ["S_MODEL"]; OUT=os.environ["S_OUT"]; NTOK=int(os.environ.get("S_NTOK","256"))
    INJECT=os.environ.get("S_INJECT","0")=="1"
    llm=LLM(model=M,enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
    sp=SamplingParams(max_tokens=NTOK,temperature=0.0,ignore_eos=True,min_tokens=NTOK)
    p=["The history of the Roman Empire begins with"]
    llm.generate(p,sp,use_tqdm=False)
    C.step=0;C.checks=0;C.first_detect=None;C.det_count=0;C.clean_checks=0;C.max_resid=0.0
    o=llm.generate(p,sp,use_tqdm=False)
    res={"N":N,"layers":len(_LAYERS),"clean_checks":C.clean_checks,
         "clean_max_resid":round(C.max_resid,5),"clean_out":o[0].outputs[0].text[:30]}
    if INJECT:
        C.step=0;C.checks=0;C.first_detect=None;C.det_count=0;C.clean_checks=0;C.max_resid=0.0
        tgt=_LAYERS[len(_LAYERS)//2]; tgt.inj.fill_(1.0)
        o2=llm.generate(p,sp,use_tqdm=False)
        res["inj_first_detect_step"]=C.first_detect; res["inj_det_count"]=C.det_count
        res["inj_checks"]=C.checks; res["inj_max_resid"]=round(C.max_resid,5)
        res["inj_out"]=o2[0].outputs[0].text[:30]
        res["inj_diverged"]= o2[0].outputs[0].text[:30]!=o[0].outputs[0].text[:30]
        tb=min(8, tgt.obuf.shape[0]); o2t=_oa(tgt._cipher_sch,tgt,tgt.xbuf[:tb],None)
        res["target_direct_resid"]=round((o2t.float()-tgt.obuf[:tb].float()).abs().max().item(),5)
        res["caught_within_N"]= C.first_detect is not None and C.first_detect<=N
    json.dump(res,open(OUT,"w"),indent=1)
    print("CGOOBv2",json.dumps(res))
if __name__=="__main__": main()
