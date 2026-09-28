# OUT-OF-BAND FP8 SDC catch validation — v3. DEFAULT cudagraph.
# v1 (in-place fault + SEPARATE copy) and v2 (functional fault feeding copy+downstream) BOTH failed:
# generation diverged (fault reached live compute) but the exposed obuf stayed clean (resid 0). Root
# cause: the copy is a SEPARATELY-TRACED op, so inductor is free to realize its source buffer from a
# pre-fault SSA value of out. In-place vs functional doesn't matter — the SPLIT between corruption and
# copy is what inductor exploits.
#
# v3 removes that freedom: corruption AND the exposing copy happen INSIDE ONE opaque custom op, so the
# copy provably reads out's memory AFTER the corruption, atomically, and there is no separate copy node
# to reorder. This is the faithful persistent-fault model: a single corruption in out's memory that
# BOTH the exposed copy and the downstream consumer observe (the op mutates out in place and returns it
# aliased; downstream consumes the returned value). flag=0 -> the add is a no-op (0 FP, normal exposure).
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
    layer.register_buffer("inj",torch.zeros(1,device="cuda"),persistent=False)
    for t in (layer.xbuf,layer.obuf,layer.inj):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    layer._cipher_sch=self
    _LAYERS.append(layer); return r
CW.create_weights=_cw
# ONE op: (optionally) corrupt out's real memory, then copy x->xbuf and out->obuf. Atomic & opaque ->
# the copy cannot be reordered before the corruption, and there is no sibling clean SSA value to read.
_lib=torch.library.Library("cexpose","FRAGMENT")
_lib.define("expose(Tensor(a!) out, Tensor(b!) xbuf, Tensor(c!) obuf, Tensor x, Tensor flag)->Tensor(a!)")
def _expose(out,xbuf,obuf,x,flag):
    out.view(-1)[:1].add_(flag.to(out.dtype)*5.0)   # corrupt out IN PLACE (flag=0 -> no-op)
    b=x.shape[0]
    xbuf.narrow(0,0,b).copy_(x)
    obuf.narrow(0,0,b).copy_(out)                    # copy AFTER corruption, SAME op -> obuf faithful
    return out
_lib.impl("expose",_expose,"CompositeExplicitAutograd")
torch.library.register_fake("cexpose::expose",lambda out,xbuf,obuf,x,flag:out)
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    if x.shape[0] <= layer.xbuf.shape[0]:            # decode(small)->expose; profile(16384)->skip
        out=torch.ops.cexpose.expose(out, layer.xbuf, layer.obuf, x, layer.inj)
    return out
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
            out2=_oa(L._cipher_sch,L,L.xbuf[:b],None)        # clean exact recompute (no expose op)
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
    print("CGOOBv3",json.dumps(res))
if __name__=="__main__": main()
