# PERIODIC out-of-band FP8 detection. Expose per-layer I/O via PLAIN bf16 copies (free, no float/
# reduction in-graph, warning 1). Every Nth step, host-side EAGER exact recompute on the buffered I/O
# (warning 2: real orchestration live). One compiled forward; check outside the graph. In-process
# (delivery proven in Add.4). MEASURED end-to-end tok/s.
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
    layer.register_buffer("inj",torch.zeros(1,device="cuda"),persistent=False)   # device fault flag (test)
    for t in (layer.xbuf,layer.obuf,layer.inj):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    layer._cipher_sch=self    # the scheme, to recompute host-side via _oa
    _LAYERS.append(layer); return r
CW.create_weights=_cw
# in-place fault op (simulates a REAL SDC in out's memory; non-reorderable so the copy captures it,
# unlike a functional add which inductor can reorder around the in-place copy)
_clf=torch.library.Library("cfault","FRAGMENT"); _clf.define("fault(Tensor(a!) out, Tensor flag)->()")
def _flt(out,flag): out.view(-1)[:1].add_(flag.to(out.dtype)*5.0)
_clf.impl("fault",_flt,"CompositeExplicitAutograd"); torch.library.register_fake("cfault::fault",lambda out,flag:None)
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    torch.ops.cfault.fault(out, layer.inj)         # in-place SDC sim (flag=0 no-op); BEFORE copy so obuf captures it
    b=x.shape[0]
    if b <= layer.xbuf.shape[0]:                    # dynamo specializes: decode(small)->copy, profile(16384)->skip
        layer.xbuf.narrow(0,0,b).copy_(x)           # PLAIN bf16 copy (free, no float/reduction)
        layer.obuf.narrow(0,0,b).copy_(out)
    return out
CW.apply_weights=_aw

import vllm.compilation.cuda_graph as cg
_orig_call=cg.CUDAGraphWrapper.__call__
TOL=float(os.environ.get("CG_TOL","0.05"))   # principled fp8 relative tol (Add.6: clean <=0.7%, 5% safe)
class C:
    step=0; checks=0; first_detect=None; det_count=0; clean_checks=0; max_resid=0.0
_MX=torch.zeros(1,device="cuda")
def checker(b):
    # host-side EAGER exact recompute on the buffered I/O; accumulate residual on-device, ONE sync/check
    if b<=0 or b>MAXB: return False
    _MX.zero_()
    with torch.no_grad():
        for L in _LAYERS:
            out2=_oa(L._cipher_sch,L,L.xbuf[:b],None)        # exact recompute (same op)
            r=(out2.float()-L.obuf[:b].float()).abs().max()/(L.obuf[:b].abs().max()+1e-6)
            torch.maximum(_MX, r.reshape(1), out=_MX)
    m=_MX.item()                                              # ONE sync per check
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
    best=0
    for _ in range(3):
        t0=time.perf_counter(); o=llm.generate(p,sp,use_tqdm=False); dt=time.perf_counter()-t0
        g=sum(len(x.outputs[0].token_ids) for x in o); best=max(best,g/dt)
    res={"N":N,"layers":len(_LAYERS),"tok_s":round(best,2),"checks":C.checks,
         "clean_checks":C.clean_checks,"clean_max_resid":round(C.max_resid,5),"out":o[0].outputs[0].text[:30]}
    if INJECT:
        C.step=0;C.checks=0;C.first_detect=None;C.det_count=0;C.max_resid=0.0
        tgt=_LAYERS[len(_LAYERS)//2]; tgt.inj.fill_(1.0)
        o2=llm.generate(p,sp,use_tqdm=False)
        res["inj_first_detect_step"]=C.first_detect; res["inj_det_count"]=C.det_count
        res["inj_checks"]=C.checks; res["inj_max_resid"]=round(C.max_resid,5)
        res["inj_out"]=o2[0].outputs[0].text[:30]
        # direct sanity: recompute target layer from its buffers vs stored output
        tb=min(8, tgt.obuf.shape[0]); o2t=_oa(tgt._cipher_sch,tgt,tgt.xbuf[:tb],None)
        res["target_direct_resid"]=round((o2t.float()-tgt.obuf[:tb].float()).abs().max().item(),5)
        res["caught_within_N"]= C.first_detect is not None and C.first_detect<=N
    json.dump(res,open(OUT,"w"),indent=1)
    print("CGOOB",json.dumps({k:res.get(k) for k in ["N","tok_s","layers","clean_checks","clean_max_resid","inj_first_detect_step","caught_within_N"]}))
if __name__=="__main__": main()
