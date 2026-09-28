# PROVE-OR-KILL: does the in-graph plain-copy of the FP8 output capture a PHYSICAL SDC?
# Physical fault = corrupt a WEIGHT in device memory host-side BETWEEN generates (invisible to
# inductor: graph already compiled/captured; replay reads the corrupted memory; the GEMM produces a
# genuinely wrong output with NO clean SSA version to snapshot -> unlike Add-9's reorderable torch-op
# inject). Discriminator on the target layer's exposed obuf:
#   resid_corrupt = |recompute(xbuf, CORRUPTED w) - obuf|  -> ~0 if obuf faithfully captured the output
#   resid_clean   = |recompute(xbuf, CLEAN w)     - obuf|  -> >tol if obuf = wrong output (CAPTURED)
# obuf faithful  <=> resid_corrupt~0 AND resid_clean>tol  -> out-of-band CAPTURES physical SDC.
# obuf lossy/clean<=> resid_corrupt>tol AND resid_clean~0 -> structurally lossy -> cheap path DEAD.
import os, json
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING"); os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import torch
MAXB=512
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8 as CW
_LAYERS=[]
_oc=CW.create_weights
def _cw(self,layer,*a,**k):
    r=_oc(self,layer,*a,**k)
    Wn=layer.weight.shape[0] if layer.weight.dim()==2 else layer.weight.shape[-1]
    Kk=layer.weight.shape[1] if layer.weight.dim()==2 else layer.weight.shape[0]
    layer.register_buffer("xbuf",torch.zeros(MAXB,Kk,dtype=torch.bfloat16,device="cuda"),persistent=False)
    layer.register_buffer("obuf",torch.zeros(MAXB,Wn,dtype=torch.bfloat16,device="cuda"),persistent=False)
    for t in (layer.xbuf,layer.obuf):
        try: torch._dynamo.mark_static_address(t)
        except Exception: pass
    layer._sch=self; _LAYERS.append(layer); return r
CW.create_weights=_cw
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    b=x.shape[0]
    if b<=layer.xbuf.shape[0]:
        layer.xbuf.narrow(0,0,b).copy_(x)     # PLAIN bf16 exposure copies (Add-9), no float/reduction
        layer.obuf.narrow(0,0,b).copy_(out)
    return out
CW.apply_weights=_aw

from vllm import LLM, SamplingParams
def recompute(layer, xb):                      # exact recompute with whatever weight is live now
    with torch.no_grad(): return _oa(layer._sch, layer, xb, None)

def main():
    M=os.environ["S_MODEL"]; OUT=os.environ["S_OUT"]
    llm=LLM(model=M,enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
    tgt=_LAYERS[len(_LAYERS)//2]
    cleanw=tgt.weight.data.clone()             # detector's clean reference (for the discriminator)
    sp=SamplingParams(max_tokens=64,temperature=0.0,ignore_eos=True,min_tokens=64)
    p=["The history of the Roman Empire begins with"]
    o1=llm.generate(p,sp,use_tqdm=False); clean_out=o1[0].outputs[0].text[:45]   # warm+capture; buffers populated
    b=1
    # baseline (no fault): obuf should match a same-weight recompute (0-FP sanity)
    o2c=recompute(tgt, tgt.xbuf[:b]); base_resid=(o2c.float()-tgt.obuf[:b].float()).abs().max().item()
    # ---- PHYSICAL SDC sanity: AGGRESSIVELY corrupt the target weight (zero it) to confirm the GEMM
    # actually reads layer.weight. Report weight attrs to locate the real kernel weight if this no-ops.
    wattrs={n:tuple(getattr(tgt,n).shape) for n in dir(tgt) if isinstance(getattr(tgt,n,None),torch.Tensor) and 'weight' in n.lower()}
    print("WATTRS", wattrs, "tgt_type", type(tgt).__name__, flush=True)
    tgt.weight.data.zero_()                              # zero the entire weight (max corruption)
    # run the forward with the CORRUPTED weight (captured graph reads the now-corrupted memory)
    o2=llm.generate(p,sp,use_tqdm=False); inj_out=o2[0].outputs[0].text[:45]
    # discriminator on the target layer's exposed obuf (post corrupted-weight forward)
    out2_corrupt=recompute(tgt, tgt.xbuf[:b])                                   # uses LIVE (corrupted) weight
    obuf=tgt.obuf[:b].float()
    resid_corrupt=(out2_corrupt.float()-obuf).abs().max().item()
    corrw=tgt.weight.data.clone()
    tgt.weight.data.copy_(cleanw)                                               # swap in CLEAN weight
    out2_clean=recompute(tgt, tgt.xbuf[:b])
    tgt.weight.data.copy_(corrw)                                                # restore corrupted
    resid_clean=(out2_clean.float()-obuf).abs().max().item()
    onorm=obuf.abs().max().item()+1e-6
    res={"clean_out":clean_out,"inj_out":inj_out,"generation_diverged":clean_out!=inj_out,
         "base_resid_0fp":round(base_resid,5),
         "resid_corrupt(obuf vs live-weight recompute)":round(resid_corrupt,5),
         "resid_clean(obuf vs CLEAN-weight recompute)":round(resid_clean,5),
         "resid_clean_rel":round(resid_clean/onorm,5),
         "obuf_CAPTURED_physical_SDC": (resid_corrupt < 1e-3) and (resid_clean/onorm > 0.02)}
    json.dump(res,open(OUT,"w"),indent=1); print("PHYS",json.dumps(res))
if __name__=="__main__": main()
