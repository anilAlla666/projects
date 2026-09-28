# GATE B (FIXED discriminator): does the in-graph plain-copy of the FP8 output capture a PHYSICAL SDC?
# Compare obuf (after corrupted-weight forward) vs obuf_clean (SAVED true clean output, SAME input).
# No live recompute -> avoids the cached-processed-weight trap (Add.10). Corrupt ONLY the target layer's
# weight, so earlier layers (hence the target's INPUT) are unchanged -> obuf_clean vs obuf_corrupt is a
# clean same-input comparison; any difference = the copy saw the physical fault.
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
    _LAYERS.append(layer); return r
CW.create_weights=_cw
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    b=x.shape[0]
    if b<=layer.xbuf.shape[0]:
        layer.xbuf.narrow(0,0,b).copy_(x)      # plain bf16 exposure copies (the out-of-band channel)
        layer.obuf.narrow(0,0,b).copy_(out)
    return out
CW.apply_weights=_aw

from vllm import LLM, SamplingParams
def main():
    M=os.environ["S_MODEL"]; OUT=os.environ["S_OUT"]
    llm=LLM(model=M,enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
    tgt=_LAYERS[len(_LAYERS)//2]
    p=["The history of the Roman Empire begins with"]
    sp1=SamplingParams(max_tokens=1,temperature=0.0,ignore_eos=True,min_tokens=1)
    sp16=SamplingParams(max_tokens=16,temperature=0.0,ignore_eos=True,min_tokens=16)
    # clean single-step: capture target's TRUE clean I/O for this input
    llm.generate(p,sp1,use_tqdm=False)
    xbuf_clean=tgt.xbuf.clone(); obuf_clean=tgt.obuf.clone()
    clean16=llm.generate(p,sp16,use_tqdm=False)[0].outputs[0].text[:50]
    # ---- PHYSICAL SDC: ZERO the target weight in device memory (max corruption; output -> ~0, so obuf
    # MUST change drastically if captured; unambiguous). In-place fp8 write, invisible to inductor.
    tgt.weight.data.zero_()
    # same single-step forward (same prompt/input); only target weight changed
    llm.generate(p,sp1,use_tqdm=False)
    obuf_corrupt=tgt.obuf.clone()
    inj16=llm.generate(p,sp16,use_tqdm=False)[0].outputs[0].text[:50]
    # discriminator (direct tensor comparison, no recompute)
    on=obuf_clean.abs().max().item()+1e-6
    resid_obuf=(obuf_corrupt.float()-obuf_clean.float()).abs().max().item()
    resid_xbuf=(tgt.xbuf.float()-xbuf_clean.float()).abs().max().item()   # target INPUT should be unchanged
    res={"clean16":clean16,"inj16":inj16,"generation_diverged":clean16!=inj16,
         "resid_obuf(corrupt vs clean output)":round(resid_obuf,5),"resid_obuf_rel":round(resid_obuf/on,5),
         "resid_xbuf(target input change, should~0)":round(resid_xbuf,5),
         "COPY_CAPTURED_physical_SDC": (resid_obuf/on > 0.02)}
    json.dump(res,open(OUT,"w"),indent=1); print("GATEB",json.dumps(res))
if __name__=="__main__": main()
