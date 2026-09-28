# OUT-OF-BAND FP8 SDC catch validation — v4, EAGER control (enforce_eager=True).
# Purpose: isolate WHY the cudagraph catch fails. In eager there is NO inductor and NO buffer reuse;
# apply_weights runs in strict program order, so an in-place fault on out BEFORE the copy is provably
# captured by the copy. If eager CATCHES, the copy+discriminator logic is correct and the cudagraph
# failure (v1/v2/v3) is purely compiler buffer-realization -> the only clean cudagraph validation is a
# CUDA .so that corrupts output memory outside the compiler's model (Add.12).
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
    layer.register_buffer("inj",torch.zeros(1,device="cuda"),persistent=False)
    layer._cipher_sch=self
    _LAYERS.append(layer); return r
CW.create_weights=_cw
_oa=CW.apply_weights
def _aw(self,layer,x,bias=None):
    out=_oa(self,layer,x,bias)
    if not hasattr(layer,"obuf"): return out
    out.view(-1)[:1].add_(layer.inj.to(out.dtype)*5.0)   # in-place fault on out (eager: strict order)
    b=x.shape[0]
    if b <= layer.xbuf.shape[0]:
        layer.xbuf.narrow(0,0,b).copy_(x)
        layer.obuf.narrow(0,0,b).copy_(out)              # copy AFTER fault -> faithful in eager
    return out
CW.apply_weights=_aw
TOL=float(os.environ.get("CG_TOL","0.05"))
def check_all(b):
    mx=0.0; worst=-1
    with torch.no_grad():
        for i,L in enumerate(_LAYERS):
            if L.obuf[:b].abs().max().item()==0: continue
            out2=_oa(L._cipher_sch,L,L.xbuf[:b],None)
            r=((out2.float()-L.obuf[:b].float()).abs().max()/(L.obuf[:b].abs().max()+1e-6)).item()
            if r>mx: mx=r; worst=i
    return mx,worst
from vllm import LLM, SamplingParams
def main():
    M=os.environ["S_MODEL"]; OUT=os.environ["S_OUT"]; NTOK=int(os.environ.get("S_NTOK","64"))
    llm=LLM(model=M,enforce_eager=True,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
    sp=SamplingParams(max_tokens=NTOK,temperature=0.0,ignore_eos=True,min_tokens=NTOK)
    p=["The history of the Roman Empire begins with"]
    # clean pass
    o=llm.generate(p,sp,use_tqdm=False)
    b=1
    clean_resid,clean_worst=check_all(b)
    res={"layers":len(_LAYERS),"clean_max_resid":round(clean_resid,5),"clean_out":o[0].outputs[0].text[:30]}
    # inject pass
    tgt=_LAYERS[len(_LAYERS)//2]; tgt.inj.fill_(1.0)
    o2=llm.generate(p,sp,use_tqdm=False)
    inj_resid,inj_worst=check_all(b)
    res["inj_max_resid"]=round(inj_resid,5); res["inj_worst_layer"]=inj_worst
    res["target_layer"]=len(_LAYERS)//2
    res["inj_out"]=o2[0].outputs[0].text[:30]
    res["inj_diverged"]= o2[0].outputs[0].text[:30]!=o[0].outputs[0].text[:30]
    res["caught"]= inj_resid>TOL
    res["clean_fp"]= clean_resid>TOL
    json.dump(res,open(OUT,"w"),indent=1)
    print("CGOOBv4eager",json.dumps(res))
if __name__=="__main__": main()
