import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING"); os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import torch
MODE=os.environ.get("MODE","copy")   # copy | off
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8 as CW
_cl=torch.library.Library("cpf","FRAGMENT"); _cl.define("sink(Tensor d, Tensor(a!) r)->()")
def _sk(d,r): torch.maximum(r,d.reshape(1),out=r)
_cl.impl("sink",_sk,"CompositeExplicitAutograd"); torch.library.register_fake("cpf::sink",lambda d,r:None)
if MODE=="copy":
    _oc=CW.create_weights
    def _cw(self,layer,*a,**k):
        r=_oc(self,layer,*a,**k)
        Wn=layer.weight.shape[0] if layer.weight.dim()==2 else layer.weight.shape[-1]
        Kk=layer.weight.shape[1] if layer.weight.dim()==2 else layer.weight.shape[0]
        layer.register_buffer("cxb",torch.zeros(1,device="cuda"),persistent=False)
        layer.register_buffer("cob",torch.zeros(1,device="cuda"),persistent=False)
        for t in (layer.cxb,layer.cob):
            try: torch._dynamo.mark_static_address(t)
            except Exception: pass
        return r
    CW.create_weights=_cw
    _oa=CW.apply_weights
    def _aw(self,layer,x,bias=None):
        out=_oa(self,layer,x,bias)
        if hasattr(layer,"cob"):
            torch.ops.cpf.sink(x.float().abs().amax(), layer.cob)     # touch all of x (exposure read)
            torch.ops.cpf.sink(out.float().abs().amax(), layer.cxb)   # touch all of out (exposure read)
        return out
    CW.apply_weights=_aw
from vllm import LLM, SamplingParams
llm=LLM(model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
sp=SamplingParams(max_tokens=256,temperature=0.0,ignore_eos=True,min_tokens=256)
p=["The history of the Roman Empire begins with"]
llm.generate(p,sp,use_tqdm=False)
best=0
for _ in range(3):
    t0=time.perf_counter(); o=llm.generate(p,sp,use_tqdm=False); dt=time.perf_counter()-t0
    g=sum(len(x.outputs[0].token_ids) for x in o); best=max(best,g/dt)
print("TIMED",json.dumps({"mode":MODE,"tok_s":round(best,2),"out":o[0].outputs[0].text[:30]}))
