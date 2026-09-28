#!/usr/bin/env python3
# Settle the advisor's catch: do the may13 fused RMSNorm/SiLU kernels (a) produce CORRECT output,
# and (b) actually survive torch.cuda.graph capture+replay — or is the run_mistral_fused "+14.5%"
# an artifact of stream=NULL (kernels fire once during capture, never captured; replay reads stale)?
# Compares greedy token streams: bf16 reference vs fused-eager vs fused-graph, with stream=NULL (pre-fix)
# AND stream=current (the documented one-line fix). Output match == correct+captured.
import os, ctypes, torch
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP

ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL("/home/ubuntu/cipher-may13-evidence/libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init(); rt.cipher_fusion_kernels_init()
rt.cipher_fused_rmsnorm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
rt.cipher_fused_silu_mul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]

MODEL="/home/ubuntu/models/Mistral-7B-v0.1"; N=24
STREAM_MODE = os.environ.get("STREAM","null")   # "null" (pre-fix) | "current" (documented fix)
def sptr():
    return None if STREAM_MODE=="null" else ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="cuda").eval()
ids=tok("The future of artificial intelligence in GPU computing is to make",return_tensors="pt").input_ids.cuda()
plen=ids.shape[1]

orig_rms=MistralRMSNorm.forward; orig_mlp=MistralMLP.forward
def fused_rms(self,x):
    out=torch.empty_like(x); flat=x.reshape(-1,x.shape[-1]).contiguous(); of=out.reshape(-1,x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(),self.weight.contiguous().data_ptr(),of.data_ptr(),
                            flat.shape[0],flat.shape[-1],float(self.variance_epsilon),sptr())
    return out
def fused_mlp(self,x):
    g=self.gate_proj(x); u=self.up_proj(x); g=g.contiguous(); u=u.contiguous(); o=torch.empty_like(g)
    rt.cipher_fused_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),g.numel(),sptr())
    return self.down_proj(o)

def greedy_eager(patched):
    if patched: MistralRMSNorm.forward=fused_rms; MistralMLP.forward=fused_mlp
    else: MistralRMSNorm.forward=orig_rms; MistralMLP.forward=orig_mlp
    with torch.no_grad():
        o=model.generate(ids,max_new_tokens=N,do_sample=False,pad_token_id=tok.eos_token_id)
    return o[0,plen:].tolist()

def greedy_graph(patched):
    if patched: MistralRMSNorm.forward=fused_rms; MistralMLP.forward=fused_mlp
    else: MistralRMSNorm.forward=orig_rms; MistralMLP.forward=orig_mlp
    cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=plen+N+4,device="cuda",dtype=torch.float16)
    with torch.no_grad():
        cp=torch.arange(plen,device="cuda"); o=model(input_ids=ids,cache_position=cp,past_key_values=cache,use_cache=True,return_dict=True)
    nt=o.logits[:,-1:].argmax(-1); input_ids=nt.detach().clone(); cache_pos=torch.tensor([plen],device="cuda")
    outl=torch.empty(1,1,model.config.vocab_size,device="cuda",dtype=torch.float16)
    side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                oo=model(input_ids=input_ids,cache_position=cache_pos,past_key_values=cache,use_cache=True,return_dict=True)
            outl.copy_(oo.logits); input_ids.copy_(outl.argmax(-1)); cache_pos+=1
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            oo=model(input_ids=input_ids,cache_position=cache_pos,past_key_values=cache,use_cache=True,return_dict=True)
        outl.copy_(oo.logits); input_ids.copy_(outl.argmax(-1))
    toks=[]
    for _ in range(N):
        cache_pos+=1; g.replay(); torch.cuda.synchronize(); toks.append(int(input_ids.item()))
    return toks

ref=greedy_eager(False)
fe =greedy_eager(True)
gg =greedy_graph(True)
gb =greedy_graph(False)
def m(a,b): return sum(1 for x,y in zip(a,b) if x==y)/len(a)
print(f"STREAM={STREAM_MODE}")
print("ref (bf16 eager):        ", tok.decode(ref))
print("fused-eager:             ", tok.decode(fe), f"  match_vs_ref={m(fe,ref):.0%}")
print("fused-graph (patched):   ", tok.decode(gg), f"  match_vs_ref={m(gg,ref):.0%}")
print("baseline-graph (no fuse):", tok.decode(gb), f"  match_vs_ref={m(gb,ref):.0%}")
