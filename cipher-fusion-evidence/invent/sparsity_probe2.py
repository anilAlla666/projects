# Real activation sparsity: hook down_proj inputs (the 14336-dim SwiGLU intermediate) during a REAL forward pass.
import torch, glob, json, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"; MIST=glob.glob("/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/")[0]
tok=AutoTokenizer.from_pretrained(MIST)
model=AutoModelForCausalLM.from_pretrained(MIST,torch_dtype=torch.float16,device_map=DEV)
model.eval()
caps={}  # layer -> list of intermediate vectors (per position)
def mk(layer):
    def hook(mod,inp): caps[layer]=inp[0].detach()  # inp to down_proj = [B,T,14336]
    return hook
for li,lyr in enumerate(model.model.layers):
    lyr.mlp.down_proj.register_forward_pre_hook(mk(li))
# down_proj column norms (importance weighting) per layer
coln={li:lyr.mlp.down_proj.weight.detach().float().norm(dim=0) for li,lyr in enumerate(model.model.layers)}  # [14336]
texts=["The transformer architecture processes tokens using attention and feed-forward networks.",
       "Paris is the capital of France and a major center of art and culture.",
       "def quicksort(arr): if len(arr)<=1: return arr"]
I=14336; res=[]
with torch.no_grad():
    for t in texts:
        ids=tok(t,return_tensors="pt").input_ids.to(DEV)
        model(ids)
        for li in [0,8,16,24,31]:
            inter=caps[li][0]  # [T,14336]
            cn=coln[li]
            for ti in range(inter.shape[0]):
                imp=(inter[ti].float().abs()*cn)
                srt,_=imp.sort(descending=True); tot=srt.sum().clamp(min=1e-9)
                cum=srt.cumsum(0)/tot
                k99=int((cum<0.99).sum())+1; k999=int((cum<0.999).sum())+1
                res.append((li,k99/I,k999/I))
f99=[r[1] for r in res]; f999=[r[2] for r in res]
print(f"REAL Mistral-7B decode MLP sparsity ({len(res)} layer,token samples):")
print(f"  keep 99.0% of down_proj output -> read {100*st.mean(f99):.1f}% of neurons (median {100*st.median(f99):.1f}%) => {1/st.mean(f99):.1f}x down-traffic cut")
print(f"  keep 99.9% of down_proj output -> read {100*st.mean(f999):.1f}% of neurons => {1/st.mean(f999):.1f}x")
print(f"  per-layer keep@99%: "+", ".join(f"L{li}:{100*st.mean([r[1] for r in res if r[0]==li]):.0f}%" for li in [0,8,16,24,31]))
json.dump(res,open("invent/sparsity_real.json","w"))
