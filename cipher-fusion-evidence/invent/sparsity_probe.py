# INVENTION PREMISE TEST: is decode activation sparsity real in Mistral-7B?
# Measure: of the 14336 SwiGLU intermediate neurons, what fraction can be SKIPPED (not read down_proj column)
# while preserving the output? Skippable fraction = potential weight-traffic reduction = potential speedup.
import torch, glob, json
from safetensors.torch import load_file
DEV="cuda"; MIST=glob.glob("/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/")[0]
H,I,L=4096,14336,32
sd={}
for s in sorted(glob.glob(MIST+"*.safetensors")): sd.update(load_file(s,device="cpu"))
def g(k): return sd[k].to(DEV,torch.float16)
# real-ish hidden states: run a few layers forward on a real prompt's embedding to get realistic decode activations.
# Simpler & robust: feed random-but-realistic-norm hidden states is NOT real. Instead use the actual embeddings +
# first-layer to produce a realistic x into the MLP. We approximate: take embedding of real tokens, RMSNorm, into MLP.
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(MIST)
emb=g("model.embed_tokens.weight")
text="The transformer architecture processes tokens using attention and feed-forward networks. Paris is the capital of France."
ids=tok(text,return_tensors="pt").input_ids[0].to(DEV)
results=[]
for layer in [0,8,16,24,31]:
    p=f"model.layers.{layer}."
    gate=g(p+"mlp.gate_proj.weight"); up=g(p+"mlp.up_proj.weight"); down=g(p+"mlp.down_proj.weight")
    ln=g(p+"post_attention_layernorm.weight" if False else p+"input_layernorm.weight")
    # use real token embeddings as a stand-in hidden state (realistic magnitude/structure), per-token
    x=emb[ids].float()                                   # [T, H]
    x=x/ x.norm(dim=-1,keepdim=True)*(H**0.5)            # rmsnorm-ish scale
    x=(x*ln.float()).half()
    gproj=torch.nn.functional.silu(x@gate.t().float())   # [T, I] SiLU(gate)
    inter=(gproj*(x@up.t().float()))                     # [T, I] SwiGLU intermediate
    # per token: what fraction of I neurons are needed to keep 99% / 99.9% of output (down @ inter) norm?
    for ti in range(inter.shape[0]):
        v=inter[ti].abs()
        # contribution of neuron j to output ~ |inter_j| * ||down[:,j]||. Use |inter_j|*colnorm as importance.
        coln=down.float().norm(dim=0)                    # [I]
        imp=(v*coln)
        srt,_=imp.sort(descending=True)
        tot=srt.sum()
        cum=srt.cumsum(0)/tot
        k99=int((cum<0.99).sum())+1; k999=int((cum<0.999).sum())+1
        results.append(dict(layer=layer,tok=ti,frac_keep_99=k99/I,frac_keep_999=k999/I,nnz_above_1pct=int((v>v.max()*0.01).sum())/I))
import statistics as st
print(f"Mistral-7B MLP intermediate sparsity (I={I} neurons), over {len(results)} (layer,token) samples:")
print(f"  to keep 99.0% of down-proj output: read only {100*st.mean([r['frac_keep_99'] for r in results]):.1f}% of neurons (median {100*st.median([r['frac_keep_99'] for r in results]):.1f}%)")
print(f"  to keep 99.9% of down-proj output: read only {100*st.mean([r['frac_keep_999'] for r in results]):.1f}% of neurons")
print(f"  => potential down_proj weight-traffic reduction at 99%: {1/st.mean([r['frac_keep_99'] for r in results]):.1f}x")
json.dump(results, open("invent/sparsity_probe.json","w"))
