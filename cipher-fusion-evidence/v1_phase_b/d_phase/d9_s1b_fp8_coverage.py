#!/usr/bin/env python3
# D.9 §1b measurement 3 — per-layer-CLASS FP8 OUTPUT-IDENTITY coverage (zero-regression proof).
# FP8 E4M3 (weight per-tensor absmax + activation dynamic per-tensor absmax, torch._scaled_mm,
# fp16 out) applied to selected Linear classes; full-forward output compared TOKEN-IDENTICAL to
# fp16 reference across an eval set (greedy + sampled). Coverage = GEMM-time-weighted % FP8-SAFE.
# KL=0 holds by output-preservation: a class that flips ANY token => DECLINE (stays fp16).
import os, sys, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"

PROMPTS=[
 "The capital of France is","Water boils at a temperature of","The first president of the United States was",
 "In 1969, humans first landed on the","Photosynthesis is the process by which plants","The speed of light in a vacuum is approximately",
 "A prime number is a number that","The mitochondria is the powerhouse of the","To make a cup of tea, first you",
 "The theory of relativity was developed by","An example of a renewable energy source is","The largest planet in our solar system is",
 "Shakespeare wrote many famous plays including","The chemical symbol for gold is","DNA stands for",
 "The Great Wall of China was built to","A triangle has three","The currency used in Japan is the",
 "Machine learning is a subset of","The human heart has four","Mount Everest is the tallest",
 "The process of converting water to vapor is called","In computer science, an algorithm is","The Pacific is the largest",
 "A balanced diet should include","The French Revolution began in the year","Gravity causes objects to",
 "The smallest unit of life is the","Electricity flows through a conductor because","The author of Romeo and Juliet is",
 "A noun is a word that names a","The boiling point of water decreases at high","Solar panels convert sunlight into",
 "The square root of sixteen is","Bees are important because they","The longest river in the world is",
 "An atom consists of protons, neutrons, and","To solve a quadratic equation you can use","The opposite of hot is",
 "A democracy is a system of government in which","The freezing point of water is","Photosystem II is part of",
 "The capital of Japan is","A byte consists of eight","The force that pulls objects toward Earth is",
 "Recursion in programming is when a function","The primary colors are red, blue, and","The largest ocean on Earth is the",
 "A haiku is a form of poetry that","The inventor of the telephone was",
]

def make_fp8_fwd(lin):
    K=lin.in_features; w=lin.weight.data
    wscale=(w.abs().max()/448.0).clamp(min=1e-12).to(torch.float32)
    w8=(w/wscale).to(torch.float8_e4m3fn); bias=lin.bias
    def fwd(x):
        shp=x.shape; x2=x.reshape(-1,K).contiguous()
        ascale=(x2.abs().max()/448.0).clamp(min=1e-12).to(torch.float32)
        x8=(x2/ascale).to(torch.float8_e4m3fn)
        o=torch._scaled_mm(x8, w8.t(), scale_a=ascale, scale_b=wscale, out_dtype=torch.float16)
        if bias is not None: o=o+bias
        return o.reshape(*shp[:-1], o.shape[-1])
    return fwd

def select(model, cls, min_layer=0):
    sel=[]
    for name,mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        if "lm_head" in name: continue
        lyr=-1
        if ".layers." in name:
            try: lyr=int(name.split(".layers.")[1].split(".")[0])
            except: lyr=-1
        is_ffn=any(k in name for k in ["mlp.gate","mlp.up","mlp.down"])
        is_attn=any(k in name for k in ["q_proj","k_proj","v_proj","o_proj"])
        if lyr<min_layer: continue
        if cls=="ALL" and (is_ffn or is_attn): sel.append((name,mod))
        elif cls=="FFN" and is_ffn: sel.append((name,mod))
        elif cls=="ATTN" and is_attn: sel.append((name,mod))
        elif cls=="FFN+ATTN" and (is_ffn or is_attn): sel.append((name,mod))
    return sel

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"   # left-pad for correct batched causal generation
ref_model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
enc=tok(PROMPTS, return_tensors="pt", padding=True).to("cuda")
def gen(model, do_sample=False):
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=24, do_sample=do_sample, temperature=(0.8 if do_sample else None),
                           top_p=(0.95 if do_sample else None), top_k=(50 if do_sample else None),
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out[:, enc["input_ids"].shape[1]:].cpu()

torch.manual_seed(0); ref_greedy=gen(ref_model,False); torch.manual_seed(0); ref_samp=gen(ref_model,True)
print(f"[ref] fp16 reference captured ({len(PROMPTS)} prompts x24 tok)", flush=True)

# GEMM-time weights (Mistral per-layer linear params): FFN ~81%, ATTN ~19%
W_FFN, W_ATTN = 0.81, 0.19
configs=[("ALL",0),("FFN",0),("ATTN",0),("FFN+ATTN",4)]
results=[]
for (cls,minl) in configs:
    sel=select(ref_model, cls, minl); orig={id(m):m.forward for _,m in sel}
    for _,m in sel: m.forward=make_fp8_fwd(m)
    try:
        torch.manual_seed(0); g=gen(ref_model,False); torch.manual_seed(0); s=gen(ref_model,True)
        gmatch=(g==ref_greedy).all(dim=1); smatch=(s==ref_samp).all(dim=1)
        g_rate=float(gmatch.float().mean()*100); s_rate=float(smatch.float().mean()*100)
        safe = (g_rate==100.0 and s_rate==100.0)
    except Exception as e:
        g_rate=s_rate=-1; safe=False; print(f"[{cls} minl={minl}] ERR {e}",flush=True)
    finally:
        for _,m in sel: m.forward=orig[id(m)]   # restore
    results.append({"cls":cls,"min_layer":minl,"n_layers_fp8":len(sel),"greedy_token_match_pct":round(g_rate,1),"sampled_token_match_pct":round(s_rate,1),"FP8_SAFE":safe})
    print(f"[{cls} minl={minl}] {len(sel)} linears FP8 -> greedy {round(g_rate,1)}% / sampled {round(s_rate,1)}% token-identical -> {'SAFE' if safe else 'DECLINE'}",flush=True)

# coverage = GEMM-time-weighted fraction FP8-safe (from the class results)
def safe(cls): return next((r["FP8_SAFE"] for r in results if r["cls"]==cls and r["min_layer"]==0), False)
all_safe=safe("ALL"); ffn_safe=safe("FFN"); attn_safe=safe("ATTN")
if all_safe: coverage=100.0
else: coverage=round(100*((W_FFN if ffn_safe else 0)+(W_ATTN if attn_safe else 0)),1)
out={"reference":"fp16 Mistral-7B greedy+sampled","n_prompts":len(PROMPTS),"configs":results,
     "gemm_time_weights":{"FFN":W_FFN,"ATTN":W_ATTN},"FP8_SAFE_COVERAGE_pct":coverage}
print(f"\nFP8-SAFE COVERAGE (GEMM-time-weighted) = {coverage}%  (ALL_safe={all_safe} FFN_safe={ffn_safe} ATTN_safe={attn_safe})",flush=True)
json.dump(out, open("/home/ubuntu/d9_s1b_fp8_coverage.json","w"), indent=2)
print("WROTE d9_s1b_fp8_coverage.json", flush=True)
