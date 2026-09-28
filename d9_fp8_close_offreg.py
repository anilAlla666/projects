#!/usr/bin/env python3
# D.9 FP8 staging-close §1 — OFF byte-identical (Mem #11 HARD STOP) + FP8-ON NaN check.
# TAG=vanilla (no inject) | cipher_fp8off (inject, CIPHER_FP8 unset) | cipher_fp8on (inject, CIPHER_FP8=on)
# Saves per-model decode logits; offline compare = byte-identical(vanilla, fp8off) + NaN-free(fp8on).
import os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS=[("tinyllama","/home/ubuntu/models/TinyLlama-1.1B"),
        ("llama32_1b","/home/ubuntu/models/Llama-3.2-1B-Instruct"),
        ("mistral7b","/home/ubuntu/models/Mistral-7B-v0.1"),
        ("llama31_8b","/home/ubuntu/models/Llama-3.1-8B")]
GEN=16; TAG=os.environ["TAG"]
nan_any=False
for name,path in MODELS:
    t=AutoTokenizer.from_pretrained(path)
    if t.pad_token is None: t.pad_token=t.eos_token
    m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ids=t("The history of computing spans several distinct", return_tensors="pt").input_ids.cuda()
    logs=[]
    with torch.no_grad():
        out=m(ids,use_cache=True); past=out.past_key_values; lg=out.logits[:,-1,:].float()
        for _ in range(GEN):
            arr=lg[0].cpu().numpy(); logs.append(arr)
            if not np.isfinite(arr).all(): nan_any=True
            nt=int(lg.argmax(-1))
            out=m(torch.tensor([[nt]],device='cuda'),past_key_values=past,use_cache=True); past=out.past_key_values
            lg=out.logits[:,-1,:].float()
    np.save("/home/ubuntu/d9clz_%s_%s.npy"%(name,TAG), np.stack(logs))
    del m; torch.cuda.empty_cache()
    print("SAVED %s %s nan_so_far=%s"%(name,TAG,nan_any),flush=True)
print("DONE TAG=%s NAN_ANY=%s"%(TAG,nan_any),flush=True)
