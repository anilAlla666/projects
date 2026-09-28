"""T4.6.1 S1.5 probe driver - runs Mistral-7B generate(max_new_tokens=8)
with cipher_attn_probe.so LD_PRELOADed. Expects hit count > 0 if SDPA
flash is the attention path."""
import os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
mp = "/home/ubuntu/models/Mistral-7B-v0.1"
print(f"[driver] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
print(f"[driver] backend probes:", flush=True)
print(f"  flash_sdp_enabled = {torch.backends.cuda.flash_sdp_enabled()}", flush=True)
print(f"  mem_efficient_sdp_enabled = {torch.backends.cuda.mem_efficient_sdp_enabled()}", flush=True)
print(f"  math_sdp_enabled = {torch.backends.cuda.math_sdp_enabled()}", flush=True)

tok = AutoTokenizer.from_pretrained(mp)
print("[driver] loading Mistral-7B fp16 ...", flush=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16,
                                         attn_implementation="sdpa").cuda()
m.train(False)  # inference mode without using .eval() shorthand
print(f"[driver] loaded in {time.time()-t0:.1f}s", flush=True)

with torch.inference_mode():
    inp = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    t0 = time.time()
    out = m.generate(inp, max_new_tokens=8, do_sample=False)
    dt = time.time() - t0
print(f"[driver] generate {dt:.2f}s output:", flush=True)
print("[driver]", repr(tok.decode(out[0], skip_special_tokens=True)), flush=True)
