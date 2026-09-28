#!/usr/bin/env python3
# run_mistral.py — Session 7
# Usage:
#   Baseline:  python3 run_mistral.py
#   CIPHER:    CIPHER_SAFE_MODE=1 LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" python3 run_mistral.py

import sys, os, time
sys.path.insert(0, '/workspace/CIPHER_final_session7')

cipher_active = 'libcipher_hook' in os.environ.get('LD_PRELOAD', '')

if cipher_active:
    import cipher_wrapper
    print('[CIPHER] Active — fp16 MLP substitution enabled')
else:
    print('[BASELINE] Running without CIPHER')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of artificial intelligence in GPU computing is"
N_TOKENS = 50

print(f'Loading {MODEL}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map='cuda'
)
model.eval()

# Install MLP Koopman hooks on all 32 layers (per-layer calibrated)
if cipher_active:
    cipher_wrapper.install_all_mlp_hooks(model)

# Warmup
inputs = tokenizer(PROMPT, return_tensors='pt').to('cuda')
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize()

# Timed run
print(f'Generating {N_TOKENS} tokens...')
start = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

tokens_per_sec = N_TOKENS / elapsed
print(f'\nResult:')
print(f'  Tokens generated: {N_TOKENS}')
print(f'  Time: {elapsed:.3f}s')
print(f'  Tokens/sec: {tokens_per_sec:.1f}')
print(f'  CIPHER active: {cipher_active}')
if cipher_active:
    print(f'  fp16 MLP substitutions: {cipher_wrapper._fp16_sub_count}')
print(f'\nGenerated text:')
print(tokenizer.decode(out[0], skip_special_tokens=True))
