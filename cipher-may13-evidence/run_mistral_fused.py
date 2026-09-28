#!/usr/bin/env python3
"""Mistral-7B with fused RMSNorm + SiLU·Mul kernels via monkey-patch.

Wires Track A Patterns 1, 2, 3 directly into HF Mistral modules. Combined
with Track-1 graph capture, measures end-to-end tok/s + power."""
import os, sys, time, ctypes, subprocess, threading
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
apply_patches = "FUSED" in os.environ.get("CIPHER_MODE", "")
rt = None
if apply_patches:
    rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
    rt.cipher_substitute_v2_init.restype = ctypes.c_int
    rt.cipher_fusion_kernels_init.restype = ctypes.c_int
    rt.cipher_fusion_kernels_enabled.restype = ctypes.c_int
    rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    rt.cipher_fused_silu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_silu_mul.restype = ctypes.c_int
    rt.cipher_substitute_v2_init()
    rt.cipher_fusion_kernels_init()


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1])
    out_flat = out.reshape(-1, x.shape[-1])
    if not (flat.is_contiguous() and weight.is_contiguous()):
        flat = flat.contiguous()
    rt.cipher_fused_rmsnorm(
        flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
        flat.shape[0], flat.shape[-1], float(eps), None)
    return out


def cipher_silu_mul(gate, up):
    if not (gate.is_contiguous() and up.is_contiguous()):
        gate = gate.contiguous()
        up = up.contiguous()
    out = torch.empty_like(gate)
    rt.cipher_fused_silu_mul(
        gate.data_ptr(), up.data_ptr(), out.data_ptr(),
        gate.numel(), None)
    return out


# ── Monkey-patch Mistral modules ────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm, MistralMLP)

# Patch RMSNorm.forward
def patched_rmsnorm_forward(self, x):
    return cipher_rmsnorm(x, self.weight, self.variance_epsilon)

# Patch MLP.forward — original: down_proj(silu(gate_proj(x)) * up_proj(x))
def patched_mlp_forward(self, x):
    gate = self.gate_proj(x)
    up   = self.up_proj(x)
    fused = cipher_silu_mul(gate, up)
    return self.down_proj(fused)


def install_patches():
    MistralRMSNorm.forward = patched_rmsnorm_forward
    MistralMLP.forward    = patched_mlp_forward
    print("[CIPHER FUSION] patched MistralRMSNorm.forward, MistralMLP.forward")

# ── Power sampler ───────────────────────────────────────────────────────────
def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception:
            pass
        time.sleep(0.2)


# ── Main ────────────────────────────────────────────────────────────────────
MODEL = "mistralai/Mistral-7B-v0.1"
N_TOKENS = 256
MAX_LEN = 384
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
nparams = sum(p.numel() for p in model.parameters())
prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids.shape[1]

if apply_patches:
    install_patches()


def measure_eager():
    with torch.no_grad():
        _ = model.generate(prompt_ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    t = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); t.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); t.join(timeout=2)
    return elapsed, samples, out


def measure_graph():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cache_position = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cache_position,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(1, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)

    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids.copy_(out_logits.argmax(-1))

    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    t = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); t.start()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        cache_pos += 1
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); t.join(timeout=2)
    return elapsed, samples, None


def report(label, elapsed, samples):
    pw = [s[0] for s in samples[1:]] if len(samples) > 1 else samples
    clk = [s[1] for s in samples[1:]] if len(samples) > 1 else samples
    mean_pw = sum(pw) / max(len(pw), 1) if pw else 0.0
    mean_clk = sum(clk) / max(len(clk), 1) if clk else 0
    tps = N_TOKENS / elapsed
    flops = 2.0 * nparams * N_TOKENS
    tflops = flops / elapsed / 1e12
    mfu = tflops / 989.0 * 100.0
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:35s} tps={tps:7.2f}  power={mean_pw:5.1f}W  clk={mean_clk:.0f}MHz  "
          f"TFLOPS={tflops:5.2f}  MFU={mfu:5.2f}%  tok/W={tok_w:6.3f}")


print(f"=== Mistral-7B fp16, {N_TOKENS} tokens, mode={os.environ.get('CIPHER_MODE','')} ===")
elapsed_e, samples_e, _ = measure_eager()
report("eager", elapsed_e, samples_e)
elapsed_g, samples_g, _ = measure_graph()
report("graph capture", elapsed_g, samples_g)
