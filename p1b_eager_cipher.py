#!/usr/bin/env python3
"""PILLAR 1 STEP 1B — Eager-mode CIPHER (no graph capture).

Stack:
  - INT4 GEMV at B=1 only (corrected binding)
  - Fused RMSNorm + fused SiLU·Mul (existing, correctness-tested)
  - NO graph capture
  - LD_PRELOAD of libcipher_hook.so so cublasGemmEx and cuLaunchKernel
    interceptors fire (kernels go through CIPHER's substitution path)

Hypothesis (Pillar 1): without graph capture, GPU sees idle gaps → clocks
down → less power. Step 1A showed clock pins at 1980 MHz even in eager
mode, so this hypothesis is shaky, but measure to confirm.

Same 4 batches, same prefill=1024, same NaN guards as Step 1A.
"""
import os, sys, ctypes, time, threading, subprocess, json, gc
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

for fn, restype, argtypes in [
    ("cipher_substitute_v2_init",        ctypes.c_int, []),
    ("cipher_weight_compress_init",      ctypes.c_int, []),
    ("cipher_fusion_kernels_init",       ctypes.c_int, []),
    ("cipher_weight_compress_observe",   ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t]),
    ("cipher_weight_compress_quantize",  ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("cipher_weight_compress_lookup_T",  ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
         ctypes.POINTER(ctypes.c_int)]),
    ("cipher_weight_compress_int4_gemv", ctypes.c_int,
        [ctypes.c_void_p]*4 + [ctypes.c_int]*3 + [ctypes.c_void_p]),
    ("cipher_fused_rmsnorm", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]),
    ("cipher_fused_silu_mul", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int, ctypes.c_void_p]),
]:
    f = getattr(rt, fn)
    if restype is not None: f.restype = restype
    f.argtypes = argtypes

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()
rt.cipher_fusion_kernels_init()

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP


class Counters:
    int4_path = 0
    int4_fallback = 0
    fused_rms = 0
    fused_silu = 0
    @classmethod
    def reset(cls):
        cls.int4_path = cls.int4_fallback = 0
        cls.fused_rms = cls.fused_silu = 0


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Int4Linear(nn.Module):
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.in_features  = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        wt = orig.weight.detach().t().contiguous()
        self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
        for _ in range(1001):
            rt.cipher_weight_compress_observe(self.wt_fp16.data_ptr(),
                                              wt.numel()*2)
        rc = rt.cipher_weight_compress_quantize(self.wt_fp16.data_ptr(),
                                                self.in_features,
                                                self.out_features)
        self._compressed = (rc == 1)
        if self._compressed:
            bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
            br = ctypes.c_int(); bc = ctypes.c_int()
            ok = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs),
                ctypes.byref(br), ctypes.byref(bc))
            if ok != 1: self._compressed = False
            else:
                self._bT_ptr = bT.value; self._bs_ptr = bs.value

    def forward(self, x):
        if (self._compressed and x.dim() >= 2
                and x.shape[-2] == 1 and x.shape[0] == 1):
            xb = x.reshape(-1, self.in_features)
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream_ptr())
            if rc == 1:
                Counters.int4_path += 1
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                Counters.int4_fallback += 1
                out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            Counters.int4_fallback += 1
            out = x @ self.wt_fp16
        if self.bias is not None: out = out + self.bias
        return out


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps),
                              stream_ptr())
    Counters.fused_rms += 1
    return out


def cipher_silu_mul_outer(gate, up):
    if not gate.is_contiguous(): gate = gate.contiguous()
    if not up.is_contiguous():   up = up.contiguous()
    out = torch.empty_like(gate)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(), stream_ptr())
    Counters.fused_silu += 1
    return out


MODEL = "mistralai/Mistral-7B-v0.1"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 10
MEASURE_SECONDS = 10.0

print("[p1b] loading Mistral-7B-v0.1...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
BASE = ("Energy efficiency means doing more useful work per watt. "
        "The future of GPU computing is to make every joule count. ")
ids_one = tok(BASE, return_tensors="pt").input_ids[0]
N_TILE = (PREFILL_LEN + len(ids_one)-1) // len(ids_one)
big = torch.cat([ids_one for _ in range(N_TILE)])[:PREFILL_LEN]
prompt_ids_1 = big.unsqueeze(0).to("cuda")

# CIPHER patches
MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(
    x, self.weight, self.variance_epsilon)
MistralMLP.forward    = lambda self, x: self.down_proj(
    cipher_silu_mul_outer(self.gate_proj(x), self.up_proj(x)))

n_int4 = 0
for layer in model.model.layers:
    for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        old = getattr(layer.self_attn, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.self_attn, attr, Int4Linear(old).to("cuda")); n_int4 += 1
    for attr in ["gate_proj", "up_proj", "down_proj"]:
        old = getattr(layer.mlp, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.mlp, attr, Int4Linear(old).to("cuda")); n_int4 += 1
print(f"[p1b] {n_int4} Int4Linear, fused RMSNorm + fused SiLU·Mul active")


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi",
                                "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.15)


def measure(B, prefill_len=PREFILL_LEN, max_decode=320):
    MAX_LEN = prefill_len + max_decode
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda",
                        dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prefill_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prefill_len], device="cuda", dtype=torch.long)

    out_logits = torch.full((B, 1, model.config.vocab_size), float("nan"),
                            dtype=torch.float16, device="cuda")

    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
    torch.cuda.synchronize()
    if torch.isnan(out_logits).any():
        return dict(error=f"NaN after warmup")

    Counters.reset()
    out_logits.fill_(float("nan"))
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples),
                          daemon=True); th.start()
    t0 = time.perf_counter(); n_steps = 0
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
        n_steps += 1
        if cache_pos.item() >= MAX_LEN - 4: break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    if torch.isnan(out_logits).any():
        return dict(error=f"NaN after timed loop")

    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])

    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw  = sum(pw)/len(pw)
    mean_clk = sum(clk)/len(clk)
    tps = (B * n_steps) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    out = dict(tps=tps, draw_w=mean_pw, clk_mhz=mean_clk, tok_w=tok_w,
               elapsed_s=elapsed, n_steps=n_steps,
               last_token=last_token, last_token_text=sample_text,
               batch=B, prefill=prefill_len,
               int4_path=Counters.int4_path,
               int4_fallback=Counters.int4_fallback,
               fused_rms=Counters.fused_rms,
               fused_silu=Counters.fused_silu)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== EAGER CIPHER (no graph capture, LD_PRELOAD active) ===")
print(f"{'B':>4} {'tps':>9} {'watts':>7} {'clk':>5} {'tok/W':>8} "
      f"{'INT4':>6} {'fallb':>6} {'RMS':>5} {'SILU':>5} {'last_tok':>14}")
results = []
for B in [1, 8, 32, 64]:
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  >> B={B} (free mem: {free:.1f} GB)", flush=True)
    try:
        r = measure(B)
    except torch.cuda.OutOfMemoryError:
        print(f"     OOM"); torch.cuda.empty_cache(); continue
    except Exception as e:
        print(f"     {type(e).__name__}: {e}"); continue
    if "error" in r:
        print(f"     SENTINEL FAIL: {r['error']}"); continue
    results.append(r)
    print(f"{B:>4} {r['tps']:>9.2f} {r['draw_w']:>7.1f} {r['clk_mhz']:>5.0f} "
          f"{r['tok_w']:>8.4f} {r['int4_path']:>6} {r['int4_fallback']:>6} "
          f"{r['fused_rms']:>5} {r['fused_silu']:>5} "
          f"{r['last_token_text']!r:>14}")

with open(os.path.join(ROOT, "p1b_eager_cipher.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[p1b] wrote p1b_eager_cipher.json")

# Side-by-side with eager baseline
print("\n=== Δ vs eager baseline (Step 1A) ===")
try:
    with open(os.path.join(ROOT, "p1a_eager_baseline.json")) as f:
        base = {r["batch"]: r for r in json.load(f)}
    print(f"{'B':>3} {'eager tok/W':>11} {'cipher tok/W':>13} "
          f"{'Δtps':>8} {'Δwatts':>8} {'Δtok/W':>8}")
    for r in results:
        b = base.get(r["batch"])
        if not b: continue
        d_tps = (r["tps"]/b["tps"] - 1) * 100
        d_w   = (r["draw_w"]/b["draw_w"] - 1) * 100
        d_tw  = (r["tok_w"]/b["tok_w"] - 1) * 100
        print(f"{r['batch']:>3} {b['tok_w']:>11.4f} {r['tok_w']:>13.4f} "
              f"{d_tps:>+7.1f}% {d_w:>+7.1f}% {d_tw:>+7.1f}%")
except FileNotFoundError:
    print("  (p1a_eager_baseline.json not found)")
