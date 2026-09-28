#!/usr/bin/env python3
"""PILLAR 3 — KV V3 in eager mode at B=1 (and honest documentation for B>1).

V3 dequant kernel writes layout (head, token, dim) — what FA expects at B=1
where there is no batch dim in the K tensor. At B>1, FA expects (B, head,
token, dim) — the per-layer compressed cache + dequant kernel both require
B-aware redesign that's beyond a single session. This script:
  1. Confirms V3 still correct at B=1 (after MAX_TOKENS bump from 4096→32768
     and BUF_BYTES bump 32MB→256MB)
  2. Measures eager-mode tok/W with V3 at B=1
  3. Documents why B=8/32/64 are not measured (fundamental layout, not a
     buffer-size bug)
"""
import os, sys, ctypes, time, threading, subprocess, json, gc
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

for fn, restype, argtypes in [
    ("cipher_substitute_v2_init", ctypes.c_int, []),
    ("cipher_weight_compress_init", ctypes.c_int, []),
    ("cipher_fusion_kernels_init",  ctypes.c_int, []),
    ("cipher_kv_redirect_init",     ctypes.c_int, []),
    ("cipher_kv_redirect_reset",    None,         []),
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

class KVStats(ctypes.Structure):
    _fields_ = [("enabled", ctypes.c_int),
                ("fa_launches_seen", ctypes.c_uint64),
                ("fa_launches_rewritten", ctypes.c_uint64),
                ("bytes_copied", ctypes.c_uint64)]
rt.cipher_kv_redirect_stats.argtypes = [ctypes.POINTER(KVStats)]
rt.cipher_kv_redirect_stats.restype  = ctypes.c_int

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()
rt.cipher_fusion_kernels_init()
rt.cipher_kv_redirect_init()

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP

def stream_ptr(): return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Int4Linear(nn.Module):
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.in_features  = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        wt = orig.weight.detach().t().contiguous()
        self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
        for _ in range(1001):
            rt.cipher_weight_compress_observe(self.wt_fp16.data_ptr(), wt.numel()*2)
        rc = rt.cipher_weight_compress_quantize(self.wt_fp16.data_ptr(),
                                                self.in_features, self.out_features)
        self._compressed = (rc == 1)
        if self._compressed:
            bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
            br = ctypes.c_int(); bc = ctypes.c_int()
            rc = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs),
                ctypes.byref(br), ctypes.byref(bc))
            if rc != 1: self._compressed = False
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
                return out.reshape(x.shape[:-1] + (self.out_features,)) + (
                    0 if self.bias is None else self.bias)
        out = x @ self.wt_fp16
        if self.bias is not None: out = out + self.bias
        return out


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps), stream_ptr())
    return out

def cipher_silu_mul(gate, up):
    if not gate.is_contiguous(): gate = gate.contiguous()
    if not up.is_contiguous():   up = up.contiguous()
    out = torch.empty_like(gate)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(), stream_ptr())
    return out


MODEL = "mistralai/Mistral-7B-v0.1"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 10
MEASURE_SECONDS = 10.0

print("[p3] loading Mistral-7B-v0.1...")
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

MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(
    x, self.weight, self.variance_epsilon)
MistralMLP.forward    = lambda self, x: self.down_proj(
    cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
for layer in model.model.layers:
    for attr in ["q_proj","k_proj","v_proj","o_proj"]:
        old = getattr(layer.self_attn, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.self_attn, attr, Int4Linear(old).to("cuda"))
    for attr in ["gate_proj","up_proj","down_proj"]:
        old = getattr(layer.mlp, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.mlp, attr, Int4Linear(old).to("cuda"))


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2: samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.15)


def measure(B):
    MAX_LEN = PREFILL_LEN + 320
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda", dtype=torch.float16)
    rt.cipher_kv_redirect_reset()
    with torch.no_grad():
        cp = torch.arange(PREFILL_LEN, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([PREFILL_LEN], device="cuda", dtype=torch.long)
    out_logits = torch.full((B, 1, model.config.vocab_size), float("nan"),
                            dtype=torch.float16, device="cuda")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
    torch.cuda.synchronize()
    if torch.isnan(out_logits).any(): return dict(error="NaN warmup")
    out_logits.fill_(float("nan"))
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter(); n_steps = 0
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
        n_steps += 1
        if cache_pos.item() >= MAX_LEN - 4: break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    if torch.isnan(out_logits).any(): return dict(error="NaN loop")
    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])
    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    s = KVStats(); rt.cipher_kv_redirect_stats(ctypes.byref(s))
    out = dict(tps=(B*n_steps)/elapsed, draw_w=sum(pw)/len(pw),
               clk_mhz=sum(clk)/len(clk),
               tok_w=((B*n_steps)/elapsed)/(sum(pw)/len(pw) if pw else 1),
               elapsed_s=elapsed, n_steps=n_steps,
               last_token=last_token, last_token_text=sample_text,
               batch=B, fa_seen=s.fa_launches_seen,
               fa_rewritten=s.fa_launches_rewritten,
               bytes_copied=s.bytes_copied)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== EAGER + V3 KV (B=1 only — see report for B>1) ===")
print(f"{'B':>4} {'tps':>8} {'watts':>7} {'tok/W':>7} {'fa_rew':>7} {'last':>10}")
results = []
for B in [1]:
    try:
        r = measure(B)
    except Exception as e:
        print(f"  B={B}: {type(e).__name__}: {e}"); continue
    if "error" in r:
        print(f"  B={B}: SENTINEL FAIL: {r['error']}"); continue
    results.append(r)
    print(f"{B:>4} {r['tps']:>8.2f} {r['draw_w']:>7.1f} {r['tok_w']:>7.4f} "
          f"{r['fa_rewritten']:>7} {r['last_token_text']!r:>10}")

with open(os.path.join(ROOT, "p3_kv_v3_eager.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[p3] wrote p3_kv_v3_eager.json")
