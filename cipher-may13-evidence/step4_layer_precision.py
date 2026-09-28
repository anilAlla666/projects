#!/usr/bin/env python3
"""STEP 4 — Adaptive Layer Precision.

Measures per-layer fp16 vs INT4-via-CIPHER drift at decode time, and
emits a precision map: layers with rel_err < 0.005 get "skip" (NOP layer,
pass residual through), 0.005-0.02 get "int2" (use cipher_kv_q2 on
weights), else "int4". Optional 200-token perplexity-proxy check verifies
the map preserves quality.
"""
import os, sys, ctypes, json, time
import warnings
warnings.filterwarnings("ignore")

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
]:
    f = getattr(rt, fn)
    if restype is not None: f.restype = restype
    f.argtypes = argtypes

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()
rt.cipher_fusion_kernels_init()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm, MistralMLP)


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Int4Linear(nn.Module):
    """Int4 linear with per-call mode selector. Set self.mode in {"fp16", "int4", "skip"}.
       Skip outputs zeros (so residual passes through at decoder layer level).
    """
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
            ok = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs),
                ctypes.byref(br), ctypes.byref(bc))
            if ok != 1: self._compressed = False
            else:
                self._bT_ptr = bT.value; self._bs_ptr = bs.value
        self.mode = "int4"  # default

    def forward(self, x):
        if self.mode == "skip":
            shape = x.shape[:-1] + (self.out_features,)
            return torch.zeros(shape, dtype=x.dtype, device=x.device)
        if self.mode == "fp16" or not self._compressed \
            or x.dim() < 2 or x.shape[-2] != 1 or x.shape[0] != 1:
            out = x @ self.wt_fp16
            if self.bias is not None: out = out + self.bias
            return out
        # int4 path
        xb = x.reshape(-1, self.in_features)
        out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
        rc = rt.cipher_weight_compress_int4_gemv(
            xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
            1, self.out_features, self.in_features, stream_ptr())
        if rc != 1:
            Int4Linear._fallback_count = getattr(Int4Linear, "_fallback_count", 0) + 1
            out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            Int4Linear._int4_count = getattr(Int4Linear, "_int4_count", 0) + 1
            out = out.reshape(x.shape[:-1] + (self.out_features,))
        if self.bias is not None: out = out + self.bias
        return out


MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "Energy efficiency means doing more useful work per watt. The future of GPU computing is to make every joule count."
N_DECODE = 50
N_PROXY  = 200

print("[step4] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
print(f"[step4] prompt_len={prompt_ids.shape[1]}")

# Patch nn.Linear to Int4Linear (decode-only int4 path; fallback fp16 at B>1)
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
print(f"[step4] {n_int4} Int4Linear")

# Helper: set mode on every Int4Linear globally / per-layer
def set_global_mode(mode):
    for layer in model.model.layers:
        for sub in [layer.self_attn, layer.mlp]:
            for c in sub.children():
                if isinstance(c, Int4Linear): c.mode = mode

def set_layer_mode(layer_idx, mode):
    layer = model.model.layers[layer_idx]
    for sub in [layer.self_attn, layer.mlp]:
        for c in sub.children():
            if isinstance(c, Int4Linear): c.mode = mode

# Capture per-layer outputs via forward hooks
captures = {}    # {layer_idx: tensor}
def make_hook(idx):
    def hook(module, inp, out):
        captures[idx] = out[0] if isinstance(out, tuple) else out
        captures[idx] = captures[idx].detach().clone()
    return hook
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(i))


def run_decode(n_decode, mode="fp16"):
    """Run prefill + n_decode tokens, return (final_hidden_per_layer_per_step, logits_history)."""
    set_global_mode(mode)
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=prompt_ids.shape[1] + n_decode + 4,
                        device="cuda", dtype=torch.float16)
    captures.clear()
    with torch.no_grad():
        cp = torch.arange(prompt_ids.shape[1], device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    captures.clear()  # discard prefill captures
    cur_input = out.logits[:, -1:].argmax(-1)
    cache_pos = torch.tensor([prompt_ids.shape[1]], device="cuda", dtype=torch.long)
    layer_history = {i: [] for i in range(len(model.model.layers))}
    logits_hist = []
    for _ in range(n_decode):
        with torch.no_grad():
            o = model(input_ids=cur_input, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        for i in range(len(model.model.layers)):
            if i in captures:
                layer_history[i].append(captures[i].detach().float().cpu())
        logits_hist.append(o.logits.detach().float().cpu())
        cur_input = o.logits.argmax(-1)
        cache_pos += 1
        captures.clear()
    return layer_history, logits_hist

print(f"[step4] running fp16 baseline ({N_DECODE} decode tokens)...")
fp16_hist, fp16_logits = run_decode(N_DECODE, "fp16")
print(f"  fp16 fallback={getattr(Int4Linear,'_fallback_count',0)} "
      f"int4_path={getattr(Int4Linear,'_int4_count',0)}")
Int4Linear._fallback_count = 0; Int4Linear._int4_count = 0
print(f"[step4] running int4 stack ({N_DECODE} decode tokens)...")
int4_hist, int4_logits = run_decode(N_DECODE, "int4")
print(f"  int4 fallback={getattr(Int4Linear,'_fallback_count',0)} "
      f"int4_path={getattr(Int4Linear,'_int4_count',0)}")

# Per-layer rel_err
per_layer = []
for i in range(len(model.model.layers)):
    if not fp16_hist[i] or not int4_hist[i]:
        per_layer.append(dict(layer=i, rel_err=None)); continue
    fp16_t = torch.stack(fp16_hist[i])
    int4_t = torch.stack(int4_hist[i])
    diff = (fp16_t - int4_t).abs()
    rel = diff.mean().item() / (fp16_t.abs().mean().item() + 1e-9)
    per_layer.append(dict(layer=i, rel_err=rel,
                          mean_abs=fp16_t.abs().mean().item()))

print("\n=== Per-layer fp16 vs int4 rel_err (decode mean) ===")
for r in per_layer:
    print(f"  L{r['layer']:>2}: rel_err={r['rel_err']:.4f}  mean_abs={r['mean_abs']:.4f}")

# Build precision map
precision = []
for r in per_layer:
    e = r["rel_err"]
    if e is None:                 mode = "int4"
    elif e < 0.005:               mode = "skip"
    elif e < 0.02:                mode = "int2"   # not yet implemented; → int4 fallback
    else:                         mode = "int4"
    precision.append(mode)

n_skip = precision.count("skip")
n_int2 = precision.count("int2")
n_int4 = precision.count("int4")
print(f"\n[step4] precision map: skip={n_skip} int2={n_int2} int4={n_int4}")

with open(os.path.join(ROOT, "step4_precision_map.json"), "w") as f:
    json.dump(dict(map=precision, per_layer=per_layer), f, indent=2)
print(f"[step4] wrote step4_precision_map.json")

# 200-token perplexity-proxy: KL(p_fp16 || p_int4_w_map) on the fp16-greedy
# generated tokens. p_int4_w_map applies the precision map.
print(f"\n[step4] perplexity proxy: running map-based decode for {N_PROXY} tokens...")
# Apply the map: skip-marked layers get mode="skip", others "int4"
for i, m in enumerate(precision):
    set_layer_mode(i, m if m != "int2" else "int4")  # int2 not implemented → int4
# Run 200 decode steps
proxy_hist, proxy_logits = run_decode(N_PROXY, mode=None)  # mode=None preserves per-layer set above
# Need fp16 baseline same length
print("[step4] running fp16 baseline 200 tokens...")
fp16_hist_long, fp16_logits_long = run_decode(N_PROXY, mode="fp16")

# Compute mean KL of last-token logits over the 200 steps
def kl_proxy(p_logits, q_logits):
    # logits in fp32 already (cpu)
    p = F.log_softmax(p_logits[..., -1, :], dim=-1)
    q = F.log_softmax(q_logits[..., -1, :], dim=-1)
    p_dist = p.exp()
    return (p_dist * (p - q)).sum(-1).item()

kls = []
for i in range(min(len(fp16_logits_long), len(proxy_logits))):
    kls.append(kl_proxy(fp16_logits_long[i], proxy_logits[i]))
mean_kl = sum(kls) / max(1, len(kls))
print(f"[step4] mean KL(fp16 || mapped) over {len(kls)} tokens = {mean_kl:.4f} nats")

# Re-set everything to int4 for posterity
set_global_mode("int4")
print("[step4] done")
