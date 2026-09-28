#!/usr/bin/env python3
"""Mistral-7B with INT4 GEMV substituted into nn.Linear forward."""
import os, ctypes, time, threading, subprocess
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup.restype = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()


class Int4Linear(nn.Module):
    """nn.Linear-equivalent that does Y = X @ W.T using cipher_int4_gemv.
    W.T (shape in_features × out_features) is the matrix we quantize. K = in
    is the reduction axis; the GEMV runs over output cols n=0..out-1."""
    def __init__(self, orig_linear: nn.Linear):
        super().__init__()
        self.in_features  = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias         = orig_linear.bias
        # Quantize W.T. We hold a contiguous fp16 copy of W.T to register
        # with the engine; the original W is kept around for fallback.
        wt = orig_linear.weight.detach().t().contiguous()      # (in, out)
        self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
        # Force compression
        for _ in range(1001):
            rt.cipher_weight_compress_observe(
                self.wt_fp16.data_ptr(), wt.numel() * 2)
        rc = rt.cipher_weight_compress_quantize(
            self.wt_fp16.data_ptr(), self.in_features, self.out_features)
        self._compressed = (rc == 1)
        if self._compressed:
            bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
            br = ctypes.c_int(); bc = ctypes.c_int()
            ok = rt.cipher_weight_compress_lookup_T(self.wt_fp16.data_ptr(),
                ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
            if ok != 1:
                self._compressed = False
            else:
                self._bT_ptr = bT.value
                self._bs_ptr = bs.value

    def forward(self, x):
        # x: (..., in)
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            # Decode path: M=1 → use INT4 GEMV
            xb = x.reshape(-1, self.in_features)
            assert xb.shape[0] == 1
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                self.out_features, self.in_features, None)
            if rc != 1:
                # Fallback
                out = (x.reshape(-1, self.in_features) @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
            else:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
        else:
            # Prefill or M>1 → fall back to fp16
            out = x @ self.wt_fp16
        if self.bias is not None:
            out = out + self.bias
        return out


from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
MODEL = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
nparams = sum(p.numel() for p in model.parameters())

apply_int4 = "INT4" in os.environ.get("CIPHER_MODE", "")
if apply_int4:
    # Replace q/k/v/o/gate/up/down projections in each layer
    n_replaced = 0
    for layer in model.model.layers:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                setattr(layer.self_attn, attr, Int4Linear(old).to("cuda"))
                n_replaced += 1
        for attr in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                setattr(layer.mlp, attr, Int4Linear(old).to("cuda"))
                n_replaced += 1
    print(f"[CIPHER] replaced {n_replaced} nn.Linear modules with Int4Linear")

PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")
N_TOKENS = 64
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    _ = model.generate(prompt_ids, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"  Mistral-7B {('INT4-GEMV' if apply_int4 else 'baseline')}: "
      f"{N_TOKENS} tokens in {elapsed:.2f}s = {N_TOKENS/elapsed:.2f} tok/s")
print(f"  output[:120]: {tok.decode(out[0])[:120]!r}")
