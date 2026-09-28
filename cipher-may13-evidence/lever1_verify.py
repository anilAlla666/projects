#!/usr/bin/env python3
"""Lever 1 batch=1 verification: confirm power cap delivers tok/W gain
without regressing tok/s by more than 5%, on the full INT4+FUSED+graph stack.

Compares 700W baseline against the auto-cap (cipher_power_cap_apply_for_batch).
"""
import os, sys, ctypes, time, threading, subprocess, json, atexit, signal
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("CIPHER_POWER_CAP", "on")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

# bindings — power cap
rt.cipher_power_cap_init.restype = ctypes.c_int
rt.cipher_power_cap_apply_for_batch.argtypes = [ctypes.c_int]
rt.cipher_power_cap_apply_for_batch.restype  = ctypes.c_int
rt.cipher_power_cap_apply_w.argtypes = [ctypes.c_int]
rt.cipher_power_cap_apply_w.restype  = ctypes.c_int
rt.cipher_power_cap_restore.restype  = ctypes.c_int

# bindings — full CIPHER stack (INT4 + fused) from run_mistral_full.py
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_fusion_kernels_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
rt.cipher_fused_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
rt.cipher_fused_rmsnorm.restype = ctypes.c_int
rt.cipher_fused_silu_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_void_p]
rt.cipher_fused_silu_mul.restype = ctypes.c_int

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()
rt.cipher_fusion_kernels_init()
rt.cipher_power_cap_init()

# Crash-safe restore
def _restore():
    try: rt.cipher_power_cap_restore()
    except Exception: pass
atexit.register(_restore)
def _sig(signum, frame): _restore(); sys.exit(128 + signum)
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP


class Int4Linear(nn.Module):
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        wt = orig.weight.detach().t().contiguous()
        self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
        for _ in range(1001):
            rt.cipher_weight_compress_observe(self.wt_fp16.data_ptr(), wt.numel() * 2)
        rc = rt.cipher_weight_compress_quantize(self.wt_fp16.data_ptr(),
                                                 self.in_features, self.out_features)
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
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            xb = x.reshape(-1, self.in_features)
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                self.out_features, self.in_features,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
            if rc == 1:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            out = x @ self.wt_fp16
        if self.bias is not None:
            out = out + self.bias
        return out


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps),
                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    return out


def cipher_silu_mul(gate, up):
    if not gate.is_contiguous(): gate = gate.contiguous()
    if not up.is_contiguous():   up = up.contiguous()
    out = torch.empty_like(gate)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(),
                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    return out


MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")
MAX_LEN = 1024
WARMUP_REPLAYS = 40
TARGET_REPLAYS = 600
B = 1

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")

# Patch RMSNorm + MLP (FUSED) and Linear (INT4)
MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight, self.variance_epsilon)
MistralMLP.forward = lambda self, x: self.down_proj(cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
n = 0
for layer in model.model.layers:
    for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        old = getattr(layer.self_attn, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.self_attn, attr, Int4Linear(old).to("cuda"))
            n += 1
    for attr in ["gate_proj", "up_proj", "down_proj"]:
        old = getattr(layer.mlp, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.mlp, attr, Int4Linear(old).to("cuda"))
            n += 1
print(f"[verify] patched {n} INT4Linear, FUSED RMSNorm + MLP active")

prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda").expand(B, -1).contiguous()
prompt_len = prompt_ids.shape[1]


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.15)


def measure(label):
    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    nt = out.logits[:, -1:].argmax(-1)
    input_ids = nt.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(B, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)
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

    for _ in range(WARMUP_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(TARGET_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw = sum(pw) / len(pw); mean_clk = sum(clk) / len(clk)
    tps = (B * TARGET_REPLAYS) / elapsed
    tok_w = tps / mean_pw
    print(f"  {label:32s} tps={tps:7.2f}  draw={mean_pw:5.1f}W  clk={mean_clk:.0f}MHz  tok/W={tok_w:6.4f}  ({elapsed:.1f}s)")
    return tps, mean_pw, tok_w


print("\n=== batch=1 verification: full INT4+FUSED+graph stack ===")
print("[verify] baseline at 700W (uncapped)")
rt.cipher_power_cap_restore()
time.sleep(0.5)
tps_b, pw_b, tw_b = measure("baseline (700W)")

print("\n[verify] auto-cap apply_for_batch(1) — expect 300W")
rt.cipher_power_cap_apply_for_batch(1)
time.sleep(0.5)
tps_c, pw_c, tw_c = measure("with auto-cap")

rt.cipher_power_cap_restore()

tps_pct = (tps_c - tps_b) / tps_b * 100
pw_pct  = (pw_c - pw_b) / pw_b * 100
tw_pct  = (tw_c - tw_b) / tw_b * 100
print(f"\n=== Δ vs baseline ===")
print(f"  tps    : {tps_pct:+.2f}%")
print(f"  power  : {pw_pct:+.2f}%")
print(f"  tok/W  : {tw_pct:+.2f}%")
print()

# Pass criteria from user: "highest tok/W without tok/s regression >5%"
ok_tps  = tps_pct >= -5.0
ok_tokw = tw_pct  > 0.0
print("PASS: tps regression within 5%" if ok_tps  else f"FAIL: tps regression {tps_pct:.2f}% > 5%")
print("PASS: tok/W increased"          if ok_tokw else f"FAIL: tok/W did not increase ({tw_pct:.2f}%)")

result = dict(baseline=dict(tps=tps_b, power_w=pw_b, tok_w=tw_b),
              with_cap=dict(tps=tps_c, power_w=pw_c, tok_w=tw_c),
              delta_tps_pct=tps_pct, delta_tok_w_pct=tw_pct)
with open(os.path.join(ROOT, "lever1_verify_results.json"), "w") as f:
    json.dump(result, f, indent=2)

if not (ok_tps and ok_tokw):
    sys.exit(1)
print("\nLEVER 1 VERIFICATION: PASS")
