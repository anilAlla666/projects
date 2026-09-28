#!/usr/bin/env python3
"""STEP 1 — Wire KV-compression V3 (with RoPE) into the benchmark and measure
tok/s at batch=1,8,32,64 with V3 on vs off.

V3 path: K/V proj outputs (cublasGemmEx) get quantized into a per-layer 2-bit
cache; on every FA launch the cache is dequantized into FA's staging buffer
with RoPE applied inline (cipher_kv_dq2_perm_rope).

Run with libcipher_hook.so + libcuda preloaded so the FA cuLaunchKernel and
cublasGemmEx interceptors fire and call into libcipher_rt.so."""
import os, sys, ctypes, time, threading, subprocess, json, atexit, signal
import warnings
warnings.filterwarnings("ignore")

# Limit power cap module to silent (we are not testing Lever 1 here).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

# bindings
for fn, restype, argtypes in [
    ("cipher_substitute_v2_init",   ctypes.c_int, []),
    ("cipher_weight_compress_init", ctypes.c_int, []),
    ("cipher_fusion_kernels_init",  ctypes.c_int, []),
    ("cipher_kv_redirect_init",     ctypes.c_int, []),
    ("cipher_kv_redirect_reset",    None,         []),
    ("cipher_weight_compress_observe",  ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t]),
    ("cipher_weight_compress_quantize", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("cipher_weight_compress_lookup_T", ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
         ctypes.POINTER(ctypes.c_int)]),
    ("cipher_weight_compress_int4_gemv", ctypes.c_int,
        [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]),
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
rt.cipher_kv_redirect_init()    # honours CIPHER_KV_REDIRECT etc.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP


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
                                              wt.numel() * 2)
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
        # GEMV path only at single-token decode (T=1) AND batch=1.
        # For B>=8 we fall through to fp16 cuBLAS so that cublasGemmEx is the
        # path under inspection (this is the same regime the V3 hook targets).
        if (self._compressed and x.dim() >= 2 and x.shape[-2] == 1
                and x.shape[0] == 1):
            xb = x.reshape(-1, self.in_features)
            out = torch.empty(1, self.out_features, dtype=torch.float16,
                              device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                self.out_features, self.in_features,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
            if rc == 1:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                out = (xb @ self.wt_fp16).reshape(
                    x.shape[:-1] + (self.out_features,))
        else:
            out = x @ self.wt_fp16
        if self.bias is not None: out = out + self.bias
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
PREFILL_LEN     = 1024
WARMUP_REPLAYS  = 8
MEASURE_SECONDS = 6.0
MAX_LEN         = 1300

print("[step1] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
BASE = ("The future of artificial intelligence in GPU computing is to make every "
        "joule of energy count. Energy efficiency means doing more useful work "
        "per watt — that is the metric that matters. ")
ids_one = tok(BASE, return_tensors="pt").input_ids[0]
N_TILE = (PREFILL_LEN + len(ids_one) - 1) // len(ids_one)
big = torch.cat([ids_one for _ in range(N_TILE)])[:PREFILL_LEN]
prompt_ids_1 = big.unsqueeze(0).to("cuda")
print(f"[step1] prompt_len={prompt_ids_1.shape[1]}")

# Patch
MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight,
                                                        self.variance_epsilon)
MistralMLP.forward    = lambda self, x: self.down_proj(
    cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
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
print(f"[step1] {n_int4} INT4Linear, FUSED RMSNorm+MLP active")


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


def measure(B):
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    prompt_len = prompt_ids.shape[1]
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda",
                        dtype=torch.float16)
    rt.cipher_kv_redirect_reset()
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(B, 1, model.config.vocab_size,
                             device="cuda", dtype=torch.float16)

    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True,
                          return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids.copy_(out_logits.argmax(-1))
    torch.cuda.synchronize()

    for _ in range(WARMUP_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()

    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples),
                          daemon=True); th.start()
    t0 = time.perf_counter(); n_replays = 0
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        for _ in range(64):
            cache_pos += 1; g.replay(); n_replays += 1
            if cache_pos.item() >= MAX_LEN - 4: break
        if cache_pos.item() >= MAX_LEN - 4: break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw = sum(pw)/len(pw); mean_clk = sum(clk)/len(clk)
    tps = (B * n_replays) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    s = KVStats(); rt.cipher_kv_redirect_stats(ctypes.byref(s))
    out = dict(tps=tps, draw_w=mean_pw, clk_mhz=mean_clk, tok_w=tok_w,
               elapsed_s=elapsed, n_replays=n_replays,
               kv_enabled=s.enabled, fa_seen=s.fa_launches_seen,
               fa_rewritten=s.fa_launches_rewritten,
               kv_bytes=s.bytes_copied)
    del g, cache, cache_pos, out_logits, input_ids, prompt_ids
    torch.cuda.empty_cache()
    return out


print("\n=== STEP 1 — V3 KV compression: tps and power ===")
print(f"{'batch':>5} {'V3':>4} {'tps':>10} {'draw_w':>7} {'clk':>6} {'tok/W':>8} "
      f"{'fa_rew':>8} {'sec':>5}")
mode = "v3_on" if (os.environ.get("CIPHER_KV_RDR_V3","").lower() in
                    ("1","on","true","yes")) else "v3_off"
results = []
for B in [1, 8, 32, 64]:
    try:
        r = measure(B)
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B}: OOM"); torch.cuda.empty_cache(); continue
    except Exception as e:
        print(f"  B={B}: {type(e).__name__}: {e}"); continue
    r["batch"] = B; r["mode"] = mode
    results.append(r)
    print(f"{B:>5} {mode:>4} {r['tps']:>10.2f} {r['draw_w']:>7.1f} "
          f"{r['clk_mhz']:>6.0f} {r['tok_w']:>8.4f} "
          f"{r['fa_rewritten']:>8} {r['elapsed_s']:>5.2f}")

out_path = os.path.join(ROOT, f"step1_{mode}.json")
with open(out_path, "w") as f: json.dump(results, f, indent=2)
print(f"\n[step1] wrote {out_path}")
