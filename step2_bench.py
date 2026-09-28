#!/usr/bin/env python3
"""STEP 2 — Deep MLP Fusion Megakernel benchmark.

Replaces MistralMLP with two fused INT4 megakernels at decode-M=1:
  - cipher_int4_silu_mul_gemv  : RMSNorm(x) → gate INT4 GEMV → up INT4 GEMV →
                                  SiLU(gate) * up                  (1 kernel)
  - cipher_int4_down_residual_gemv : silu_mul → down INT4 GEMV + residual add (1 kernel)

Suppresses 5 of the original 7 MLP launches (gate, up, silu, mul, down, add → fused
into 2). RMSNorm of MLP-input (LayerNorm-2) still uses the existing fused kernel.

Measured at batch=1,8,32,64. At B≥2 the MLP falls back to fp16 cuBLAS
(matches existing INT4 path's M=1 gating).
"""
import os, sys, ctypes, time, threading, subprocess, json
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
        [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]),
    ("cipher_weight_compress_int4_silu_mul", ctypes.c_int,
        [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_void_p]),
    ("cipher_weight_compress_int4_down_residual", ctypes.c_int,
        [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.c_void_p]),
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
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm, MistralMLP, MistralDecoderLayer)


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Int4Linear(nn.Module):
    """Single-row decode INT4 linear with fp16 fallback for M>1."""
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.in_features = orig.in_features
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

    def forward(self, x):
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1 and x.shape[0] == 1:
            xb = x.reshape(-1, self.in_features)
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                self.out_features, self.in_features, stream_ptr())
            if rc == 1:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            out = x @ self.wt_fp16
        if self.bias is not None: out = out + self.bias
        return out


class FusedMistralMLP(nn.Module):
    """Full-MLP wrapper using two INT4 megakernels for B=1, M=1 decode.
       Falls back to standard fp16 path otherwise."""
    def __init__(self, mlp: MistralMLP):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj   = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn    = mlp.act_fn
        # Pre-resolve compressed weight pointers when each linear is an
        # Int4Linear with a successful quant+T-lookup.
        self._gate_ok = isinstance(mlp.gate_proj, Int4Linear) and mlp.gate_proj._compressed
        self._up_ok   = isinstance(mlp.up_proj,   Int4Linear) and mlp.up_proj._compressed
        self._down_ok = isinstance(mlp.down_proj, Int4Linear) and mlp.down_proj._compressed
        self._megakernel_active = self._gate_ok and self._up_ok and self._down_ok

    def forward(self, x):
        # MLP receives RMSNorm(hidden); residual was added by the decoder
        # layer caller. Megakernel path: produce only the MLP output;
        # decoder layer will add residual = hidden_pre_norm + mlp_out itself.
        # That separation keeps drop-in semantics with HF's MistralMLP.
        if (self._megakernel_active and x.dim() >= 2 and
                x.shape[-2] == 1 and x.shape[0] == 1):
            B = x.shape[0]; T = x.shape[-2]; D = x.shape[-1]
            in_flat = x.reshape(-1, D)                 # (1, 4096)
            N_int   = self.gate_proj.out_features      # 14336
            silu_mul = torch.empty(1, N_int, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_silu_mul(
                in_flat.data_ptr(),
                self.gate_proj._bT_ptr, self.gate_proj._bs_ptr,
                self.up_proj._bT_ptr,   self.up_proj._bs_ptr,
                silu_mul.data_ptr(),
                1, N_int, D, stream_ptr())
            if rc != 1:
                return self._fallback(x)
            out = torch.empty(1, D, dtype=torch.float16, device=x.device)
            # We use the down megakernel WITHOUT residual by passing a zero
            # residual buffer — the decoder layer adds residual already and
            # we want pure MLP output here. Cost: the kernel pays one extra
            # add of zero. (Simpler than splitting kernels.)
            zero = torch.zeros(1, D, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_down_residual(
                silu_mul.data_ptr(),
                self.down_proj._bT_ptr, self.down_proj._bs_ptr,
                zero.data_ptr(),
                out.data_ptr(),
                1, D, N_int, stream_ptr())
            if rc != 1:
                return self._fallback(x)
            return out.reshape(x.shape[:-1] + (D,))
        return self._fallback(x)

    def _fallback(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        # Use cipher_fused_silu_mul to keep parity with the non-megakernel path
        if not gate.is_contiguous(): gate = gate.contiguous()
        if not up.is_contiguous():   up   = up.contiguous()
        sm = torch.empty_like(gate)
        rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), sm.data_ptr(),
                                  gate.numel(), stream_ptr())
        return self.down_proj(sm)


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps),
                              stream_ptr())
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
WARMUP_REPLAYS  = 8
MEASURE_SECONDS = 6.0
MAX_LEN         = 1300

print("[step2] loading model...")
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
print(f"[step2] prompt_len={prompt_ids_1.shape[1]}")

# Patch the model
MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(
    x, self.weight, self.variance_epsilon)
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

# Replace MistralMLP modules with FusedMistralMLP wrappers when megakernel
# requested (env CIPHER_MLP_MEGAKERNEL=on).
megakernel = os.environ.get("CIPHER_MLP_MEGAKERNEL", "").lower() in ("1","on","true","yes")
if megakernel:
    n_fused = 0
    for layer in model.model.layers:
        layer.mlp = FusedMistralMLP(layer.mlp).to("cuda")
        n_fused += 1
    print(f"[step2] {n_int4} INT4Linear, {n_fused} FusedMistralMLP MEGAKERNEL")
else:
    print(f"[step2] {n_int4} INT4Linear, MLP unfused (baseline)")


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
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids  = next_token.detach().clone()
    cache_pos  = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
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
    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw = sum(pw)/len(pw); mean_clk = sum(clk)/len(clk)
    tps = (B * n_replays) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    out = dict(tps=tps, draw_w=mean_pw, clk_mhz=mean_clk, tok_w=tok_w,
               elapsed_s=elapsed, n_replays=n_replays)
    del g, cache, cache_pos, out_logits, input_ids, prompt_ids
    torch.cuda.empty_cache()
    return out


print("\n=== STEP 2 — Megakernel: tps and power ===")
print(f"{'batch':>5} {'mode':>10} {'tps':>10} {'draw_w':>7} {'clk':>6} "
      f"{'tok/W':>8} {'sec':>5}")
mode = "mega" if megakernel else "ref"
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
    print(f"{B:>5} {mode:>10} {r['tps']:>10.2f} {r['draw_w']:>7.1f} "
          f"{r['clk_mhz']:>6.0f} {r['tok_w']:>8.4f} {r['elapsed_s']:>5.2f}")

out_path = os.path.join(ROOT, f"step2_{mode}.json")
with open(out_path, "w") as f: json.dump(results, f, indent=2)
print(f"\n[step2] wrote {out_path}")
