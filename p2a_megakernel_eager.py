#!/usr/bin/env python3
"""PILLAR 2 STEPS 2A+2B+2C — Megakernel in eager mode at all batches.

Step 2A/2B: Megakernel + INT4 + fused norms in eager mode (no graph capture).
            Compare to eager baseline at every B.

Step 2C: Extend megakernel gating to M>1.

The existing kernel signature `cipher_int4_silu_mul_gemv(..., M, N, K)` already
launches gridDim.y = M, so the M=1 gate in FusedMistralMLP is the only restriction.
This script:
  1. Quickly verifies megakernel output at M=8 vs fp16 reference (correctness)
  2. Benchmarks eager-mode decode at B=1, 8, 32, 64 with megakernel always on

Hypothesis: at M>1 the GEMV-style megakernel (one warp per output column) does
8-64× more serial work per warp than at M=1. cuBLAS uses tensor cores at M>=8,
so megakernel will likely lose at high B. Measure to confirm.
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
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP


class Counters:
    int4_path = 0
    int4_fallback = 0
    mlp_megakernel = 0
    mlp_fallback = 0
    @classmethod
    def reset(cls):
        cls.int4_path = cls.int4_fallback = 0
        cls.mlp_megakernel = cls.mlp_fallback = 0


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


# ---- M>1 megakernel correctness probe (Step 2C) -----------------------------
print("[p2a] Step 2C correctness probe at M=8...")
torch.manual_seed(0)
M_test, K_test, N_test = 8, 4096, 14336
x_t  = torch.randn(M_test, K_test, dtype=torch.float16, device="cuda") * 0.1
Wg_t = torch.randn(K_test, N_test, dtype=torch.float16, device="cuda") * 0.05
Wu_t = torch.randn(K_test, N_test, dtype=torch.float16, device="cuda") * 0.05
for W in [Wg_t, Wu_t]:
    for _ in range(1001):
        rt.cipher_weight_compress_observe(W.data_ptr(), W.numel()*2)
    rt.cipher_weight_compress_quantize(W.data_ptr(), K_test, N_test)
def lookup(W):
    bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
    br = ctypes.c_int(); bc = ctypes.c_int()
    rt.cipher_weight_compress_lookup_T(W.data_ptr(),
        ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
    return bT.value, bs.value
bgT, bgs = lookup(Wg_t); buT, bus = lookup(Wu_t)
gate_ref = (x_t @ Wg_t).float()
up_ref   = (x_t @ Wu_t).float()
silu_ref = (gate_ref * torch.sigmoid(gate_ref)) * up_ref
out_t = torch.full((M_test, N_test), float("nan"), dtype=torch.float16, device="cuda")
rc = rt.cipher_weight_compress_int4_silu_mul(
    x_t.data_ptr(), bgT, bgs, buT, bus, out_t.data_ptr(),
    M_test, N_test, K_test, stream_ptr())
torch.cuda.synchronize()
nans = torch.isnan(out_t).sum().item()
diff = (out_t.float() - silu_ref).abs()
rel  = diff.mean().item() / silu_ref.abs().mean().item()
print(f"  M=8 silu_mul megakernel: rc={rc} NaNs={nans}/{M_test*N_test} "
      f"rel_err={rel:.3f}")
M_GATED = "1+"  # default — try megakernel at any M
if rc != 1 or nans > 0 or rel > 0.30:
    print(f"  ✗ M=8 not safe → restricting megakernel to M=1")
    M_GATED = "1"
else:
    print(f"  ✓ M=8 safe → megakernel enabled at any M")


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
        # GEMV path at any T=1 step (allow M>1 via batch-dim flattening)
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            xb = x.reshape(-1, self.in_features)
            M = xb.shape[0]
            out = torch.empty(M, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                M, self.out_features, self.in_features, stream_ptr())
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


class FusedMistralMLP(nn.Module):
    def __init__(self, mlp: MistralMLP, allow_m_gt_1):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj   = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn    = mlp.act_fn
        self._megakernel_active = (
            isinstance(mlp.gate_proj, Int4Linear) and mlp.gate_proj._compressed and
            isinstance(mlp.up_proj,   Int4Linear) and mlp.up_proj._compressed   and
            isinstance(mlp.down_proj, Int4Linear) and mlp.down_proj._compressed)
        self._allow_m_gt_1 = allow_m_gt_1

    def forward(self, x):
        # Activate megakernel for any T=1 decode step (any batch)
        if self._megakernel_active and x.dim() >= 2 and x.shape[-2] == 1 \
                and (self._allow_m_gt_1 or x.shape[0] == 1):
            in_flat = x.reshape(-1, x.shape[-1])
            M = in_flat.shape[0]; D = in_flat.shape[-1]
            N_int = self.gate_proj.out_features
            sm_buf  = torch.empty(M, N_int, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_silu_mul(
                in_flat.data_ptr(),
                self.gate_proj._bT_ptr, self.gate_proj._bs_ptr,
                self.up_proj._bT_ptr,   self.up_proj._bs_ptr,
                sm_buf.data_ptr(),
                M, N_int, D, stream_ptr())
            if rc != 1:
                Counters.mlp_fallback += 1
                return self._fallback(x)
            out = torch.empty(M, D, dtype=torch.float16, device=x.device)
            zero = torch.zeros(M, D, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_down_residual(
                sm_buf.data_ptr(),
                self.down_proj._bT_ptr, self.down_proj._bs_ptr,
                zero.data_ptr(),
                out.data_ptr(),
                M, D, N_int, stream_ptr())
            if rc != 1:
                Counters.mlp_fallback += 1
                return self._fallback(x)
            Counters.mlp_megakernel += 1
            return out.reshape(x.shape[:-1] + (D,))
        Counters.mlp_fallback += 1
        return self._fallback(x)

    def _fallback(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
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


def cipher_silu_mul_outer(gate, up):
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

print("[p2a] loading Mistral-7B-v0.1...")
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

allow_mgt1 = (M_GATED == "1+")
n_fused = 0
for layer in model.model.layers:
    layer.mlp = FusedMistralMLP(layer.mlp, allow_mgt1).to("cuda")
    n_fused += 1
print(f"[p2a] {n_int4} Int4Linear, {n_fused} FusedMLP "
      f"(M_gate={'M>=1' if allow_mgt1 else 'M=1 only'})")


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
               batch=B,
               int4_path=Counters.int4_path,
               int4_fallback=Counters.int4_fallback,
               mlp_megakernel=Counters.mlp_megakernel,
               mlp_fallback=Counters.mlp_fallback)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== EAGER MEGAKERNEL (CIPHER stack, no graph capture) ===")
print(f"{'B':>4} {'tps':>9} {'watts':>7} {'clk':>5} {'tok/W':>8} "
      f"{'INT4':>7} {'fallb':>7} {'mega':>5} {'mfb':>5} {'last':>10}")
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
          f"{r['tok_w']:>8.4f} {r['int4_path']:>7} {r['int4_fallback']:>7} "
          f"{r['mlp_megakernel']:>5} {r['mlp_fallback']:>5} "
          f"{r['last_token_text']!r:>10}")

with open(os.path.join(ROOT, "p2a_megakernel_eager.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[p2a] wrote p2a_megakernel_eager.json")

# Side-by-side
print("\n=== Δ vs eager baseline (Step 1A) ===")
try:
    with open(os.path.join(ROOT, "p1a_eager_baseline.json")) as f:
        base = {r["batch"]: r for r in json.load(f)}
    print(f"{'B':>3} {'eager tok/W':>11} {'mega tok/W':>11} "
          f"{'Δtps':>8} {'Δwatts':>8} {'Δtok/W':>8}")
    for r in results:
        b = base.get(r["batch"])
        if not b: continue
        d_tps = (r["tps"]/b["tps"] - 1) * 100
        d_w   = (r["draw_w"]/b["draw_w"] - 1) * 100
        d_tw  = (r["tok_w"]/b["tok_w"] - 1) * 100
        print(f"{r['batch']:>3} {b['tok_w']:>11.4f} {r['tok_w']:>11.4f} "
              f"{d_tps:>+7.1f}% {d_w:>+7.1f}% {d_tw:>+7.1f}%")
except FileNotFoundError:
    print("  (p1a_eager_baseline.json not found)")
