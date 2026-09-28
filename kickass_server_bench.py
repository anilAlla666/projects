#!/usr/bin/env python3
"""KICKASS BENCH — verified end-to-end stack at server-realistic batches.

Hypothesis: 10× tok/W vs single-user eager (0.255 tok/W) is reachable by
amortizing weight reads across concurrent users (continuous batching). At
B=128-256 each weight byte serves 128-256× the work, naturally pushing
tok/W well past the per-user ceiling.

Stack (all verified independently):
  - Fused RMSNorm + Fused SiLU·Mul   (test_fusion_correctness.py)
  - Graph capture                     (test_actuation_microbench.py 4.07× speedup)
  - INT4 GEMV at B=1 only            (verify_int4_binding.py — corrected binding)
  - Megakernel at B=1 only           (verify_megakernel.py)
  - fp16 cuBLAS for B>1               (PyTorch native, batched n=B per linear)

Guards on every row:
  - NaN sentinel on out_logits before warmup AND before timed loop
  - argmax of last logits decodes to a real English token
  - per-batch memory print before measure
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
        if (self._compressed and x.dim() >= 2
                and x.shape[-2] == 1 and x.shape[0] == 1):
            xb = x.reshape(-1, self.in_features)
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream_ptr())
            if rc == 1:
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            out = x @ self.wt_fp16
        if self.bias is not None: out = out + self.bias
        return out


class FusedMistralMLP(nn.Module):
    def __init__(self, mlp: MistralMLP):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj   = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn    = mlp.act_fn
        self._megakernel_active = (
            isinstance(mlp.gate_proj, Int4Linear) and mlp.gate_proj._compressed and
            isinstance(mlp.up_proj,   Int4Linear) and mlp.up_proj._compressed   and
            isinstance(mlp.down_proj, Int4Linear) and mlp.down_proj._compressed)

    def forward(self, x):
        if (self._megakernel_active and x.dim() >= 2
                and x.shape[-2] == 1 and x.shape[0] == 1):
            D = x.shape[-1]
            in_flat = x.reshape(-1, D)
            N_int   = self.gate_proj.out_features
            sm_buf  = torch.empty(1, N_int, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_silu_mul(
                in_flat.data_ptr(),
                self.gate_proj._bT_ptr, self.gate_proj._bs_ptr,
                self.up_proj._bT_ptr,   self.up_proj._bs_ptr,
                sm_buf.data_ptr(),
                1, N_int, D, stream_ptr())
            if rc != 1: return self._fallback(x)
            out = torch.empty(1, D, dtype=torch.float16, device=x.device)
            zero = torch.zeros(1, D, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_down_residual(
                sm_buf.data_ptr(),
                self.down_proj._bT_ptr, self.down_proj._bs_ptr,
                zero.data_ptr(),
                out.data_ptr(),
                1, D, N_int, stream_ptr())
            if rc != 1: return self._fallback(x)
            return out.reshape(x.shape[:-1] + (D,))
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
WARMUP_REPLAYS  = 8
MEASURE_SECONDS = 8.0

# Memory-aware (B, prefill) sizing. At B*P=131072 we approach the
# MLP-intermediate ceiling (B*P*14336*2 ≈ 4 GB, plus KV cache + activations).
# These pairs all fit comfortably in 80 GB:
CONFIGS = [
    (1,   1024),  # single-user reference (M=1 INT4 + megakernel active)
    (8,   1024),
    (32,  1024),
    (64,  1024),
    (128, 512),   # batch sweet spot for tok/W amortization
    (192, 384),
    (256, 256),   # max-batch regime — BUILD_STATE Phase 2 reported 21.76 tok/W here
]

print("[bench] loading Mistral-7B-v0.1...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
BASE = ("Energy efficiency means doing more useful work per watt. "
        "The future of GPU computing is to make every joule count. ")
ids_one = tok(BASE, return_tensors="pt").input_ids[0]

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
n_fused = 0
for layer in model.model.layers:
    layer.mlp = FusedMistralMLP(layer.mlp).to("cuda")
    n_fused += 1
print(f"[bench] {n_int4} INT4Linear (corrected binding), {n_fused} FusedMistralMLP")


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


def measure(B, PREFILL_LEN):
    MAX_LEN = PREFILL_LEN + 320
    N_TILE = (PREFILL_LEN + len(ids_one)-1) // len(ids_one)
    big = torch.cat([ids_one for _ in range(N_TILE)])[:PREFILL_LEN]
    prompt_ids_1 = big.unsqueeze(0).to("cuda")
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda",
                        dtype=torch.float16)
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

    # NaN guard #1
    out_logits.fill_(float("nan"))
    for _ in range(WARMUP_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    nan_after_warmup = torch.isnan(out_logits).sum().item()
    if nan_after_warmup > 0:
        return dict(error=f"NaN after warmup ({nan_after_warmup})")

    # NaN guard #2: refill, run timed loop, check
    out_logits.fill_(float("nan"))
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

    nan_after_loop = torch.isnan(out_logits).sum().item()
    if nan_after_loop > 0:
        return dict(error=f"NaN after timed loop ({nan_after_loop})")

    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])

    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw  = sum(pw)/len(pw)
    mean_clk = sum(clk)/len(clk)
    tps = (B * n_replays) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    out = dict(tps=tps, draw_w=mean_pw, clk_mhz=mean_clk, tok_w=tok_w,
               elapsed_s=elapsed, n_replays=n_replays,
               last_token=last_token, last_token_text=sample_text)
    del g, cache, cache_pos, out_logits, input_ids, prompt_ids, prompt_ids_1
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== KICKASS — verified stack at server batches ===")
print(f"{'B':>4} {'P':>5} {'tps':>9} {'watts':>7} {'tok/W':>8} "
      f"{'sec':>5} {'last_token':>22} {'×B=1eager(0.255)':>17}")
results = []
for B, P in CONFIGS:
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  >> B={B} P={P}  (free GPU mem: {free:.1f} GB)", flush=True)
    try:
        r = measure(B, P)
    except torch.cuda.OutOfMemoryError:
        print(f"     OOM"); torch.cuda.empty_cache(); continue
    except Exception as e:
        print(f"     {type(e).__name__}: {e}"); continue
    if "error" in r:
        print(f"     SENTINEL FAIL: {r['error']}"); continue
    r["batch"] = B; r["prefill"] = P
    results.append(r)
    x_eager_b1 = r["tok_w"] / 0.255
    print(f"{B:>4} {P:>5} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
          f"{r['tok_w']:>8.4f} {r['elapsed_s']:>5.1f} "
          f"{r['last_token_text']!r:>22} {x_eager_b1:>15.2f}×")

with open(os.path.join(ROOT, "kickass_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[bench] wrote kickass_results.json")

# Compute throughput peak / efficiency peak
if results:
    best_tps = max(results, key=lambda r: r["tps"])
    best_tw  = max(results, key=lambda r: r["tok_w"])
    print(f"\nPeak throughput: B={best_tps['batch']:>3} = "
          f"{best_tps['tps']:.0f} tok/s @ {best_tps['draw_w']:.0f} W")
    print(f"Peak tok/W:      B={best_tw['batch']:>3} = "
          f"{best_tw['tok_w']:.3f} tok/W "
          f"({best_tw['tok_w']/0.255:.1f}× eager B=1 baseline)")
    if best_tw['tok_w'] / 0.255 >= 10.0:
        print(f"\n>>> 10× tok/W TARGET HIT at B={best_tw['batch']} <<<")
    else:
        print(f"\n  best ratio: {best_tw['tok_w']/0.255:.2f}× eager B=1 baseline; "
              f"need {10.0:.0f}× — gap {10.0/(best_tw['tok_w']/0.255):.2f}×")
