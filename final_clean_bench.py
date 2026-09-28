#!/usr/bin/env python3
"""ONE clean end-to-end benchmark with primary-evidence guards.

Stack (all verified):
  - INT4 GEMV with corrected (M, N, K, stream) binding [verify_int4_binding.py]
  - silu_mul megakernel + down_residual megakernel at B=1 [verify_megakernel.py]
  - Graph capture
  - Fused RMSNorm + fused SiLU·Mul (existing, correctness-tested)

Guards on every measurement:
  1. NaN-sentinel: out_logits filled with NaN before each measure(). After
     warmup and after sustained loop we assert no NaN remains in last token's
     logits — proves the full stack actually computed something.
  2. Path-fire counters at the Python boundary:
        - n_int4_path:     INT4 GEMV (rc==1) calls
        - n_int4_fallback: rc!=1 fallthroughs
        - n_megakernel:    Step-2 megakernel (rc==1) calls
        - n_mlp_fallback:  megakernel rc!=1 or B>1 fallthroughs
     If a config claims "INT4+megakernel active" the counter must be
     non-zero. Otherwise the row is flagged.
  3. Sanity vs decoded text: print 16 generated tokens to confirm the model
     is generating English, not garbage.
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
    # CORRECTED 8-arg binding — (4 ptr, M, N, K, stream)
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


# Path-fire counters — module-global so they survive monkey-patches
class Counters:
    n_int4_path = 0
    n_int4_fallback = 0
    n_mlp_megakernel = 0
    n_mlp_fallback = 0
    def reset(self):
        Counters.n_int4_path = 0
        Counters.n_int4_fallback = 0
        Counters.n_mlp_megakernel = 0
        Counters.n_mlp_fallback = 0
C = Counters()


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
                Counters.n_int4_path += 1
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                Counters.n_int4_fallback += 1
                out = (xb @ self.wt_fp16).reshape(x.shape[:-1] + (self.out_features,))
        else:
            Counters.n_int4_fallback += 1
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
            if rc != 1:
                Counters.n_mlp_fallback += 1
                return self._fallback(x)
            out = torch.empty(1, D, dtype=torch.float16, device=x.device)
            zero = torch.zeros(1, D, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_down_residual(
                sm_buf.data_ptr(),
                self.down_proj._bT_ptr, self.down_proj._bs_ptr,
                zero.data_ptr(),
                out.data_ptr(),
                1, D, N_int, stream_ptr())
            if rc != 1:
                Counters.n_mlp_fallback += 1
                return self._fallback(x)
            Counters.n_mlp_megakernel += 1
            return out.reshape(x.shape[:-1] + (D,))
        Counters.n_mlp_fallback += 1
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
WARMUP_REPLAYS  = 8
MEASURE_SECONDS = 8.0
MAX_LEN         = 1300

print("[bench] loading Mistral-7B-v0.1...")
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
print(f"[bench] prompt_len={prompt_ids_1.shape[1]}")

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

# Quick generated-text sanity: 16 tokens at B=1
ids = tok("The future of GPU computing is", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=16, do_sample=False, use_cache=True)
text = tok.decode(gen[0, ids.shape[1]:].tolist())
print(f"[bench] sanity gen: {text!r}")
del gen


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
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)

    # NaN-sentinel: every output buffer pre-filled with NaN before any
    # decode work; if the captured graph silently failed we'd see NaN.
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

    # NaN sentinel check #1: re-fill out_logits with NaN, run warmup, assert
    # no NaN remains. This proves the captured graph wrote real logits.
    out_logits.fill_(float("nan"))
    for _ in range(WARMUP_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    nan_after_warmup = torch.isnan(out_logits).sum().item()
    if nan_after_warmup > 0:
        return dict(error=f"NaN in logits after warmup ({nan_after_warmup} elements)")

    # Re-fill NaN once more and run the timed loop. After the loop, last
    # logits must again be all-finite.
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
        return dict(error=f"NaN in logits after timed loop ({nan_after_loop})")

    # Logits sanity: argmax should give a valid token id; sample one.
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
    del g, cache, cache_pos, out_logits, input_ids, prompt_ids
    torch.cuda.empty_cache()
    return out


print("\n=== FINAL CLEAN BENCHMARK ===")
print(f"{'B':>3} {'tps':>8} {'watts':>7} {'clk':>5} {'tok/W':>7} "
      f"{'INT4':>6} {'fallb':>6} {'mega':>6} {'mlpfb':>6} {'sec':>4} "
      f"{'last_tok':>20}")

results = []
for B in [1, 8, 32, 64]:
    C.reset()
    t_start = time.perf_counter()
    try:
        r = measure(B)
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B}: OOM"); torch.cuda.empty_cache(); continue
    except Exception as e:
        print(f"  B={B}: {type(e).__name__}: {e}"); continue
    t_elapsed = time.perf_counter() - t_start

    if "error" in r:
        print(f"  B={B}: SENTINEL FAILURE: {r['error']}")
        continue

    n_int4 = Counters.n_int4_path
    n_fb   = Counters.n_int4_fallback
    n_mk   = Counters.n_mlp_megakernel
    n_mfb  = Counters.n_mlp_fallback
    r["batch"] = B
    r["counters"] = dict(int4_path=n_int4, int4_fallback=n_fb,
                         mlp_megakernel=n_mk, mlp_fallback=n_mfb)
    results.append(r)
    print(f"{B:>3} {r['tps']:>8.2f} {r['draw_w']:>7.1f} "
          f"{r['clk_mhz']:>5.0f} {r['tok_w']:>7.4f} "
          f"{n_int4:>6} {n_fb:>6} {n_mk:>6} {n_mfb:>6} "
          f"{r['elapsed_s']:>4.1f} {r['last_token_text']!r:>20}")

with open(os.path.join(ROOT, "final_clean_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[bench] wrote final_clean_results.json")

# Per-row interpretation
print("\n=== Path-fire interpretation ===")
for r in results:
    B = r["batch"]; c = r["counters"]
    if B == 1:
        if c["int4_path"] > 0 and c["mlp_megakernel"] > 0:
            print(f"  B={B}: ✓ INT4 GEMV fired {c['int4_path']}× ; "
                  f"megakernel fired {c['mlp_megakernel']}×")
        else:
            print(f"  B={B}: ✗ FAIL — INT4 ({c['int4_path']}) or megakernel "
                  f"({c['mlp_megakernel']}) did not fire")
    else:
        if c["int4_path"] == 0 and c["mlp_megakernel"] == 0:
            print(f"  B={B}: ✓ M=1 paths correctly dormant; "
                  f"fp16 cuBLAS ({c['int4_fallback']} linear fallbacks, "
                  f"{c['mlp_fallback']} MLP fallbacks)")
        else:
            print(f"  B={B}: ⚠ unexpected M=1 path firing at B>1: "
                  f"int4={c['int4_path']} mega={c['mlp_megakernel']}")
