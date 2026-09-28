#!/usr/bin/env python3
"""FINAL — combined verified-positive levers per batch.

Levers active (each only where its sign is positive):
  ALL B: eager mode (no graph capture)             [Pillar 1A baseline]
  ALL B: clock-locked to per-batch optimum         [Pillar 5B]  ← biggest win
  B=1:   INT4 GEMV (corrected binding)             [Pillar 1B / Step 5 fix]
  B=1:   MLP megakernel                            [Pillar 2A]
  ALL B: fused RMSNorm + fused SiLU·Mul            [pre-existing]
  ALL B: NaN sentinel + counter verification

Excluded (verified-negative or unsafe):
  V3 KV redirect  — −1.9% tok/W at B=1, layout broken at B>1
  Adaptive depth  — KL=9.96 nats per token, output incoherent
  Idle injection  — H100 doesn't clock down on µs gaps (verified)
  M>1 megakernel  — −60-73% tok/W (GEMV-style, no tensor cores)

Per-batch clock plan (from p5b_clock_sweep.json, "best tok/W ≥95% peak tps"):
  B=1  → 1000 MHz
  B=8  → 1600 MHz   (compromise: tighter tps loss vs more aggressive 1200)
  B=32 → 1980 MHz   (boost; tps too critical to drop)
  B=64 → 1980 MHz

Optionally also reports the maximally-aggressive (best tok/W period) clock
for each batch as a separate row.
"""
import os, sys, ctypes, time, threading, subprocess, json, gc, atexit, signal
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

for fn, restype, argtypes in [
    ("cipher_substitute_v2_init", ctypes.c_int, []),
    ("cipher_weight_compress_init", ctypes.c_int, []),
    ("cipher_fusion_kernels_init", ctypes.c_int, []),
    ("cipher_weight_compress_observe", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t]),
    ("cipher_weight_compress_quantize", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("cipher_weight_compress_lookup_T", ctypes.c_int,
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


def lock_clock(mhz):
    return subprocess.run(["sudo","-n","nvidia-smi","-lgc",str(mhz)],
                          capture_output=True, text=True).returncode == 0
def restore_clock():
    subprocess.run(["sudo","-n","nvidia-smi","-rgc"],
                   capture_output=True, text=True)
atexit.register(restore_clock)
def _sig(s,f): restore_clock(); sys.exit(128+s)
signal.signal(signal.SIGINT,_sig); signal.signal(signal.SIGTERM,_sig)


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP


class Counters:
    int4_path = 0
    mlp_megakernel = 0
    fallback = 0
    @classmethod
    def reset(cls):
        cls.int4_path = cls.mlp_megakernel = cls.fallback = 0


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
                if self.bias is not None: out = out + self.bias
                return out
        Counters.fallback += 1
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
        # Gate megakernel strictly to M=1 (verified positive there)
        if (self._megakernel_active and x.dim() >= 2
                and x.shape[-2] == 1 and x.shape[0] == 1):
            D = x.shape[-1]
            in_flat = x.reshape(-1, D)
            N_int = self.gate_proj.out_features
            sm_buf = torch.empty(1, N_int, dtype=torch.float16, device=x.device)
            rc = rt.cipher_weight_compress_int4_silu_mul(
                in_flat.data_ptr(),
                self.gate_proj._bT_ptr, self.gate_proj._bs_ptr,
                self.up_proj._bT_ptr,   self.up_proj._bs_ptr,
                sm_buf.data_ptr(),
                1, N_int, D, stream_ptr())
            if rc == 1:
                out = torch.empty(1, D, dtype=torch.float16, device=x.device)
                zero = torch.zeros(1, D, dtype=torch.float16, device=x.device)
                rc = rt.cipher_weight_compress_int4_down_residual(
                    sm_buf.data_ptr(),
                    self.down_proj._bT_ptr, self.down_proj._bs_ptr,
                    zero.data_ptr(), out.data_ptr(),
                    1, D, N_int, stream_ptr())
                if rc == 1:
                    Counters.mlp_megakernel += 1
                    return out.reshape(x.shape[:-1] + (D,))
        return self._fallback(x)

    def _fallback(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        if not gate.is_contiguous(): gate = gate.contiguous()
        if not up.is_contiguous():   up = up.contiguous()
        sm = torch.empty_like(gate)
        rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), sm.data_ptr(),
                                  gate.numel(), stream_ptr())
        return self.down_proj(sm)


def cipher_rmsnorm(x, weight, eps):
    out = torch.empty_like(x)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out_flat = out.reshape(-1, x.shape[-1])
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps), stream_ptr())
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

# Per-batch clock plan: best tok/W with ≤5% tps loss
SAFE_CLK = {1: 1000, 8: 1600, 32: 1980, 64: 1980}
# Aggressive (best tok/W period, accepting tps loss)
AGGRESSIVE_CLK = {1: 1000, 8: 1200, 32: 1000, 64: 1000}

print("[final] loading Mistral-7B...")
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

MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight,
                                                        self.variance_epsilon)
MistralMLP.forward    = lambda self, x: self.down_proj(
    cipher_silu_mul_outer(self.gate_proj(x), self.up_proj(x)))
n_int4 = 0
for layer in model.model.layers:
    for attr in ["q_proj","k_proj","v_proj","o_proj"]:
        old = getattr(layer.self_attn, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.self_attn, attr, Int4Linear(old).to("cuda")); n_int4 += 1
    for attr in ["gate_proj","up_proj","down_proj"]:
        old = getattr(layer.mlp, attr)
        if old.in_features % 128 == 0 and old.out_features % 8 == 0:
            setattr(layer.mlp, attr, Int4Linear(old).to("cuda")); n_int4 += 1
n_fused = 0
for layer in model.model.layers:
    layer.mlp = FusedMistralMLP(layer.mlp).to("cuda"); n_fused += 1
print(f"[final] {n_int4} Int4Linear, {n_fused} FusedMLP, fused norms active")


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
    Counters.reset()
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
    out = dict(tps=(B*n_steps)/elapsed, draw_w=sum(pw)/len(pw),
               clk_mhz=sum(clk)/len(clk),
               tok_w=((B*n_steps)/elapsed)/(sum(pw)/len(pw) if pw else 1),
               elapsed_s=elapsed, n_steps=n_steps, batch=B,
               last_token=last_token, last_token_text=sample_text,
               int4_path=Counters.int4_path,
               mlp_mega=Counters.mlp_megakernel,
               fallback=Counters.fallback)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


# Load eager baselines
with open(os.path.join(ROOT, "p1a_eager_baseline.json")) as f:
    EAGER = {r["batch"]: r for r in json.load(f)}

print(f"\n=== FINAL: per-batch optimal clock + verified-positive levers ===")
print(f"{'mode':>10} {'B':>3} {'clk':>5} {'tps':>9} {'watts':>7} {'clk_act':>8} "
      f"{'tok/W':>8} {'INT4':>6} {'mega':>5} {'×eager':>7} {'last':>8}")

results = []
for label, plan in [("safe", SAFE_CLK), ("aggressive", AGGRESSIVE_CLK)]:
    for B in [1, 8, 32, 64]:
        c = plan[B]
        if not lock_clock(c):
            print(f"   B={B}: clock-lock failed"); continue
        time.sleep(0.5)
        try:
            r = measure(B)
        except torch.cuda.OutOfMemoryError:
            print(f"   B={B}: OOM"); torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"   B={B}: {type(e).__name__}: {e}"); continue
        if "error" in r:
            print(f"   B={B}: SENTINEL FAIL: {r['error']}"); continue
        eb = EAGER.get(B)
        x_eager = r["tok_w"] / eb["tok_w"] if eb else 1.0
        r["mode"] = label; r["clk_set"] = c; r["x_eager"] = x_eager
        results.append(r)
        print(f"{label:>10} {B:>3} {c:>5} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
              f"{r['clk_mhz']:>8.0f} {r['tok_w']:>8.4f} {r['int4_path']:>6} "
              f"{r['mlp_mega']:>5} {x_eager:>7.2f}× {r['last_token_text']!r:>8}")

restore_clock()

with open(os.path.join(ROOT, "p_final_combined.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[final] wrote p_final_combined.json")

# Final summary
print("\n=== HONEST FINAL ===")
print(f"{'B':>3} {'eager tok/W':>11} {'safe×':>6} {'aggressive×':>11}")
for B in [1, 8, 32, 64]:
    eb = EAGER.get(B);
    safe = next((r for r in results if r["batch"] == B and r["mode"] == "safe"), None)
    aggr = next((r for r in results if r["batch"] == B and r["mode"] == "aggressive"), None)
    if not eb: continue
    s = f"{safe['x_eager']:.2f}×" if safe else "—"
    a = f"{aggr['x_eager']:.2f}×" if aggr else "—"
    print(f"{B:>3} {eb['tok_w']:>11.4f} {s:>6} {a:>11}")
