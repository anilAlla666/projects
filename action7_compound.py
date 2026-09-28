"""ACTION 7: Power cap + best CIPHER stack at batch=1, graph capture.

Combine the only verified levers:
  - fp16 baseline + graph capture
  - + fusion kernels (CIPHER RMSNorm, SiLU·Mul) — verified ~5% tok/W
  - + INT4 substitution (gates to fp16 fallback at M>1; only fires at M=1)
  - + 200W power cap (clock locks ~1200 MHz)

Configurations measured at batch=1:
  C1: fp16 graph                     (no power cap)
  C2: fp16 graph + fusion            (no power cap)
  C3: fp16 graph + fusion + INT4     (no power cap)
  C4: fp16 graph                     (200W cap)
  C5: fp16 graph + fusion            (200W cap)
  C6: fp16 graph + fusion + INT4     (200W cap)
"""
import os, ctypes, time, threading, subprocess, json
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

# Init
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype = ctypes.c_int
# FIXED 8-arg
rt.cipher_weight_compress_int4_gemv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
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
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1 and x.shape[0] == 1:
            xb = x.reshape(-1, self.in_features).contiguous()
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream)
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
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rt.cipher_fused_rmsnorm(flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                              flat.shape[0], flat.shape[-1], float(eps), stream)
    return out


def cipher_silu_mul(gate, up):
    if not gate.is_contiguous(): gate = gate.contiguous()
    if not up.is_contiguous(): up = up.contiguous()
    out = torch.empty_like(gate)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(), stream)
    return out


from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of artificial intelligence in GPU computing is to make every joule of energy count."
N_TOKENS = 256
MAX_LEN = 384


def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except: pass
        time.sleep(0.2)


def set_power_cap(watts):
    """Set or unset GPU power cap. Returns True on success."""
    if watts is None:
        # Reset to default
        r = subprocess.run(["nvidia-smi", "-pl", "700"], capture_output=True, text=True, timeout=5)
    else:
        r = subprocess.run(["nvidia-smi", "-pl", str(watts)], capture_output=True, text=True, timeout=5)
    return r.returncode == 0


def run_graph_decode(model, prompt_ids, batch=1):
    cfg = model.config
    cache = StaticCache(config=cfg, max_batch_size=batch, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(batch, 1, cfg.vocab_size, device="cuda", dtype=torch.float16)

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
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        cache_pos += 1
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    return elapsed, samples


def build(use_fusion=False, use_int4=False):
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    m.train(False)
    if use_fusion:
        MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight, self.variance_epsilon)
        MistralMLP.forward = lambda self, x: self.down_proj(cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
    if use_int4:
        for layer in m.model.layers:
            for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                old = getattr(layer.self_attn, attr)
                if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                    setattr(layer.self_attn, attr, Int4Linear(old).to("cuda"))
            for attr in ["gate_proj", "up_proj", "down_proj"]:
                old = getattr(layer.mlp, attr)
                if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                    setattr(layer.mlp, attr, Int4Linear(old).to("cuda"))
    return m


def measure(label, model, tok, results):
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    elapsed, samples = run_graph_decode(model, prompt_ids, batch=1)
    powers = [s[0] for s in samples[1:]]
    clocks = [s[1] for s in samples[1:]]
    mean_pw = sum(powers)/len(powers) if powers else 0
    mean_clk = sum(clocks)/len(clocks) if clocks else 0
    tps = N_TOKENS / elapsed
    tokw = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:<48}  tps={tps:>7.2f}  W={mean_pw:>6.1f}  clk={mean_clk:>5.0f}  tok/W={tokw:>6.3f}", flush=True)
    results.append({"label": label, "tps": tps, "watts": mean_pw, "clk_mhz": mean_clk, "tokw": tokw})


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    results = []
    print(f"\n=== ACTION 7: stacked levers at batch=1 ===\n", flush=True)

    print(f"--- Power cap: 700W (default) ---", flush=True)
    set_power_cap(700)
    m1 = build(False, False); measure("C1: fp16 graph",                          m1, tok, results); del m1; torch.cuda.empty_cache()
    m2 = build(True,  False); measure("C2: fp16 graph + fusion",                 m2, tok, results); del m2; torch.cuda.empty_cache()
    m3 = build(True,  True);  measure("C3: fp16 graph + fusion + INT4",          m3, tok, results); del m3; torch.cuda.empty_cache()

    print(f"\n--- Power cap: 200W ---", flush=True)
    if set_power_cap(200):
        m4 = build(False, False); measure("C4: fp16 graph (200W)",                m4, tok, results); del m4; torch.cuda.empty_cache()
        m5 = build(True,  False); measure("C5: fp16 graph + fusion (200W)",       m5, tok, results); del m5; torch.cuda.empty_cache()
        m6 = build(True,  True);  measure("C6: fp16 graph + fusion + INT4 (200W)", m6, tok, results); del m6; torch.cuda.empty_cache()
        set_power_cap(700)  # reset
    else:
        print("  power cap setting failed (need root?); skipping 200W rows", flush=True)

    print(f"\n=== Best tok/W: {max(results, key=lambda r: r['tokw'])['label']} ===", flush=True)
    best = max(results, key=lambda r: r['tokw'])
    print(f"  {best['label']}  tok/W={best['tokw']:.3f}", flush=True)

    with open("/home/ubuntu/op31-prod-fix/action7_compound_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
