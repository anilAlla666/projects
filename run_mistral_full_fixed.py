#!/usr/bin/env python3
"""Mistral-7B full stack — INT4 GEMV + fused kernels + graph capture, with the
ctypes signature bug FIXED (cipher_weight_compress_int4_gemv takes 8 args:
a, b_T, scales, c, M, N, K, stream — original harness was missing M).

Also sets CIPHER_WEIGHT_COMPRESS=on / CIPHER_SUBSTITUTE_V2=on BEFORE loading the
runtime so the engine's gate flag is on at constructor time.
"""
import os, ctypes, time, threading, subprocess
# Must be set before lib load:
os.environ.setdefault("CIPHER_SUBSTITUTE_V2",   "on")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
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
# FIXED: 8 args including M
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
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            xb = x.reshape(-1, self.in_features).contiguous()
            out = torch.empty(1, self.out_features, dtype=torch.float16, device=x.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                1, self.out_features, self.in_features, stream)   # FIXED: M=1 explicit
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
    if not up.is_contiguous():   up = up.contiguous()
    out = torch.empty_like(gate)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                              gate.numel(), stream)
    return out


from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP

MODEL = "mistralai/Mistral-7B-v0.1"
N_TOKENS = 256
MAX_LEN = 384
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
nparams = sum(p.numel() for p in model.parameters())
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids.shape[1]

mode = os.environ.get("CIPHER_MODE", "")

if "FUSED" in mode:
    MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight, self.variance_epsilon)
    MistralMLP.forward = lambda self, x: self.down_proj(cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
    print("[FUSE] patched MistralRMSNorm + MistralMLP")

if "INT4" in mode:
    n = 0
    n_compressed = 0
    for layer in model.model.layers:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                new = Int4Linear(old).to("cuda")
                if new._compressed: n_compressed += 1
                setattr(layer.self_attn, attr, new)
                n += 1
        for attr in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                new = Int4Linear(old).to("cuda")
                if new._compressed: n_compressed += 1
                setattr(layer.mlp, attr, new)
                n += 1
    print(f"[INT4] replaced {n} nn.Linear modules with Int4Linear ({n_compressed} actually compressed)")


def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.2)


def run_eager():
    with torch.no_grad():
        _ = model.generate(prompt_ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    return elapsed, samples, out


def run_graph():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(1, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)

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


def report(label, elapsed, samples):
    pw = [s[0] for s in samples[1:]] or [0]
    clk = [s[1] for s in samples[1:]] or [0]
    mean_pw = sum(pw) / len(pw) if pw else 0
    mean_clk = sum(clk) / len(clk) if clk else 0
    tps = N_TOKENS / elapsed
    flops = 2.0 * nparams * N_TOKENS
    tflops = flops / elapsed / 1e12
    mfu = tflops / 989.0 * 100.0
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:25s} tps={tps:7.2f}  power={mean_pw:5.1f}W  TFLOPS={tflops:5.2f}  MFU={mfu:5.2f}%  tok/W={tok_w:6.3f}")


print(f"=== Mistral-7B mode={mode!r} N={N_TOKENS} ===")
e_eager, s_eager, out_text = run_eager()
report("eager", e_eager, s_eager)
gen_ids = out_text[0, prompt_ids.shape[1]:].tolist()
print(f"  eager output[:60]: {tok.decode(gen_ids)[:120]!r}")
e_g, s_g = run_graph()
report("graph", e_g, s_g)
