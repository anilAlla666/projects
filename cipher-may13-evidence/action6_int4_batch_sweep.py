"""ACTION 6: INT4 + FUSED + graph at batch=1, 8, 32, 64 with FIXED binding.

The INT4 GEMV kernel is M=1 only (decode). At batch>1 with seq_len=1, M=batch.
The kernel may handle this; if not, fp16 fallback. We measure both INT4 and
fp16 baselines at each batch and report.

Critically: verify INT4 actually fires at every batch (no silent fallback) by
counting calls vs fallback events in the Int4Linear forward.
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
# FIXED 8-arg binding
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


# Counter for how often the INT4 GEMV fires vs falls back
class Counter:
    int4_calls = 0
    fp16_fallback = 0


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
        # x: [B, 1, in] during decode (or [B, S, in])
        if self._compressed and x.dim() >= 2 and x.shape[-2] == 1:
            B = x.shape[0] if x.dim() == 3 else 1
            xb = x.reshape(-1, self.in_features).contiguous()
            out = torch.empty(B, self.out_features, dtype=torch.float16, device=x.device)
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            # M = B (each batch row needs its own GEMV)
            rc = rt.cipher_weight_compress_int4_gemv(
                xb.data_ptr(), self._bT_ptr, self._bs_ptr, out.data_ptr(),
                B, self.out_features, self.in_features, stream)
            if rc == 1:
                Counter.int4_calls += 1
                out = out.reshape(x.shape[:-1] + (self.out_features,))
            else:
                Counter.fp16_fallback += 1
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


def run_graph_decode(model, prompt_ids_batched, batch_size):
    """Graph capture replay decode for batched sequences. Returns elapsed and power samples."""
    cfg = model.config
    cache = StaticCache(config=cfg, max_batch_size=batch_size, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    prompt_len = prompt_ids_batched.shape[1]
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids_batched, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(batch_size, 1, cfg.vocab_size, device="cuda", dtype=torch.float16)

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


def measure(label, build_fn, batch_sizes, results, tok):
    print(f"\n=== {label} ===", flush=True)
    print(f"  {'batch':>5}  {'tps':>8}  {'tps/seq':>8}  {'power_W':>8}  {'tok/W':>7}  {'INT4 calls':>11}  {'fp16 fall':>10}", flush=True)
    print("  " + "-" * 80, flush=True)
    for B in batch_sizes:
        try:
            model = build_fn()
            prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
            prompt_ids_b = prompt_ids.expand(B, -1).contiguous()
            Counter.int4_calls = 0; Counter.fp16_fallback = 0
            elapsed, samples = run_graph_decode(model, prompt_ids_b, B)
            powers = [s[0] for s in samples[1:]]
            mean_pw = sum(powers)/len(powers) if powers else 0
            tps_total = (B * N_TOKENS) / elapsed
            tps_per_seq = N_TOKENS / elapsed
            tokw = tps_total / mean_pw if mean_pw > 0 else 0
            print(f"  {B:>5}  {tps_total:>8.1f}  {tps_per_seq:>8.1f}  {mean_pw:>8.1f}  {tokw:>7.3f}  "
                  f"{Counter.int4_calls:>11d}  {Counter.fp16_fallback:>10d}", flush=True)
            results.setdefault(label, []).append({
                "batch": B, "tps_total": tps_total, "tps_per_seq": tps_per_seq,
                "power_W": mean_pw, "tokw": tokw,
                "int4_calls": Counter.int4_calls, "fp16_fallback": Counter.fp16_fallback,
            })
            del model
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {B:>5}  OOM ({type(e).__name__})", flush=True)
            torch.cuda.empty_cache()


def build_fp16_baseline():
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    m.train(False)
    return m


def build_int4_fused():
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    m.train(False)
    # Patch fusion
    MistralRMSNorm.forward = lambda self, x: cipher_rmsnorm(x, self.weight, self.variance_epsilon)
    MistralMLP.forward = lambda self, x: self.down_proj(cipher_silu_mul(self.gate_proj(x), self.up_proj(x)))
    # Patch INT4
    n_replaced = 0; n_compressed = 0
    for layer in m.model.layers:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                new = Int4Linear(old).to("cuda")
                if new._compressed: n_compressed += 1
                setattr(layer.self_attn, attr, new); n_replaced += 1
        for attr in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, attr)
            if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                new = Int4Linear(old).to("cuda")
                if new._compressed: n_compressed += 1
                setattr(layer.mlp, attr, new); n_replaced += 1
    print(f"  [build] replaced {n_replaced} linears ({n_compressed} actually compressed)", flush=True)
    return m


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    results = {}
    BATCHES = [1, 8, 32, 64]
    measure("fp16 baseline (no CIPHER, graph)",       build_fp16_baseline, BATCHES, results, tok)
    measure("INT4+FUSED (FIXED binding, graph)",       build_int4_fused,     BATCHES, results, tok)

    # Compare
    print(f"\n=== COMPARISON: tok/W ratio (INT4+FUSED / fp16) ===", flush=True)
    base_rows = {r["batch"]: r for r in results.get("fp16 baseline (no CIPHER, graph)", [])}
    int4_rows = {r["batch"]: r for r in results.get("INT4+FUSED (FIXED binding, graph)", [])}
    for B in BATCHES:
        if B in base_rows and B in int4_rows:
            ratio = int4_rows[B]["tokw"] / base_rows[B]["tokw"] if base_rows[B]["tokw"] > 0 else 0
            print(f"  batch={B}: fp16 tok/W={base_rows[B]['tokw']:.3f}  INT4 tok/W={int4_rows[B]['tokw']:.3f}  ratio={ratio:.3f}x", flush=True)

    with open("/home/ubuntu/op31-prod-fix/action6_int4_batches.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[save] action6_int4_batches.json", flush=True)


if __name__ == "__main__":
    main()
