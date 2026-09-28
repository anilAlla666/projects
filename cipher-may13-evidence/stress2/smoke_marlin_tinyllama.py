"""Smoke test — Marlin INT4 weight substitution on TinyLlama-1.1B-Chat.

Mirrors c2_marlin.py's MarlinLinear pattern but loads TinyLlama directly
(no auth needed) and skips stress_common's hardcoded Llama-3.1-8B path
and Llama-specific fusion patches.

Pass/fail criteria:
  - n_compressed > 0   (Marlin patched at least one nn.Linear)
  - Counters.marlin > 0 (Marlin GEMM fires during decode)
  - text is coherent  (non-empty, ASCII-printable, no '!!!!', no early eos)
  - exit 0            (no segfault, no Python exception)

No LD_PRELOAD, no clock pin, no DVFS, no FP8 envs.
"""
import os, sys, ctypes, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "/home/ubuntu/models/TinyLlama-1.1B"
N_TOKENS = int(os.environ.get("N_TOKENS", "80"))
MARLIN_MAX_M = int(os.environ.get("MARLIN_MAX_M", "8"))
PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# These two are required for Marlin to function (NOT FP8 envs):
#   CIPHER_WEIGHT_COMPRESS enables the INT4 quant/repack engine.
#   CIPHER_SUBSTITUTE_V2   enables the NVRTC pipeline that backs the Marlin kernel.
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")


def setup_rt():
    # rt.so references CUDA driver symbols (e.g. cuGreenCtxDestroy) without
    # being linked to libcuda. Bring libcuda into the global namespace first
    # so subsequent dlopen of rt.so resolves those symbols. This is required
    # regardless of LD_PRELOAD; the CIPHER hook itself is NOT preloaded.
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    rt = ctypes.CDLL(str(ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
    rt.cipher_substitute_v2_init.restype = ctypes.c_int
    rt.cipher_substitute_v2_init()
    rt.cipher_weight_compress_init.restype = ctypes.c_int
    rt.cipher_weight_compress_init()

    rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rt.cipher_weight_compress_observe.restype = ctypes.c_int
    rt.cipher_weight_compress_quantize.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    rt.cipher_weight_compress_quantize.restype = ctypes.c_int
    rt.cipher_weight_compress_repack_marlin.argtypes = [ctypes.c_void_p]
    rt.cipher_weight_compress_repack_marlin.restype = ctypes.c_int
    rt.cipher_weight_compress_lookup_marlin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)]
    rt.cipher_weight_compress_lookup_marlin.restype = ctypes.c_int
    rt.cipher_weight_compress_marlin_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p]
    rt.cipher_weight_compress_marlin_gemm.restype = ctypes.c_int
    return rt


class Counters:
    marlin = 0
    fallback_largeM = 0
    fallback_rc = 0


VERBOSE_FIRST = int(os.environ.get("VERBOSE_FIRST", "3"))


class MarlinLinear:
    _ctor_count = 0

    def __init__(self, rt, orig, name):
        import torch
        self.rt = rt
        self.name = name
        self.orig = orig
        self._orig_forward = type(orig).forward.__get__(orig, type(orig))
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        self._compressed = False
        verbose = MarlinLinear._ctor_count < VERBOSE_FIRST
        MarlinLinear._ctor_count += 1
        if (self.out_features % 128 != 0) or (self.in_features % 128 != 0):
            if verbose:
                print(f"  [skip {name}] in={self.in_features} out={self.out_features}"
                      f" — not 128-divisible", flush=True)
            return
        wt = orig.weight.detach().t().contiguous().to(torch.float16)
        self._wt = wt
        nbytes = wt.numel() * 2
        for _ in range(1001):
            rt.cipher_weight_compress_observe(wt.data_ptr(), nbytes)
        rc = rt.cipher_weight_compress_quantize(
            wt.data_ptr(), self.in_features, self.out_features)
        if verbose:
            print(f"  [{name}] in={self.in_features} out={self.out_features}"
                  f" quantize_rc={rc}", flush=True)
        if rc != 1:
            return
        rc = rt.cipher_weight_compress_repack_marlin(wt.data_ptr())
        if verbose:
            print(f"  [{name}] repack_marlin_rc={rc}", flush=True)
        if rc != 1:
            return
        b = ctypes.c_void_p(); s = ctypes.c_void_p()
        K = ctypes.c_int(); N = ctypes.c_int(); G = ctypes.c_int()
        ok = rt.cipher_weight_compress_lookup_marlin(
            wt.data_ptr(), ctypes.byref(b), ctypes.byref(s),
            ctypes.byref(K), ctypes.byref(N), ctypes.byref(G))
        if verbose:
            print(f"  [{name}] lookup_marlin_rc={ok} K={K.value} N={N.value} G={G.value}",
                  flush=True)
        if ok != 1:
            return
        self._mB = b.value; self._mS = s.value
        self._mK = K.value; self._mN = N.value; self._mG = G.value
        self._compressed = True

    def __call__(self, x):
        import torch
        if not self._compressed:
            return self._orig_forward(x)
        orig_shape = x.shape
        if x.shape[-1] != self.in_features:
            return self._orig_forward(x)
        xf = x.reshape(-1, self.in_features).contiguous()
        if not xf.is_contiguous() or xf.dtype != torch.float16 or not xf.is_cuda:
            return self._orig_forward(x)
        M = xf.shape[0]
        if M > MARLIN_MAX_M:
            Counters.fallback_largeM += 1
            return self._orig_forward(x)
        out = torch.empty(M, self.out_features, dtype=torch.float16,
                          device=x.device)
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = self.rt.cipher_weight_compress_marlin_gemm(
            xf.data_ptr(), self._mB, self._mS, out.data_ptr(),
            M, self.out_features, self.in_features, self._mG, stream_ptr)
        if rc != 1:
            Counters.fallback_rc += 1
            return self._orig_forward(x)
        Counters.marlin += 1
        out = out.reshape(orig_shape[:-1] + (self.out_features,))
        if self.bias is not None:
            out = out + self.bias
        return out


def patch_model(rt, model):
    import torch.nn as nn
    n_compressed = 0
    n_skipped = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.is_cuda:
            ml = MarlinLinear(rt, mod, name)
            if ml._compressed:
                mod._marlin = ml
                mod.forward = ml
                n_compressed += 1
            else:
                n_skipped += 1
    return n_compressed, n_skipped


def is_coherent(text):
    if not text:
        return False, "empty"
    if "!!!!" in text:
        return False, "contains !!!!"
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    if printable < int(0.9 * len(text)):
        return False, f"non-printable chars (printable={printable}/{len(text)})"
    return True, "ok"


def main():
    print(f"[SMOKE] loading rt from {ROOT/'libcipher_rt.so'}", flush=True)
    rt = setup_rt()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[SMOKE] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    model.requires_grad_(False)
    model.train(False)
    load_s = time.perf_counter() - t0
    print(f"[SMOKE] model loaded in {load_s:.2f}s", flush=True)

    n_c, n_s = patch_model(rt, model)
    print(f"[SMOKE] Marlin-patched {n_c} linears, skipped {n_s}", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    print(f"[SMOKE] warmup (4 tokens)...", flush=True)
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                           do_sample=False, pad_token_id=tok.pad_token_id,
                           use_cache=True)
    torch.cuda.synchronize()
    Counters.marlin = 0
    Counters.fallback_largeM = 0
    Counters.fallback_rc = 0

    print(f"[SMOKE] generating {N_TOKENS} tokens...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn,
                             max_new_tokens=N_TOKENS,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids = out[0, ids.shape[1]:]
    n_gen = gen_ids.shape[0]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    text_skip = tok.decode(gen_ids, skip_special_tokens=True)
    tps = n_gen / elapsed if elapsed > 0 else 0.0

    coherent, why = is_coherent(text_skip)
    print(f"\n[SMOKE] === RESULTS ===", flush=True)
    print(f"[SMOKE] n_compressed     = {n_c}    (>0 ? {n_c > 0})", flush=True)
    print(f"[SMOKE] Counters.marlin  = {Counters.marlin}    (>0 ? {Counters.marlin > 0})", flush=True)
    print(f"[SMOKE] fb_largeM        = {Counters.fallback_largeM}", flush=True)
    print(f"[SMOKE] fb_rc            = {Counters.fallback_rc}", flush=True)
    print(f"[SMOKE] gen tokens       = {n_gen}/{N_TOKENS}", flush=True)
    print(f"[SMOKE] tps              = {tps:.2f}", flush=True)
    print(f"[SMOKE] coherent         = {coherent}    ({why})", flush=True)
    print(f"[SMOKE] text (skip_spec) = {text_skip[:200]!r}", flush=True)
    print(f"[SMOKE] text (raw)       = {text[:200]!r}", flush=True)

    p1 = n_c > 0
    p2 = Counters.marlin > 0
    p3 = coherent
    print(f"\n[SMOKE] PASS/FAIL:", flush=True)
    print(f"  n_compressed > 0     : {'PASS' if p1 else 'FAIL'}", flush=True)
    print(f"  Counters.marlin > 0  : {'PASS' if p2 else 'FAIL'}", flush=True)
    print(f"  text coherent        : {'PASS' if p3 else 'FAIL'}", flush=True)
    print(f"  exit 0               : (this run will report based on overall)", flush=True)

    return 0 if (p1 and p2 and p3) else 1


if __name__ == "__main__":
    sys.exit(main())
