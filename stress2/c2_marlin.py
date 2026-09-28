"""C2 — Marlin INT4 weight substitution for Llama-3.1-8B on system torch 2.7.

Patches every nn.Linear in the model:
  1. Transpose weight to (in, out) row-major.
  2. Observe 1001x to cross cipher_weight_compress stability threshold.
  3. cipher_weight_compress_quantize(wt, in, out)  → INT4 + groupwise scales.
  4. cipher_weight_compress_repack_marlin(wt)      → Marlin XOR-swizzled layout.
  5. cipher_weight_compress_lookup_marlin(wt)      → (B, S, K, N, G).
  6. forward(x): if M <= MARLIN_MAX_M, call cipher_weight_compress_marlin_gemm.
                 Else fall back to original Linear.

Reports:
    - tps, watts, tok/W
    - n_marlin (calls routed to Marlin) vs n_fallback
    - first_text coherence

Compatible with both CIPHER hook loaded and not — Marlin is invoked through
the rt's exported symbols regardless. Use `CIPHER=0 PYTHONPATH=...` to run
the pure-Python harness without the LD_PRELOAD'd cublasGemmEx shim.
"""
import os, sys, json, time, ctypes, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_HOOK = os.environ.get("CIPHER", "1") != "0"
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")
MARLIN_MAX_M = int(os.environ.get("MARLIN_MAX_M", "8"))


def setup_rt():
    """Load libcipher_rt.so and bind Marlin functions."""
    rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

    rt.cipher_substitute_v2_init.restype = ctypes.c_int
    rt.cipher_substitute_v2_init()
    rt.cipher_weight_compress_init.restype = ctypes.c_int
    rt.cipher_weight_compress_init()

    rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rt.cipher_weight_compress_observe.restype  = ctypes.c_int
    rt.cipher_weight_compress_quantize.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    rt.cipher_weight_compress_quantize.restype  = ctypes.c_int
    rt.cipher_weight_compress_repack_marlin.argtypes = [ctypes.c_void_p]
    rt.cipher_weight_compress_repack_marlin.restype  = ctypes.c_int
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


class MarlinLinear:
    """Wraps an nn.Linear with Marlin INT4 substitution.

    Mutates the original module in-place: replaces forward() and stows the
    transposed fp16 weight as an attribute (kept alive so its pointer stays
    valid for the rt's compressed-weight cache lookup)."""
    def __init__(self, rt, orig, name):
        import torch
        self.rt = rt
        self.name = name
        self.orig = orig
        # Capture the original forward BEFORE we replace it on the module —
        # otherwise self.orig(x) recurses through the patched forward.
        self._orig_forward = type(orig).forward.__get__(orig, type(orig))
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        self._compressed = False
        self._bias_added = False

        # Marlin requires N (out_features) divisible by 128 and K (in_features)
        # divisible by 128. Anything else falls back.
        if (self.out_features % 128 != 0) or (self.in_features % 128 != 0):
            return
        # Marlin's m_blocks supports M in {1,2,3,4} blocks of 16, so M up to 64.

        # Transposed weight: (in, out) row-major.
        wt = orig.weight.detach().t().contiguous().to(torch.float16)
        self._wt = wt  # keep alive
        # Observe 1001 times to cross stability gate.
        nbytes = wt.numel() * 2
        for _ in range(1001):
            rt.cipher_weight_compress_observe(wt.data_ptr(), nbytes)
        rc = rt.cipher_weight_compress_quantize(
            wt.data_ptr(), self.in_features, self.out_features)
        if rc != 1:
            return
        rc = rt.cipher_weight_compress_repack_marlin(wt.data_ptr())
        if rc != 1:
            return
        b = ctypes.c_void_p(); s = ctypes.c_void_p()
        K = ctypes.c_int(); N = ctypes.c_int(); G = ctypes.c_int()
        ok = rt.cipher_weight_compress_lookup_marlin(
            wt.data_ptr(), ctypes.byref(b), ctypes.byref(s),
            ctypes.byref(K), ctypes.byref(N), ctypes.byref(G))
        if ok != 1: return
        self._mB = b.value; self._mS = s.value
        self._mK = K.value; self._mN = N.value; self._mG = G.value
        self._compressed = True

    def __call__(self, x):
        import torch
        if not self._compressed:
            return self._orig_forward(x)
        # x shape: (..., in). Flatten leading dims to (M, in).
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


class Counters:
    marlin = 0
    fallback_largeM = 0
    fallback_rc = 0
    fallback_shape = 0


def patch_model(rt, model):
    import torch.nn as nn
    n_compressed = 0
    n_skipped = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.is_cuda:
            ml = MarlinLinear(rt, mod, name)
            if ml._compressed:
                # Replace forward.
                mod._marlin = ml
                mod.forward = ml
                n_compressed += 1
            else:
                n_skipped += 1
    return n_compressed, n_skipped


def main():
    rt = setup_rt()
    import torch
    print(f"[C2] use_hook={USE_HOOK} torch={torch.__version__}", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=False, rt=None)
    n_c, n_s = patch_model(rt, model)
    print(f"[C2] Marlin-patched {n_c} linears, skipped {n_s}", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    # Warmup.
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    Counters.marlin = Counters.fallback_largeM = Counters.fallback_rc = 0

    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn, max_new_tokens=N_TOKENS,
                              do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    n = out.shape[1] - ids.shape[1]
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    tps = n / elapsed
    payload = dict(
        n_compressed_layers=n_c, n_skipped_layers=n_s,
        marlin_calls=Counters.marlin,
        fallback_largeM=Counters.fallback_largeM,
        fallback_rc=Counters.fallback_rc,
        target_tokens=N_TOKENS, generated=n,
        elapsed_s=elapsed, tps=tps, mean_w=mean_w,
        tok_w=tps/mean_w if mean_w > 0 else 0,
        coherent=("!!!!" not in text),
        first_text=text[:120])
    out_path = os.path.join(os.path.dirname(__file__), "c2_marlin.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[C2] gen={n}/{N_TOKENS} tps={tps:.2f} W={mean_w:.0f} tok/W={tps/mean_w if mean_w>0 else 0:.4f}",
          flush=True)
    print(f"[C2] marlin={Counters.marlin} fb_largeM={Counters.fallback_largeM} "
          f"fb_rc={Counters.fallback_rc}", flush=True)
    print(f"[C2] first_text: {text[:120]!r}", flush=True)


if __name__ == "__main__":
    main()
