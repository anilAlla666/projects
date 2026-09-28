"""CIPHER density-sweep harness — shared Mistral-7B model in a single
process, N worker threads, per-thread StaticCache and CUDA Stream.

Marlin workspace race mitigation:
    The rt's Marlin GEMM uses a module-scope cudaMalloc'd workspace.
    Two threads concurrently calling cipher_weight_compress_marlin_gemm
    with different per-thread streams would have their GPU kernels touch
    the same workspace concurrently => race.

    Mitigation (v1 LD_PRELOAD prototype): route ALL Marlin work through
    a single shared CUDA stream _MARLIN_STREAM. Each tenant's calling
    stream waits for _MARLIN_STREAM after enqueue. A Python-level
    _MARLIN_LOCK serializes the host enqueue so the wait_stream /
    enqueue / wait_stream triple is atomic. This is GPU-serial for
    Marlin but lets non-Marlin ops (RMSNorm/SiLU/SDPA/elementwise)
    overlap across tenant streams.

    Architecturally consistent with the device-layer-hypervisor framing:
    CIPHER multiplexes a single physical Marlin pipeline across N
    logical tenants. The Green Context / kernel-module work removes
    this limitation in subsequent phases.
"""
import os, ctypes, threading
from pathlib import Path

ROOT = Path("/workspace")
MODEL_PATH = "/home/ubuntu/models/Mistral-7B-v0.1"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")
os.environ.setdefault("CIPHER_FAIRNESS", "on")
os.environ.setdefault("CIPHER_FUSION_KERNELS", "on")

PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")


# Shared Marlin synchronization
_MARLIN_LOCK = threading.Lock()
_MARLIN_STREAM = None   # torch.cuda.Stream; created lazily


def _get_marlin_stream():
    global _MARLIN_STREAM
    if _MARLIN_STREAM is None:
        with _MARLIN_LOCK:
            if _MARLIN_STREAM is None:
                import torch
                _MARLIN_STREAM = torch.cuda.Stream(device="cuda:0")
    return _MARLIN_STREAM


def setup_rt():
    """Load libcuda into global namespace, then load + init libcipher_rt."""
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    rt = ctypes.CDLL(str(ROOT / "libcipher_rt.so"),
                     mode=ctypes.RTLD_GLOBAL)
    rt.cipher_substitute_v2_init.restype = ctypes.c_int
    rt.cipher_substitute_v2_init()
    rt.cipher_weight_compress_init.restype = ctypes.c_int
    rt.cipher_weight_compress_init()

    rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_size_t]
    rt.cipher_weight_compress_observe.restype = ctypes.c_int
    rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p,
                                                    ctypes.c_int, ctypes.c_int]
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


class MarlinLinear:
    """Thread-safe Marlin INT4 substitution for nn.Linear.

    All Marlin GEMMs go through a shared CUDA stream to serialize the
    module-scope workspace access on the GPU. After enqueue, the
    calling thread's current stream waits on the Marlin stream so
    subsequent ops on the tenant stream see the GEMM output.
    """

    MAX_M = 8  # Marlin parallel-M ceiling; M>MAX_M falls back to cuBLAS

    def __init__(self, rt, orig, name):
        import torch
        import torch.nn as nn
        self.rt = rt
        self.name = name
        self.orig = orig
        self._orig_forward = type(orig).forward.__get__(orig, type(orig))
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias
        self._compressed = False
        if (self.out_features % 128 != 0) or (self.in_features % 128 != 0):
            return
        wt = orig.weight.detach().t().contiguous().to(torch.float16)
        self._wt = wt
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
        if ok != 1:
            return
        self._mB, self._mS = b.value, s.value
        self._mK, self._mN, self._mG = K.value, N.value, G.value
        self._compressed = True
        # NOTE: do NOT free orig.weight. Prefill batches at M=128 exceed
        # Marlin's MAX_M=8 and fall back to F.linear, which reads
        # orig.weight directly. Per CLAUDE.md, "M≥128 has no compatible
        # Marlin kernel (parallel-M branch isn't compiled)." Memory cost:
        # 14 GB orig + 14 GB transposed _wt (shared, single process).

    def __call__(self, x):
        import torch
        if not self._compressed:
            return self._orig_forward(x)
        orig_shape = x.shape
        if x.shape[-1] != self.in_features:
            return self._orig_forward(x)
        xf = x.reshape(-1, self.in_features).contiguous()
        if (not xf.is_contiguous() or xf.dtype != torch.float16
                or not xf.is_cuda):
            return self._orig_forward(x)
        M = xf.shape[0]
        if M > self.MAX_M:
            return self._orig_forward(x)
        out = torch.empty(M, self.out_features,
                          dtype=torch.float16, device=x.device)

        marlin_stream = _get_marlin_stream()
        cur_stream = torch.cuda.current_stream(device=x.device)
        with _MARLIN_LOCK:
            # Marlin stream waits for any pending work on cur_stream
            # that this kernel depends on (xf was produced by it).
            marlin_stream.wait_stream(cur_stream)
            rc = self.rt.cipher_weight_compress_marlin_gemm(
                xf.data_ptr(), self._mB, self._mS, out.data_ptr(),
                M, self.out_features, self.in_features, self._mG,
                ctypes.c_void_p(marlin_stream.cuda_stream))
            # cur_stream waits for the marlin output before its next op
            cur_stream.wait_stream(marlin_stream)
        if rc != 1:
            return self._orig_forward(x)
        out = out.reshape(orig_shape[:-1] + (self.out_features,))
        if self.bias is not None:
            out = out + self.bias
        return out


def patch_model_marlin(rt, model):
    """Replace every CUDA nn.Linear with a MarlinLinear (where dims allow)."""
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


def load_model_shared():
    """Load Mistral-7B fp16 on cuda:0, inference mode, grads off."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": "cuda:0"})
    model.requires_grad_(False)
    model.train(False)
    return model, tok


def make_prompt_ids(tok, prefill_len: int, device="cuda:0"):
    import torch
    ids_one = tok(PROMPT, return_tensors="pt").input_ids[0]
    n_tile = (prefill_len + len(ids_one) - 1) // len(ids_one)
    return ids_one.repeat(n_tile)[:prefill_len].unsqueeze(0).to(device)


def tenant_decode_worker(tenant_id, model, prompt_ids, prefill_len,
                          duration_s, collector, stream=None,
                          max_decode=200, warmup_tokens=5):
    """One tenant's continuous-decode worker thread."""
    import time, torch
    from transformers import StaticCache

    if stream is None:
        stream = torch.cuda.Stream(device="cuda:0")

    try:
        with torch.cuda.stream(stream):
            MAX_LEN = prefill_len + max_decode + 16
            cache = StaticCache(config=model.config,
                                 max_cache_len=MAX_LEN)
            with torch.no_grad():
                cp = torch.arange(prefill_len, device="cuda:0",
                                   dtype=torch.long)
                o = model(input_ids=prompt_ids, cache_position=cp,
                           past_key_values=cache, use_cache=True,
                           return_dict=True)
            input_ids = o.logits[:, -1:].argmax(-1).clone()
            cache_pos = torch.tensor([prefill_len], device="cuda:0",
                                       dtype=torch.long)
            stream.synchronize()

            # Warmup
            for _ in range(warmup_tokens):
                with torch.no_grad():
                    o = model(input_ids=input_ids,
                               cache_position=cache_pos,
                               past_key_values=cache, use_cache=True,
                               return_dict=True)
                input_ids.copy_(o.logits.argmax(-1))
                cache_pos += 1
            stream.synchronize()

            # Timed loop
            latencies_us = []
            n_tokens = 0
            t_start = time.perf_counter()
            while True:
                t_tok = time.perf_counter()
                with torch.no_grad():
                    o = model(input_ids=input_ids,
                               cache_position=cache_pos,
                               past_key_values=cache, use_cache=True,
                               return_dict=True)
                input_ids.copy_(o.logits.argmax(-1))
                cache_pos += 1
                stream.synchronize()
                latencies_us.append(
                    (time.perf_counter() - t_tok) * 1e6)
                n_tokens += 1
                if cache_pos.item() >= MAX_LEN - 4:
                    cache_pos = torch.tensor([prefill_len],
                                               device="cuda:0",
                                               dtype=torch.long)
                    with torch.no_grad():
                        o = model(input_ids=prompt_ids,
                                   cache_position=torch.arange(
                                       prefill_len, device="cuda:0",
                                       dtype=torch.long),
                                   past_key_values=cache,
                                   use_cache=True, return_dict=True)
                    input_ids.copy_(o.logits[:, -1:].argmax(-1))
                    cache_pos += 1
                    stream.synchronize()
                if time.perf_counter() - t_start >= duration_s:
                    break
            elapsed = time.perf_counter() - t_start
            collector.record_tokens(tenant_id, n_tokens, elapsed,
                                     latencies_us)
    except Exception as e:
        import traceback
        print(f"[tenant {tenant_id}] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        collector.record_tokens(tenant_id, 0, 0.0, [])


def tenant_burst_worker(tenant_id, model, prompt_ids, prefill_len,
                         duration_s, collector, stream=None,
                         burst_tokens=50, sleep_s=5.0):
    """One tenant's bursty agentic worker. Repeated (prefill + decode
    burst_tokens) then sleep sleep_s. Tracks compute vs idle time."""
    import time, torch
    from transformers import StaticCache

    if stream is None:
        stream = torch.cuda.Stream(device="cuda:0")

    try:
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < duration_s:
            with torch.cuda.stream(stream):
                MAX_LEN = prefill_len + burst_tokens + 4
                cache = StaticCache(config=model.config,
                                     max_cache_len=MAX_LEN)
                t_compute_start = time.perf_counter()
                with torch.no_grad():
                    cp = torch.arange(prefill_len, device="cuda:0",
                                       dtype=torch.long)
                    o = model(input_ids=prompt_ids, cache_position=cp,
                               past_key_values=cache, use_cache=True,
                               return_dict=True)
                input_ids = o.logits[:, -1:].argmax(-1).clone()
                cache_pos = torch.tensor([prefill_len],
                                           device="cuda:0",
                                           dtype=torch.long)
                for _ in range(burst_tokens):
                    with torch.no_grad():
                        o = model(input_ids=input_ids,
                                   cache_position=cache_pos,
                                   past_key_values=cache,
                                   use_cache=True, return_dict=True)
                    input_ids.copy_(o.logits.argmax(-1))
                    cache_pos += 1
                stream.synchronize()
                compute_s = time.perf_counter() - t_compute_start
            t_idle_start = time.perf_counter()
            time.sleep(sleep_s)
            idle_s = time.perf_counter() - t_idle_start
            collector.record_burst(tenant_id, burst_tokens,
                                    compute_s, idle_s)
    except Exception as e:
        import traceback
        print(f"[burst tenant {tenant_id}] EXCEPTION: {e}", flush=True)
        traceback.print_exc()


def run_step_continuous(run_id, model, tok, n_tenants, duration_s,
                         collector_factory, prefill_len=128,
                         max_decode=200):
    """Run one step of continuous-decode density sweep. Returns receipt."""
    import time, threading, torch
    mc = collector_factory(run_id)
    mc.configure(
        model_path=MODEL_PATH,
        dtype="fp16+marlin_int4",
        cipher_envs={k: os.environ.get(k, "")
                     for k in ["CIPHER_WEIGHT_COMPRESS",
                                "CIPHER_SUBSTITUTE_V2", "CIPHER_FAIRNESS",
                                "CIPHER_FUSION_KERNELS"]},
        test_kind="continuous",
        n_tenants=n_tenants,
        duration_s=duration_s,
        prefill_len=prefill_len,
        max_decode=max_decode,
    )
    prompt_ids = make_prompt_ids(tok, prefill_len)
    for i in range(n_tenants):
        mc.register_tenant(i)
    mc.start_sampler()
    threads = []
    for i in range(n_tenants):
        t = threading.Thread(
            target=tenant_decode_worker,
            args=(i, model, prompt_ids, prefill_len, duration_s, mc),
            kwargs=dict(max_decode=max_decode),
            name=f"tenant-{i}",
            daemon=False)
        threads.append(t)
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    mc.stop_sampler()
    receipt = mc.write_receipt(stop_reason="completed")
    receipt["wall_s"] = round(wall, 2)
    return receipt


def run_step_burst(run_id, model, tok, n_tenants, duration_s,
                    collector_factory, prefill_len=128,
                    burst_tokens=50, sleep_s=5.0):
    """Run one step of agentic-burst density sweep. Returns receipt."""
    import time, threading, torch
    mc = collector_factory(run_id)
    mc.configure(
        model_path=MODEL_PATH,
        dtype="fp16+marlin_int4",
        cipher_envs={k: os.environ.get(k, "")
                     for k in ["CIPHER_WEIGHT_COMPRESS",
                                "CIPHER_SUBSTITUTE_V2", "CIPHER_FAIRNESS",
                                "CIPHER_FUSION_KERNELS"]},
        test_kind="burst",
        n_tenants=n_tenants,
        duration_s=duration_s,
        prefill_len=prefill_len,
        burst_tokens=burst_tokens,
        sleep_s=sleep_s,
        max_decode=burst_tokens,
    )
    prompt_ids = make_prompt_ids(tok, prefill_len)
    for i in range(n_tenants):
        mc.register_tenant(i)
    mc.start_sampler()
    threads = []
    for i in range(n_tenants):
        t = threading.Thread(
            target=tenant_burst_worker,
            args=(i, model, prompt_ids, prefill_len, duration_s, mc),
            kwargs=dict(burst_tokens=burst_tokens, sleep_s=sleep_s),
            name=f"burst-tenant-{i}",
            daemon=False)
        threads.append(t)
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    mc.stop_sampler()
    receipt = mc.write_receipt(stop_reason="completed")
    receipt["wall_s"] = round(wall, 2)
    return receipt
