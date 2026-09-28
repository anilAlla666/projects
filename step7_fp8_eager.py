#!/usr/bin/env python3
"""Step 7 — Mistral-7B fp16 EAGER decode with / without CIPHER FP8 substitute.

The harness self-execs into two cipher modes via env so each measurement
runs in a clean process (no module-loading order ambiguity for the rt
DSO):

    python3 step7_fp8_eager.py            # spawns both runs, prints summary
    python3 step7_fp8_eager.py --mode=baseline   # one mode, single batch
    python3 step7_fp8_eager.py --mode=fp8

Required env at launch (so the LD_PRELOAD hook + rt see them):
    CIPHER_SUBSTITUTE_V2=on           # NVRTC pipeline (always)
    CIPHER_FP8_COMPUTE=on             # only for the fp8 run; baseline omits

Required LD_PRELOAD layout:
    LD_PRELOAD="<cipher_hook.so> /usr/lib/x86_64-linux-gnu/libcuda.so"

The rt DSO is ctypes.CDLL'd here so its symbol table is visible to the
hook's dlsym(RTLD_DEFAULT, ...) lookups.
"""
import argparse, ctypes, gc, json, os, subprocess, sys, threading, time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))


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
        except Exception:
            pass
        time.sleep(0.15)


def run_mode(mode: str, batches, prefill_len=512,
             warmup_steps=10, measure_seconds=10.0,
             out_path: str = ""):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    # ── Optional Int4Linear monkey-patch for "stack" / "full" modes ───
    # M=1 -> INT4 GEMV (Python path, never reaches cublasGemmEx)
    # M>=2 -> standard nn.Linear (TransA=T -> FP8 hook fires)
    Int4Linear = None
    if mode in ("stack", "full"):
        # The Int4Linear class needs the rt DSO pre-loaded with weight-
        # compress symbols.  Pre-load and resolve.
        rt_for_int4 = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                                    mode=ctypes.RTLD_GLOBAL)
        rt_for_int4.cipher_substitute_v2_init.restype = ctypes.c_int
        rt_for_int4.cipher_weight_compress_init.restype = ctypes.c_int
        rt_for_int4.cipher_weight_compress_observe.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t]
        rt_for_int4.cipher_weight_compress_observe.restype = ctypes.c_int
        rt_for_int4.cipher_weight_compress_quantize.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        rt_for_int4.cipher_weight_compress_quantize.restype = ctypes.c_int
        rt_for_int4.cipher_weight_compress_lookup_T.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),  ctypes.POINTER(ctypes.c_int)]
        rt_for_int4.cipher_weight_compress_lookup_T.restype = ctypes.c_int
        rt_for_int4.cipher_weight_compress_int4_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        rt_for_int4.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
        rt_for_int4.cipher_substitute_v2_init()
        rt_for_int4.cipher_weight_compress_init()

        class _Int4Linear(nn.Module):
            """M=1 -> INT4 GEMV.  M>=2 -> F.linear (which is x @ W.T,
            TransA=T in cuBLAS, so the FP8 hook substitutes there)."""
            def __init__(self, orig: nn.Linear):
                super().__init__()
                self.in_features  = orig.in_features
                self.out_features = orig.out_features
                self.weight       = orig.weight        # original W (out, in)
                self.bias         = orig.bias
                # Quantize transposed copy for INT4 GEMV.
                wt = orig.weight.detach().t().contiguous()
                self.wt_fp16 = nn.Parameter(wt, requires_grad=False)
                for _ in range(1001):
                    rt_for_int4.cipher_weight_compress_observe(
                        self.wt_fp16.data_ptr(), wt.numel() * 2)
                rc = rt_for_int4.cipher_weight_compress_quantize(
                    self.wt_fp16.data_ptr(), self.in_features, self.out_features)
                self._compressed = (rc == 1)
                if self._compressed:
                    bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
                    br = ctypes.c_int(); bc = ctypes.c_int()
                    ok = rt_for_int4.cipher_weight_compress_lookup_T(
                        self.wt_fp16.data_ptr(),
                        ctypes.byref(bT), ctypes.byref(bs),
                        ctypes.byref(br), ctypes.byref(bc))
                    self._compressed = (ok == 1)
                    if self._compressed:
                        self._bT_ptr = bT.value
                        self._bs_ptr = bs.value

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # M=1 path: INT4 GEMV via ctypes.
                if (self._compressed and x.dim() >= 2
                        and x.shape[-2] == 1
                        and x.is_cuda and x.dtype == torch.float16):
                    xb = x.reshape(-1, self.in_features)
                    out = torch.empty(1, self.out_features,
                                       dtype=torch.float16, device=x.device)
                    # Stream MUST be torch's current stream (capture-aware)
                    # — passing None lands the launch on the legacy default
                    # stream and the kernel never makes it into a captured
                    # graph.
                    s_ptr = ctypes.c_void_p(
                        torch.cuda.current_stream().cuda_stream)
                    rc = rt_for_int4.cipher_weight_compress_int4_gemv(
                        xb.data_ptr(), self._bT_ptr, self._bs_ptr,
                        out.data_ptr(),
                        self.out_features, self.in_features, s_ptr)
                    if rc == 1:
                        out = out.reshape(x.shape[:-1] + (self.out_features,))
                        if self.bias is not None:
                            out = out + self.bias
                        return out
                # M>=2 path: standard nn.Linear semantics.  This goes
                # through cublasGemmEx with TransA=T, hits the FP8 hook.
                return torch.nn.functional.linear(x, self.weight, self.bias)
        Int4Linear = _Int4Linear

    use_fusion = (mode in ("fp8", "stack", "full") and
                  os.environ.get("STEP7_FUSION", "1") != "0")
    rt_handle = None
    if mode in ("fp8", "stack", "full"):
        # Load rt so the hook's dlsym for cipher_fp8_compute_* can resolve.
        rt_handle = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                                 mode=ctypes.RTLD_GLOBAL)
        rt_handle.cipher_fp8_compute_init.restype = ctypes.c_int
        rt_handle.cipher_fp8_compute_init()

        class _Stats(ctypes.Structure):
            _fields_ = [
                ("enabled",              ctypes.c_int),
                ("weights_quantized",    ctypes.c_uint64),
                ("matmul_calls",         ctypes.c_uint64),
                ("passthroughs",         ctypes.c_uint64),
                ("correctness_failures", ctypes.c_uint64),
                ("bytes_fp16",           ctypes.c_size_t),
                ("bytes_fp8",            ctypes.c_size_t),
            ]
        rt_handle.cipher_fp8_compute_stats.argtypes = [ctypes.POINTER(_Stats)]
        rt_handle.cipher_fp8_compute_stats.restype = ctypes.c_int

        def get_fp8_stats():
            s = _Stats(); rt_handle.cipher_fp8_compute_stats(ctypes.byref(s))
            return dict(weights=s.weights_quantized, calls=s.matmul_calls,
                         passthroughs=s.passthroughs)
    else:
        def get_fp8_stats(): return {}

    MODEL = "mistralai/Mistral-7B-v0.1"
    print(f"[step7:{mode}] loading {MODEL}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                  device_map="cuda")

    if use_fusion:
        # Wire fused RMSNorm + SiLU·Mul + Residual Add + RoPE.  Each is an
        # NVRTC-compiled kernel collapsing 1+ HBM round-trips into one
        # kernel launch.  In graph-capture mode the launch-overhead win
        # is amortized but the HBM-bandwidth win remains.
        from transformers.models.mistral import modeling_mistral as MM
        from transformers.models.mistral.modeling_mistral import (
            MistralRMSNorm, MistralMLP, MistralDecoderLayer)
        rt_handle.cipher_fused_rmsnorm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        rt_handle.cipher_fused_rmsnorm.restype = ctypes.c_int
        rt_handle.cipher_fused_silu_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p]
        rt_handle.cipher_fused_silu_mul.restype = ctypes.c_int
        rt_handle.cipher_fused_residual_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p]
        rt_handle.cipher_fused_residual_add.restype = ctypes.c_int
        try:
            rt_handle.cipher_fusion_kernels_init.restype = ctypes.c_int
            rt_handle.cipher_fusion_kernels_init()
        except AttributeError:
            pass

        def _stream_ptr():
            return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def _fused_rmsnorm(x, weight, eps):
            out = torch.empty_like(x)
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            out_flat = out.reshape(-1, x.shape[-1])
            rt_handle.cipher_fused_rmsnorm(
                flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                flat.shape[0], flat.shape[-1], float(eps), _stream_ptr())
            return out

        def _fused_silu_mul(gate, up):
            if not gate.is_contiguous(): gate = gate.contiguous()
            if not up.is_contiguous():   up   = up.contiguous()
            out = torch.empty_like(gate)
            rt_handle.cipher_fused_silu_mul(
                gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                gate.numel(), _stream_ptr())
            return out

        def _fused_residual_add(x, residual):
            if not x.is_contiguous():        x = x.contiguous()
            if not residual.is_contiguous(): residual = residual.contiguous()
            out = torch.empty_like(x)
            rt_handle.cipher_fused_residual_add(
                x.data_ptr(), residual.data_ptr(), out.data_ptr(),
                x.numel(), _stream_ptr())
            return out

        MistralRMSNorm.forward = lambda self, x: _fused_rmsnorm(
            x, self.weight, self.variance_epsilon)
        MistralMLP.forward = lambda self, x: self.down_proj(
            _fused_silu_mul(self.gate_proj(x), self.up_proj(x)))

        # Replace the two `residual + hidden_states` adds inside
        # MistralDecoderLayer.forward with our fused kernel.  Mirrors the
        # default forward exactly otherwise.
        def _decoder_forward(self,
                              hidden_states,
                              attention_mask=None,
                              position_ids=None,
                              past_key_values=None,
                              use_cache=False,
                              position_embeddings=None,
                              **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = _fused_residual_add(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _fused_residual_add(hidden_states, residual)
            return hidden_states
        MistralDecoderLayer.forward = _decoder_forward

        # ---- Fused RoPE on Q+K (kernel from rope_fused_kernel.py) ----
        ROPE_SRC = b"""
#include <cuda_fp16.h>
extern "C" __global__ void cipher_fused_rope_qk(
    __half* __restrict__ Q_out, __half* __restrict__ K_out,
    const __half* __restrict__ Q_in, const __half* __restrict__ K_in,
    const __half* __restrict__ cos, const __half* __restrict__ sin,
    int B, int Hq, int Hkv, int S, int D, int cos_batch_stride)
{
    int half_D = D / 2;
    int d = threadIdx.x;
    if (d >= D) return;
    int b = blockIdx.y;
    int row_idx = blockIdx.x;
    int total_q_rows = Hq * S;
    bool is_q = row_idx < total_q_rows;
    int local = is_q ? row_idx : row_idx - total_q_rows;
    int H = is_q ? Hq : Hkv;
    int h = local / S;
    int s = local % S;
    const __half* x_in;
    __half* x_out;
    if (is_q) {
        x_in  = Q_in  + ((b * Hq  + h) * S + s) * D;
        x_out = Q_out + ((b * Hq  + h) * S + s) * D;
    } else {
        x_in  = K_in  + ((b * Hkv + h) * S + s) * D;
        x_out = K_out + ((b * Hkv + h) * S + s) * D;
    }
    int cos_offset = b * cos_batch_stride + s * D + d;
    float c  = __half2float(cos[cos_offset]);
    float si = __half2float(sin[cos_offset]);
    extern __shared__ float shm[];
    shm[d] = __half2float(x_in[d]);
    __syncthreads();
    float x_self = shm[d];
    float x_pair, sign;
    if (d < half_D) { x_pair = shm[d + half_D]; sign = -1.0f; }
    else            { x_pair = shm[d - half_D]; sign =  1.0f; }
    x_out[d] = __float2half(x_self * c + sign * x_pair * si);
}
"""
        rt_handle.cipher_substitute_v2_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p]
        rt_handle.cipher_substitute_v2_compile.restype = ctypes.c_ulong
        rt_handle.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
        rt_handle.cipher_substitute_v2_get_function.restype = ctypes.c_void_p
        _rope_cubin = rt_handle.cipher_substitute_v2_compile(
            ROPE_SRC, b"cipher_fused_rope_qk")
        _rope_fn = rt_handle.cipher_substitute_v2_get_function(_rope_cubin)
        _libcuda = ctypes.CDLL("libcuda.so.1")
        _cu_launch = _libcuda.cuLaunchKernel
        _cu_launch.argtypes = [ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        _cu_launch.restype = ctypes.c_int
        if _rope_fn:
            print(f"[step7:{mode}] fused RoPE compiled cubin={_rope_cubin}",
                  flush=True)

        def _fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            # cos / sin enter as [B, S, D] (or broadcastable) BEFORE the
            # unsqueeze_dim insertion the standard apply_rotary_pos_emb
            # would do.  Our kernel handles the broadcast itself.
            B, Hq, S, D = q.shape
            _, Hkv, _, _ = k.shape
            cos_c = cos.contiguous()
            sin_c = sin.contiguous()
            if cos_c.dim() == 2:
                cos_bs = 0
            elif cos_c.shape[0] == 1:
                cos_bs = 0
            else:
                cos_bs = S * D
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            args_list = [
                ctypes.c_void_p(q_out.data_ptr()),
                ctypes.c_void_p(k_out.data_ptr()),
                ctypes.c_void_p(q.data_ptr()),
                ctypes.c_void_p(k.data_ptr()),
                ctypes.c_void_p(cos_c.data_ptr()),
                ctypes.c_void_p(sin_c.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(Hq), ctypes.c_int(Hkv),
                ctypes.c_int(S), ctypes.c_int(D), ctypes.c_int(cos_bs),
            ]
            args_arr = (ctypes.c_void_p * len(args_list))(*[
                ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
                for a in args_list])
            grid_x = (Hq + Hkv) * S
            grid_y = B
            block_x = D
            shared_bytes = D * 4
            rc = _cu_launch(_rope_fn, grid_x, grid_y, 1, block_x, 1, 1,
                             shared_bytes, _stream_ptr(), args_arr, None)
            if rc != 0:
                # fall back to default eager implementation
                cos_u = cos.unsqueeze(unsqueeze_dim)
                sin_u = sin.unsqueeze(unsqueeze_dim)
                def rotate_half(x):
                    x1 = x[..., :x.shape[-1]//2]
                    x2 = x[..., x.shape[-1]//2:]
                    return torch.cat((-x2, x1), dim=-1)
                return ((q * cos_u) + (rotate_half(q) * sin_u),
                        (k * cos_u) + (rotate_half(k) * sin_u))
            return q_out, k_out

        if _rope_fn:
            MM.apply_rotary_pos_emb = _fused_apply_rotary_pos_emb
        print(f"[step7:{mode}] fused RMSNorm + SiLU·Mul + Residual + "
              f"RoPE patched", flush=True)

    if mode in ("stack", "full") and Int4Linear is not None:
        n_replaced = 0
        for layer in model.model.layers:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                old = getattr(layer.self_attn, attr)
                if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                    setattr(layer.self_attn, attr, Int4Linear(old).to("cuda"))
                    n_replaced += 1
            for attr in ("gate_proj", "up_proj", "down_proj"):
                old = getattr(layer.mlp, attr)
                if old.in_features % 128 == 0 and old.out_features % 8 == 0:
                    setattr(layer.mlp, attr, Int4Linear(old).to("cuda"))
                    n_replaced += 1
        print(f"[step7:{mode}] replaced {n_replaced} nn.Linear with Int4Linear",
              flush=True)

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    N_TILE = (prefill_len + len(ids_one) - 1) // len(ids_one)
    big = (ids_one.repeat(N_TILE))[:prefill_len]
    prompt_ids_1 = big.unsqueeze(0).to("cuda")

    def measure(B):
        MAX_LEN = prefill_len + 320
        prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
        cache = StaticCache(config=model.config, max_batch_size=B,
                             max_cache_len=MAX_LEN, device="cuda",
                             dtype=torch.float16)
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda", dtype=torch.long)
            o = model(input_ids=prompt_ids, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        next_token = o.logits[:, -1:].argmax(-1)
        input_ids = next_token.detach().clone()
        cache_pos = torch.tensor([prefill_len], device="cuda", dtype=torch.long)

        out_logits = torch.full((B, 1, model.config.vocab_size),
                                  float("nan"), dtype=torch.float16, device="cuda")
        for _ in range(warmup_steps):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))   # in-place copy keeps
                                                     # tensor pointer stable
                                                     # so a subsequent graph
                                                     # capture can bind to it
            cache_pos += 1
        torch.cuda.synchronize()
        if torch.isnan(out_logits).any().item():
            return dict(error="NaN after warmup")

        # ── Graph-capture branch (mode == "full") ───────────────────────
        # All state (FP8 weight buffers, INT4 quant cache, fused-kernel
        # cubins) was warmed by the eager loop above, so the capture pass
        # itself does no cudaMalloc.  Three side-stream warmup steps then
        # one captured decode step replayed for the timing window.
        graph = None
        if mode == "full":
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    with torch.no_grad():
                        o = model(input_ids=input_ids,
                                  cache_position=cache_pos,
                                  past_key_values=cache,
                                  use_cache=True, return_dict=True)
                    out_logits.copy_(o.logits)
                    input_ids.copy_(out_logits.argmax(-1))
                    cache_pos += 1
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    with torch.no_grad():
                        o = model(input_ids=input_ids,
                                  cache_position=cache_pos,
                                  past_key_values=cache,
                                  use_cache=True, return_dict=True)
                    out_logits.copy_(o.logits)
                    input_ids.copy_(out_logits.argmax(-1))
                torch.cuda.synchronize()
                print(f"[step7:full] graph captured for B={B}", flush=True)
            except Exception as e:
                print(f"[step7:full] graph capture failed: {type(e).__name__}: {e}",
                      flush=True)
                graph = None

        out_logits.fill_(float("nan"))
        stats_pre = get_fp8_stats()
        samples = []; stop = threading.Event()
        th = threading.Thread(target=power_sampler, args=(stop, samples),
                              daemon=True); th.start()
        t0 = time.perf_counter(); n_steps = 0
        if graph is not None:
            # Replay path: cache_pos was already advanced inside the
            # captured step; we increment again per replay.
            while time.perf_counter() - t0 < measure_seconds:
                graph.replay()
                cache_pos += 1
                n_steps += 1
                if cache_pos.item() >= MAX_LEN - 4: break
        else:
            while time.perf_counter() - t0 < measure_seconds:
                with torch.no_grad():
                    o = model(input_ids=input_ids, cache_position=cache_pos,
                              past_key_values=cache, use_cache=True,
                              return_dict=True)
                out_logits.copy_(o.logits)
                input_ids.copy_(out_logits.argmax(-1))
                cache_pos += 1
                n_steps += 1
                if cache_pos.item() >= MAX_LEN - 4: break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)
        stats_post = get_fp8_stats()
        if torch.isnan(out_logits).any().item():
            return dict(error="NaN after timed loop")

        pw  = [s[0] for s in samples[2:]] or [0]
        clk = [s[1] for s in samples[2:]] or [0]
        mean_pw = sum(pw)/len(pw)
        tps = (B * n_steps) / elapsed
        tok_w = tps / mean_pw if mean_pw > 0 else 0
        last_token = int(out_logits[0, 0].argmax().item())
        out = dict(mode=mode, batch=B, prefill=prefill_len, tps=tps,
                   draw_w=mean_pw, tok_w=tok_w, elapsed_s=elapsed,
                   n_steps=n_steps, last_token=last_token,
                   last_text=tok.decode([last_token]),
                   fp8_pre=stats_pre, fp8_post=stats_post,
                   fp8_delta=({k: stats_post[k]-stats_pre[k]
                                for k in stats_pre} if stats_pre else {}))
        del cache, cache_pos, out_logits, input_ids, prompt_ids
        gc.collect(); torch.cuda.empty_cache()
        return out

    print(f"\n=== {mode.upper()} (eager, prefill={prefill_len}, "
          f"measure={measure_seconds}s) ===", flush=True)
    print(f"{'B':>3} {'tps':>9} {'watts':>7} {'tok/W':>8} {'sec':>5} "
          f"{'fp8_calls':>10} {'last':>14}")
    rows = []
    for B in batches:
        try:
            r = measure(B)
        except torch.cuda.OutOfMemoryError:
            print(f"   B={B}: OOM"); torch.cuda.empty_cache(); continue
        if "error" in r:
            print(f"   B={B}: {r['error']}"); continue
        rows.append(r)
        fp8c = r["fp8_delta"].get("calls", "-")
        print(f"{r['batch']:>3} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
              f"{r['tok_w']:>8.4f} {r['elapsed_s']:>5.1f} {fp8c!s:>10} "
              f"{r['last_text']!r:>14}")
    if out_path:
        json.dump({"mode": mode, "rows": rows}, open(out_path, "w"), indent=2)
        print(f"[step7:{mode}] wrote {out_path}")
    return rows


def lock_clock(mhz: int) -> bool:
    if mhz <= 0: return False
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-lgc", str(mhz)],
                       capture_output=True, text=True)
    return r.returncode == 0


def restore_clock():
    subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"],
                   capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                     choices=["baseline", "fp8", "stack", "full"], default="")
    ap.add_argument("--batches", default="1")
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--measure", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--lock-clock", type=int, default=0,
                     help="lock GPU graphics clock to N MHz (sudo nvidia-smi -lgc)")
    ap.add_argument("--modes", default="baseline,fp8",
                     help="comma list of modes to run when --mode is unset")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    if args.mode:
        run_mode(args.mode, batches, prefill_len=args.prefill,
                  measure_seconds=args.measure, out_path=args.out)
        return

    if args.lock_clock:
        if lock_clock(args.lock_clock):
            print(f"[step7] locked GPU clock to {args.lock_clock} MHz")
            import atexit; atexit.register(restore_clock)
        else:
            print(f"[step7] WARNING: failed to lock clock to {args.lock_clock}")

    # Master: spawn each requested mode in its own process.
    LD = (f"{ROOT}/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so")
    common_env = dict(os.environ)
    common_env["LD_PRELOAD"] = LD
    common_env["CIPHER_SUBSTITUTE_V2"] = "on"

    print("=" * 78)
    print(" Step 7/8 — Mistral-7B fp16 eager decode")
    if args.lock_clock:
        print(f"  GPU clock locked at {args.lock_clock} MHz")
    print("=" * 78)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    outs: dict = {}
    for m in modes:
        env = dict(common_env)
        if m == "baseline":
            env.pop("CIPHER_FP8_COMPUTE", None)
        elif m == "fp8":
            env["CIPHER_FP8_COMPUTE"] = "on"
        elif m == "stack":
            env["CIPHER_FP8_COMPUTE"]      = "on"
            env["CIPHER_WEIGHT_COMPRESS"]  = "on"
        elif m == "full":
            env["CIPHER_FP8_COMPUTE"]      = "on"
            env["CIPHER_WEIGHT_COMPRESS"]  = "on"
            env["CIPHER_FUSION_KERNELS"]   = "on"
            env["CIPHER_SUBSTITUTE_V2"]    = "on"
        out_p = os.path.join(ROOT, f"step7_{m}.json")
        outs[m] = out_p
        cmd = [sys.executable, __file__, f"--mode={m}",
               f"--batches={args.batches}", f"--prefill={args.prefill}",
               f"--measure={args.measure}", f"--out={out_p}"]
        print(f"\n→ {m} run:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, env=env)
        print(f"  rc={rc}", flush=True)

    print("\n" + "=" * 78)
    print(" Summary")
    print("=" * 78)
    rows: dict = {}
    for m, p in outs.items():
        try:
            rows[m] = json.load(open(p))["rows"]
        except FileNotFoundError:
            print(f"  missing output: {p}")
            rows[m] = []

    # Print a row per batch with all modes.
    base = rows.get("baseline", [])
    if not base:
        return
    others = [m for m in modes if m != "baseline"]
    hdr = f"{'B':>3}  {'baseline tps':>13}  {'baseline tok/W':>14}  "
    for m in others:
        hdr += f"{m+' tps':>10}  {'×tps':>5}  {m+' tok/W':>12}  {'×tok/W':>7}  fp8_calls  "
    print(hdr)
    for br in base:
        line = f"{br['batch']:>3}  {br['tps']:>13.2f}  {br['tok_w']:>14.4f}  "
        for m in others:
            mr = next((x for x in rows[m] if x["batch"] == br["batch"]), None)
            if mr is None:
                line += f"{'-':>10}  {'-':>5}  {'-':>12}  {'-':>7}  {'-':>9}  "
                continue
            s_tps = mr["tps"] / br["tps"] if br["tps"] > 0 else 0
            s_tw  = mr["tok_w"] / br["tok_w"] if br["tok_w"] > 0 else 0
            calls = mr.get("fp8_delta", {}).get("calls", 0)
            line += (f"{mr['tps']:>10.2f}  {s_tps:>5.2f}  "
                      f"{mr['tok_w']:>12.4f}  {s_tw:>7.2f}  {calls!s:>9}  ")
        print(line)


if __name__ == "__main__":
    main()
