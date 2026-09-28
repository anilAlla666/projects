#!/usr/bin/env python3
"""Step 8 - Mixtral-8x22B 4-bit (bnb nf4 + double-quant) decode with / without
CIPHER (FP8 substitution + fused RMSNorm + SiLU.Mul + residual_add + RoPE).

Two modes (each in its own subprocess so LD_PRELOAD + env stay clean):
    baseline  no CIPHER
    full      LD_PRELOAD hook + rt + CIPHER_FP8_COMPUTE + fused kernels

Run:
    python3 step8_mixtral_8x22b.py                           # both modes
    python3 step8_mixtral_8x22b.py --mode=baseline --batches=1,8
    python3 step8_mixtral_8x22b.py --mode=full     --batches=1,8

Power is sampled from BOTH H100s and summed. tok/W = tps / sum(watts).
"""
import argparse, ctypes, gc, json, os, subprocess, sys, threading, time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
MIXTRAL_PATH = "/home/ubuntu/models/Mixtral-8x22B-v0.1"


def power_sampler(stop, samples, n_gpus):
    """Sample summed power across all GPUs every ~150 ms."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]
        while not stop.is_set():
            try:
                w = sum(pynvml.nvmlDeviceGetPowerUsage(h)/1000.0 for h in handles)
                samples.append(w)
            except Exception:
                pass
            time.sleep(0.15)
        pynvml.nvmlShutdown()
    except Exception:
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi",
                                    "--query-gpu=power.draw",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=1)
                vals = [float(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
                if vals:
                    samples.append(sum(vals))
            except Exception:
                pass
            time.sleep(0.15)


def run_mode(mode: str, batches, prefill_len=128,
             warmup_tokens=8, measure_seconds=10.0,
             out_path: str = ""):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn as nn
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    rt_handle = None
    fp8_stats_fn = lambda: {}

    # Sub-flags within "full" mode for ablation:
    #   STEP8_FP8=0     -> skip CIPHER_FP8 init (still preload hook)
    #   STEP8_FUSION=0  -> skip the fused-kernel monkey-patches
    use_fp8    = (mode == "full" and os.environ.get("STEP8_FP8",    "1") != "0")
    use_fusion = (mode == "full" and os.environ.get("STEP8_FUSION", "1") != "0")

    if mode == "full" and (use_fp8 or use_fusion):
        rt_handle = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                                mode=ctypes.RTLD_GLOBAL)
        if use_fp8:
            rt_handle.cipher_fp8_compute_init.restype = ctypes.c_int
            rt_handle.cipher_fp8_compute_init()

        # Fusion stats — independent of FP8 flag.
        if use_fusion:
            class _FusionStats(ctypes.Structure):
                _fields_ = [
                    ("enabled",            ctypes.c_int),
                    ("rmsnorm_calls",      ctypes.c_uint64),
                    ("silu_mul_calls",     ctypes.c_uint64),
                    ("residual_add_calls", ctypes.c_uint64),
                ]
            rt_handle.cipher_fusion_kernels_stats.argtypes = [
                ctypes.POINTER(_FusionStats)]
            rt_handle.cipher_fusion_kernels_stats.restype = ctypes.c_int

            def _fusion_stats():
                s = _FusionStats()
                rt_handle.cipher_fusion_kernels_stats(ctypes.byref(s))
                return dict(rmsnorm=s.rmsnorm_calls,
                            silu_mul=s.silu_mul_calls,
                            residual_add=s.residual_add_calls)
            globals()['_fusion_stats_fn'] = _fusion_stats
        else:
            globals()['_fusion_stats_fn'] = lambda: {}

        if use_fp8:
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

            def _stats():
                s = _Stats(); rt_handle.cipher_fp8_compute_stats(ctypes.byref(s))
                return dict(weights=s.weights_quantized, calls=s.matmul_calls,
                            passthroughs=s.passthroughs,
                            failures=s.correctness_failures)
            fp8_stats_fn = _stats

    print(f"[step8:{mode}] loading Mixtral-8x22B 4-bit...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MIXTRAL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MIXTRAL_PATH,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"[step8:{mode}] loaded in {time.perf_counter()-t_load:.1f}s; "
          f"GPU0={torch.cuda.memory_allocated(0)/1e9:.1f}GB "
          f"GPU1={torch.cuda.memory_allocated(1)/1e9:.1f}GB", flush=True)

    if use_fusion:
        from transformers.models.mixtral import modeling_mixtral as MM
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralRMSNorm, MixtralBlockSparseTop2MLP, MixtralDecoderLayer)

        # Save the originals BEFORE we patch.  When the rt's NVRTC cubin
        # cannot be launched on the current device (it was loaded into
        # cuda:0's primary context), we fall back to these — PyTorch's
        # actual native fp16 paths, not a slower fp32 reimplementation.
        _orig_rmsnorm_forward = MixtralRMSNorm.forward
        _orig_expert_forward  = MixtralBlockSparseTop2MLP.forward
        _orig_apply_rope      = MM.apply_rotary_pos_emb
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

        def _stream_ptr_for(dev):
            return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

        # Try-fused helpers return (out_tensor, ok) where ok=False means
        # "kernel failed, caller should fall back to PyTorch's native path".
        def _try_fused_rmsnorm(x, weight, eps):
            dev = x.device
            with torch.cuda.device(dev):
                flat = x.reshape(-1, x.shape[-1]).contiguous()
                out = torch.empty_like(x)
                out_flat = out.reshape(-1, x.shape[-1])
                rc = rt_handle.cipher_fused_rmsnorm(
                    flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                    flat.shape[0], flat.shape[-1], float(eps),
                    _stream_ptr_for(dev))
            return out, (rc == 1)

        def _try_fused_silu_mul(gate, up):
            dev = gate.device
            if not gate.is_contiguous(): gate = gate.contiguous()
            if not up.is_contiguous():   up   = up.contiguous()
            with torch.cuda.device(dev):
                out = torch.empty_like(gate)
                rc = rt_handle.cipher_fused_silu_mul(
                    gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                    gate.numel(), _stream_ptr_for(dev))
            return out, (rc == 1)

        def _try_fused_residual_add(x, residual):
            dev = x.device
            if not x.is_contiguous():        x = x.contiguous()
            if not residual.is_contiguous(): residual = residual.contiguous()
            with torch.cuda.device(dev):
                out = torch.empty_like(x)
                rc = rt_handle.cipher_fused_residual_add(
                    x.data_ptr(), residual.data_ptr(), out.data_ptr(),
                    x.numel(), _stream_ptr_for(dev))
            return out, (rc == 1)

        def _patched_rmsnorm_forward(self, x):
            out, ok = _try_fused_rmsnorm(x, self.weight, self.variance_epsilon)
            if ok:
                return out
            return _orig_rmsnorm_forward(self, x)
        MixtralRMSNorm.forward = _patched_rmsnorm_forward

        def _patched_expert_forward(self, x):
            gate = self.w1(x)
            up   = self.w3(x)
            mul, ok = _try_fused_silu_mul(gate, up)
            if not ok:
                mul = torch.nn.functional.silu(gate) * up
            return self.w2(mul)
        MixtralBlockSparseTop2MLP.forward = _patched_expert_forward

        # Accelerate's add_hook_to_module wraps each instance's `forward`
        # in `functools.partial(new_forward, module)` and stashes the
        # ORIGINAL (pre-hook) bound method at `module._old_forward`.  The
        # hook calls `_old_forward(*args, **kwargs)` — which is the bound
        # method captured *before* we patched the class.  So our class-
        # level patch is invisible to every layer that accelerate hooked
        # (every layer, when device_map='auto').
        # Fix: walk the model and rebind `_old_forward` per instance to
        # our patched function.  Bound via types.MethodType so `self` is
        # passed correctly.
        import types
        n_rms = n_exp = 0
        for m in model.modules():
            if isinstance(m, MixtralRMSNorm) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_rmsnorm_forward, m)
                n_rms += 1
            elif isinstance(m, MixtralBlockSparseTop2MLP) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_expert_forward, m)
                n_exp += 1
        print(f"[step8:{mode}] re-bound {n_rms} RMSNorm and {n_exp} expert MLP "
              f"_old_forward (accelerate hook bypass)", flush=True)

        def _residual_add(x, residual):
            out, ok = _try_fused_residual_add(x, residual)
            if ok:
                return out
            return x + residual  # native fp16 elementwise

        def _decoder_forward(self, hidden_states, position_embeddings,
                             attention_mask=None, position_ids=None,
                             past_key_values=None, cache_position=None,
                             **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = _residual_add(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            moe_out = self.block_sparse_moe(hidden_states)
            hidden_states = moe_out[0] if isinstance(moe_out, tuple) else moe_out
            hidden_states = _residual_add(hidden_states, residual)
            return hidden_states
        MixtralDecoderLayer.forward = _decoder_forward
        # Same accelerate-hook bypass for the decoder layer.
        n_dec = 0
        for m in model.modules():
            if isinstance(m, MixtralDecoderLayer) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_decoder_forward, m)
                n_dec += 1
        print(f"[step8:{mode}] re-bound {n_dec} decoder layer _old_forward",
              flush=True)

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
        _libcuda = ctypes.CDLL("libcuda.so.1")
        _cu_launch = _libcuda.cuLaunchKernel
        _cu_launch.argtypes = [ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        _cu_launch.restype = ctypes.c_int

        # Per-device RoPE CUfunction.  cipher_substitute_v2_compile loads
        # the cubin into whatever CUDA context is current at compile time.
        # When the model is split across GPUs, each device needs its own
        # CUmodule/CUfunction.  Lazily compile on first use per device.
        _rope_fns: dict = {}
        def _rope_fn_for(dev):
            idx = dev.index if dev.index is not None else 0
            fn = _rope_fns.get(idx)
            if fn is not None:
                return fn
            with torch.cuda.device(dev):
                # cudaFree(0) inside compile() ensures the primary context
                # is current — but we already pushed it via torch's context
                # manager.  A no-op tensor op forces the context up, too.
                _ = torch.empty(1, device=dev)
                cubin = rt_handle.cipher_substitute_v2_compile(
                    ROPE_SRC, b"cipher_fused_rope_qk")
                fn_ptr = rt_handle.cipher_substitute_v2_get_function(cubin)
            _rope_fns[idx] = fn_ptr
            print(f"[step8:{mode}] fused RoPE compiled for cuda:{idx} "
                  f"cubin={cubin} fn={hex(fn_ptr) if fn_ptr else 'NULL'}",
                  flush=True)
            return fn_ptr

        def _fused_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            dev = q.device
            fn = _rope_fn_for(dev)
            if not fn:
                return _orig_apply_rope(q, k, cos, sin, unsqueeze_dim)
            B, Hq, S, D = q.shape
            _, Hkv, _, _ = k.shape
            cos_c = cos.contiguous()
            sin_c = sin.contiguous()
            if cos_c.dim() == 2 or cos_c.shape[0] == 1:
                cos_bs = 0
            else:
                cos_bs = S * D
            # PyTorch ships q/k as transposed views; force contiguous.
            q_in = q.contiguous()
            k_in = k.contiguous()
            with torch.cuda.device(dev):
                q_out = torch.empty_like(q_in)
                k_out = torch.empty_like(k_in)
                args_list = [
                    ctypes.c_void_p(q_out.data_ptr()),
                    ctypes.c_void_p(k_out.data_ptr()),
                    ctypes.c_void_p(q_in.data_ptr()),
                    ctypes.c_void_p(k_in.data_ptr()),
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
                rc = _cu_launch(fn, grid_x, grid_y, 1, block_x, 1, 1,
                                shared_bytes, _stream_ptr_for(dev),
                                args_arr, None)
            if rc != 0:
                return _orig_apply_rope(q, k, cos, sin, unsqueeze_dim)
            return q_out, k_out

        MM.apply_rotary_pos_emb = _fused_apply_rotary_pos_emb
        print(f"[step8:{mode}] fused RMSNorm + SiLU.Mul + Residual + "
              f"RoPE (per-device) patched on Mixtral", flush=True)

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    N_TILE = (prefill_len + len(ids_one) - 1) // len(ids_one)
    big = (ids_one.repeat(N_TILE))[:prefill_len]
    prompt_ids_1 = big.unsqueeze(0).to("cuda:0")

    n_gpus = torch.cuda.device_count()

    def measure(B):
        prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
        attn_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            _ = model.generate(prompt_ids, attention_mask=attn_mask,
                               max_new_tokens=warmup_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               use_cache=True)
        torch.cuda.synchronize()

        EST_N = 16
        t_e = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(prompt_ids, attention_mask=attn_mask,
                               max_new_tokens=EST_N, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               use_cache=True)
        torch.cuda.synchronize()
        elapsed_est = time.perf_counter() - t_e
        tps_est = (B * EST_N) / max(elapsed_est, 1e-3)
        target_tokens = max(32, int(tps_est * measure_seconds / B))
        print(f"[step8:{mode}] B={B} estimate {tps_est:.1f} tps -> "
              f"target_tokens={target_tokens} per seq", flush=True)

        stats_pre = fp8_stats_fn()
        samples = []; stop = threading.Event()
        th = threading.Thread(target=power_sampler,
                              args=(stop, samples, n_gpus), daemon=True)
        th.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(prompt_ids, attention_mask=attn_mask,
                                     max_new_tokens=target_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id,
                                     use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)
        stats_post = fp8_stats_fn()

        new_tokens = out_ids.shape[1] - prompt_ids.shape[1]
        tps = (B * new_tokens) / elapsed

        if new_tokens <= 0:
            return dict(error=f"no new tokens: shape={out_ids.shape}")
        last_text = tok.decode(out_ids[0, -16:], skip_special_tokens=True)

        pw = samples[3:] or [0]
        mean_pw = sum(pw)/len(pw)
        tok_w = tps / mean_pw if mean_pw > 0 else 0

        try:
            fus = _fusion_stats_fn()
        except (NameError, AttributeError):
            fus = {}
        result = dict(mode=mode, batch=B, prefill=prefill_len, tps=tps,
                      draw_w_total=mean_pw, tok_w=tok_w, elapsed_s=elapsed,
                      new_tokens=new_tokens, last_text=last_text,
                      fp8_pre=stats_pre, fp8_post=stats_post,
                      fp8_delta=({k: stats_post[k]-stats_pre[k]
                                  for k in stats_pre} if stats_pre else {}),
                      fusion_calls=fus)
        del out_ids, prompt_ids, attn_mask
        gc.collect(); torch.cuda.empty_cache()
        return result

    print(f"\n=== {mode.upper()} (Mixtral-8x22B 4-bit, prefill={prefill_len}, "
          f"measure~{measure_seconds}s) ===", flush=True)
    print(f"{'B':>3} {'tps':>9} {'watts(sum)':>11} {'tok/W':>8} {'sec':>5} "
          f"{'fp8_calls':>10} {'tail_text':>40}")
    rows = []
    for B in batches:
        try:
            r = measure(B)
        except torch.cuda.OutOfMemoryError as e:
            print(f"   B={B}: OOM ({e})"); torch.cuda.empty_cache(); continue
        if "error" in r:
            print(f"   B={B}: {r['error']}"); continue
        rows.append(r)
        fp8c = r["fp8_delta"].get("calls", "-")
        tail = r["last_text"].replace("\n"," ")[:40]
        print(f"{r['batch']:>3} {r['tps']:>9.2f} {r['draw_w_total']:>11.1f} "
              f"{r['tok_w']:>8.4f} {r['elapsed_s']:>5.1f} {fp8c!s:>10} "
              f"{tail!r:>40}")
    if out_path:
        json.dump({"mode": mode, "rows": rows}, open(out_path, "w"), indent=2)
        print(f"[step8:{mode}] wrote {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "full"], default="")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--measure", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--modes", default="baseline,full")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    if args.mode:
        run_mode(args.mode, batches, prefill_len=args.prefill,
                 measure_seconds=args.measure, out_path=args.out)
        return

    LD = f"{ROOT}/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
    common_env = dict(os.environ)
    common_env["LD_PRELOAD"] = LD
    common_env["CIPHER_SUBSTITUTE_V2"] = "on"

    print("=" * 78)
    print(" Step 8 - Mixtral-8x22B 4-bit decode (CIPHER FP8 + fusion vs baseline)")
    print("=" * 78)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    outs = {}
    for m in modes:
        env = dict(common_env)
        if m == "baseline":
            env.pop("LD_PRELOAD", None)
            env.pop("CIPHER_FP8_COMPUTE", None)
            env.pop("CIPHER_SUBSTITUTE_V2", None)
        elif m == "full":
            env["CIPHER_FP8_COMPUTE"]    = "on"
            env["CIPHER_SUBSTITUTE_V2"]  = "on"
            env["CIPHER_FUSION_KERNELS"] = "on"
        out_p = os.path.join(ROOT, f"step8_{m}.json")
        outs[m] = out_p
        cmd = [sys.executable, __file__, f"--mode={m}",
               f"--batches={args.batches}", f"--prefill={args.prefill}",
               f"--measure={args.measure}", f"--out={out_p}"]
        print(f"\n-> {m} run:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, env=env)
        print(f"  rc={rc}", flush=True)

    print("\n" + "=" * 78)
    print(" Summary - Mixtral-8x22B 4-bit")
    print("=" * 78)
    rows = {}
    for m, p in outs.items():
        try:
            rows[m] = json.load(open(p))["rows"]
        except FileNotFoundError:
            print(f"  missing output: {p}"); rows[m] = []
    base = rows.get("baseline", [])
    full = rows.get("full", [])
    if not base or not full:
        print("  cannot build comparison table"); return
    print(f"{'B':>3}  {'base tps':>9}  {'full tps':>9}  {'xtps':>5}  "
          f"{'base W':>8}  {'full W':>8}  "
          f"{'base tok/W':>10}  {'full tok/W':>10}  {'xtok/W':>7}  "
          f"{'fp8_calls':>10}")
    for br in base:
        fr = next((x for x in full if x["batch"] == br["batch"]), None)
        if fr is None:
            print(f"{br['batch']:>3}  {br['tps']:>9.2f}  "
                  f"{'-':>9}  {'-':>5}  {br['draw_w_total']:>8.1f}  "
                  f"{'-':>8}  {br['tok_w']:>10.4f}  {'-':>10}  {'-':>7}  {'-':>10}")
            continue
        s_tps = fr["tps"] / br["tps"] if br["tps"] > 0 else 0
        s_tw  = fr["tok_w"] / br["tok_w"] if br["tok_w"] > 0 else 0
        calls = fr.get("fp8_delta", {}).get("calls", 0)
        print(f"{br['batch']:>3}  {br['tps']:>9.2f}  {fr['tps']:>9.2f}  "
              f"{s_tps:>5.2f}  {br['draw_w_total']:>8.1f}  "
              f"{fr['draw_w_total']:>8.1f}  "
              f"{br['tok_w']:>10.4f}  {fr['tok_w']:>10.4f}  "
              f"{s_tw:>7.2f}  {calls!s:>10}")


if __name__ == "__main__":
    main()
