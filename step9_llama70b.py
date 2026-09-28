#!/usr/bin/env python3
"""Step 9 - Llama-3.1-70B FP16 decode with / without CIPHER full stack
(FP8 substitution + fused RMSNorm/SiLU.Mul/residual_add/RoPE + clock lock).

The model lives in fp16 across both H100s (~70.5 GB each) via accelerate
device_map='auto'.  No bitsandbytes — every cublasGemmEx call has stable
fp16 weights, so FP8 substitution can fire on every linear.

Two modes (each in its own subprocess):
    baseline  default clocks, no CIPHER
    full      LD_PRELOAD hook + rt + CIPHER_FP8 + fused kernels;
              caller is responsible for `nvidia-smi -lgc 1200` BEFORE
              launching this mode and `-rgc` after.
"""
import argparse, ctypes, gc, json, os, subprocess, sys, threading, time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
# H100 FP16 peak per GPU.
H100_FP16_TFLOPS = 989.4

# Defaults are 70B; override via --model and --params.
DEFAULT_MODEL_PATH = "/home/ubuntu/models/Llama-3.1-70B"
DEFAULT_PARAMS     = 70.55e9


def power_sampler(stop, samples, n_gpus):
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
                if vals: samples.append(sum(vals))
            except Exception:
                pass
            time.sleep(0.15)


def run_mode(mode: str, batches, prefill_len=128,
             warmup_tokens=8, measure_seconds=10.0,
             out_path: str = "",
             model_path: str = DEFAULT_MODEL_PATH,
             params: float = DEFAULT_PARAMS,
             device_map: str = "auto"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Limit cuBLAS workspace.  CIPHER's hook adds ~10 GB of driver-side
    # workspace pressure on Llama-70B fp16; capping cuBLAS to 8 MB keeps
    # us under the 79 GB per-GPU cap.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rt_handle = None
    fp8_stats_fn   = lambda: {}
    fusion_stats_fn = lambda: {}

    use_fp8    = (mode == "full" and os.environ.get("STEP9_FP8",    "1") != "0")
    use_fusion = (mode == "full" and os.environ.get("STEP9_FUSION", "1") != "0")

    if mode == "full" and (use_fp8 or use_fusion):
        rt_handle = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                                mode=ctypes.RTLD_GLOBAL)
        if use_fp8:
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

            def _stats():
                s = _Stats(); rt_handle.cipher_fp8_compute_stats(ctypes.byref(s))
                return dict(weights=s.weights_quantized, calls=s.matmul_calls,
                            passthroughs=s.passthroughs,
                            failures=s.correctness_failures)
            fp8_stats_fn = _stats

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

            def _fstats():
                s = _FusionStats()
                rt_handle.cipher_fusion_kernels_stats(ctypes.byref(s))
                return dict(rmsnorm=s.rmsnorm_calls,
                            silu_mul=s.silu_mul_calls,
                            residual_add=s.residual_add_calls)
            fusion_stats_fn = _fstats

    flops_per_token = 2 * params
    print(f"[step9:{mode}] loading {model_path} fp16 "
          f"(params={params/1e9:.2f}B, device_map={device_map})...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t_load = time.perf_counter()
    if device_map in ("cuda:0", "0"):
        load_dm = {"": 0}
    else:
        load_dm = device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=load_dm,
    )
    print(f"[step9:{mode}] loaded in {time.perf_counter()-t_load:.1f}s; "
          f"GPU0={torch.cuda.memory_allocated(0)/1e9:.1f}GB "
          f"GPU1={torch.cuda.memory_allocated(1)/1e9:.1f}GB", flush=True)

    if use_fusion:
        from transformers.models.llama import modeling_llama as LM
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer)

        _orig_rmsnorm_forward = LlamaRMSNorm.forward
        _orig_mlp_forward     = LlamaMLP.forward
        _orig_apply_rope      = LM.apply_rotary_pos_emb

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
            if ok: return out
            return _orig_rmsnorm_forward(self, x)
        LlamaRMSNorm.forward = _patched_rmsnorm_forward

        def _patched_mlp_forward(self, x):
            gate = self.gate_proj(x)
            up   = self.up_proj(x)
            mul, ok = _try_fused_silu_mul(gate, up)
            if not ok:
                mul = torch.nn.functional.silu(gate) * up
            return self.down_proj(mul)
        LlamaMLP.forward = _patched_mlp_forward

        def _residual_add(x, residual):
            out, ok = _try_fused_residual_add(x, residual)
            if ok: return out
            return x + residual

        def _decoder_forward(self, hidden_states, attention_mask=None,
                             position_ids=None, past_key_values=None,
                             use_cache=False, cache_position=None,
                             position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = _residual_add(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _residual_add(hidden_states, residual)
            return hidden_states
        LlamaDecoderLayer.forward = _decoder_forward

        # Accelerate hook bypass — rebind _old_forward per-instance so the
        # patched class methods actually run (accelerate captured the
        # originals at hook-install time).
        import types
        n_rms = n_mlp = n_dec = 0
        for m in model.modules():
            if isinstance(m, LlamaRMSNorm) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_rmsnorm_forward, m)
                n_rms += 1
            elif isinstance(m, LlamaMLP) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_mlp_forward, m)
                n_mlp += 1
            elif isinstance(m, LlamaDecoderLayer) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_decoder_forward, m)
                n_dec += 1
        print(f"[step9:{mode}] re-bound {n_rms} RMSNorm + {n_mlp} MLP + "
              f"{n_dec} decoder layers (accelerate hook bypass)", flush=True)

        # Per-device RoPE NVRTC compile.
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

        _rope_fns = {}
        def _rope_fn_for(dev):
            idx = dev.index if dev.index is not None else 0
            fn = _rope_fns.get(idx)
            if fn is not None: return fn
            with torch.cuda.device(dev):
                _ = torch.empty(1, device=dev)
                cubin = rt_handle.cipher_substitute_v2_compile(
                    ROPE_SRC, b"cipher_fused_rope_qk")
                fn_ptr = rt_handle.cipher_substitute_v2_get_function(cubin)
            _rope_fns[idx] = fn_ptr
            print(f"[step9:{mode}] RoPE compiled for cuda:{idx} cubin={cubin}",
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
            # PyTorch ships q/k as transposed views of [B, S, H, D]
            # underlying (q_proj(...).view(...).transpose(1,2)).  Our
            # kernel reads as if [B, H, S, D] is contiguous; force it.
            q_in = q.contiguous()
            k_in = k.contiguous()
            cos_bs = 0 if (cos_c.dim() == 2 or cos_c.shape[0] == 1) else S * D
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
                rc = _cu_launch(fn, grid_x, grid_y, 1, block_x, 1, 1,
                                D * 4, _stream_ptr_for(dev),
                                args_arr, None)
            if rc != 0:
                return _orig_apply_rope(q, k, cos, sin, unsqueeze_dim)
            return q_out, k_out

        LM.apply_rotary_pos_emb = _fused_apply_rotary_pos_emb
        print(f"[step9:{mode}] fused RMSNorm + SiLU.Mul + Residual + "
              f"RoPE patched on Llama", flush=True)

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

        for i in range(torch.cuda.device_count()):
            print(f"[step9 mem B={B} pre-measure] GPU{i} alloc="
                  f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB "
                  f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB",
                  flush=True)
        # Llama-70B fp16 leaves ~8 GB headroom per GPU, and CIPHER's hook
        # adds ~10 GB driver-side workspace pressure on cuda:0 that PyTorch
        # doesn't account for.  Doing warmup + measure as separate
        # model.generate() calls reliably OOMs on the second call (cuBLAS
        # workspace from generate #1 won't release).  So we collapse both
        # into a single generate: the first WARMUP_TOKENS tokens are
        # untimed, the rest are the measurement.
        # All batches generate ≥50 tokens so we can verify output
        # coherence past any short EOS-prefix region.
        target_tokens_table = {1: 200, 8: 96, 32: 64, 64: 56}
        target_tokens = target_tokens_table.get(B, 56)
        print(f"[step9:{mode}] B={B} target_tokens={target_tokens} "
              f"(single-shot warmup+measure)", flush=True)

        # Untimed warmup via 1 short call (small enough that cuBLAS
        # workspace allocations are bounded — ≤2 tokens, no big shape change).
        with torch.no_grad():
            _ = model.generate(prompt_ids, attention_mask=attn_mask,
                               max_new_tokens=2, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()

        stats_pre = fp8_stats_fn()
        fus_pre   = fusion_stats_fn()
        samples = []; stop = threading.Event()
        th = threading.Thread(target=power_sampler,
                              args=(stop, samples, n_gpus), daemon=True)
        th.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out_ids = model.generate(prompt_ids, attention_mask=attn_mask,
                                         max_new_tokens=target_tokens,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id,
                                         use_cache=True)
            torch.cuda.synchronize()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            stop.set(); th.join(timeout=2)
            for i in range(torch.cuda.device_count()):
                print(f"[step9 OOM @ measure B={B}] GPU{i} alloc="
                      f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB "
                      f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB",
                      flush=True)
            print(f"[step9 OOM] {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            return dict(error=f"OOM B={B}", batch=B)
        elapsed = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)
        stats_post = fp8_stats_fn()
        fus_post   = fusion_stats_fn()

        new_tokens = out_ids.shape[1] - prompt_ids.shape[1]
        tps = (B * new_tokens) / elapsed
        if new_tokens <= 0:
            return dict(error=f"no new tokens: shape={out_ids.shape}")
        # Decode WITHOUT skipping special tokens so we see EOS/pad if present.
        last_text = tok.decode(out_ids[0, prompt_ids.shape[1]:],
                               skip_special_tokens=False)[:60]

        pw = samples[3:] or [0]
        mean_pw = sum(pw)/len(pw)
        tok_w = tps / mean_pw if mean_pw > 0 else 0
        # MFU = (tps * flops_per_token) / (per-GPU peak * n_gpus_used)
        # Single-GPU when load_dm = {"": 0}; both GPUs when 'auto'.
        n_used = 1 if device_map in ("cuda:0", "0") else max(1, n_gpus)
        peak_flops = H100_FP16_TFLOPS * 1e12 * n_used
        mfu = 100.0 * (tps * flops_per_token) / peak_flops

        result = dict(mode=mode, batch=B, prefill=prefill_len, tps=tps,
                      draw_w_total=mean_pw, tok_w=tok_w, mfu_pct=mfu,
                      elapsed_s=elapsed, new_tokens=new_tokens,
                      last_text=last_text,
                      fp8_pre=stats_pre, fp8_post=stats_post,
                      fp8_delta=({k: stats_post[k]-stats_pre[k]
                                  for k in stats_pre} if stats_pre else {}),
                      fusion_delta=({k: fus_post[k]-fus_pre[k]
                                     for k in fus_pre} if fus_pre else {}))
        del out_ids, prompt_ids, attn_mask
        gc.collect(); torch.cuda.empty_cache()
        return result

    print(f"\n=== {mode.upper()} (Llama-3.1-70B fp16, prefill={prefill_len}, "
          f"measure~{measure_seconds}s) ===", flush=True)
    print(f"{'B':>3} {'tps':>7} {'watts':>6} {'tok/W':>7} {'MFU%':>5} {'sec':>4} "
          f"{'fp8':>6} {'rms':>5} {'silu':>5} {'add':>5} {'tail':>30}")
    rows = []
    for B in batches:
        try:
            r = measure(B)
        except torch.cuda.OutOfMemoryError as e:
            print(f"   B={B}: OOM"); torch.cuda.empty_cache(); continue
        if "error" in r:
            print(f"   B={B}: {r['error']}"); continue
        rows.append(r)
        fp8c = r["fp8_delta"].get("calls", "-")
        fus  = r.get("fusion_delta", {})
        tail = r["last_text"].replace("\n"," ")[:30]
        print(f"{r['batch']:>3} {r['tps']:>7.2f} {r['draw_w_total']:>6.1f} "
              f"{r['tok_w']:>7.4f} {r['mfu_pct']:>5.2f} {r['elapsed_s']:>4.1f} "
              f"{fp8c!s:>6} {fus.get('rmsnorm','-')!s:>5} "
              f"{fus.get('silu_mul','-')!s:>5} "
              f"{fus.get('residual_add','-')!s:>5} {tail!r:>30}")
    if out_path:
        json.dump({"mode": mode, "rows": rows}, open(out_path, "w"), indent=2)
        print(f"[step9:{mode}] wrote {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "full"], default="")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--measure", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--params", type=float, default=DEFAULT_PARAMS)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    if args.mode:
        run_mode(args.mode, batches, prefill_len=args.prefill,
                 measure_seconds=args.measure, out_path=args.out,
                 model_path=args.model, params=args.params,
                 device_map=args.device_map)
        return


if __name__ == "__main__":
    main()
