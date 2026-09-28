#!/usr/bin/env python3
"""Llama-3.1-8B fp16 single-GPU CIPHER full stack with CUDA graph capture.

Same patches as step9 (fused RMSNorm + SiLU.Mul + residual_add + RoPE +
FP8 substitution) plus a CUDAGraph capture around one decode step.  After
capture, replay the graph for measure_seconds, counting steps.

Coherence verification: keep every predicted token in a host buffer during
the replay loop, decode them all at the end, and abort on early EOS.
"""
import argparse, ctypes, gc, json, os, subprocess, sys, threading, time
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/home/ubuntu/models/Llama-3.1-8B"
H100_FP16_TFLOPS = 989.4
PARAMS = 8.03e9
FLOPS_PER_TOKEN = 2 * PARAMS


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "-i", "0",
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1)
            samples.append(float(r.stdout.strip()))
        except Exception:
            pass
        time.sleep(0.15)


def setup_rt():
    rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                     mode=ctypes.RTLD_GLOBAL)
    rt.cipher_fp8_compute_init.restype = ctypes.c_int
    rt.cipher_fp8_compute_init()
    rt.cipher_fused_rmsnorm.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    rt.cipher_fused_silu_mul.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_silu_mul.restype = ctypes.c_int
    rt.cipher_fused_residual_add.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_residual_add.restype = ctypes.c_int
    rt.cipher_substitute_v2_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p]
    rt.cipher_substitute_v2_compile.restype = ctypes.c_ulong
    rt.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
    rt.cipher_substitute_v2_get_function.restype = ctypes.c_void_p
    try:
        rt.cipher_fusion_kernels_init.restype = ctypes.c_int
        rt.cipher_fusion_kernels_init()
    except AttributeError:
        pass
    return rt


def apply_patches(rt, model, mode):
    if mode == "baseline":
        return
    import torch
    from transformers.models.llama import modeling_llama as LM
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer)

    _orig_rms = LlamaRMSNorm.forward
    _orig_mlp = LlamaMLP.forward
    _orig_rope = LM.apply_rotary_pos_emb

    def _stream_ptr(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def _rms(self, x):
        out = torch.empty_like(x)
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        of = out.reshape(-1, x.shape[-1])
        rc = rt.cipher_fused_rmsnorm(
            flat.data_ptr(), self.weight.data_ptr(), of.data_ptr(),
            flat.shape[0], flat.shape[-1], float(self.variance_epsilon),
            _stream_ptr(x.device))
        return out if rc == 1 else _orig_rms(self, x)
    LlamaRMSNorm.forward = _rms

    def _mlp(self, x):
        g = self.gate_proj(x); u = self.up_proj(x)
        if not g.is_contiguous(): g = g.contiguous()
        if not u.is_contiguous(): u = u.contiguous()
        out = torch.empty_like(g)
        rc = rt.cipher_fused_silu_mul(
            g.data_ptr(), u.data_ptr(), out.data_ptr(),
            g.numel(), _stream_ptr(g.device))
        if rc != 1:
            out = torch.nn.functional.silu(g) * u
        return self.down_proj(out)
    LlamaMLP.forward = _mlp

    def _residual_add(x, residual):
        if not x.is_contiguous(): x = x.contiguous()
        if not residual.is_contiguous(): residual = residual.contiguous()
        out = torch.empty_like(x)
        rc = rt.cipher_fused_residual_add(
            x.data_ptr(), residual.data_ptr(), out.data_ptr(),
            x.numel(), _stream_ptr(x.device))
        return out if rc == 1 else x + residual

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
            **kwargs)
        hidden_states = _residual_add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _residual_add(hidden_states, residual)
        return hidden_states
    LlamaDecoderLayer.forward = _decoder_forward

    # Per-device RoPE NVRTC compile (single GPU here, but keep the dict).
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
    libcuda = ctypes.CDLL("libcuda.so.1")
    cu_launch = libcuda.cuLaunchKernel
    cu_launch.argtypes = [ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    cu_launch.restype = ctypes.c_int

    cubin = rt.cipher_substitute_v2_compile(ROPE_SRC, b"cipher_fused_rope_qk")
    rope_fn = rt.cipher_substitute_v2_get_function(cubin)
    print(f"[step9_graph:{mode}] RoPE compiled fn={hex(rope_fn) if rope_fn else 'NULL'}", flush=True)

    def _rope(q, k, cos, sin, unsqueeze_dim=1):
        if not rope_fn:
            return _orig_rope(q, k, cos, sin, unsqueeze_dim)
        B, Hq, S, D = q.shape
        _, Hkv, _, _ = k.shape
        cos_c = cos.contiguous(); sin_c = sin.contiguous()
        cos_bs = 0 if (cos_c.dim() == 2 or cos_c.shape[0] == 1) else S * D
        # FIXED: q/k come from view+transpose; force contiguous.
        q_in = q.contiguous(); k_in = k.contiguous()
        q_out = torch.empty_like(q_in); k_out = torch.empty_like(k_in)
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
            ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_list])
        rc = cu_launch(rope_fn, (Hq+Hkv)*S, B, 1, D, 1, 1, D*4,
                        _stream_ptr(q.device), args_arr, None)
        if rc != 0:
            return _orig_rope(q, k, cos, sin, unsqueeze_dim)
        return q_out, k_out
    LM.apply_rotary_pos_emb = _rope


def run(mode, batches, prefill_len, measure_seconds, out_path):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    print(f"[step9_graph:{mode}] loading Llama-3.1-8B fp16 on cuda:0...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map={"": 0})
    print(f"[step9_graph:{mode}] loaded in {time.perf_counter()-t_load:.1f}s; "
          f"GPU0={torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)

    rt = None
    if mode == "full":
        rt = setup_rt()
        apply_patches(rt, model, mode)

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    N_TILE = (prefill_len + len(ids_one) - 1) // len(ids_one)
    big = ids_one.repeat(N_TILE)[:prefill_len]
    prompt_ids_1 = big.unsqueeze(0).to("cuda:0")

    def measure(B):
        MAX_LEN = prefill_len + 256
        prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
        cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)

        # Prefill — establishes cache shape on first .update() call.
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
            o = model(input_ids=prompt_ids, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        next_token = o.logits[:, -1:].argmax(-1)
        input_ids = next_token.detach().clone()
        cache_pos = torch.tensor([prefill_len], device="cuda:0", dtype=torch.long)

        out_logits = torch.full((B, 1, model.config.vocab_size),
                                 float("nan"), dtype=torch.float16,
                                 device="cuda:0")

        # Eager warmup — needed both to populate caching allocator and to
        # trigger CIPHER's lazy NVRTC compiles before capture.
        for _ in range(10):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True,
                          return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
        torch.cuda.synchronize()
        if torch.isnan(out_logits).any().item():
            return dict(error="NaN after eager warmup", batch=B)

        # Side-stream warmup before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                with torch.no_grad():
                    o = model(input_ids=input_ids, cache_position=cache_pos,
                              past_key_values=cache, use_cache=True,
                              return_dict=True)
                out_logits.copy_(o.logits)
                input_ids.copy_(out_logits.argmax(-1))
                cache_pos += 1
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Capture one decode step.
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    o = model(input_ids=input_ids, cache_position=cache_pos,
                              past_key_values=cache, use_cache=True,
                              return_dict=True)
                out_logits.copy_(o.logits)
                input_ids.copy_(out_logits.argmax(-1))
            torch.cuda.synchronize()
            print(f"[step9_graph:{mode}] graph captured for B={B}", flush=True)
        except Exception as e:
            return dict(error=f"graph capture failed: {type(e).__name__}: {e}",
                        batch=B)

        # Coherence buffer: predicted token id at each replay step (row 0).
        token_log = []

        # Read FP8 / fusion stats before timed loop.
        def _read(name, struct_def):
            try:
                Stats = type("Stats", (ctypes.Structure,), {"_fields_": struct_def})
                fn = getattr(rt, name)
                fn.argtypes = [ctypes.POINTER(Stats)]
                fn.restype = ctypes.c_int
                s = Stats(); fn(ctypes.byref(s))
                return {f[0]: getattr(s, f[0]) for f in struct_def}
            except Exception:
                return {}
        fp8_pre = _read("cipher_fp8_compute_stats", [
            ("enabled", ctypes.c_int),
            ("weights_quantized", ctypes.c_uint64),
            ("matmul_calls", ctypes.c_uint64),
            ("passthroughs", ctypes.c_uint64),
            ("correctness_failures", ctypes.c_uint64),
            ("bytes_fp16", ctypes.c_size_t),
            ("bytes_fp8", ctypes.c_size_t),
        ]) if rt else {}
        fus_pre = _read("cipher_fusion_kernels_stats", [
            ("enabled", ctypes.c_int),
            ("rmsnorm_calls", ctypes.c_uint64),
            ("silu_mul_calls", ctypes.c_uint64),
            ("residual_add_calls", ctypes.c_uint64),
        ]) if rt else {}

        out_logits.fill_(float("nan"))
        samples = []; stop = threading.Event()
        th = threading.Thread(target=power_sampler, args=(stop, samples),
                              daemon=True); th.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter(); n_steps = 0
        while time.perf_counter() - t0 < measure_seconds:
            graph.replay()
            cache_pos += 1
            n_steps += 1
            if cache_pos.item() >= MAX_LEN - 4: break
            # Capture predicted token for row 0 every step (cheap host copy).
            if n_steps <= 80:  # keep it bounded for log
                token_log.append(int(input_ids[0, 0].item()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)

        if torch.isnan(out_logits).any().item():
            return dict(error="NaN after timed loop", batch=B)

        fp8_post = _read("cipher_fp8_compute_stats", [
            ("enabled", ctypes.c_int),
            ("weights_quantized", ctypes.c_uint64),
            ("matmul_calls", ctypes.c_uint64),
            ("passthroughs", ctypes.c_uint64),
            ("correctness_failures", ctypes.c_uint64),
            ("bytes_fp16", ctypes.c_size_t),
            ("bytes_fp8", ctypes.c_size_t),
        ]) if rt else {}
        fus_post = _read("cipher_fusion_kernels_stats", [
            ("enabled", ctypes.c_int),
            ("rmsnorm_calls", ctypes.c_uint64),
            ("silu_mul_calls", ctypes.c_uint64),
            ("residual_add_calls", ctypes.c_uint64),
        ]) if rt else {}

        # Coherence verification — decode the captured token log.
        eos_id = tok.eos_token_id
        eos_idx = next((i for i, t in enumerate(token_log) if t == eos_id), -1)
        verify_text = tok.decode(token_log[:80], skip_special_tokens=False)[:120]

        pw = samples[3:] or [0]
        mean_pw = sum(pw) / len(pw)
        tps = (B * n_steps) / elapsed
        tok_w = tps / mean_pw if mean_pw > 0 else 0
        peak_flops = H100_FP16_TFLOPS * 1e12
        mfu = 100.0 * (tps * FLOPS_PER_TOKEN) / peak_flops

        result = dict(mode=mode, batch=B, prefill=prefill_len, tps=tps,
                      draw_w=mean_pw, tok_w=tok_w, mfu_pct=mfu,
                      elapsed_s=elapsed, n_steps=n_steps,
                      verify_text=verify_text, verify_eos_idx=eos_idx,
                      verify_n_logged=len(token_log),
                      fp8_delta={k: fp8_post.get(k,0)-fp8_pre.get(k,0)
                                 for k in fp8_pre},
                      fusion_delta={k: fus_post.get(k,0)-fus_pre.get(k,0)
                                    for k in fus_pre})
        del graph, cache, prompt_ids, input_ids, cache_pos, out_logits
        gc.collect(); torch.cuda.empty_cache()
        return result

    print(f"\n=== {mode.upper()} (Llama-3.1-8B fp16, prefill={prefill_len}, "
          f"graph-replay, ~{measure_seconds}s) ===", flush=True)
    print(f"{'B':>3} {'tps':>8} {'W':>6} {'tok/W':>7} {'MFU%':>5} {'sec':>4} "
          f"{'fp8':>6} {'rms':>5} verify (≤80 tokens)")
    rows = []
    for B in batches:
        try:
            r = measure(B)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  B={B}: OOM"); torch.cuda.empty_cache(); continue
        if "error" in r:
            print(f"  B={B}: {r['error']}"); continue
        rows.append(r)
        fp8c = r["fp8_delta"].get("matmul_calls", "-")
        rmsc = r["fusion_delta"].get("rmsnorm_calls", "-")
        v = r["verify_text"][:55].replace("\n", " ")
        eos_marker = (f" eos@{r['verify_eos_idx']}"
                      if r["verify_eos_idx"] >= 0 else "")
        print(f"{B:>3} {r['tps']:>8.2f} {r['draw_w']:>6.1f} "
              f"{r['tok_w']:>7.4f} {r['mfu_pct']:>5.2f} {r['elapsed_s']:>4.1f} "
              f"{fp8c!s:>6} {rmsc!s:>5} {v!r}{eos_marker}")
        if r["verify_eos_idx"] >= 0 and r["verify_eos_idx"] < 5:
            print(f"  EARLY EOS at index {r['verify_eos_idx']} — STOP",
                  flush=True)
            sys.exit(1)
    if out_path:
        json.dump({"mode": mode, "rows": rows}, open(out_path, "w"), indent=2)
        print(f"[step9_graph:{mode}] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "full"], required=True)
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--measure", type=float, default=10.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    run(args.mode, batches, prefill_len=args.prefill,
        measure_seconds=args.measure, out_path=args.out)


if __name__ == "__main__":
    main()
