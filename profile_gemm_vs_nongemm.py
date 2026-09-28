#!/usr/bin/env python3
"""Task 2: profile one decode step at B=32, B=64 — count GEMM vs non-GEMM
GPU time, report percentages.

Run this AFTER `step7_fp8_eager.py` style warmup so cublasLt/cuDNN
algorithms are settled.  Uses torch.profiler.profile to capture every
CUDA kernel launch + duration; then bucketizes by kernel name into
GEMM (cublas/cutlass/wgmma/sgemm/hgemm/MEMcpyD2D for Q@K_T paths) vs
non-GEMM (RMSNorm/SiLU/RoPE/index_*/elementwise/etc).

Run:
    python3 profile_gemm_vs_nongemm.py --batches=32,64 --prefill=128
"""
import argparse, ctypes, gc, os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from torch.profiler import profile, record_function, ProfilerActivity


GEMM_KEYS = (
    "gemm", "GEMM",
    "sgemm", "hgemm", "bgemm",
    "ampere_h", "ampere_s", "ampere_b", "ampere_g",
    "cutlass", "CUTLASS",
    "tensorop", "wgmma",
    "matmul",
    "Kernel2",          # cutlass kernel wrappers
    "cutlass3x", "cutlass2x",
    "splitKreduce", "splitK",
    # Hopper-optimized cuBLAS hgemm kernels (CUDA 12.2+).
    "nvjet_hsh", "nvjet_tsh", "nvjet_bsh",
)
ATTN_KEYS = (
    "flash", "Flash",
    "memory_efficient", "mha_fwd",
    "scaled_dot",
    "softmax", "Softmax",
    "PyTorchMemEffAttention",
    "fmha_cutlass", "fmha_kernel",
)
RMSNORM_KEYS = ("rmsnorm", "rms_norm", "RMSNorm")
ELEMWISE_KEYS = (
    "silu", "SiLU",
    "swiglu",
    "elementwise", "Elementwise",
    "binary_op",
    "BinaryFunctor",
    "vectorized_elementwise",
    "index_copy", "index_select",
    "rope", "RoPE",
    "mul_kernel", "add_kernel",
    "embedding",
    "argmax",
    "view", "reshape",
    "fill",
)


def classify(name: str) -> str:
    # ATTN check BEFORE GEMM: "fmha_cutlassF" matches both "cutlass" (GEMM
    # bucket) and "fmha_cutlass" (ATTN bucket); attention should win since
    # the kernel is logically attention even if its tile machinery is
    # cutlass-based.
    if any(k in name for k in ATTN_KEYS):
        return "ATTN/Softmax"
    if any(k in name for k in GEMM_KEYS):
        return "GEMM"
    if any(k in name for k in RMSNORM_KEYS):
        return "RMSNorm"
    if any(k in name for k in ELEMWISE_KEYS):
        return "Elem/Cast/Other"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,64")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--steps",   type=int, default=8,
                     help="how many decode steps to profile")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    MODEL = "mistralai/Mistral-7B-v0.1"
    print(f"[profile] loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda")

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile = (args.prefill + len(ids_one) - 1) // len(ids_one)
    big = (ids_one.repeat(n_tile))[:args.prefill]
    prompt_ids_1 = big.unsqueeze(0).to("cuda")

    for B in batches:
        MAX_LEN = args.prefill + args.steps + 16
        prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
        cache = StaticCache(config=model.config, max_batch_size=B,
                             max_cache_len=MAX_LEN, device="cuda",
                             dtype=torch.float16)
        with torch.no_grad():
            cp = torch.arange(args.prefill, device="cuda", dtype=torch.long)
            out = model(input_ids=prompt_ids, cache_position=cp,
                        past_key_values=cache, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        next_token = out.logits[:, -1:].argmax(-1)
        input_ids = next_token.detach().clone()
        cache_pos = torch.tensor([args.prefill], device="cuda", dtype=torch.long)

        # Warmup so cuBLAS picks its kernels.
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            input_ids = o.logits[:, -1:].argmax(-1)
            cache_pos += 1
        torch.cuda.synchronize()

        # Profile N decode steps.
        with profile(activities=[ProfilerActivity.CUDA],
                     record_shapes=False, with_stack=False) as prof:
            for _ in range(args.steps):
                with torch.no_grad():
                    o = model(input_ids=input_ids, cache_position=cache_pos,
                              past_key_values=cache, use_cache=True,
                              return_dict=True)
                input_ids = o.logits[:, -1:].argmax(-1)
                cache_pos += 1
            torch.cuda.synchronize()

        events = prof.key_averages()
        # Aggregate cuda time by category.
        buckets = {"GEMM": 0, "ATTN/Softmax": 0, "RMSNorm": 0,
                    "Elem/Cast/Other": 0, "Other": 0}
        bucket_counts = {k: 0 for k in buckets}
        bucket_examples = {k: [] for k in buckets}
        # Detailed top-10 unbucketed.
        unbucketed = []
        total_cuda_us = 0
        for ev in events:
            t = ev.device_time_total       # microseconds
            if t <= 0: continue
            cat = classify(ev.key)
            buckets[cat] += t
            bucket_counts[cat] += ev.count
            total_cuda_us += t
            if cat == "Other" and len(bucket_examples["Other"]) < 25:
                bucket_examples["Other"].append((ev.key, t, ev.count))
            unbucketed.append((ev.key, t, cat))

        unbucketed.sort(key=lambda x: -x[1])
        print(f"\n{'='*86}")
        print(f"  B = {B}    decode steps profiled = {args.steps}")
        print(f"{'='*86}")
        print(f"  Total CUDA time (all steps): {total_cuda_us:>12.0f} µs")
        print(f"  Per-step:                     {total_cuda_us/args.steps:>12.0f} µs")
        print()
        print(f"  {'category':<18}  {'cuda_us':>12}  {'% total':>9}  "
              f"{'#kernels':>10}")
        print(f"  {'-'*18}  {'-'*12}  {'-'*9}  {'-'*10}")
        for cat, us in buckets.items():
            pct = 100.0 * us / max(total_cuda_us, 1)
            print(f"  {cat:<18}  {us:>12.0f}  {pct:>8.2f}%  "
                  f"{bucket_counts[cat]:>10}")

        print()
        print(f"  Top-15 kernels by cuda_us (any category):")
        for name, us, cat in unbucketed[:15]:
            pct = 100.0 * us / max(total_cuda_us, 1)
            print(f"    [{cat:<16}] {us:>10.0f} µs  ({pct:>5.2f}%)  {name[:60]}")

        if buckets["Other"] > 0.05 * total_cuda_us:
            print()
            print(f"  Top-15 unclassified kernels (>'Other' bucket):")
            other_examples = [(n, t, c) for n, t, _ in unbucketed
                               for c in [classify(n)] if c == "Other"][:15]
            for name, us, _ in other_examples:
                pct = 100.0 * us / max(total_cuda_us, 1)
                print(f"    {us:>10.0f} µs  ({pct:>5.2f}%)  {name[:70]}")

        del cache, prompt_ids, input_ids, cache_pos, o, out
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
