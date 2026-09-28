#!/usr/bin/env python3
"""
Phase 4.0 (vLLM baseline) — apples-to-apples comparison against the original
~55% MFU prediction. Uses vLLM's offline batched decode (continuous batching,
paged attention, fused QKV, CUDA graphs all on by default).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/vllm_install")

from vllm import LLM, SamplingParams

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCHES = [32, 64, 128, 192, 256]
PROMPT_LEN = 512        # tokens of context per request
DECODE_LEN = 128        # tokens to generate per request

H100_FP16_PEAK_TFLOPS = 989.0
H100_FP16_SUSTAINED_TFLOPS = 700.0
H100_HBM3_PEAK_GB_S = 3350.0


def main():
    print(f"[load] {MODEL_ID} fp16 via vLLM")
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=PROMPT_LEN + DECODE_LEN + 32,
        enforce_eager=False,           # let vLLM use CUDA graphs
        disable_log_stats=True,
    )
    params = 1_100_048_384  # TinyLlama 1.1B (fixed; vLLM internal API churns)
    print(f"[load] params = {params/1e9:.3f} B (TinyLlama-1.1B fixed constant)")

    # Pre-tokenize a fake prompt of exactly PROMPT_LEN tokens so each batch
    # sees identical context length (continuous-batching shape stable).
    tok = llm.get_tokenizer()
    fake_text = (tok.decode([1000]) + " ") * PROMPT_LEN
    prompt_ids = tok.encode(fake_text)[:PROMPT_LEN]

    rows = []
    for B in BATCHES:
        sp = SamplingParams(temperature=0.0, max_tokens=DECODE_LEN, ignore_eos=True)
        prompts = [{"prompt_token_ids": prompt_ids} for _ in range(B)]
        # Warm up
        _ = llm.generate(prompts, sp, use_tqdm=False)
        # Time
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        toks_per_s = gen_tokens / dt
        # Decode-step FLOPs/HBM (per token)
        flops_per_tok = 2.0 * params
        hbm_bytes_per_tok = 2.0 * params / B  # weights amortized across batch
        achieved_tflops = toks_per_s * flops_per_tok / 1e12
        hbm_gb_s = toks_per_s * hbm_bytes_per_tok / 1e9
        r = {
            "batch": B,
            "wall_s": dt,
            "tokens_total": gen_tokens,
            "tokens_per_s": toks_per_s,
            "tflops": achieved_tflops,
            "mfu_spec": achieved_tflops / H100_FP16_PEAK_TFLOPS,
            "mfu_sustained": achieved_tflops / H100_FP16_SUSTAINED_TFLOPS,
            "hbm_gb_s": hbm_gb_s,
            "hbm_util": hbm_gb_s / H100_HBM3_PEAK_GB_S,
        }
        rows.append(r)
        print(f"[bench] B={B:3d}  wall={dt:.2f}s  toks/s={toks_per_s:.1f}  "
              f"TFLOPS={achieved_tflops:.1f}  MFU(spec)={r['mfu_spec']*100:.1f}%  "
              f"HBM={hbm_gb_s:.0f} GB/s ({r['hbm_util']*100:.1f}%)")

    print("\n=== Phase 4.0 vLLM baseline (continuous batching + CUDA graphs) ===")
    print(f"{'B':>4} | {'wall s':>7} | {'tok/s':>9} | {'TFLOPS':>7} | "
          f"{'MFU spec':>8} | {'MFU sust':>8} | {'HBM GB/s':>8} | {'HBM util':>8}")
    print("-" * 86)
    for r in rows:
        print(f"{r['batch']:>4} | {r['wall_s']:>7.2f} | {r['tokens_per_s']:>9.1f} | "
              f"{r['tflops']:>7.1f} | {r['mfu_spec']*100:>7.1f}% | "
              f"{r['mfu_sustained']*100:>7.1f}% | {r['hbm_gb_s']:>8.0f} | "
              f"{r['hbm_util']*100:>7.1f}%")

    out = {
        "model": MODEL_ID,
        "params": params,
        "engine": "vllm",
        "prompt_len": PROMPT_LEN,
        "decode_len": DECODE_LEN,
        "rows": rows,
    }
    out_path = Path(__file__).parent / "bench_vllm_baseline_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[json] wrote {out_path}")


if __name__ == "__main__":
    main()
