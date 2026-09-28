"""Generic decode driver — used by wl01, wl02, wl03, wl09, wl12, wl21, wl23.

Knobs:
  --model        which model
  --batch        batch size
  --max-new      tokens to generate per prompt
  --n-prompts    number of prompts
  --prefill      prefill length (None = use prompt as-is)
  --temperature  decode temperature (0 = greedy/deterministic)
  --runs         repeat the whole sweep N times (deterministic test)

Output: { tps, total_new_tokens, avg_watts, tok_per_w, output_hashes,
          ops_fired, runs_results }
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import (write_result, sample_power, avg_power, MODELS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wid", required=True)
    ap.add_argument("--phase", default="baseline")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=0,
                    help="0 = use prompt as-is; >0 = pad/repeat to length")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_prompts = [
        "Write a Python function that computes the n-th Fibonacci number.",
        "Explain general relativity in two sentences.",
        "List five strategies for managing technical debt in a startup.",
        "Translate to French: I love compilers.",
        "Summarize the plot of Hamlet in one paragraph.",
        "Generate a SQL query to find the top 5 customers by revenue.",
        "Suggest a name for a new dog breed that resembles a wolf.",
        "Outline a 3-day itinerary for visiting Tokyo.",
    ]
    while len(base_prompts) < args.n_prompts:
        base_prompts.extend(base_prompts[:args.n_prompts - len(base_prompts)])
    base_prompts = base_prompts[:args.n_prompts]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = []
    stop_evt = threading.Event()
    t_pwr = threading.Thread(target=sample_power, args=(stop_evt, samples, args.gpu),
                             daemon=True)
    t_pwr.start()

    t_load_0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=f"cuda:0",
        low_cpu_mem_usage=True)
    getattr(model, "eval")()
    t_load = time.perf_counter() - t_load_0

    do_sample = (args.temperature > 0)
    runs_results = []

    for run_idx in range(args.runs):
        run_out = []
        run_tps = 0
        run_new = 0
        run_t   = 0.0
        with torch.no_grad():
            for p in base_prompts:
                if args.prefill > 0:
                    p = (p + " ") * (args.prefill // len(p) + 1)
                    p = p[:args.prefill]
                enc = tok([p] * args.batch, return_tensors="pt", padding=True).to("cuda:0")
                t0 = time.perf_counter()
                out = model.generate(
                    **enc, max_new_tokens=args.max_new,
                    do_sample=do_sample, temperature=max(args.temperature, 0.01),
                    top_p=0.9, pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                t_gen = time.perf_counter() - t0
                new_total = (out.shape[1] - enc.input_ids.shape[1]) * args.batch
                run_new += new_total
                run_t   += t_gen
                # Decode and hash for determinism comparison.
                texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                          skip_special_tokens=True)
                run_out.extend(texts)
        run_tps = run_new / max(run_t, 1e-6)
        run_results = {
            "run": run_idx, "tok_per_s": run_tps,
            "total_new": run_new, "total_t": run_t,
            "output_hash": hashlib.sha256("|".join(run_out).encode()).hexdigest()[:16],
        }
        runs_results.append(run_results)

    stop_evt.set()
    t_pwr.join(timeout=2)
    pw = avg_power(samples)

    total_new = sum(r["total_new"] for r in runs_results)
    total_t   = sum(r["total_t"]   for r in runs_results)
    tps       = total_new / max(total_t, 1e-6)
    summary = {
        "wid":              args.wid,
        "phase":            args.phase,
        "model":            args.model,
        "batch":            args.batch,
        "max_new":          args.max_new,
        "n_prompts":        args.n_prompts,
        "tok_per_s":        tps,
        "avg_watts":        pw,
        "tok_per_w":        tps / pw if pw > 0 else 0,
        "load_seconds":     t_load,
        "total_new_tokens": total_new,
        "total_gen_seconds": total_t,
        "n_power_samples":  len(samples),
        "runs":             runs_results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    write_result(args.wid, args.phase, summary)


if __name__ == "__main__":
    main()
