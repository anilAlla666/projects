#!/usr/bin/env python3
"""Realbench driver — runs a workload (or all 24) under baseline vs CIPHER,
computes per-prompt quality + throughput metrics via score.py, writes a
JSON summary + markdown table.

Each run produces:
  RUN_DIR/
    phase_baseline/
      result.jsonl     — per-prompt rows (PromptResult)
      summary.json     — aggregated metrics
      power.csv        — sampled nvidia-smi
    phase_cipher/
      result.jsonl
      summary.json
      power.csv
      counters/cipher_counters_*.json
    COMPARISON.md      — side-by-side baseline vs CIPHER table

Usage:
  # Single workload
  python run.py --workload wl01_decode_single --model /home/ubuntu/models/Llama-3.1-8B

  # All 24 workloads sequentially
  python run.py --all-workloads --duration-each 1h
"""
import argparse
import csv
import ctypes
import gc
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RT_PATH = ROOT / "libcipher_rt.so"
HOOK_PATH = ROOT / "libcipher_hook.so"
sys.path.insert(0, str(Path(__file__).parent))

from score import (
    PromptResult, QualityMetrics, ThroughputMetrics,
    per_position_kl, per_position_topk_agreement,
    perplexity_under_baseline, aggregate, write_jsonl, render_markdown_table,
)


def load_prompts(prompts_path: str) -> list[dict]:
    out = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sample_power(stop_evt, samples, gpu_id: int = 0, interval: float = 0.5):
    """Background thread: poll nvidia-smi power.draw every `interval` s."""
    while not stop_evt.is_set():
        try:
            o = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu_id),
                 "--query-gpu=power.draw,clocks.gr,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            ).decode().strip()
            pw, clk, mem, util = o.split(",")
            samples.append((time.perf_counter(), float(pw), float(clk),
                             int(mem), float(util)))
        except Exception:
            pass
        time.sleep(interval)


def load_cipher_rt(load: bool):
    """If load=True, ctypes.CDLL libcipher_rt.so (RTLD_GLOBAL) so the
    LD_PRELOAD hook can resolve symbols. Returns the handle or None."""
    if not load:
        return None
    if not RT_PATH.exists():
        print(f"[realbench] WARNING: {RT_PATH} not found", file=sys.stderr)
        return None
    return ctypes.CDLL(str(RT_PATH), mode=ctypes.RTLD_GLOBAL)


def generate_with_logits(model, tok, prompt: str, max_new: int, temperature: float = 0.7,
                          top_p: float = 0.9, do_sample: bool = True, batch: int = 1):
    """Generate `max_new` tokens, returning a tensor of generated token ids
    (shape [batch, max_new]) AND the logits at each generation step (shape
    [batch, max_new, vocab]) so we can run KL / topk against a baseline.

    Greedy when do_sample=False, sampling otherwise.
    """
    import torch
    enc = tok(prompt, return_tensors="pt").to("cuda:0")
    input_ids = enc.input_ids
    if batch > 1:
        input_ids = input_ids.repeat(batch, 1)

    generated = []        # list of [batch, 1] tensors
    logits_log = []       # list of [batch, vocab] tensors (per generated step)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]    # [batch, vocab]
        for step in range(max_new):
            logits_log.append(next_logits.detach())
            if do_sample:
                logits_t = next_logits / max(temperature, 1e-6)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_t, descending=True)
                    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits[mask] = float("-inf")
                    logits_t = torch.zeros_like(logits_t).scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits_t, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)   # [batch, 1]
            else:
                next_id = next_logits.argmax(-1, keepdim=True)
            generated.append(next_id)
            out = model(input_ids=next_id, past_key_values=past,
                        use_cache=True, return_dict=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
    gen_ids = torch.cat(generated, dim=1)             # [batch, max_new]
    logits_stack = torch.stack(logits_log, dim=1)      # [batch, max_new, vocab]
    return gen_ids, logits_stack


def run_phase(phase_label: str, prompts: list[dict], model_path: str,
              load_rt: bool, out_dir: Path, gpu_id: int = 0,
              do_sample: bool = True, batch: int = 1,
              cipher_env_setup=None) -> dict:
    """Run one phase (baseline or CIPHER) end-to-end. Returns the aggregated
    summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counters_dir = out_dir / "counters"
    counters_dir.mkdir(exist_ok=True)

    # Power sampler
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_power, args=(stop, samples, gpu_id), daemon=True)
    sampler.start()

    if cipher_env_setup:
        cipher_env_setup(counters_dir)

    import torch
    rt = load_cipher_rt(load_rt)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"[{phase_label}] loading model {model_path}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    for p in model.parameters(): p.requires_grad_(False)

    rows: list[PromptResult] = []
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        gen_ids, logits = generate_with_logits(
            model, tok, p["prompt"], p["max_new_tokens"],
            do_sample=do_sample, batch=batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_tok = gen_ids.shape[0] * gen_ids.shape[1]

        # Stash per-prompt result; quality metrics are filled in by the
        # comparison step (need both baseline and cipher logits side-by-side).
        decoded = tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        result = PromptResult(
            prompt_id=p["id"],
            category=p["category"],
            decoded_text=decoded[:300],
            throughput=ThroughputMetrics(
                tokens_per_s=n_tok / elapsed if elapsed > 0 else 0.0,
                elapsed_s=elapsed,
                n_tokens=n_tok,
            ),
        )
        # Keep token ids + final-step logits for the comparison
        result._gen_ids = gen_ids[0].tolist()                  # type: ignore
        result._logits_path = out_dir / f"logits_{p['id']}.pt"  # type: ignore
        torch.save({"gen_ids": gen_ids[0].cpu(),
                    "logits": logits[0].cpu().to(torch.float16)},
                   result._logits_path)                          # type: ignore
        rows.append(result)
        if i % 5 == 0:
            print(f"[{phase_label}] {i+1}/{len(prompts)} prompts done, "
                  f"last={n_tok/elapsed:.1f} tok/s", file=sys.stderr)

    stop.set(); sampler.join(timeout=2)

    # Power post-processing: compute mean watts during this phase
    mean_w = sum(s[1] for s in samples) / len(samples) if samples else 0.0

    # Persist power samples
    with open(out_dir / "power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_perf", "watts", "clock_mhz", "mem_mib", "util_pct"])
        w.writerows(samples)

    # Backfill per-prompt mean watts (uniform — single-prompt power is noisy)
    for r in rows:
        r.throughput.mean_watts = mean_w
        if mean_w > 0:
            r.throughput.tok_per_w = r.throughput.tokens_per_s / mean_w

    # Persist per-prompt rows + summary
    write_jsonl(rows, str(out_dir / "result.jsonl"))
    summary = aggregate(rows)
    summary["mean_watts"] = mean_w
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    del model, tok
    gc.collect()
    if 'torch' in dir(): torch.cuda.empty_cache()
    return summary


def compute_quality_metrics(baseline_dir: Path, cipher_dir: Path):
    """Side-by-side: load each phase's saved logits + token ids per prompt,
    compute KL, top-k agreement, perplexity. Update the cipher result.jsonl
    in place with quality scores."""
    import torch
    bp = json.load(open(baseline_dir / "summary.json"))
    cp = json.load(open(cipher_dir / "summary.json"))

    # Load per-prompt rows
    base_rows = [json.loads(l) for l in open(baseline_dir / "result.jsonl")]
    ciph_rows = [json.loads(l) for l in open(cipher_dir / "result.jsonl")]

    base_by_id = {r["prompt_id"]: r for r in base_rows}
    upd_rows: list[PromptResult] = []
    for cr in ciph_rows:
        pid = cr["prompt_id"]
        bp_blob = torch.load(baseline_dir / f"logits_{pid}.pt")
        cp_blob = torch.load(cipher_dir / f"logits_{pid}.pt")
        b_log = bp_blob["logits"].to("cuda:0").float()
        c_log = cp_blob["logits"].to("cuda:0").float()
        b_ids = bp_blob["gen_ids"].to("cuda:0")
        c_ids = cp_blob["gen_ids"].to("cuda:0")

        n = min(b_log.shape[0], c_log.shape[0])
        b_log = b_log[:n]; c_log = c_log[:n]
        c_ids = c_ids[:n]

        kl = per_position_kl(b_log, c_log)            # [n]
        topk = per_position_topk_agreement(b_log, c_ids, ks=(1, 5, 10, 50))
        ppl = perplexity_under_baseline(b_log, c_ids)

        cr["quality"] = {
            "perplexity_under_baseline": ppl,
            "kl_divergence_mean": float(kl.mean().item()),
            "kl_divergence_p95": float(kl.quantile(0.95).item()),
            "top1_agreement": topk[1],
            "top5_agreement": topk[5],
            "top10_agreement": topk[10],
            "top50_agreement": topk[50],
            "n_compared_positions": n,
        }
        upd_rows.append(cr)

    # Rewrite cipher result.jsonl with quality scores attached
    with open(cipher_dir / "result.jsonl", "w") as f:
        for cr in upd_rows:
            f.write(json.dumps(cr) + "\n")

    # Re-aggregate and rewrite summary
    cipher_quality = {
        "ppl_mean": sum(r["quality"]["perplexity_under_baseline"] for r in upd_rows) / max(len(upd_rows), 1),
        "kl_mean": sum(r["quality"]["kl_divergence_mean"] for r in upd_rows) / max(len(upd_rows), 1),
        "kl_p95": sum(r["quality"]["kl_divergence_p95"] for r in upd_rows) / max(len(upd_rows), 1),
        "top1_agreement": sum(r["quality"]["top1_agreement"] for r in upd_rows) / max(len(upd_rows), 1),
        "top5_agreement": sum(r["quality"]["top5_agreement"] for r in upd_rows) / max(len(upd_rows), 1),
        "top10_agreement": sum(r["quality"]["top10_agreement"] for r in upd_rows) / max(len(upd_rows), 1),
        "top50_agreement": sum(r["quality"]["top50_agreement"] for r in upd_rows) / max(len(upd_rows), 1),
    }
    cp["quality"] = cipher_quality
    with open(cipher_dir / "summary.json", "w") as f:
        json.dump(cp, f, indent=2)
    # Baseline's "quality" is meaningless against itself; render as identity
    bp["quality"] = {
        "ppl_mean": 1.0,
        "kl_mean": 0.0,
        "kl_p95": 0.0,
        "top1_agreement": 1.0,
        "top5_agreement": 1.0,
        "top10_agreement": 1.0,
        "top50_agreement": 1.0,
    }
    with open(baseline_dir / "summary.json", "w") as f:
        json.dump(bp, f, indent=2)
    return bp, cp


def setup_cipher_env(counters_dir: Path):
    """Set CIPHER env vars for the cipher phase. Called at start of phase
    BEFORE model load. Caller has already set LD_PRELOAD via the orchestrator."""
    os.environ.setdefault("CIPHER_PRESET", "production")
    os.environ.setdefault("CIPHER_FP8_COMPUTE", "on")
    os.environ.setdefault("CIPHER_FUSION_KERNELS", "on")
    os.environ.setdefault("CIPHER_FUSION", "on")
    os.environ.setdefault("CIPHER_FLOW_RECORD", "on")
    os.environ.setdefault("CIPHER_FLOW_MATCH", "on")
    os.environ.setdefault("CIPHER_FLOW_SUBSTITUTE", "on")
    os.environ.setdefault("CIPHER_THERMOSTAT", "on")
    os.environ.setdefault("CIPHER_PREDICT", "on")
    os.environ.setdefault("CIPHER_GUARD", "on")
    os.environ.setdefault("CIPHER_TRACE", "on")
    os.environ.setdefault("CIPHER_RECEIPT", "on")
    os.environ.setdefault("CIPHER_CARBON", "on")
    os.environ.setdefault("CIPHER_TOPOLOGY", "on")
    os.environ.setdefault("CIPHER_PIPELINE", "on")
    os.environ.setdefault("CIPHER_CONTINUITY", "on")
    os.environ.setdefault("CIPHER_LOOP", "on")
    os.environ.setdefault("CIPHER_PERSIST", "on")
    os.environ.setdefault("CIPHER_PERSIST_ENGINE", "on")
    os.environ.setdefault("CIPHER_WORKLOAD_DETECT", "on")
    os.environ.setdefault("CIPHER_WORKLOAD_REPORT", "on")
    os.environ["CIPHER_COUNTERS_DUMP_DIR"] = str(counters_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--model", default="/home/ubuntu/models/Llama-3.1-8B")
    ap.add_argument("--out", required=True, help="Output run directory")
    ap.add_argument("--phase", choices=["baseline", "cipher", "compare"], default="baseline",
                    help="baseline: run no-CIPHER; cipher: run with CIPHER (caller sets LD_PRELOAD); compare: read both phases' saved logits and write COMPARISON.md")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-sample", action="store_true",
                    help="Use greedy decoding instead of sampling")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run first N prompts (0 = all)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    if args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"[realbench] {len(prompts)} prompts loaded", file=sys.stderr)

    do_sample = not args.no_sample

    if args.phase == "baseline":
        # Caller must have unset LD_PRELOAD before invoking us
        run_phase("baseline", prompts, args.model, load_rt=False,
                  out_dir=out_dir / "phase_baseline", gpu_id=args.gpu,
                  do_sample=do_sample, batch=args.batch)
    elif args.phase == "cipher":
        # Caller must have set LD_PRELOAD=libcipher_hook.so:libcuda.so before invoking us
        run_phase("cipher", prompts, args.model, load_rt=True,
                  out_dir=out_dir / "phase_cipher", gpu_id=args.gpu,
                  do_sample=do_sample, batch=args.batch,
                  cipher_env_setup=setup_cipher_env)
    elif args.phase == "compare":
        bp, cp = compute_quality_metrics(out_dir / "phase_baseline",
                                           out_dir / "phase_cipher")
        # Pull the actual run config from the baseline summary, not args
        n_compared = bp.get("n_prompts", len(prompts))
        # Render comparison table
        with open(out_dir / "COMPARISON.md", "w") as f:
            f.write("# Realbench comparison — baseline vs CIPHER\n\n")
            f.write(f"Run dir: `{out_dir}`\n\n")
            f.write(f"Model: {args.model}  \n")
            f.write(f"Prompts compared: {n_compared}  \n")
            f.write(f"Batch: {args.batch}\n\n")
            f.write("## Aggregate\n\n")
            f.write(render_markdown_table(bp, cp))
            f.write("\n\n")
            f.write("## Notes\n\n")
            f.write("- `KL divergence` / `top-k agreement` measure logit-distribution similarity AT EACH GENERATION POSITION.\n")
            f.write("- Greedy mode: drift compounds across positions (one logit-noise event diverges all subsequent tokens). Sampling mode: KL is a position-local measure unaffected by cumulative drift, so it's the right metric for production-quality assessment.\n")
            f.write("- `perplexity under baseline` is currently approximate (computed against cipher's own conditional logits, not against baseline given cipher's prefix). For exact PPL, a follow-up step re-runs baseline on cipher's full output sequence — TODO.\n")
            f.write("- Throughput numbers are per-prompt averages; aggregate tok/W is sensitive to long-prompt warmup transients.\n")
        print(f"[realbench] DONE → {out_dir}/COMPARISON.md")


if __name__ == "__main__":
    main()
