"""Stage 0 on the GPU: training throughput per config, eager and torch.compile, at equal tokens per step
(65,536: 32 x 2048 or 128 x 512), three repetitions with median and spread; peak memory; verdict forward time
at 8,192 tokens; hours and dollars per 1B tokens at the given hourly price.

    python scripts/stage0_gpu.py [--quick] [--out experiments/stage0_gpu] [--steps 5] [--price 1.29]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm.model.config import DEV, MAIN, MID, TINY  # noqa: E402
from vmt.model import Judge, n_params  # noqa: E402
from vmt.train import Trainer, build_optimizers, lm_loss_with  # noqa: E402


def train_speed(name, cfg, batch, seq, steps, compiled, price, reps=3):
    torch.manual_seed(0)
    model = Judge(cfg).cuda()
    run_model = torch.compile(model) if compiled else model
    opts, scheds = build_optimizers(model, 0.02, 3e-4, total_steps=10_000, warmup=10)
    trainer = Trainer(model, opts, scheds)
    if compiled:
        trainer.loss_fn = lambda _m, tok, seg: lm_loss_with(run_model, model, tok, seg)
    tok = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda")
    seg = torch.zeros_like(tok)
    for _ in range(2):
        trainer.step(tok, seg)  # warm up (and compile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rates = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(steps):
            trainer.step(tok, seg)
        torch.cuda.synchronize()
        rates.append(batch * seq * steps / (time.perf_counter() - t))
    tps = statistics.median(rates)
    hours = 1e9 / tps / 3600
    return {"config": name, "params": n_params(model), "compiled": compiled, "batch": batch, "seq": seq,
            "tokens_per_s": round(tps), "tokens_per_s_min": round(min(rates)), "tokens_per_s_max": round(max(rates)),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "hours_per_1b_tokens": round(hours, 2), "dollars_per_1b_tokens": round(hours * price, 2)}


@torch.no_grad()
def verdict_speed(name, cfg, seq, reps=5):
    model = Judge(cfg).cuda().eval()
    tok = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")
    seg = torch.zeros_like(tok)
    pos = torch.tensor([seq - 1], device="cuda")

    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.verdict_logit(model(tok, seg), pos)

    run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    return {"config": name, "seq": seq, "sec_per_forward": round((time.perf_counter() - t) / reps, 4)}


def measure(name, cfg, batch, seq, steps, comp, price):
    """train_speed, halving the batch when a row runs out of GPU memory (the halving is recorded)."""
    note = None
    while True:
        try:
            row = train_speed(name, cfg, batch, seq, steps, comp, price)
            if note:
                row["note"] = note
            return row
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            note = f"batch {batch} ran out of memory, halved to {batch // 2}"
            batch //= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "stage0_gpu")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--price", type=float, default=1.29, help="dollars per GPU hour")
    ap.add_argument("--configs", default="dev,mid,main", help="comma-separated subset of dev, mid, main")
    a = ap.parse_args()
    keep = set(a.configs.split(","))
    if a.quick:
        plans = [("tiny", TINY, 4, 128, False), ("tiny", TINY, 4, 128, True)]
        verdicts = [("tiny", TINY, 256)]
    else:
        plans = [(n, c, b, s, comp) for n, c in (("dev", DEV), ("mid", MID), ("main", MAIN))
                 for b, s in ((32, 2048), (128, 512)) for comp in (False, True)]
        verdicts = [("dev", DEV, 8192), ("mid", MID, 8192), ("main", MAIN, 8192)]
        plans = [p for p in plans if p[0] in keep]
        verdicts = [v for v in verdicts if v[0] in keep]
    out = a.out / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "price_per_hour": a.price,
              "train": [], "verdict": []}

    def save():  # after every row, so a crash keeps what was measured
        (out / "stage0_gpu.json").write_text(json.dumps(report, indent=1))

    for name, cfg, batch, seq, comp in plans:
        report["train"].append(measure(name, cfg, batch, seq, a.steps, comp, a.price))
        print(json.dumps(report["train"][-1]), flush=True)
        save()
        gc.collect()
        torch.cuda.empty_cache()
    for v in verdicts:
        report["verdict"].append(verdict_speed(*v))
        save()
    rows = report["train"]
    print("\n| config | params | compiled | batch x seq | tokens/s (median) | min to max | peak GB | hours per 1B | $ per 1B |\n"
          "|---|---:|---|---|---:|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['config']} | {r['params']:,} | {'yes' if r['compiled'] else 'no'} | {r['batch']} x {r['seq']} | "
              f"{r['tokens_per_s']:,} | {r['tokens_per_s_min']:,} to {r['tokens_per_s_max']:,} | {r['peak_gb']} | "
              f"{r['hours_per_1b_tokens']} | {r['dollars_per_1b_tokens']} |")
    for v in report["verdict"]:
        print(f"verdict forward, {v['config']}, {v['seq']} tokens: {v['sec_per_forward']} s")
    print("report", out / "stage0_gpu.json")


if __name__ == "__main__":
    main()
