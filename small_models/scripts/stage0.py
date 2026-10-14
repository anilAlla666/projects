"""Stage 0: is the architecture trainable on this Mac, and how fast?

1. Numeric check: chunked_delta against mlx-lm's step-by-step gated_delta_ops at the production head shape,
   at a mild gate floor (-0.1) and the hard floor (-1.0).
2. Training throughput: compiled train steps for each config and dtype; tokens per second, peak memory,
   and the days 1B tokens would take at that rate.
3. Verdict time: one forward pass at batch 1 over the full 8,192-token context (model only; the harness
   that gathers evidence is timed in plan 2).

    python scripts/stage0.py [--quick] [--out experiments/stage0] [--steps 3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm.model.config import DEV, MAIN, MID, TINY  # noqa: E402
from vm.model.delta import chunked_delta  # noqa: E402
from vm.model.model import Judge  # noqa: E402
from vm.train.optim import build_optimizer  # noqa: E402
from vm.train.step import make_train_step  # noqa: E402

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}


def numeric_check(T: int, H: int, D: int) -> list[dict]:
    rows = []
    for lo in (-0.1, -1.0):
        mx.random.seed(0)
        q = mx.random.normal((1, T, H, D))
        k = mx.random.normal((1, T, H, D))
        v = mx.random.normal((1, T, H, D))
        q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * D**-0.5
        k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
        g = mx.random.uniform(lo, 0.0, (1, T, H, D))
        b = mx.sigmoid(mx.random.normal((1, T, H)))
        o1, s1 = chunked_delta(q, k, v, g, b)
        o2, s2 = gated_delta_ops(q, k, v, mx.exp(g), b)
        rows.append({"gate_floor": lo, "T": T, "H": H, "D": D,
                     "max_out_diff": mx.abs(o1 - o2).max().item(),
                     "max_state_diff": mx.abs(s1 - s2.swapaxes(-1, -2)).max().item()})
    return rows


def train_speed(name, cfg, dtype, batch, seq, steps, compile=True, reps=3) -> dict:
    """Time `steps` steps `reps` times after a warm-up; report the median rate and the spread, with the
    load average before and after so a contended measurement is visible in the record."""
    load_before = os.getloadavg()[0]
    mx.random.seed(0)
    model = Judge(cfg)
    model.set_dtype(DTYPES[dtype])
    mx.eval(model.parameters())
    opt = build_optimizer(0.02, 3e-4, total_steps=1000, warmup=10)
    step, state = make_train_step(model, opt, compile=compile)
    tok = mx.random.randint(0, cfg.vocab_size, (batch, seq + 1))
    seg = mx.zeros((batch, seq + 1), dtype=mx.int32)
    mx.eval(step(tok, seg), state)  # compile and warm up
    mx.reset_peak_memory()
    rates = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(steps):
            loss = step(tok, seg)
            mx.eval(loss, state)
        rates.append(batch * seq * steps / (time.perf_counter() - t))
    tps = statistics.median(rates)
    return {"config": name, "dtype": dtype, "compile": compile, "batch": batch, "seq": seq,
            "sec_per_step": round(batch * seq / tps, 3), "tokens_per_s": round(tps, 1),
            "tokens_per_s_min": round(min(rates), 1), "tokens_per_s_max": round(max(rates), 1),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "days_per_1b_tokens": round(1e9 / tps / 86400, 2),
            "load_before": round(load_before, 2), "load_after": round(os.getloadavg()[0], 2)}


def verdict_speed(name, cfg, seq, reps=3) -> dict:
    model = Judge(cfg)
    model.eval()
    mx.eval(model.parameters())
    tok = mx.random.randint(0, cfg.vocab_size, (1, seq))
    seg = mx.zeros((1, seq), dtype=mx.int32)
    pos = mx.array([seq - 1])
    run = lambda: model.verdict_logit(model(tok, seg), pos)  # noqa: E731
    mx.eval(run())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(run())
    return {"config": name, "seq": seq, "sec_per_forward": round((time.perf_counter() - t) / reps, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny config and short sequences (for the test)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "stage0")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    if a.quick:
        check = numeric_check(T=64, H=2, D=64)
        plans = [("tiny", TINY, "float32", 2, 128, True), ("tiny", TINY, "float32", 2, 128, False)]
        verdicts = [("tiny", TINY, 256)]
        reps = 2
    else:
        check = numeric_check(T=512, H=12, D=64)
        # Every row trains on 8,192 tokens per step (4 x 2048 or 16 x 512), so configs and modes compare on
        # equal work. bfloat16 only (it tracks float32; see docs/stage0_report.md). Compiled and eager are
        # both recorded because they disagreed in the first, contended measurement.
        plans = []
        for n, c in (("dev", DEV), ("mid", MID), ("main", MAIN)):
            plans += [(n, c, "bfloat16", 4, 2048, True), (n, c, "bfloat16", 4, 2048, False),
                      (n, c, "bfloat16", 16, 512, False)]
        verdicts = [("dev", DEV, 8192), ("mid", MID, 8192), ("main", MAIN, 8192)]
        reps = 3
    rows = []
    for name, cfg, dtype, batch, seq, compile in plans:
        rows.append(train_speed(name, cfg, dtype, batch, seq, a.steps, compile, reps))
        print(json.dumps(rows[-1]), flush=True)
    report = {"mlx_version": mx.__version__, "machine": platform.platform(), "numeric_check": check,
              "train": rows, "verdict": [verdict_speed(n, c, s) for n, c, s in verdicts]}
    out = a.out / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "stage0.json").write_text(json.dumps(report, indent=1))
    print("\n| config | dtype | compiled | batch x seq | s/step | tokens/s (median) | min to max | peak GB | "
          "days per 1B tokens | load before, after |\n|---|---|---|---|---:|---:|---|---:|---:|---|")
    for r in rows:
        print(f"| {r['config']} | {r['dtype']} | {'yes' if r['compile'] else 'no'} | {r['batch']} x {r['seq']} | "
              f"{r['sec_per_step']} | {r['tokens_per_s']:,.0f} | {r['tokens_per_s_min']:,.0f} to {r['tokens_per_s_max']:,.0f} | "
              f"{r['peak_gb']} | {r['days_per_1b_tokens']} | {r['load_before']}, {r['load_after']} |")
    for v in report["verdict"]:
        print(f"verdict forward, {v['config']}, {v['seq']} tokens: {v['sec_per_forward']} s")
    print("numeric check max diffs:", [(c["gate_floor"], c["max_out_diff"], c["max_state_diff"]) for c in check])
    print("report", out / "stage0.json")


if __name__ == "__main__":
    main()
