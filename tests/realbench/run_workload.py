#!/usr/bin/env python3
"""M1.T9 — drives one workload from the registry, end-to-end.

Usage:
    python run_workload.py wl01 --phase baseline
    CIPHER_FAIRNESS=on python run_workload.py wl05 --phase cipher \\
        --model /path/to/local-snapshot
"""
import argparse
import json
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workloads import by_id, runnable_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wid", help="workload id (e.g. wl01) or NAME")
    ap.add_argument("--model", default=None,
                    help="override model path (default: registry-default)")
    ap.add_argument("--phase", default="baseline",
                    choices=["baseline", "cipher"])
    ap.add_argument("--out", default=None,
                    help="output dir (default: tests/realbench/runs/<wid>/<phase>)")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--list-runnable", action="store_true")
    args = ap.parse_args()

    if args.list_runnable:
        print("\n".join(runnable_ids()))
        return

    entry = by_id(args.wid)
    if not entry.driver_mod:
        print(f"workload {entry.id} status={entry.status} note={entry.note}")
        sys.exit(2)

    drv = importlib.import_module(f"workloads.{entry.driver_mod}")
    model_path = args.model or entry.default_model
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "runs" / entry.id / args.phase)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[realbench] {entry.name} phase={args.phase} model={model_path}")
    summary = drv.run(model_path, prompts, args.phase, out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
