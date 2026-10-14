import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from vm.model.config import TINY
from vmt.model import Judge

ROOT = Path(__file__).resolve().parents[1]


def synth(path, n=64, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2
        T = rng.randint(40, 90)
        toks = [rng.randint(8, TINY.vocab_size - 1) for _ in range(T)]
        toks[5] = 100 + label  # a token that decides the label, so the task is learnable
        line_starts = [10, 20, 30]
        code = [2, 4]
        rows.append({"example_id": f"e{i}", "library": f"l{i % 4}", "tokens": toks, "segments": [0] * T, "verdict_pos": T - 1,
                     "line_starts": line_starts, "code_line_starts": code, "label": label, "pointer": label,
                     "hidden_bucket": label, "bug_line": 1, "attempt_kind": "revert", "operator": "compare"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_finetune_learns_and_scores(tmp_path):
    torch.manual_seed(0)
    torch.save({"model": Judge(TINY).state_dict(), "config": {"model": "TINY"}}, tmp_path / "pre.pt")
    for name in ("code", "english"):
        np.tile(np.arange(8, 60, dtype=np.uint16), 2000).tofile(tmp_path / f"{name}.bin")
    synth(tmp_path / "train.jsonl")
    cfg = f'''name = "t"
pretrained = "{tmp_path / 'pre.pt'}"
train = "{tmp_path / 'train.jsonl'}"
dev = "{tmp_path / 'train.jsonl'}"
pretrain_data = "{tmp_path}"
tokens_per_batch = 1024
epochs = 15
lr_muon = 0.02
lr_adam = 0.003
warmup_frac = 0.05
weight_decay = 0.0
grad_clip = 1.0
replay_every = 10
replay_weight = 0.1
seed = 0
'''
    (tmp_path / "ft.toml").write_text(cfg)
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(ROOT / "vmt/finetune.py"),
                        str(tmp_path / "ft.toml"), "--name", f"test_{tmp_path.name}"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    run_dir = Path(r.stdout.split("RUN_DIR ")[1].split()[0])
    out = tmp_path / "scores.jsonl"
    r2 = subprocess.run([sys.executable, str(ROOT / "vmt/score.py"), str(run_dir / "final.pt"), str(tmp_path / "train.jsonl"), str(out)],
                        cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-3000:]
    scores = {s["example_id"]: s["p"] for s in map(json.loads, out.read_text().splitlines())}
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    acc = sum((scores[x["example_id"]] > 0.5) == bool(x["label"]) for x in rows) / len(rows)
    assert acc >= 0.9, acc
