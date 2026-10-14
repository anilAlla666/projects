"""Fine-tune the pretrained judge under torchrun: verdict, evidence pointer, hidden-failure bucket and bug line,
with a small share of pretraining batches replayed so it keeps reading code and English.

    torchrun --nproc_per_node 4 vmt/finetune.py CONFIG [--weights mined.json] [--init CKPT] [--name NAME]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.model import config as model_config  # noqa: E402
from vmt.data import MixedSampler, TokenStream  # noqa: E402
from vmt.heads import AuxHeads  # noqa: E402
from vmt.judge_data import JudgeDataset, batches, judge_losses  # noqa: E402
from vmt.model import Judge  # noqa: E402
from vmt.train import build_optimizers, lm_loss_with  # noqa: E402


class Both(nn.Module):
    def __init__(self, judge, heads):
        super().__init__()
        self.judge, self.heads = judge, heads


def average_gradients(model: nn.Module, world: int) -> None:
    """Data parallelism by hand: every rank starts from the same weights and seed, so averaging gradients after
    each backward pass keeps them identical. The heads are used outside a single forward call, which a DDP
    wrapper would track as unused."""
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        distributed.all_reduce(p.grad, op=distributed.ReduceOp.AVG)


def load_judge(path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = getattr(model_config, state["config"]["model"]) if "model" in state["config"] else model_config.MAIN
    judge = Judge(cfg).to(device)
    judge.load_state_dict(state["model"] if "model" in state else state["judge"])
    heads = AuxHeads(cfg.d_model).to(device)
    if "heads" in state:
        heads.load_state_dict(state["heads"])
    return judge, heads, state["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--init", type=Path)
    ap.add_argument("--name")
    a = ap.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    distributed.init_process_group("nccl")
    rank, world = distributed.get_rank(), distributed.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(cfg["seed"])
    judge, heads, pre_cfg = load_judge(a.init or cfg["pretrained"])
    model = Both(judge, heads)
    weights = json.loads(a.weights.read_text()) if a.weights else None
    train = JudgeDataset(cfg["train"], weights)
    est_batches = max(1, sum(len(r["tokens"]) for r in train.rows) // cfg["tokens_per_batch"] // world)
    total_steps = est_batches * cfg["epochs"] + 1
    opts, scheds = build_optimizers(model, cfg["lr_muon"], cfg["lr_adam"], total_steps,
                                    max(1, int(total_steps * cfg["warmup_frac"])), cfg["weight_decay"])
    for p in model.parameters():  # identical start on every rank
        distributed.broadcast(p.data, src=0)
    data = Path(cfg["pretrain_data"])
    replay = MixedSampler({n: TokenStream(data / f"{n}.bin") for n in ("code", "english")}, {"code": 0.6, "english": 0.4}, cfg["seed"], rank)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("/home/ubuntu/verifier_model/experiments/finetune") / (a.name or cfg["name"]) / stamp
    if rank == 0:
        run_dir.mkdir(parents=True)
        (run_dir / "config.toml").write_text(a.config.read_text())
        log = open(run_dir / "log.jsonl", "a")

    step = 0
    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        for b in batches(train, cfg["tokens_per_batch"], cfg["seed"] + epoch, rank, world):
            loss, parts = judge_losses(judge, heads, b)
            if step % cfg["replay_every"] == 0:
                rb = replay.batch(8, 2048).cuda()
                loss = loss + cfg["replay_weight"] * lm_loss_with(judge, judge, rb, torch.zeros_like(rb))
            loss.backward()
            average_gradients(model, world)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            for s in scheds:
                s.step()
            step += 1
            if rank == 0 and step % 10 == 0:
                log.write(json.dumps({"epoch": epoch, "step": step, "loss": round(loss.item(), 4), **{k: round(v, 4) for k, v in parts.items()}}) + "\n")
                log.flush()
            if step >= total_steps:
                break
        distributed.barrier()
        if rank == 0:
            torch.save({"judge": judge.state_dict(), "heads": heads.state_dict(), "config": pre_cfg, "epoch": epoch},
                       run_dir / f"epoch_{epoch}.pt")
            log.write(json.dumps({"epoch_done": epoch, "seconds": round(time.perf_counter() - t0, 1)}) + "\n")
            log.flush()
    if rank == 0:
        torch.save({"judge": judge.state_dict(), "heads": heads.state_dict(), "config": pre_cfg}, run_dir / "final.pt")
        print("RUN_DIR", run_dir, flush=True)
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
