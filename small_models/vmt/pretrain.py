"""Pretraining under torchrun: data-parallel, bfloat16 autocast, Muon and AdamW, held-out validation per source,
checkpoints and resume. The last long_frac of tokens use long_seq with the same tokens per step.

    torchrun --nproc_per_node 4 vmt/pretrain.py configs/pretrain_main.toml [--resume RUN_DIR --extra-steps N]
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
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.model import config as model_config  # noqa: E402
from vmt.data import MixedSampler, TokenStream  # noqa: E402
from vmt.model import Judge  # noqa: E402
from vmt.train import build_optimizers, lm_loss_with  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--extra-steps", type=int, default=0)
    a = ap.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    distributed.init_process_group("nccl")
    rank, world = distributed.get_rank(), distributed.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(cfg["seed"])

    mcfg = getattr(model_config, cfg["model"])
    model = Judge(mcfg).cuda()
    # the verdict and pointer heads are trained later; frozen here so data-parallel sees no unused parameters
    for p in [*model.verdict_head.parameters(), *model.pointer_query.parameters()]:
        p.requires_grad_(False)
    tokens_per_step = cfg["batch_per_gpu"] * cfg["seq"] * world
    total_steps = cfg["tokens"] // tokens_per_step + a.extra_steps
    long_from = int(total_steps * (1 - cfg["long_frac"]))
    opts, scheds = build_optimizers(model, cfg["lr_muon"], cfg["lr_adam"], total_steps,
                                    max(1, int(total_steps * cfg["warmup_frac"])), cfg["weight_decay"])
    step = 0
    if a.resume:
        run_dir = a.resume
        ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        state = torch.load(ckpts[-1], map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(scheds, state["scheds"]):
            sc.load_state_dict(s)
        step = state["step"]
        total_steps = step + a.extra_steps if a.extra_steps else total_steps
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(cfg["out_dir"]) / cfg["name"] / stamp
        if rank == 0:
            run_dir.mkdir(parents=True)
            (run_dir / "config.toml").write_text(a.config.read_text())
    ddp = DDP(model, device_ids=[local])
    run_model = torch.compile(ddp) if cfg["compile"] else ddp

    data = Path(cfg["data_dir"])
    train = MixedSampler({n: TokenStream(data / f"{n}.bin") for n in cfg["sources"]}, cfg["sources"], cfg["seed"] + step, rank)
    vals = {n: MixedSampler({n: TokenStream(data / f"{n}_val.bin")}, {n: 1.0}, 12345, rank) for n in cfg["sources"]}
    log = open(run_dir / "log.jsonl", "a") if rank == 0 else None
    t0, tok0 = time.perf_counter(), 0

    def save(name):
        if rank == 0:
            torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                        "scheds": [s.state_dict() for s in scheds], "step": step, "config": cfg}, run_dir / name)

    while step < total_steps:
        seq = cfg["long_seq"] if step >= long_from else cfg["seq"]
        bsz = max(1, cfg["batch_per_gpu"] * cfg["seq"] // seq)
        batch = train.batch(bsz, seq).cuda(non_blocking=True)
        seg = torch.zeros_like(batch)
        loss = lm_loss_with(run_model, model, batch, seg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        for s in scheds:
            s.step()
        step += 1
        tok0 += bsz * seq * world
        if step % cfg["log_every"] == 0 or step == total_steps:
            lv = loss.detach()
            distributed.all_reduce(lv, op=distributed.ReduceOp.AVG)
            if rank == 0:
                el = time.perf_counter() - t0
                log.write(json.dumps({"step": step, "tokens": step * tokens_per_step, "loss": round(lv.item(), 4),
                                      "lr_muon": scheds[0].get_last_lr()[0], "seq": seq,
                                      "tokens_per_s": round(tok0 / el)}) + "\n")
                log.flush()
                t0, tok0 = time.perf_counter(), 0
        if step % cfg["val_every"] == 0 or step == total_steps:
            model.eval()
            res = {}
            with torch.no_grad():
                for n, v in vals.items():
                    ls = [lm_loss_with(model, model, v.batch(max(1, bsz // 4), cfg["seq"]).cuda(), torch.zeros(max(1, bsz // 4), cfg["seq"] + 1, dtype=torch.long, device="cuda"))
                          for _ in range(cfg["val_batches"])]
                    x = torch.stack(ls).mean()
                    distributed.all_reduce(x, op=distributed.ReduceOp.AVG)
                    res[n] = round(x.item(), 4)
            model.train()
            if rank == 0:
                with open(run_dir / "val.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, **res}) + "\n")
        if step % cfg["ckpt_every"] == 0:
            save(f"ckpt_{step}.pt")
    save("final.pt")
    if rank == 0:
        save(f"ckpt_{step}.pt")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
