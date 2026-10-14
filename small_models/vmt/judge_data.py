"""Serialised judge examples: length-bucketed, class-balanced batches under a token budget."""
from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F

PAD = 1


class JudgeDataset:
    def __init__(self, path, weights: dict | None = None):
        self.rows = [json.loads(l) for l in open(path)]
        for r in self.rows:
            r["weight"] = (weights or {}).get(r["example_id"], 1.0)

    def __len__(self):
        return len(self.rows)


def _collate(rows, device):
    T = max(len(r["tokens"]) for r in rows)
    B = len(rows)
    tok = torch.full((B, T), PAD, dtype=torch.long)
    seg = torch.zeros((B, T), dtype=torch.long)
    line_mask = torch.zeros((B, T), dtype=torch.bool)
    code_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["tokens"])
        tok[i, :n] = torch.tensor(r["tokens"])
        seg[i, :n] = torch.tensor(r["segments"])
        line_mask[i, r["line_starts"]] = True
        if r["code_line_starts"]:
            code_mask[i, r["code_line_starts"]] = True
    t = lambda xs: torch.tensor(xs)  # noqa: E731
    return {"tokens": tok.to(device), "segments": seg.to(device), "line_mask": line_mask.to(device),
            "code_mask": code_mask.to(device),
            "verdict_pos": t([r["verdict_pos"] for r in rows]).to(device),
            "pointer_pos": t([r["line_starts"][r["pointer"]] for r in rows]).to(device),
            "bug_pos": t([r["code_line_starts"][min(r["bug_line"], len(r["code_line_starts"]) - 1)] if r["code_line_starts"] else 0 for r in rows]).to(device),
            "has_code": t([bool(r["code_line_starts"]) for r in rows]).to(device),
            "bucket": t([r["hidden_bucket"] for r in rows]).to(device),
            "label": t([float(r["label"]) for r in rows]).to(device),
            "weight": t([float(r["weight"]) for r in rows]).to(device)}


def batches(ds: JudgeDataset, tokens_per_batch: int, seed: int, rank: int = 0, world: int = 1, device="cuda"):
    """One epoch. Examples are sorted into length buckets, each batch draws from both classes alternately, and
    batch i goes to rank i % world."""
    rng = random.Random(seed)
    pos = [r for r in ds.rows if r["label"] == 1]
    neg = [r for r in ds.rows if r["label"] == 0]
    for xs in (pos, neg):
        rng.shuffle(xs)
    merged = []
    while pos or neg:
        if pos:
            merged.append(pos.pop())
        if neg:
            merged.append(neg.pop())
    chunks = [merged[i : i + 256] for i in range(0, len(merged), 256)]
    out = []
    for ch in chunks:
        ch.sort(key=lambda r: len(r["tokens"]))
        cur = []
        for r in ch:
            if cur and (len(cur) + 1) * max(len(r["tokens"]), max(len(c["tokens"]) for c in cur)) > tokens_per_batch:
                out.append(cur)
                cur = []
            cur.append(r)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    if world > 1:  # every rank must take the same number of steps, or the gradient average waits forever
        out = out[: len(out) - len(out) % world]
    for i, b in enumerate(out):
        if i % world == rank:
            yield _collate(b, device)


def judge_losses(model, heads, b, w=(1.0, 0.5, 0.25, 0.25)):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(b["tokens"], b["segments"])
        v = model.verdict_logit(h, b["verdict_pos"]).float()
        p = model.pointer_logits(h, b["verdict_pos"], b["line_mask"]).float()
        k = heads.bucket_logits(h, b["verdict_pos"]).float()
        g = heads.bug_logits(h, b["verdict_pos"], b["code_mask"]).float()
    wt = b["weight"] / b["weight"].mean()
    verdict = (F.binary_cross_entropy_with_logits(v, b["label"], reduction="none") * wt).mean()
    pointer = (F.cross_entropy(p, b["pointer_pos"], reduction="none") * wt).mean()
    bucket = (F.cross_entropy(k, b["bucket"], reduction="none") * wt).mean()
    bug_l = F.cross_entropy(g, b["bug_pos"], reduction="none") * b["has_code"].float()
    bug = (bug_l * wt).sum() / b["has_code"].float().sum().clamp(min=1)
    total = w[0] * verdict + w[1] * pointer + w[2] * bucket + w[3] * bug
    acc = ((v > 0).float() == b["label"]).float().mean()
    return total, {"verdict": verdict.item(), "pointer": pointer.item(), "bucket": bucket.item(), "bug": bug.item(), "acc": acc.item()}
