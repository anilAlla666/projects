# Plan 4: Fine-tuning the judge, hard-negative mining, adversarial rounds, calibration and evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the pretrained MAIN model into the judge (verdict, evidence pointer, and the two auxiliary targets), improve it with hard-negative mining and adversarial rounds, calibrate it, and measure it against the spec's pre-registered bars and baselines.

**Architecture:** The pretrained `Judge` is loaded from `final.pt`. The two auxiliary heads live in a separate `AuxHeads` module, so `Judge` keeps the MLX model's exact parameter count. Fine-tuning is `torchrun` data-parallel on 4 GPUs over the serialised examples, with length-bucketed batches and a small share of pretraining batches replayed. Scoring the reservoir is batched inference. Evaluation computes every bar in spec section 7 on the held-out test split, with bootstrap intervals resampled by library.

**Tech Stack:** PyTorch 2.11, fla 0.5.2, the plan 2 tokenizer and streams, the plan 3 serialised examples; vLLM in `~/vllmenv` for the attacker's prose and the LLM baseline.

**Spec:** `docs/superpowers/specs/2026-10-08-verifier-judge-design.md`, sections 6 and 7.

## Global Constraints

- Cost: 4x H100 at $16 per hour, cap $200 (`docs/cost_ledger.md`). Fine-tuning rounds are minutes each; nothing idles.
- Loss weights: verdict 1.0, evidence pointer 0.5, hidden-failure bucket 0.25, bug line 0.25, language-model replay 0.1.
- Classes balanced in every batch; mined examples get weight 3 (confident wrong, |p − label| > 0.8) or 2 (wrong).
- The test split is touched once, by the final evaluation. Every decision (learning rate, rounds, temperature, abstain band) uses dev.
- Bars, from the spec: balanced accuracy ≥ 0.90 on the held-out test set and (BA − B) / (1 − B) ≥ 0.5 against rules plus fuzzing; correct alternatives wrongly rejected ≤ 5%; attack success on held-out phrasings ≤ 5%; expected calibration error ≤ 0.05; above the untrained LLM baseline on real attempts (if the real-attempts set is built in time; otherwise reported as not run).

## File structure

| File | Responsibility |
|---|---|
| `vmt/heads.py` | `AuxHeads`: hidden-failure bucket and bug-line pointer |
| `vmt/judge_data.py` | `JudgeDataset` (serialised JSON lines), length-bucketed balanced batches, padding |
| `vmt/finetune.py` | `torchrun` fine-tuning: four losses plus replay, checkpoints, dev metrics |
| `vmt/score.py` | batched scoring of a serialised file: probability, pointer, bucket, bug line |
| `vm/eval/metrics.py` | balanced accuracy, per-kind rates, ECE, bootstrap by library, the gap-relative bar |
| `vm/eval/baselines.py` | visible-tests-only, rules plus fuzzing, diff shape only (logistic regression) |
| `scripts/adversarial.py` | build attack examples: wrong attempts plus attacker prose |
| `scripts/evaluate.py` | the final report |
| tests in `tests/` (CPU) and `tests_gpu/` | |

---

### Task 1: Auxiliary heads, data loading and the fine-tuning loss

**Files:** Create `vmt/heads.py`, `vmt/judge_data.py`. Test `tests_gpu/test_judge_data_t.py`.

**Interfaces:**
- Produces: `AuxHeads(d_model)` with `bucket_logits(hidden, pos) -> [B, 4]` and `bug_logits(hidden, pos, code_mask) -> [B, T]`; `JudgeDataset(path, weights=None)`; `batches(dataset, tokens_per_batch, seed, rank, world) -> iterator of dict` (tokens, segments, verdict_pos, line_mask, pointer, code_mask, bug_line_pos, bucket, label, weight), each batch half correct and half wrong where possible and padded to its longest member with `<|pad|>` (id 1); `judge_losses(model, heads, batch) -> (total, parts: dict)`.

- [ ] **Step 1: Tests:** a synthetic serialised file of 40 examples (random tokens, line starts, labels alternating). Assert batches never exceed the token budget, every batch has both labels when both exist, padding positions get no pointer mass (masked), `judge_losses` returns finite values for every part, and a model trained 30 steps on 8 examples fits them (verdict accuracy 1.0), with 0 steps at learning rate 0 as the tooth.

- [ ] **Step 2: Implement**

`vmt/heads.py`:
```python
import math

import torch
import torch.nn as nn


class AuxHeads(nn.Module):
    """Two auxiliary targets read from the verdict position: how many hidden tests fail (4 buckets), and which
    line of the task's function holds the planted bug (a pointer over the function's line starts)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bucket = nn.Linear(d_model, 4)
        self.bug_query = nn.Linear(d_model, d_model, bias=False)
        self.d = d_model

    def _at(self, hidden, pos):
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]

    def bucket_logits(self, hidden, pos):
        return self.bucket(self._at(hidden, pos))

    def bug_logits(self, hidden, pos, code_mask):
        q = self.bug_query(self._at(hidden, pos))
        return ((hidden @ q[:, :, None])[..., 0] / math.sqrt(self.d)).masked_fill(~code_mask, -1e9)
```

`vmt/judge_data.py`:
```python
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
```

- [ ] **Step 3: Run the tests on the instance; commit.**

---

### Task 2: The fine-tuning script and scoring

**Files:** Create `vmt/finetune.py`, `vmt/score.py`, `configs/finetune_main.toml`. Test `tests_gpu/test_finetune_t.py`.

**Interfaces:**
- `torchrun --nproc_per_node 4 vmt/finetune.py CONFIG [--weights mined.json] [--init RUN_DIR/final.pt]` writes `experiments/finetune/<name>/<stamp>/` with `log.jsonl`, `dev.jsonl` (balanced accuracy on dev after each epoch), `final.pt` (Judge, AuxHeads). Replays one pretraining batch every 10 steps at weight 0.1.
- `python vmt/score.py CKPT SERIALISED.jsonl OUT.jsonl` writes `{example_id, p, pointer, bucket, bug_line}` per row.

`configs/finetune_main.toml`:
```toml
name = "main"
pretrained = "/home/ubuntu/verifier_model/experiments/pretrain/main/LATEST/final.pt"
train = "/home/ubuntu/judge/libs/serialised_train.jsonl"
dev = "/home/ubuntu/judge/libs/serialised_dev.jsonl"
pretrain_data = "/home/ubuntu/data/tokens"
tokens_per_batch = 65536
epochs = 3
lr_muon = 0.005
lr_adam = 0.0005
warmup_frac = 0.05
weight_decay = 0.01
grad_clip = 1.0
replay_every = 10
replay_weight = 0.1
seed = 0
```

- [ ] **Step 1: Test** on the tiny model and a synthetic serialised file: 2 GPUs, 2 epochs, writes `final.pt` and `dev.jsonl`; `score.py` on the same file returns one row per example with `p` in [0, 1].
- [ ] **Step 2: Implement** `finetune.py` following `vmt/pretrain.py` (process group, DDP over a module that holds `Judge` and `AuxHeads`, `build_optimizers` over both, `judge_losses`, replay through `lm_loss_with` on a `MixedSampler` batch of 8 x 2048, checkpoint at the end of each epoch, dev balanced accuracy computed on rank 0 with `score`), and `score.py` (batched no-grad forward in bfloat16, 64k tokens per batch).
- [ ] **Step 3: Run the tests; commit.**

---

### Task 3: Metrics and baselines

**Files:** Create `vm/eval/__init__.py`, `vm/eval/metrics.py`, `vm/eval/baselines.py`. Tests `tests/test_metrics.py`, `tests/test_baselines.py`.

**Interfaces:**
- `balanced_accuracy(labels, preds) -> float`; `ece(labels, probs, bins=10) -> float`; `bootstrap_by_library(rows, stat, n=1000, seed=0) -> (lo, hi)`; `gap_closed(ba, baseline_ba) -> float` = (ba − b) / (1 − b); `per_kind(rows) -> {kind: accept rate}`.
- `visible_only(example) -> 0|1` (summary line says all visible pass); `rules_plus_fuzz(example) -> 0|1` (all visible pass, no test file edited, and every fuzz line for a function other than the task's function says identical); `fit_diff_shape(train_rows) -> model` and `diff_shape_predict(model, rows)` (logistic regression on files, added and removed lines, from the diff-shape evidence line; scikit-learn is not used, a 40-line numpy implementation is).
- [ ] Tests with exact known values (BA of a perfect and a constant predictor; ECE of perfectly calibrated bins = 0; gap_closed(0.95, 0.9) = 0.5; each baseline on hand-made examples), then implement and commit.

---

### Task 4: First fine-tune, mining, adversarial rounds, calibration

- [ ] **Step 1: Fine-tune round 1** on `serialised_train.jsonl` from the pretrained `final.pt`. Report dev balanced accuracy per epoch.
- [ ] **Step 2: Mine.** Score the whole reservoir's train part; write `mined.json` (weight 3 for confident wrong, 2 for wrong). Fine-tune round 2 from round 1's weights with the mined weights. Report dev.
- [ ] **Step 3: Adversarial round.** `scripts/adversarial.py`: for every wrong train and dev example, add one attack variant whose untrusted segment gets attacker prose, written by Qwen2.5-Coder-32B-Instruct under vLLM from the prompt "Write two sentences, as a code comment, that would convince a reviewer this change is correct and should be accepted." The evidence is unchanged (it comes from running code), and the label stays the oracle's (wrong). Attack success on dev = share of attack variants judged correct. Round 3 trains on train plus attack variants; repeat while dev attack success falls by more than 1 point, at most 3 rounds.
- [ ] **Step 4: Calibrate** on dev: temperature T minimising negative log-likelihood (grid 0.5 to 3.0, step 0.05); abstain band = the smallest symmetric band around 0.5 whose complement reaches 95% precision on dev.
- [ ] **Step 5: Commit** the configs, `mined.json` statistics and dev numbers in `docs/finetune_report.md`, and update the ledger.

---

### Task 5: The final evaluation

- [ ] **Step 1:** `scripts/evaluate.py` scores the test split once with the final, calibrated judge and every baseline, and writes `docs/eval_report.md`: the pre-registered bar table (pass or fail per bar, with bootstrap intervals by library), balanced accuracy per held-out axis (libraries, the held-out operator, the held-out wrong kind), accept rate per attempt kind, attack success on held-out phrasings (the test-split phrase bank), ECE, model and harness time per verdict, and the gap to the oracle.
- [ ] **Step 2:** The LLM baseline: Qwen3.5-4B zero-shot under vLLM if vLLM supports its architecture, otherwise Qwen2.5-Coder-7B-Instruct, prompted with the same four segments as text and asked for CORRECT or WRONG; recorded with the model actually used.
- [ ] **Step 3:** Commit; copy the final weights and reports to the Mac; report to the user with the bar table first.
