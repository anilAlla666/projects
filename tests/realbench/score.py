"""Realbench scoring utilities — quality + throughput metrics for CIPHER vs baseline.

Quality metrics (per-prompt):
  - perplexity: exp(mean(NLL)) over generated tokens, computed against the
    BASELINE model's logits (i.e. how likely is CIPHER's generation under
    the baseline model)
  - kl_divergence: mean KL(baseline_logits || cipher_logits) per position
  - top_k_agreement: fraction of positions where baseline's top-k contains
    cipher's chosen token (for k=1, 5, 10, 50)

Throughput metrics:
  - tokens_per_s
  - mean_watts (from sampled nvidia-smi)
  - tok_per_w

A row in the scoring table looks like:
  {prompt_id, category, quality.{ppl, kl, topk}, throughput.{tps, watts, tpw}, ops.{...}}
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QualityMetrics:
    perplexity_under_baseline: float = 0.0
    kl_divergence_mean: float = 0.0
    kl_divergence_p95: float = 0.0
    top1_agreement: float = 0.0
    top5_agreement: float = 0.0
    top10_agreement: float = 0.0
    top50_agreement: float = 0.0
    n_compared_positions: int = 0


@dataclass
class ThroughputMetrics:
    tokens_per_s: float = 0.0
    mean_watts: float = 0.0
    tok_per_w: float = 0.0
    p50_burst_us: float = 0.0
    p99_burst_us: float = 0.0
    elapsed_s: float = 0.0
    n_tokens: int = 0


@dataclass
class PromptResult:
    prompt_id: str
    category: str
    decoded_text: str = ""
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    ops: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Quality metric primitives — operate on torch tensors, no model calls
# ----------------------------------------------------------------------

def per_position_kl(p_logits, q_logits):
    """KL(P||Q) per position. Inputs are [seq_len, vocab] tensors. Returns
    a 1D tensor of length seq_len with KL divergence at each position."""
    import torch
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1)


def per_position_topk_agreement(baseline_logits, cipher_token_ids, ks=(1, 5, 10, 50)):
    """For each position, was the cipher-chosen token in baseline's top-k?
    baseline_logits: [seq_len, vocab]
    cipher_token_ids: [seq_len] of token ids actually chosen by cipher
    Returns dict {k: fraction}."""
    import torch
    out = {}
    seq_len = cipher_token_ids.shape[0]
    if seq_len == 0:
        for k in ks:
            out[k] = 0.0
        return out
    for k in ks:
        topk = baseline_logits.topk(k, dim=-1).indices  # [seq_len, k]
        agree = (topk == cipher_token_ids.unsqueeze(-1)).any(dim=-1)
        out[k] = float(agree.float().mean().item())
    return out


def perplexity_under_baseline(baseline_logits_at_t, cipher_chosen_t):
    """Perplexity of CIPHER's generated tokens under the baseline model's
    next-token distribution. Lower = CIPHER's choices align with what
    baseline would consider likely.
    baseline_logits_at_t: [seq_len, vocab]
    cipher_chosen_t: [seq_len] (the token cipher actually picked at each step)
    """
    import torch
    log_probs = torch.log_softmax(baseline_logits_at_t.float(), dim=-1)
    nll = -log_probs.gather(-1, cipher_chosen_t.unsqueeze(-1)).squeeze(-1)
    return float(math.exp(nll.mean().item()))


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(rows: list[PromptResult]) -> dict:
    """Average quality + throughput across rows; return a single summary dict."""
    if not rows:
        return {"n_rows": 0}
    def _avg(xs): return statistics.mean(xs) if xs else 0.0

    return {
        "n_prompts": len(rows),
        "quality": {
            "ppl_mean": _avg([r.quality.perplexity_under_baseline for r in rows
                              if r.quality.perplexity_under_baseline > 0]),
            "kl_mean": _avg([r.quality.kl_divergence_mean for r in rows]),
            "kl_p95": _avg([r.quality.kl_divergence_p95 for r in rows]),
            "top1_agreement": _avg([r.quality.top1_agreement for r in rows]),
            "top5_agreement": _avg([r.quality.top5_agreement for r in rows]),
            "top10_agreement": _avg([r.quality.top10_agreement for r in rows]),
            "top50_agreement": _avg([r.quality.top50_agreement for r in rows]),
        },
        "throughput": {
            "tok_per_s_mean": _avg([r.throughput.tokens_per_s for r in rows]),
            "mean_watts": _avg([r.throughput.mean_watts for r in rows]),
            "tok_per_w_mean": _avg([r.throughput.tok_per_w for r in rows]),
            "elapsed_s_total": sum(r.throughput.elapsed_s for r in rows),
            "n_tokens_total": sum(r.throughput.n_tokens for r in rows),
        },
        "by_category": {},
    }


def write_jsonl(rows: list[PromptResult], path: str):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def render_markdown_table(baseline_summary: dict, cipher_summary: dict) -> str:
    """Two-column comparison table, baseline vs CIPHER, for the report."""
    lines = []
    lines.append("| metric | baseline | CIPHER | ratio |")
    lines.append("|---|---:|---:|---:|")

    def _row(label, b, c):
        ratio = (c / b) if b not in (0, 0.0) else 0.0
        return f"| {label} | {b:.4f} | {c:.4f} | {ratio:.3f}× |"

    bq = baseline_summary["quality"]; cq = cipher_summary["quality"]
    bt = baseline_summary["throughput"]; ct = cipher_summary["throughput"]
    lines.append(_row("perplexity (under baseline)", bq["ppl_mean"], cq["ppl_mean"]))
    lines.append(_row("KL divergence mean", bq["kl_mean"], cq["kl_mean"]))
    lines.append(_row("KL divergence p95", bq["kl_p95"], cq["kl_p95"]))
    lines.append(_row("top-1 agreement", bq["top1_agreement"], cq["top1_agreement"]))
    lines.append(_row("top-5 agreement", bq["top5_agreement"], cq["top5_agreement"]))
    lines.append(_row("top-10 agreement", bq["top10_agreement"], cq["top10_agreement"]))
    lines.append(_row("top-50 agreement", bq["top50_agreement"], cq["top50_agreement"]))
    lines.append(_row("tokens/s (mean per prompt)", bt["tok_per_s_mean"], ct["tok_per_s_mean"]))
    lines.append(_row("mean watts", bt["mean_watts"], ct["mean_watts"]))
    lines.append(_row("tok/W (mean per prompt)", bt["tok_per_w_mean"], ct["tok_per_w_mean"]))
    return "\n".join(lines)
