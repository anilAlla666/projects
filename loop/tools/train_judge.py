"""Step 13: QLoRA training of our judge model (Qwen 3.8 27B) on the A100 40 GB.

Each conversation from tools/sft_build.py is cut into examples of up to 8 assistant turns. An example is the
conversation up to the end of its segment, rendered exactly as the local judge sees it at test time (the model's chat
template with the bash tool, older outputs cut by writer_api.trimmed). Loss is on that segment's assistant turns only:
reasoning, text and tool calls for our model's own rounds; text and tool calls for Opus's (the API returns no Opus
reasoning, so its empty think block is not trained on). Examples longer than 32k tokens are split further.

Memory (measured 11 Oct): 4-bit base 16.5 GiB; at 32k tokens the step fits at 32 GiB only with gradient
checkpointing, the checkpoints offloaded to host RAM and the feed-forward layers tiled into 4k-token slices.

    python train_judge.py --data sft.jsonl --out adapter_dir [--epochs 2 --lr 1e-4]
"""
from __future__ import annotations

import argparse
import contextlib
import os
import json
import math
import random
import sys
import time

import torch
import torch.utils.checkpoint as ck

KEEP_FULL, TRIM_TO, MAX_LEN, SEG = 4, 400, 32768, 4
VIEW = {"cut": TRIM_TO, "max_len": MAX_LEN, "seg": SEG}  # step 15b: --view outputs renders every command and output whole (only old reasoning dropped)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_a", "in_proj_b", "out_proj"]


def trimmed(msgs: list, keep: int = KEEP_FULL, cut: int = TRIM_TO) -> list:
    """The conversation as a local model sees it (step 13, amendment 5): the most recent turns are whole, and every
    turn before a block boundary is shortened - its command output and its command cut to `cut` characters, its
    reasoning dropped. Boundaries move in blocks of `keep` tool calls, so the start of the conversation (the server's
    cached prefix) changes once every `keep` calls and `keep` to 2*`keep`-1 recent outputs stay whole. Measured on a
    50-turn round: the longest context fell from 78,663 to 28,795 tokens, inside the 32k the GPU can train on.
    Training renders through this same function."""
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    cut_n = ((len(tool_idx) - keep) // keep) * keep if keep and len(tool_idx) > keep else 0
    old = set(tool_idx[:cut_n])
    bound = tool_idx[cut_n - 1] if cut_n else -1
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content") or ""
        if i in old and len(c) > cut:
            m = m | {"content": c[:cut] + f"\n... [older output cut to {cut} of {len(c)} characters; run the command "
                                          f"again to see it]"}
        if i < bound and m.get("role") == "assistant":
            m = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
            calls = []
            for x in m.get("tool_calls") or []:
                try:
                    cmd = json.loads(x["function"].get("arguments") or "{}").get("command", "")
                except (json.JSONDecodeError, AttributeError):
                    cmd = str(x["function"].get("arguments"))
                if len(cmd) > cut:
                    cmd = cmd[:cut] + "\n# ... [older command cut]"
                calls.append({**x, "function": {**x["function"], "arguments": json.dumps({"command": cmd})}})
            if calls:
                m["tool_calls"] = calls
        out.append(m)
    return out


def for_template(m: dict) -> dict:
    """The message as the chat template wants it: tool-call arguments as a mapping, reasoning as reasoning_content."""
    m = dict(m)
    r = m.pop("reasoning", None)
    if r and not m.get("reasoning_content"):
        m["reasoning_content"] = r
    if m.get("tool_calls"):
        calls = []
        for c in m["tool_calls"]:
            f = dict(c["function"])
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"] or "{}")
                except json.JSONDecodeError:
                    f["arguments"] = {"command": f["arguments"]}
            calls.append({"type": "function", "id": c.get("id", ""), "function": f})
        m["tool_calls"] = calls
    if m.get("content") is None:
        m["content"] = ""
    return m


def examples(conv: dict, tok) -> list[tuple[list[int], list[int]]]:
    msgs, own = conv["messages"], conv["source"] in ("own", "teacher")  # step 19: the teacher's reasoning is trained
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    segs = [a_idx[i:i + VIEW["seg"]] for i in range(0, len(a_idx), VIEW["seg"])]
    out = []
    while segs:
        seg = segs.pop(0)
        end = seg[-1] + 1
        sys_ = [{"role": "system", "content": conv["system"]}] if conv.get("system") else []  # step 18: no system turn
        ctx = sys_ + [for_template(m) for m in trimmed(msgs[:end], cut=VIEW["cut"])]
        text = tok.apply_chat_template(ctx, tools=conv["tools"], tokenize=False, reasoning_effort="medium")
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        if len(enc["input_ids"]) > VIEW["max_len"]:
            if len(seg) > 1:
                h = len(seg) // 2
                segs[:0] = [seg[:h], seg[h:]]
            continue  # a single turn over the limit is dropped
        # the assistant blocks in the rendered text, in order; train on this segment's
        spans, pos = [], 0
        while True:
            s = text.find("<|im_start|>assistant\n", pos)
            if s < 0:
                break
            e = text.find("<|im_end|>", s)
            spans.append((s + len("<|im_start|>assistant\n"), e + len("<|im_end|>")))
            pos = e
        mine = spans[len(spans) - len(seg):]
        ranges = []
        for s, e in mine:
            if not own:  # Opus: no reasoning was returned; train from the end of the empty think block
                k = text.find("</think>\n\n", s, e)
                s = k + len("</think>\n\n") if k >= 0 else s
            ranges.append((s, e))
        labels = [t if any(s <= a < e for s, e in ranges) else -100
                  for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
        if any(l != -100 for l in labels):
            out.append((enc["input_ids"], labels))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/ubuntu/models/qwen38-27b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dry", action="store_true", help="build and count the examples only")
    ap.add_argument("--no-offload", action="store_true", help="keep saved activations on the GPU (80 GB GPUs); "
                    "segments longer than 32,768 tokens are still offloaded")
    ap.add_argument("--view", choices=["trim", "outputs"], default="trim")
    ap.add_argument("--init-adapter", default=None, help="step 18: continue from a saved adapter (training keys)")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--seg", type=int, default=SEG, help="assistant turns trained per example")
    ap.add_argument("--neg-cap", type=float, default=None,
                    help="step 17b: a negative-advantage segment stops pushing once its loss passes this")
    a = ap.parse_args()
    VIEW["max_len"], VIEW["seg"] = a.max_len, a.seg
    VIEW["cut"] = 10**9 if a.view == "outputs" else TRIM_TO
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    convs = [json.loads(l) for l in open(a.data)]
    exs = []
    for c in convs:
        exs += [(c["source"], c["task"], x, float(c.get("weight", 1.0))) for x in examples(c, tok)]  # step 17: RL advantage
    n_tok = sum(len(x[2][0]) for x in exs)
    n_lab = sum(sum(l != -100 for l in x[2][1]) for x in exs)
    print(json.dumps({"conversations": len(convs), "examples": len(exs), "tokens": n_tok, "trained_tokens": n_lab,
                      "by_source": {s: sum(1 for x in exs if x[0] == s) for s in ("opus", "own")},
                      "max_len": max(len(x[2][0]) for x in exs),
                      "weights": {"positive": sum(1 for x in exs if x[3] > 0), "negative": sum(1 for x in exs if x[3] < 0)}}),
          flush=True)
    if a.dry:
        return
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head", "visual", "mtp"])
    # data parallel under torchrun (step 14, amendment 1): each rank takes its share of every accumulation group and
    # the LoRA gradients are summed across ranks, which is the same update as one GPU accumulating the whole group
    ws, rk, lrk = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if ws > 1:
        import torch.distributed as dist
        torch.cuda.set_device(lrk)
        dist.init_process_group("nccl")
    m = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, dtype=torch.bfloat16,
                                             device_map={"": lrk})
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=TARGETS))
    if a.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(m, load_file(f"{a.init_adapter}/adapter_model.safetensors"))
        assert not [k for k in res.unexpected_keys if "lora" in k], res.unexpected_keys[:5]

    def tiled(fwd, chunk=4096):
        def f(x):
            if x.shape[1] <= chunk:
                return fwd(x)
            return torch.cat([ck.checkpoint(fwd, x[:, i:i + chunk], use_reentrant=False)
                              for i in range(0, x.shape[1], chunk)], 1)
        return f
    for n, mod in m.named_modules():
        if n.endswith(".mlp") and hasattr(mod, "gate_proj"):
            mod.forward = tiled(mod.forward)
    base = m.base_model.model
    inner = base.model.language_model if hasattr(base.model, "language_model") else base.model
    head, ce = base.lm_head, LigerFusedLinearCrossEntropyLoss()
    params = [p for p in m.parameters() if p.requires_grad]
    if ws > 1:  # LoRA's random init must be the same on every rank
        for p in params:
            dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps = math.ceil(len(exs) * a.epochs / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng, order, t0, step, seen = random.Random(0), [], time.time(), 0, 0
    total = int(len(exs) * a.epochs)
    while len(order) < total:
        idx = list(range(len(exs)))
        rng.shuffle(idx)
        order += idx
    order = order[:total]
    def offload(n):
        if a.no_offload and n <= 32768:
            return contextlib.nullcontext()
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    for g0 in range(0, len(order), a.accum):
        grp = range(g0, min(g0 + a.accum, len(order)))
        for j, k in enumerate(grp):
            if j % ws != rk:
                continue
            ids, labels = exs[order[k]][2]
            w = exs[order[k]][3]
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor(labels[1:], device="cuda")
            with offload(len(ids)):
                h = inner(input_ids=x).last_hidden_state
                ce_val = ce(head.weight, h[0, :-1].contiguous(), y)
                if w < 0 and a.neg_cap is not None:  # bounded unlikelihood: no reward for driving a loss to infinity
                    ce_val = torch.clamp(ce_val, max=a.neg_cap)
                loss = ce_val / a.accum * w
            loss.backward()
            seen += len(ids)
        if ws > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        el = time.time() - t0
        if rk == 0:
            print(json.dumps({"step": step, "of": steps, "loss": round(loss.item() * a.accum, 4), "lr": sched.get_last_lr()[0],
                              "tok_s": round(seen * ws / el), "eta_min": round(el / (grp[-1] + 1) * (len(order) - grp[-1] - 1) / 60, 1),
                              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
            for k in grp:
                if (k + 1) % len(exs) == 0:
                    m.save_pretrained(f"{a.out}/epoch{(k + 1) // len(exs)}")
    if rk == 0:
        m.save_pretrained(a.out)
        print(json.dumps({"DONE": a.out, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    if ws > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
