"""Train the requirement model: LoRA on an open instruct model, loss on the target requirement list only (prompt
tokens masked). One A100 40GB is enough for a 7B model with gradient checkpointing.

    python gpu/train_req.py DATA_DIR OUT_DIR [--model Qwen/Qwen2.5-7B-Instruct] [--max-len 8192] [--epochs 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-adapter", type=Path, help="continue training this LoRA adapter instead of starting a new one")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in (a.data / "train.jsonl").read_text().splitlines()]
    feats = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=True)
        if not isinstance(p, list):
            p = p["input_ids"]
        c = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > a.max_len:
            continue
        feats.append({"input_ids": p + c, "labels": [-100] * len(p) + c})

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(batch), n), pad)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args = TrainingArguments(output_dir=str(a.out), per_device_train_batch_size=1, gradient_accumulation_steps=8,
                             num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                             bf16=True, logging_steps=10, save_strategy="no", report_to=[], ddp_find_unused_parameters=False,
                             dataloader_num_workers=2, remove_unused_columns=False)
    tr = Trainer(model=model, args=args, train_dataset=feats, data_collator=collate)
    if tr.is_world_process_zero():
        print(json.dumps({"examples": len(feats), "dropped_too_long": len(rows) - len(feats)}), flush=True)
    tr.train()
    if tr.is_world_process_zero():
        model.save_pretrained(a.out / "adapter")
        tok.save_pretrained(a.out / "adapter")
        print("SAVED", a.out / "adapter", flush=True)


if __name__ == "__main__":
    main()
