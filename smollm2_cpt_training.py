#!/usr/bin/env python3
"""
SmolLM2 MMLU continual pre-training experiment
Anil Kumar Alla - Tether Assessment

NOTE: auxiliary_train split from cais/mmlu doesn't load properly (known HF issue)
Had to supplement with ARC + SciQ to get enough training data
"""

import torch
import json
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
use_bf16 = torch.cuda.get_device_capability()[0] >= 8  # ampere or newer

OUTPUT_DIR = Path("/content/smollm2_mmlu_direct")
OUTPUT_DIR.mkdir(exist_ok=True)

# === hyperparameters ===
# tested a few LR values, 5e-5 worked best
# 1e-5 was too slow, 1e-4 caused instability
LR = 5e-5
BATCH_SIZE = 16
GRAD_ACCUM = 2  # effective batch = 32
EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LEN = 256

# all 57 MMLU subjects
SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# subset for faster eval during iteration
EVAL_SUBSET = SUBJECTS[:20]


def load_model_and_tokenizer(model_name):
    """load smollm2 with bf16 if available"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
    )
    model.gradient_checkpointing_enable()  # save memory
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name}: {n_params/1e6:.1f}M params")
    
    return model, tokenizer


def format_example(question, choices, answer_idx, subject):
    """
    format QA pair to match eval prompt structure
    important: no letter choices (A/B/C/D) - we use cloze evaluation
    """
    topic = subject.replace("_", " ")
    answer = choices[answer_idx]
    return f"The following are questions about {topic}.\nQuestion: {question}\nAnswer: {answer}"


def load_training_data():
    """
    try to load MMLU auxiliary_train, fall back to other QA datasets if needed
    
    NOTE: auxiliary_train has issues on HuggingFace - returns empty for all subjects
    see: https://huggingface.co/datasets/cais/mmlu/discussions/22
    """
    from datasets import load_dataset
    
    training_texts = []
    
    # try auxiliary_train first (this usually fails/returns 0)
    print("Attempting to load auxiliary_train...")
    aux_count = 0
    for subj in tqdm(SUBJECTS, desc="aux_train"):
        try:
            ds = load_dataset("cais/mmlu", subj)
            if "auxiliary_train" in ds:
                for ex in ds["auxiliary_train"]:
                    text = format_example(ex["question"], ex["choices"], ex["answer"], subj)
                    training_texts.append(text)
                    aux_count += 1
        except:
            continue
    print(f"Got {aux_count} from auxiliary_train")
    
    # load validation + dev (these work)
    print("Loading validation split...")
    for subj in tqdm(SUBJECTS, desc="validation"):
        try:
            ds = load_dataset("cais/mmlu", subj)
            for ex in ds.get("validation", []):
                text = format_example(ex["question"], ex["choices"], ex["answer"], subj)
                training_texts.append(text)
        except:
            continue
    
    print("Loading dev split...")
    for subj in tqdm(SUBJECTS, desc="dev"):
        try:
            ds = load_dataset("cais/mmlu", subj)
            for ex in ds.get("dev", []):
                text = format_example(ex["question"], ex["choices"], ex["answer"], subj)
                training_texts.append(text)
        except:
            continue
    
    print(f"MMLU total: {len(training_texts)}")
    
    # if we don't have enough, add ARC + SciQ
    # this makes the data science-heavy but better than nothing
    if len(training_texts) < 5000:
        print("Not enough data from MMLU, supplementing with ARC + SciQ...")
        
        # ARC (science QA)
        try:
            for arc_split in ["ARC-Challenge", "ARC-Easy"]:
                ds = load_dataset("allenai/ai2_arc", arc_split, split="train")
                for ex in ds:
                    try:
                        key = ex["answerKey"]
                        # handle both numeric (1,2,3,4) and letter (A,B,C,D) keys
                        if key.isdigit():
                            idx = int(key) - 1
                        else:
                            idx = ord(key) - ord('A')
                        answer = ex["choices"]["text"][idx]
                        text = f"The following are questions about science.\nQuestion: {ex['question']}\nAnswer: {answer}"
                        training_texts.append(text)
                    except:
                        continue  # skip malformed examples
            print(f"After ARC: {len(training_texts)}")
        except Exception as e:
            print(f"ARC load failed: {e}")
        
        # SciQ
        try:
            ds = load_dataset("allenai/sciq", split="train")
            for ex in ds:
                text = f"The following are questions about science.\nQuestion: {ex['question']}\nAnswer: {ex['correct_answer']}"
                training_texts.append(text)
            print(f"After SciQ: {len(training_texts)}")
        except Exception as e:
            print(f"SciQ load failed: {e}")
    
    random.shuffle(training_texts)
    print(f"Final training set: {len(training_texts)} examples")
    return training_texts


class SimpleDataset(torch.utils.data.Dataset):
    """basic dataset wrapper for tokenized text"""
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx].clone()
        }


def train(model, tokenizer, training_texts):
    """main training loop"""
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    from accelerate import Accelerator
    
    # tokenize everything upfront
    print("Tokenizing...")
    tokenized = tokenizer(
        training_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    dataset = SimpleDataset(tokenized["input_ids"], tokenized["attention_mask"])
    
    # setup accelerator for mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision="bf16" if use_bf16 else "fp16"
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # cosine schedule with warmup
    total_steps = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # prepare for distributed/mixed precision
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    print(f"Training: {len(dataset)} examples, {total_steps} steps, {warmup_steps} warmup")
    
    model.train()
    loss_history = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    
    # save checkpoint
    print("Saving checkpoint...")
    ckpt_dir = OUTPUT_DIR / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    
    return ckpt_dir, loss_history


def evaluate_mmlu(model_path, subjects, desc="eval"):
    """
    run cloze-style MMLU evaluation
    
    cloze = score each answer by log-likelihood, pick highest
    this is more reliable for small models than letter prediction
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    device = torch.device("cuda")
    
    # load model
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    def get_completion_logprob(prompt, completion):
        """compute avg log prob of completion tokens given prompt"""
        full_text = prompt + completion
        
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = tok(full_text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]
        
        if full_len <= prompt_len:
            return -1e10  # completion didn't add any tokens
        
        full_ids = full_ids.to(device)
        
        with torch.no_grad():
            logits = model(full_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
        
        # sum log probs for completion tokens only
        total_ll = 0.0
        n_tokens = 0
        for i in range(prompt_len - 1, full_len - 1):
            next_token = full_ids[0, i + 1]
            total_ll += log_probs[0, i, next_token].item()
            n_tokens += 1
        
        return total_ll / n_tokens if n_tokens > 0 else -1e10
    
    # run eval
    total_correct = 0
    total_count = 0
    per_subject = {}
    
    for subj in tqdm(subjects, desc=desc):
        try:
            ds = load_dataset("cais/mmlu", subj)
            test_split = ds["test"]
            
            correct = 0
            for ex in test_split:
                topic = subj.replace("_", " ")
                prompt = f"The following are questions about {topic}.\nQuestion: {ex['question']}\nAnswer:"
                
                # score each choice
                scores = []
                for choice in ex["choices"]:
                    score = get_completion_logprob(prompt, f" {choice}")
                    scores.append(score)
                
                pred = scores.index(max(scores))
                if pred == ex["answer"]:
                    correct += 1
            
            acc = 100 * correct / len(test_split)
            per_subject[subj] = acc
            total_correct += correct
            total_count += len(test_split)
            
        except Exception as e:
            print(f"Error on {subj}: {e}")
            continue
    
    overall_acc = 100 * total_correct / total_count if total_count > 0 else 0
    
    # cleanup
    del model
    torch.cuda.empty_cache()
    
    return overall_acc, per_subject


def main():
    # install deps (for colab)
    import subprocess
    subprocess.run(["pip", "install", "-q", "transformers>=4.35.0", "datasets>=2.14.0", "accelerate>=0.24.0"])
    
    print("="*60)
    print("SmolLM2-135M MMLU Continual Pre-training")
    print("="*60)
    
    # load model
    model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M")
    
    # load training data
    training_texts = load_training_data()
    
    # train
    ckpt_dir, losses = train(model, tokenizer, training_texts)
    print(f"Loss curve: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    # cleanup training model
    del model
    torch.cuda.empty_cache()
    
    # evaluate
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    print("\nBaseline (original model):")
    base_acc, base_results = evaluate_mmlu("HuggingFaceTB/SmolLM2-135M", EVAL_SUBSET, "baseline")
    print(f"Baseline accuracy: {base_acc:.2f}%")
    
    print("\nTrained model:")
    trained_acc, trained_results = evaluate_mmlu(str(ckpt_dir), EVAL_SUBSET, "trained")
    print(f"Trained accuracy: {trained_acc:.2f}%")
    
    # results summary
    improvement = trained_acc - base_acc
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline:    {base_acc:.2f}%")
    print(f"Trained:     {trained_acc:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")
    
    # per-subject breakdown
    print(f"\n{'Subject':<35} {'Base':>7} {'Train':>7} {'Delta':>7}")
    print("-"*60)
    
    deltas = []
    for subj in EVAL_SUBSET:
        if subj in base_results and subj in trained_results:
            b = base_results[subj]
            t = trained_results[subj]
            d = t - b
            deltas.append((subj, d))
            print(f"{subj:<35} {b:>6.1f}% {t:>6.1f}% {d:>+6.1f}%")
    
    # top gains/drops
    deltas.sort(key=lambda x: x[1], reverse=True)
    print("\nTop gains:", [(s, f"{d:+.1f}%") for s, d in deltas[:3] if d > 0])
    print("Top drops:", [(s, f"{d:+.1f}%") for s, d in deltas[-3:] if d < 0])
    
    # save results
    results = {
        "baseline_acc": base_acc,
        "trained_acc": trained_acc,
        "improvement": improvement,
        "n_training_examples": len(training_texts),
        "loss_history": losses,
        "per_subject_baseline": base_results,
        "per_subject_trained": trained_results
    }
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")
    print("Done!")


if __name__ == "__main__":
    main()
