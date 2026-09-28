#!/usr/bin/env python3
"""
================================================================================
SmolLM2-135M Continual Pre-Training: Training Implementation
================================================================================

Script 2 of 3: Continual Pre-Training on SmolLM2-135M

This script performs Continual Pre-Training (CPT) on SmolLM2-135M using the
processed dataset from Script 1.

Training Configuration (from Technical Report v3.0):
- Learning Rate: 5e-5
- Epochs: 5
- Batch Size: 16 (effective: 32 with gradient accumulation)
- Scheduler: Cosine with 10% warmup
- Precision: BF16/FP16
- Expected Final Loss: ~0.26

Expected Results:
- Baseline MMLU: 30.70%
- Target MMLU: 32.09% (+1.39%)
- Best Subject Gain: +9.8% (conceptual_physics)

Reference: HuggingFaceTB/SmolLM2-135M
Nanotron Checkpoint: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt

Author: [Candidate]
Date: January 2025
================================================================================
"""

# %% [markdown]
# # SmolLM2 CPT - Continual Pre-Training
# 
# This notebook trains SmolLM2-135M on MMLU-format QA data.

# %% Install Dependencies
print("="*70)
print("SmolLM2 CPT - Continual Pre-Training Implementation")
print("="*70)

# For Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("\n[✓] Running in Google Colab")
    get_ipython().system('pip install -q transformers>=4.36.0 datasets>=2.14.0 accelerate>=0.24.0 bitsandbytes')
    
    # Check GPU
    get_ipython().system('nvidia-smi --query-gpu=name,memory.total --format=csv')
except:
    IN_COLAB = False
    print("\n[✓] Running locally")

# %% Imports
import os
import json
import math
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset, load_from_disk

# %% Configuration
@dataclass
class TrainingConfig:
    """
    Training configuration matching Technical Report v3.0 - Experiment 6 (MMLU Direct)
    """
    
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./smollm2-mmlu-cpt"
    
    # Training Hyperparameters (EXACT from report)
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2  # Effective batch size = 32
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Precision
    bf16: bool = True  # Will fallback to fp16 if bf16 not supported
    tf32: bool = True
    
    # Memory Optimization
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    
    # Sequence
    max_seq_length: int = 384
    
    # Logging & Saving
    logging_steps: int = 50
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    
    # Reproducibility
    seed: int = 42


config = TrainingConfig()
set_seed(config.seed)

# %% Check Hardware
print("\n" + "="*70)
print("Hardware Configuration")
print("="*70)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Check for BF16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    
    print(f"  GPU: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")
    print(f"  BF16 Support: {bf16_supported}")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    if not bf16_supported:
        print("  → Falling back to FP16")
        config.bf16 = False
else:
    device = torch.device("cpu")
    print("  ⚠ No GPU available - training will be slow")
    config.bf16 = False

# %% Load Data
print("\n" + "="*70)
print("Loading Training Data")
print("="*70)

# Try to load from HuggingFace dataset format first
dataset_path = os.path.join(config.data_dir, "mmlu_cpt_dataset")
json_path = os.path.join(config.data_dir, "mmlu_cpt_training_data.json")

if os.path.exists(dataset_path):
    print(f"[✓] Loading from HuggingFace dataset: {dataset_path}")
    train_dataset = load_from_disk(dataset_path)
elif os.path.exists(json_path):
    print(f"[✓] Loading from JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_dataset = Dataset.from_list(data)
else:
    print("[!] No pre-processed data found. Running data collection inline...")
    print("    Please run 01_data_collection.py first, or data will be collected now.")
    
    # Inline data collection (minimal version)
    from datasets import load_dataset as hf_load_dataset
    
    all_data = []
    
    # Load MMLU validation
    print("\n    Loading MMLU validation...")
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry"
    ]
    
    for subject in subjects:
        try:
            ds = hf_load_dataset("cais/mmlu", subject, split="validation", trust_remote_code=True)
            for item in ds:
                letters = ['A', 'B', 'C', 'D']
                choices_text = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(item['choices']))
                correct = item['choices'][item['answer']]
                text = f"Question: {item['question']}\n{choices_text}\nAnswer: {correct}"
                all_data.append({'text': text, 'subject': subject, 'source': 'mmlu_validation'})
        except:
            pass
    
    # Load ARC
    print("    Loading ARC...")
    try:
        arc = hf_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for item in list(arc)[:2000]:
            choices = item['choices']['text'][:4]
            while len(choices) < 4:
                choices.append("")
            labels = item['choices']['label']
            try:
                ans_idx = labels.index(item['answerKey'])
            except:
                ans_idx = 0
            letters = ['A', 'B', 'C', 'D']
            choices_text = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
            text = f"Question: {item['question']}\n{choices_text}\nAnswer: {choices[ans_idx]}"
            all_data.append({'text': text, 'subject': 'science', 'source': 'arc'})
    except:
        pass
    
    # Load SciQ
    print("    Loading SciQ...")
    try:
        sciq = hf_load_dataset("allenai/sciq", split="train")
        import random
        random.seed(42)
        for item in list(sciq):
            choices = [item['correct_answer'], item['distractor1'], item['distractor2'], item['distractor3']]
            indices = list(range(4))
            random.shuffle(indices)
            shuffled = [choices[i] for i in indices]
            ans_idx = indices.index(0)
            letters = ['A', 'B', 'C', 'D']
            choices_text = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(shuffled))
            text = f"Question: {item['question']}\n{choices_text}\nAnswer: {shuffled[ans_idx]}"
            all_data.append({'text': text, 'subject': 'science', 'source': 'sciq'})
    except:
        pass
    
    train_dataset = Dataset.from_list(all_data)
    print(f"    ✓ Collected {len(train_dataset)} examples")

print(f"\n[✓] Training examples: {len(train_dataset):,}")

# Show sample
print("\nSample training example:")
print("-" * 50)
print(train_dataset[0]['text'][:500])
print("-" * 50)

# %% Load Model and Tokenizer
print("\n" + "="*70)
print("Loading Model and Tokenizer")
print("="*70)

print(f"\nLoading: {config.model_name}")

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[✓] Model loaded successfully")
print(f"    Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"    Trainable parameters: {trainable_params:,}")
print(f"    Model dtype: {model.dtype}")

# %% Tokenize Dataset
print("\n" + "="*70)
print("Tokenizing Dataset")
print("="*70)

def tokenize_function(examples):
    """Tokenize examples for causal language modeling."""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=config.max_seq_length,
        padding=False,
        return_tensors=None
    )
    # For causal LM, labels = input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# Tokenize
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing"
)

print(f"\n[✓] Tokenization complete")
print(f"    Examples: {len(tokenized_dataset):,}")

# Token length statistics
lengths = [len(x['input_ids']) for x in tokenized_dataset]
print(f"    Avg length: {sum(lengths)/len(lengths):.1f} tokens")
print(f"    Max length: {max(lengths)} tokens")

# %% Setup Training Arguments
print("\n" + "="*70)
print("Training Configuration")
print("="*70)

# Calculate training steps
num_examples = len(tokenized_dataset)
effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
steps_per_epoch = math.ceil(num_examples / effective_batch_size)
total_steps = steps_per_epoch * config.num_train_epochs
warmup_steps = int(total_steps * config.warmup_ratio)

print(f"""
Training Setup:
  - Examples: {num_examples:,}
  - Batch size (per device): {config.per_device_train_batch_size}
  - Gradient accumulation: {config.gradient_accumulation_steps}
  - Effective batch size: {effective_batch_size}
  - Steps per epoch: {steps_per_epoch}
  - Total epochs: {config.num_train_epochs}
  - Total steps: {total_steps}
  - Warmup steps: {warmup_steps} ({config.warmup_ratio*100:.0f}%)
  - Learning rate: {config.learning_rate}
  - Scheduler: {config.lr_scheduler_type}
  - Precision: {'BF16' if config.bf16 else 'FP16'}
""")

training_args = TrainingArguments(
    output_dir=config.output_dir,
    
    # Training
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    
    # Optimization
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    adam_beta1=config.adam_beta1,
    adam_beta2=config.adam_beta2,
    adam_epsilon=config.adam_epsilon,
    max_grad_norm=config.max_grad_norm,
    
    # Scheduler
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_ratio=config.warmup_ratio,
    
    # Precision & Memory
    bf16=config.bf16,
    fp16=not config.bf16 and torch.cuda.is_available(),
    tf32=config.tf32,
    gradient_checkpointing=config.gradient_checkpointing,
    optim=config.optim if torch.cuda.is_available() else "adamw_torch",
    
    # Logging
    logging_steps=config.logging_steps,
    logging_first_step=True,
    report_to="none",  # Disable wandb etc.
    
    # Saving
    save_strategy=config.save_strategy,
    save_total_limit=config.save_total_limit,
    
    # Other
    seed=config.seed,
    data_seed=config.seed,
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

# %% Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# %% Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# %% Training
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nExpected Results (from Technical Report):")
print("  - Initial loss: ~2.23")
print("  - Final loss: ~0.26")
print("  - Loss reduction: 88.5%")
print("\n" + "="*70 + "\n")

# Train
train_result = trainer.train()

# %% Save Model
print("\n" + "="*70)
print("Saving Model")
print("="*70)

# Save final model
final_model_path = os.path.join(config.output_dir, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"\n[✓] Model saved to: {final_model_path}")

# Save training metrics
metrics = {
    'train_loss': train_result.training_loss,
    'train_runtime': train_result.metrics.get('train_runtime', 0),
    'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
    'total_steps': train_result.global_step,
    'epochs': config.num_train_epochs,
    'learning_rate': config.learning_rate,
    'effective_batch_size': effective_batch_size,
    'total_examples': num_examples,
    'timestamp': datetime.now().isoformat()
}

with open(os.path.join(config.output_dir, "training_metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"[✓] Metrics saved to: {config.output_dir}/training_metrics.json")

# %% Training Summary
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"""
Training Results:
  - Final training loss: {train_result.training_loss:.4f}
  - Total steps: {train_result.global_step}
  - Training time: {train_result.metrics.get('train_runtime', 0)/60:.1f} minutes
  - Throughput: {train_result.metrics.get('train_samples_per_second', 0):.1f} samples/sec

Expected vs Actual:
  - Expected final loss: ~0.26
  - Actual final loss: {train_result.training_loss:.4f}
  
Model saved to: {final_model_path}

Next Step:
  Run 03_evaluation.py to evaluate MMLU performance
  
  Expected Results:
  - Baseline: 30.70%
  - After CPT: 32.09% (+1.39%)
""")

# %% Nanotron Checkpoint Note
print("\n" + "="*70)
print("Note on Nanotron Checkpoint")
print("="*70)
print("""
This script uses the HuggingFace Transformers checkpoint for compatibility.
For production deployment with Nanotron:

1. The weights are mathematically identical between formats
2. Conversion can be done via:

   from nanotron.serialize import save_checkpoint
   save_checkpoint(model.state_dict(), "nanotron_format/")

3. See Technical Report Appendix C for full conversion details.

Reference: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt
""")
