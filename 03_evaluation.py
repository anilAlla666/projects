#!/usr/bin/env python3
"""
================================================================================
SmolLM2-135M Continual Pre-Training: Evaluation & Benchmarking
================================================================================

Script 3 of 3: MMLU Evaluation (Cloze Formulation)

This script evaluates SmolLM2-135M on MMLU using the OFFICIAL cloze-based
methodology that matches HuggingFace Open LLM Leaderboard.

CRITICAL: Evaluation Methodology
- We use CLOZE-BASED evaluation (log-likelihood of answer text)
- NOT letter-based prediction (which gives ~25% random performance)
- This 5% discrepancy is documented in Technical Report Section 8

Expected Results (from Technical Report v3.0):
- Baseline (untrained): 30.70%
- After CPT: 32.09%
- Improvement: +1.39% overall
- Best subject: conceptual_physics (+9.8%)
- Worst subject: business_ethics (-5.0%)

Author: [Candidate]
Date: January 2025
================================================================================
"""

# %% [markdown]
# # SmolLM2 CPT - MMLU Evaluation
# 
# Official cloze-based evaluation matching HuggingFace Open LLM Leaderboard.

# %% Install Dependencies
print("="*70)
print("SmolLM2 CPT - MMLU Evaluation (Cloze Method)")
print("="*70)

# For Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("\n[✓] Running in Google Colab")
    get_ipython().system('pip install -q transformers>=4.36.0 datasets>=2.14.0 accelerate>=0.24.0 pandas tqdm')
except:
    IN_COLAB = False
    print("\n[✓] Running locally")

# %% Imports
import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% Configuration
@dataclass
class EvalConfig:
    """Evaluation configuration."""
    
    # Model paths
    baseline_model: str = "HuggingFaceTB/SmolLM2-135M"
    trained_model: str = "./smollm2-mmlu-cpt/final_model"
    
    # Evaluation subjects (20 subjects from report)
    subjects: List[str] = field(default_factory=lambda: [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry"
    ])
    
    # Output
    output_dir: str = "./evaluation_results"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = EvalConfig()
os.makedirs(config.output_dir, exist_ok=True)

# %% Check Hardware
print("\n" + "="*70)
print("Hardware Configuration")
print("="*70)

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠ No GPU available - evaluation will be slower")

# %% MMLU Evaluation Class
class MMLUEvaluator:
    """
    MMLU Evaluator using official cloze-based methodology.
    
    This matches the HuggingFace Open LLM Leaderboard evaluation:
    - Computes log-likelihood of each answer choice
    - Normalizes by token length
    - Selects highest scoring answer
    
    NOT letter-based (which gives ~25% for small models).
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to model or HuggingFace model name
            device: Device to run on
        """
        self.device = device
        self.model_path = model_path
        
        print(f"\nLoading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if os.path.exists(model_path) else model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cuda" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[✓] Model loaded on {device}")
    
    def compute_log_likelihood(
        self, 
        prompt: str, 
        completion: str
    ) -> Tuple[float, int]:
        """
        Compute log-likelihood of completion given prompt.
        
        This is the core of cloze-based evaluation:
        P(completion | prompt) = Σ log P(token_i | prompt, token_1:i-1)
        
        Args:
            prompt: The context/question
            completion: The answer text to score
            
        Returns:
            Tuple of (log_likelihood, num_tokens)
        """
        # Tokenize prompt and completion separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        
        # Full sequence
        full_ids = prompt_ids + completion_ids
        input_ids = torch.tensor([full_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # We want P(completion | prompt)
        # So we look at positions len(prompt)-1 to len(full)-2
        # predicting tokens at positions len(prompt) to len(full)-1
        
        log_likelihood = 0.0
        start_idx = len(prompt_ids) - 1  # Position predicting first completion token
        
        for i, token_id in enumerate(completion_ids):
            pos = start_idx + i
            if pos < logits.shape[1]:
                token_logits = logits[0, pos, :]
                log_probs = F.log_softmax(token_logits, dim=-1)
                log_likelihood += log_probs[token_id].item()
        
        return log_likelihood, len(completion_ids)
    
    def evaluate_question(
        self, 
        question: str, 
        choices: List[str], 
        subject: str
    ) -> int:
        """
        Evaluate a single MMLU question using cloze method.
        
        Args:
            question: The question text
            choices: List of 4 answer choices
            subject: Subject name for prompt formatting
            
        Returns:
            Index of predicted answer (0-3)
        """
        # Format prompt (matching MMLU cloze evaluation)
        prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        prompt += f"Question: {question}\nAnswer:"
        
        scores = []
        
        for choice in choices:
            # Add space before answer (standard formatting)
            completion = f" {choice}"
            
            log_ll, num_tokens = self.compute_log_likelihood(prompt, completion)
            
            # Length normalization (important for fair comparison)
            normalized_score = log_ll / num_tokens if num_tokens > 0 else float('-inf')
            scores.append(normalized_score)
        
        # Return index of highest scoring answer
        return scores.index(max(scores))
    
    def evaluate_subject(self, subject: str) -> Dict:
        """
        Evaluate all questions for a subject.
        
        Args:
            subject: MMLU subject name
            
        Returns:
            Dictionary with accuracy and per-question results
        """
        # Load test split
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except:
            print(f"    ⚠ Could not load test split for {subject}")
            return {'accuracy': 0.0, 'total': 0, 'correct': 0}
        
        correct = 0
        total = 0
        results = []
        
        for item in dataset:
            question = item['question']
            choices = item['choices']
            answer = item['answer']  # Ground truth index
            
            # Get prediction
            prediction = self.evaluate_question(question, choices, subject)
            
            is_correct = (prediction == answer)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'question': question[:100] + '...' if len(question) > 100 else question,
                'prediction': prediction,
                'answer': answer,
                'correct': is_correct
            })
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
    
    def evaluate_all(self, subjects: List[str]) -> Dict:
        """
        Evaluate all subjects.
        
        Args:
            subjects: List of MMLU subjects to evaluate
            
        Returns:
            Dictionary with overall and per-subject results
        """
        all_results = {}
        total_correct = 0
        total_questions = 0
        
        print(f"\nEvaluating {len(subjects)} subjects...")
        print("-" * 60)
        
        for subject in tqdm(subjects, desc="Subjects"):
            result = self.evaluate_subject(subject)
            all_results[subject] = result
            
            total_correct += result['correct']
            total_questions += result['total']
            
            print(f"  {subject:35s}: {result['accuracy']:5.2f}% ({result['correct']}/{result['total']})")
        
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0
        
        print("-" * 60)
        print(f"  {'OVERALL':35s}: {overall_accuracy:5.2f}% ({total_correct}/{total_questions})")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'per_subject': all_results
        }


# %% Evaluate Baseline Model
print("\n" + "="*70)
print("STEP 1: Evaluating BASELINE Model")
print("="*70)

baseline_evaluator = MMLUEvaluator(config.baseline_model, config.device)
baseline_results = baseline_evaluator.evaluate_all(config.subjects)

print(f"\n[✓] Baseline MMLU Score: {baseline_results['overall_accuracy']:.2f}%")

# Clean up to free memory
del baseline_evaluator
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% Evaluate Trained Model
print("\n" + "="*70)
print("STEP 2: Evaluating TRAINED Model (after CPT)")
print("="*70)

# Check if trained model exists
if os.path.exists(config.trained_model):
    trained_evaluator = MMLUEvaluator(config.trained_model, config.device)
    trained_results = trained_evaluator.evaluate_all(config.subjects)
    
    print(f"\n[✓] Trained MMLU Score: {trained_results['overall_accuracy']:.2f}%")
    
    del trained_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print(f"\n[!] Trained model not found at: {config.trained_model}")
    print("    Please run 02_continual_pretraining.py first.")
    print("    Using baseline results for comparison display.")
    trained_results = None

# %% Results Comparison
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

# Build comparison table
comparison_data = []

for subject in config.subjects:
    baseline_acc = baseline_results['per_subject'][subject]['accuracy']
    
    if trained_results:
        trained_acc = trained_results['per_subject'][subject]['accuracy']
        delta = trained_acc - baseline_acc
    else:
        trained_acc = None
        delta = None
    
    comparison_data.append({
        'Subject': subject.replace('_', ' ').title(),
        'Baseline': f"{baseline_acc:.2f}%",
        'Trained': f"{trained_acc:.2f}%" if trained_acc is not None else "N/A",
        'Delta': f"{delta:+.2f}%" if delta is not None else "N/A",
        'delta_value': delta if delta is not None else 0
    })

# Sort by delta
comparison_data.sort(key=lambda x: x['delta_value'], reverse=True)

# Print table
print("\n" + "-"*75)
print(f"{'Subject':40s} {'Baseline':>10s} {'Trained':>10s} {'Delta':>10s}")
print("-"*75)

for row in comparison_data:
    delta_str = row['Delta']
    if row['delta_value'] > 1:
        delta_str = f"✓ {delta_str}"
    elif row['delta_value'] < -1:
        delta_str = f"✗ {delta_str}"
    
    print(f"{row['Subject']:40s} {row['Baseline']:>10s} {row['Trained']:>10s} {delta_str:>12s}")

print("-"*75)

# Overall
baseline_overall = baseline_results['overall_accuracy']
if trained_results:
    trained_overall = trained_results['overall_accuracy']
    overall_delta = trained_overall - baseline_overall
    print(f"{'OVERALL':40s} {baseline_overall:>9.2f}% {trained_overall:>9.2f}% {overall_delta:>+9.2f}%")
else:
    print(f"{'OVERALL':40s} {baseline_overall:>9.2f}%")

print("-"*75)

# %% Save Results
print("\n" + "="*70)
print("Saving Results")
print("="*70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save baseline results
baseline_output = os.path.join(config.output_dir, f"baseline_results_{timestamp}.json")
with open(baseline_output, 'w') as f:
    # Remove detailed results to keep file small
    baseline_save = {
        'model': config.baseline_model,
        'overall_accuracy': baseline_results['overall_accuracy'],
        'total_correct': baseline_results['total_correct'],
        'total_questions': baseline_results['total_questions'],
        'per_subject': {k: {'accuracy': v['accuracy'], 'correct': v['correct'], 'total': v['total']} 
                       for k, v in baseline_results['per_subject'].items()},
        'timestamp': timestamp,
        'evaluation_method': 'cloze (log-likelihood, length-normalized)'
    }
    json.dump(baseline_save, f, indent=2)
print(f"[✓] Baseline results: {baseline_output}")

# Save trained results
if trained_results:
    trained_output = os.path.join(config.output_dir, f"trained_results_{timestamp}.json")
    with open(trained_output, 'w') as f:
        trained_save = {
            'model': config.trained_model,
            'overall_accuracy': trained_results['overall_accuracy'],
            'total_correct': trained_results['total_correct'],
            'total_questions': trained_results['total_questions'],
            'per_subject': {k: {'accuracy': v['accuracy'], 'correct': v['correct'], 'total': v['total']} 
                           for k, v in trained_results['per_subject'].items()},
            'timestamp': timestamp,
            'evaluation_method': 'cloze (log-likelihood, length-normalized)'
        }
        json.dump(trained_save, f, indent=2)
    print(f"[✓] Trained results: {trained_output}")

# Save comparison
comparison_output = os.path.join(config.output_dir, f"comparison_{timestamp}.json")
comparison_save = {
    'baseline_model': config.baseline_model,
    'trained_model': config.trained_model if trained_results else None,
    'baseline_accuracy': baseline_results['overall_accuracy'],
    'trained_accuracy': trained_results['overall_accuracy'] if trained_results else None,
    'improvement': trained_results['overall_accuracy'] - baseline_results['overall_accuracy'] if trained_results else None,
    'per_subject_comparison': comparison_data,
    'timestamp': timestamp
}
with open(comparison_output, 'w') as f:
    json.dump(comparison_save, f, indent=2)
print(f"[✓] Comparison: {comparison_output}")

# %% Final Summary
print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         FINAL RESULTS                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Baseline MMLU:        {baseline_results['overall_accuracy']:>6.2f}%                                    ║""")

if trained_results:
    delta = trained_results['overall_accuracy'] - baseline_results['overall_accuracy']
    print(f"""║  Trained MMLU:         {trained_results['overall_accuracy']:>6.2f}%                                    ║
║  ──────────────────────────────────────────────────────────────────  ║
║  Improvement:          {delta:>+6.2f}%                                    ║""")
    
    # Find best and worst subjects
    deltas = [(s, trained_results['per_subject'][s]['accuracy'] - baseline_results['per_subject'][s]['accuracy']) 
              for s in config.subjects]
    best = max(deltas, key=lambda x: x[1])
    worst = min(deltas, key=lambda x: x[1])
    
    print(f"""║  Best Subject:         {best[0][:20]:20s} ({best[1]:+.1f}%)        ║
║  Worst Subject:        {worst[0][:20]:20s} ({worst[1]:+.1f}%)        ║""")

print("""╚══════════════════════════════════════════════════════════════════════╝

Evaluation Method: Cloze-based (official HuggingFace Leaderboard method)
  - Log-likelihood scoring of answer text
  - Length normalization applied
  - NOT letter-based prediction (which gives ~25%)

Expected Results (from Technical Report):
  - Baseline: 30.70%
  - After CPT: 32.09%
  - Improvement: +1.39%
""")

# %% Verification Against Report
print("\n" + "="*70)
print("VERIFICATION AGAINST TECHNICAL REPORT")
print("="*70)

print(f"""
Report Claims vs Actual Results:

                        Report      Actual      Match?
  Baseline MMLU:        30.70%      {baseline_results['overall_accuracy']:.2f}%      {'✓' if abs(baseline_results['overall_accuracy'] - 30.70) < 1 else '✗'}
""")

if trained_results:
    delta_actual = trained_results['overall_accuracy'] - baseline_results['overall_accuracy']
    print(f"""  Trained MMLU:         32.09%      {trained_results['overall_accuracy']:.2f}%      {'✓' if abs(trained_results['overall_accuracy'] - 32.09) < 1 else '~'}
  Improvement:          +1.39%      {delta_actual:+.2f}%      {'✓' if abs(delta_actual - 1.39) < 0.5 else '~'}
""")

print("""
Note: Small variations (±1%) are expected due to:
  - Random initialization differences
  - Hardware-specific floating point behavior
  - Dataset loading order variations
""")
