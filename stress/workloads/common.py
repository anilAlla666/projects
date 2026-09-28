"""Shared utilities for the 24-workload sweep.

Every wlNN_*.py driver imports from here. Keep minimal — sub-second import.
"""
from __future__ import annotations
import json, os, subprocess, threading, time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
RUN_SH     = REPO_ROOT / "cipher_run.sh"
RESULTS    = REPO_ROOT / "stress/results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("/home/ubuntu/models")
MODEL_LLAMA_1B = MODELS_DIR / "Llama-3.2-1B"
MODEL_LLAMA_8B = "/root/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B"
MODEL_MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
MODEL_AWQ_8B = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def sample_power(stop_evt, samples, gpu_id=0, interval=0.5):
    while not stop_evt.is_set():
        try:
            o = subprocess.check_output([
                "nvidia-smi", "-i", str(gpu_id),
                "--query-gpu=power.draw,clocks.gr,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], timeout=2).decode().strip()
            pw, clk, mem, util = o.split(",")
            samples.append((time.perf_counter(), float(pw), float(clk),
                            int(mem), float(util)))
        except Exception:
            pass
        time.sleep(interval)


def avg_power(samples):
    if not samples:
        return 0.0
    return sum(s[1] for s in samples) / len(samples)


def write_result(wid, phase, summary):
    fp = RESULTS / f"{wid}_{phase}.json"
    fp.write_text(json.dumps(summary, indent=2, default=str))
    return fp


def load_summary(wid, phase):
    fp = RESULTS / f"{wid}_{phase}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())
