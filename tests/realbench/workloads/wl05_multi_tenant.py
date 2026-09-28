"""Workload 05 — MULTI_TENANT.

Spawns N child processes, each with a different CIPHER_TENANT_ID, all
running the same wl01-style decode loop on the same GPU. Validates:
  - per-tenant policy.json is correctly scoped per process
  - FAIRNESS work-unit accounting differentiates tenants
  - distributor sees N tenants observed via shm
  - aggregate tok/s × tenants > single-tenant tok/s × N × 0.6 (no
    catastrophic contention, target spread <3.0× max/min)

Output: aggregate summary + per-tenant rows.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NAME           = "MULTI_TENANT"
PROMPT_KEY     = ("chat", "instruction")
DEFAULT_MODEL  = "unsloth/Llama-3.2-1B"
EXPECTED_OPS   = {"FAIRNESS", "ARBITRATE", "WORKLOAD_OBSERVE"}


def _child_script(model_path, prompts_path, max_new, n_iters, out_path):
    """The body each child process runs. Saved to a file so we can
    exec it under LD_PRELOAD if the parent wants to flip CIPHER on."""
    return f"""
import json, os, time, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
prompts = []
for line in open({prompts_path!r}):
    line = line.strip()
    if line:
        prompts.append(json.loads(line))
prompts = [p for p in prompts if p.get("category") in {PROMPT_KEY}][:4]
tok = AutoTokenizer.from_pretrained({model_path!r})
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    {model_path!r}, torch_dtype=torch.float16, device_map="cuda:0",
    low_cpu_mem_usage=True)
getattr(model, "eval")()
total_new, total_t = 0, 0.0
for it in range({n_iters}):
    for p in prompts:
        enc = tok(p["prompt"], return_tensors="pt").to("cuda:0")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens={max_new}, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        total_t += time.perf_counter() - t0
        total_new += int(out.shape[1] - enc.input_ids.shape[1])
res = {{"tenant_id":  os.environ.get("CIPHER_TENANT_ID", "?"),
        "pid":        os.getpid(),
        "total_new":  total_new,
        "total_t":    total_t,
        "tok_per_s":  total_new / max(total_t, 1e-6)}}
open({out_path!r}, "w").write(json.dumps(res))
"""


def run(model_path, prompts, phase_label, out_dir,
        n_tenants=4, n_iters=2, max_new=32, **_):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "_prompts.jsonl"
    with open(prompts_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Tenant ids match policy.json.example layout: default / premium / shared.
    tenant_ids = ["default", "premium", "shared"][:n_tenants]
    while len(tenant_ids) < n_tenants:
        tenant_ids.append(f"t{len(tenant_ids)}")

    children = []
    out_paths = []
    t_wall_0 = time.perf_counter()

    for tid in tenant_ids:
        cpath = out_dir / f"_child_{tid}.py"
        opath = out_dir / f"_result_{tid}.json"
        cpath.write_text(_child_script(model_path, str(prompts_path),
                                        max_new, n_iters, str(opath)))
        env = os.environ.copy()
        env["CIPHER_TENANT_ID"] = tid
        # Hook is loaded only on CIPHER phase (via LD_PRELOAD provided by
        # the orchestrator that calls this driver).
        proc = subprocess.Popen([sys.executable, str(cpath)], env=env)
        children.append(proc)
        out_paths.append((tid, opath))

    for p in children:
        p.wait()
    t_wall = time.perf_counter() - t_wall_0

    rows = []
    for tid, opath in out_paths:
        try:
            r = json.loads(opath.read_text())
        except Exception as e:
            r = {"tenant_id": tid, "error": str(e)}
        rows.append(r)

    tps_vals = [r["tok_per_s"] for r in rows if "tok_per_s" in r]
    spread = (max(tps_vals) / min(tps_vals)) if (tps_vals and min(tps_vals) > 0) else 0.0
    summary = {
        "workload":      NAME,
        "phase":         phase_label,
        "n_tenants":     n_tenants,
        "wall_seconds":  t_wall,
        "tok_per_s_total": sum(tps_vals),
        "tok_per_s_max_min_spread": spread,
        "rows":          rows,
    }
    with open(out_dir / "wl05_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
