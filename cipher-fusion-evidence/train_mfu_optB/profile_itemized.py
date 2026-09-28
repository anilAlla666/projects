#!/usr/bin/env python3
# OptB B1/B3: training-step kernel-time decomposition with ITEMIZED pointwise bucket.
# Method = the build's profile_train.py (copied alongside, unedited), extended with:
#  - RTLD_NOLOAD substrate probe (must be substrate_loaded:False — PURE TORCH run)
#  - pointwise sub-buckets (cast/copy, add/mul, optimizer-foreach, activation, reduce, other)
#  - op-level (aten) producer table to attribute kernels to framework ops
#  - --base-dtype fp16|bf16, --adapter-dtype fp32|native  (B1 = fp16+fp32, the build's config;
#    B3 lean cell = bf16+native: same-dtype adapters, no fp32<->fp16 cast source)
# Pure torch: this script never sets injection env; engagement impossible by construction.
import os, json, time, argparse, ctypes
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
ap = argparse.ArgumentParser()
ap.add_argument("--base-dtype", choices=["fp16", "bf16"], default="fp16")
ap.add_argument("--adapter-dtype", choices=["fp32", "native"], default="fp32")
ap.add_argument("--result", required=True)
ap.add_argument("--stability-steps", type=int, default=20)
args = ap.parse_args()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.profiler import profile, ProfilerActivity
from collections import defaultdict

def substrate_probe():
    try:
        lib = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        return {"substrate_loaded": True}
    except OSError:
        return {"substrate_loaded": False}

B, S = 2, 2048
DT = torch.float16 if args.base_dtype == "fp16" else torch.bfloat16
torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=DT, attn_implementation="sdpa")
lc = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
m = get_peft_model(m, lc); m.cuda(); m.train()
if args.adapter_dtype == "fp32":
    for n, p in m.named_parameters():
        if p.requires_grad: p.data = p.data.float()
else:  # "native" = force adapters to the BASE dtype. NOT a no-op: peft's get_peft_model
    # creates LoRA weights in fp32 even on a bf16 base (verified: first run of this cell
    # left adapter_param_dtypes=['torch.float32'] and was discarded as not-lean).
    for n, p in m.named_parameters():
        if p.requires_grad: p.data = p.data.to(DT)
adapter_dtypes = sorted({str(p.dtype) for p in m.parameters() if p.requires_grad})
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
ids = tok("The history of computing began "*400, return_tensors="pt")["input_ids"][0][:S].unsqueeze(0).repeat(B,1).cuda()
tr = [p for p in m.parameters() if p.requires_grad]
def step():
    o = m(input_ids=ids, labels=ids); l = o.loss; opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); return float(l.detach())
losses = [step() for _ in range(5)]
torch.cuda.synchronize()
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3): losses.append(step())
    torch.cuda.synchronize()
wall3 = time.perf_counter() - t0
# stability tail (cheap; loss curve continuity for the lean cell's numerics note)
for _ in range(args.stability_steps): losses.append(step())
torch.cuda.synchronize()
clk = int(open("/sys/class/drm/card0/device/../..//dev/null", "w") and 0) if False else None
import subprocess
clk = subprocess.run(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()

ev = prof.key_averages()
kern = defaultdict(float)
ops = []
for e in ev:
    if e.device_time_total > 0 and getattr(e, "device_type", None) is not None:
        pass
for e in ev:
    k = e.key
    if e.device_time_total > 0:
        if k.startswith("aten::") or k.startswith("Optimizer") or k.startswith("autograd::") or "Backward" in k or k.startswith("torch::"):
            ops.append((k, e.device_time_total, e.count))
        elif not k.startswith("cuda") and not k.startswith("Memcpy") and not k.startswith("Memset") and not k.startswith("Profiler") and "empty" not in k:
            kern[k] += e.device_time_total
tot = sum(kern.values())

def coarse(n):
    s = n.lower()
    if any(t in s for t in ("flash","fmha","fwd_kernel","attention","sdpa","scaled_dot","softmax","_attn","mha")): return "ATTENTION"
    if any(t in s for t in ("nvjet","gemm","cutlass","wgmma","cublas","sm90_xmma","s16816","ampere_")): return "GEMM"
    if any(t in s for t in ("adam","fused_adam","optimizer","l2norm","clip","foreach","_amp","grad_scale","multi_tensor")): return "OPTIMIZER"
    if any(t in s for t in ("rmsnorm","rms_norm","norm","silu","swiglu","gelu","activation","rotary","rope","elementwise","add","mul","cast","convert","copy","memcpy","memset","fill","cat","_to_copy")): return "POINTWISE"
    return "OTHER"
def fine(n):
    # ORDER MATTERS: "nocast" (same-dtype fast path) must not substring-match "cast";
    # LoadWithCast/StoreWithCast = mixed-dtype elementwise (cast fused into load/store).
    s = n.lower()
    if any(t in s for t in ("multi_tensor","foreach","adam")): return "PW_OPTIMIZER_FOREACH"
    if "silu" in s or "gelu" in s or "swiglu" in s: return "PW_ACTIVATION"
    if any(t in s for t in ("direct_copy","float16_copy","bfloat16_copy","_to_copy","convert")):
        return "PW_COPY_SAMEDTYPE" if "nocast" in s else "PW_CAST_COPY"
    if "nocast" in s: return "PW_ADD_MUL_SAMEDTYPE"
    if "withcast" in s: return "PW_MIXED_DTYPE_ADD_MUL"
    if any(t in s for t in ("functor_add","aunaryfunctor","binaryfunctor","bunaryfunctor","cudafunctor","pow_tensor","gpu_kernel_impl","elementwise_kernel")): return "PW_ADD_MUL_VECTORIZED_SAMEDTYPE"
    if any(t in s for t in ("fill","cat","memset")): return "PW_FILL_CAT"
    return None

coarse_b = defaultdict(float); fine_b = defaultdict(float); fine_members = defaultdict(list)
for k, v in kern.items():
    cb = coarse(k); coarse_b[cb] += v
    if cb == "POINTWISE":
        fb = fine(k) or "PW_UNMATCHED"
        fine_b[fb] += v; fine_members[fb].append((k, v))
res = {
    "config": {"B": B, "S": S, "base_dtype": args.base_dtype, "adapter_dtype": args.adapter_dtype,
               "adapter_param_dtypes": adapter_dtypes, "attn": "sdpa", "opt": "AdamW(foreach default)",
               "method": "build's profile_train.py method: 5 warmup + 3 profiled steps, key_averages device_time"},
    "substrate_probe": substrate_probe(), "injection_env": os.environ.get("CUDA_INJECTION64_PATH", "unset"),
    "clocks_sm_after": clk,
    "wall_s_3steps": round(wall3, 4), "gpu_kernel_s_3steps": round(tot/1e6, 4),
    "sec_per_step_wall": round(wall3/3, 4),
    "buckets_pct": {b: round(100*v/tot, 2) for b, v in sorted(coarse_b.items(), key=lambda x: -x[1])},
    "pointwise_sub_buckets_pct_of_total": {b: round(100*v/tot, 2) for b, v in sorted(fine_b.items(), key=lambda x: -x[1])},
    "pointwise_members": {b: [{"k": k[:300], "pct_of_total": round(100*v/tot, 2)} for k, v in sorted(l, key=lambda x: -x[1])] for b, l in fine_members.items()},
    "all_kernels_ge_0p1pct": [{"k": k[:300], "bucket": coarse(k), "pct": round(100*v/tot, 2)} for k, v in sorted(kern.items(), key=lambda x: -x[1]) if 100*v/tot >= 0.1],
    "op_table_top40": [{"op": k[:80], "device_ms_per_step": round(v/1e3/3, 2), "count_per_3steps": c} for k, v, c in sorted(ops, key=lambda x: -x[1])[:40]],
    "loss_first": losses[0], "loss_last": losses[-1], "n_steps_total": len(losses),
    "nan_steps": sum(1 for l in losses if l != l),
}
json.dump(res, open(args.result, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("substrate_probe","injection_env","buckets_pct","pointwise_sub_buckets_pct_of_total","sec_per_step_wall","loss_first","loss_last","nan_steps","clocks_sm_after")}, indent=1))
