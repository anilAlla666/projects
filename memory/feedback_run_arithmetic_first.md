---
name: Run the arithmetic before committing to a plan
description: Three plan resets in session 7 all came from proposing a mechanism before running the physics arithmetic; always compute the roofline FIRST and derive the plan from the numbers
type: feedback
---

**The rule**: Before committing to any performance plan, **run the roofline arithmetic for the actual workload envelope** (hardware peak compute, HBM bandwidth, model weight size, batch size, precision). Derive the plan from the physics, not from hunches about what technique is trendy.

**Why**: Session 7 on 2026-04-05 had **three plan resets in one conversation**, and every one was caused by skipping the arithmetic step:

1. **First miss (batch=1 anchoring)**: proposed graph-capture + allocator interceptor as the load-bearing path. Never checked that batch=1 MFU ceiling is ~0.5% regardless of software quality. Wasted ~200 lines of pool-observer code chasing a hypothesis the arithmetic would have rejected in 30 seconds.

2. **Second miss (INT4 overcorrection)**: when user pointed out nobody runs batch=1 in production, I anchored on batch=32–64 and proposed INT4 quantization as the primary lever. Didn't check that fp16 at batch=256 already hits the 85% ceiling on its own. Proposed a plan that violated the "no quantization" constraint before checking if quantization was even needed.

3. **Third miss (literal batch=32)**: when user rejected INT4 and fixed precision at fp16/bf16, I anchored on batch=32 literally and concluded the gate was physically impossible without exotic mechanisms. Didn't ask whether the "batch=32" meant a single point or a range. It was a range (32–256), and the upper end trivially hits the gate.

All three were avoidable with one discipline: **compute the arithmetic first, propose the plan second.**

**How to apply**:

Before proposing ANY performance plan, write out:

1. **Peak compute** of target hardware at target precision (e.g. H100 fp16 = 989 TFLOPs)
2. **Peak bandwidth** (HBM3 on H100 = 3.35 TB/s)
3. **Ridge point** (peak compute ÷ peak bandwidth = FLOPs/byte at crossover). For H100 fp16 this is 295.
4. **Per-step HBM bytes fetched** for the workload (weight size + activation traffic)
5. **Per-step compute done** (batch × 2 × params for decode steps)
6. **Binding constraint**: whichever takes longer, HBM time or compute time
7. **Theoretical MFU ceiling**: compute_done / (peak_compute × elapsed)
8. **Roofline table across the batch range the user actually runs**

THEN propose mechanisms that attack whichever of (HBM, compute, overhead) is the binding constraint at the relevant batch points.

**For CIPHER specifically** (H100 fp16 stable envelope):
- Ridge point: 295 FLOPs/byte
- At batch ≥ 256: compute-bound (or nearly so); bottleneck is overhead and kernel efficiency
- At batch 32–128: HBM-bound hard; ceiling is batch/295 × 100% MFU
- CIPHER's primary job is closing the gap between measured PyTorch and the ceiling at each batch point
- The 85% gate is physically reachable only at or near batch=256 at fp16 — or at lower batches with HBM-reducing techniques that preserve fp16 (2:4 sparsity, structural compression)

**Red flag**: any plan that proposes a mechanism (quant, fusion, capture, etc.) before I've written out the roofline numbers. Stop and do the arithmetic. It takes 5 minutes and prevents full-day resets.

**Related memories**:
- `project_operating_envelope.md` — has the roofline table; keep it updated
- `feedback_batch_size_range.md` — the correct envelope shape
- `feedback_batch_size_miss.md` — first anchoring mistake
