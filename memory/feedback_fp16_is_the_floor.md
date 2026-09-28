---
name: fp16/bf16 is CIPHER's compute precision floor — do not lower for the load-bearing path
description: User has locked fp16/bf16 as the minimum compute precision; quantization (INT4/INT8/FP8) is deferred to optional tiers only; weight STORAGE tricks that preserve fp16 numerical behaviour are permitted
type: feedback
---

**The rule**: fp16 and bf16 are CIPHER's **compute precision floor**. Never lower compute precision in any load-bearing instance. Weight STORAGE tricks (sparsity, structural compression) that preserve fp16 numerical behaviour during the matmul are permitted; weight COMPUTE precision reduction (INT4/INT8/FP8) is deferred to optional post-session-7 tiers.

**Why**: User stated this explicitly and firmly on 2026-04-05:
> "see we nned to maintain fp16 or bf16. int4 is toooooo lowwww and will fuck us up. we are gonna make something that we cant sell if we go for int4."

The reasoning is commercial, not technical: **for a new primitive establishing enterprise trust**, launching with quantization as the headline mechanism immediately positions CIPHER as a cost tier. Accuracy-critical customers (code gen, agents, tool-use, legal, medical) reject quantized variants by default. The reputation cost exceeds the throughput benefit for a v1 product. Even FP8 — which NVIDIA promotes as H100's native inference precision — carries the conversation overhead of "what's the quality hit?" and the user judged this unacceptable for initial positioning.

**How to apply**:

✅ **PERMITTED** (preserves fp16/bf16 compute):
- **2:4 structured sparsity** — weights are still fp16 values, just structurally zero at 50% of positions. H100 tensor cores have native sparse path. No precision loss. Pending explicit user approval before shipping.
- **Structural weight compression** (Koopman rank-r, Monarch, butterfly, block-low-rank) — factored weight matrices reconstruct to fp16 values in register before matmul; the matmul itself runs at fp16.
- **Graph capture / megakernel fusion / kernel-specialized substitution** — these don't touch precision at all.
- **Persistent kernels and L2 residency hints** — purely orthogonal to precision.
- **Speculative decoding** — uses a draft model but final compute is still fp16; would require non-drop-in cooperation so it is not the current plan.

❌ **NOT PERMITTED in the load-bearing path**:
- INT4 weight-only quant (including AWQ, GPTQ, Marlin)
- INT8 weight-only quant
- FP8 weight-only (even though NVIDIA ships it as H100 native — user rejected anyway)
- Any activation quantization
- Any compute-precision reduction on the matmul path

**What to do when arithmetic suggests quantization is needed**:
- Re-check the envelope first. Three times in session 7 I proposed quantization because I had the batch size or range wrong. The correct envelope (batch range 32–256 fp16 on H100) makes the 85% MFU gate reachable without quantization via overhead elimination.
- If the envelope is correct and arithmetic still demands precision reduction, STOP and tell the user. Do not propose a quant solution in memory and expect it to stick.
- 2:4 sparsity is the only "free" HBM lever that respects the fp16 promise. Propose it explicitly and get approval.

**When quant may return (later, never load-bearing)**:
- Post-session-7, as an opt-in throughput tier for customers who explicitly want it.
- Order if approved: FP8 weight-only → INT8 → INT4. Each as a separate optional instance with its own verification gate.
- See `project_instance_2_deferred.md`.

**Red flag to watch for**: any plan that uses phrases like "the INT4 path", "quant-enabled baseline", "low-precision tier as primary" in the context of the 85% MFU gate. Stop immediately. The gate is reached via fp16 overhead elimination, not quantization.
