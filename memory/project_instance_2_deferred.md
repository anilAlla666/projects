---
name: Instance 2 (quantization) deferred — optional tier, NOT load-bearing
description: INT4/INT8/FP8 weight quantization is explicitly deferred after user rejection; may return as an opt-in throughput tier post-session-7 but is never on the critical path to the 85% MFU gate
type: project
---

**Status as of 2026-04-05**: Instance 2 (precision-reducing quantization) is **DEFERRED** and **NOT** part of the load-bearing path to CIPHER's 85% MFU gate.

## Why it was proposed

Under the (incorrect) literal-batch-1 and literal-batch-32 envelopes I briefly ran arithmetic on, fp16 could not reach 85% MFU at the stated batch size. INT4 weight-only quant was proposed as the primary lever.

## Why it was rejected by the user

Direct quote: *"see we nned to maintain fp16 or bf16. int4 is toooooo lowwww and will fuck us up. we are gonna make something that we cant sell if we go for int4."*

Reasoning (user's, validated): for a **new primitive trying to establish enterprise trust**, launching with "quantized" as the headline mechanism immediately pigeonholes CIPHER as a cost-tier product. Accuracy-critical customers (code generation, agents, legal/medical, tool-use) reject the quantized tier by default. The reputation cost is larger than the throughput benefit for a v1 product.

## Why it is no longer needed

After the batch-range correction (32–256 instead of fixed 32) on 2026-04-05, the physics at batch=256 fp16 on H100 permits MFU ≥85% through overhead elimination alone (Instances 4+3). Quantization is not on the critical path for the gate. See `project_operating_envelope.md` roofline table.

## When (if ever) it may return

Post-session-7, as an **optional opt-in tier** for customers who explicitly want further throughput and accept the accuracy tradeoff. Order of re-introduction if approved:

1. **FP8 weight-only** (H100 native, <0.1% perplexity drift, NVIDIA Transformer Engine default). Closest to fp16 quality. Cleanest reputation.
2. **INT8 via cuBLASLt** (2× HBM, ~0.5% drift with proper per-channel scaling). Standard throughput tier.
3. **INT4 via AWQ/GPTQ + Marlin-style kernel** (4× HBM, 1–3% drift). Cost tier only.

Each would be a separate instance with its own verification gate and explicit user opt-in. None are session-7 work.

## Code state

Zero code written for Instance 2. Nothing to unwind. No `cipher_int4*` or `cipher_quant*` files exist.

## How to apply when resuming

- **Do not propose Instance 2 as a path to the 85% gate.** The gate is reached via Instances 4+3 at fp16.
- **Do not re-run the arithmetic that suggests quant is needed.** That was based on the wrong envelope. See `project_operating_envelope.md` for the correct roofline.
- If the user explicitly asks for a throughput tier beyond the 85% gate, this file is the starting point for that work — but only then.
- **2:4 structured sparsity is NOT quantization** and is permitted (pending user approval) — see `feedback_fp16_is_the_floor.md`.
