---
name: Check real production serving before anchoring on batch size
description: Always verify the actual production workload shape before committing to an envelope — "batch=1" is almost never what a real inference server runs
type: feedback
---

**The rule**: Before committing to any performance envelope, verify what production inference serving stacks actually do at the workload-shape level. Never anchor on "batch=1" without first checking whether that's the real workload.

**Why**: In session 7 (2026-04-05) I spent most of the session optimizing CIPHER against literal batch=1 TinyLlama `model.generate()`. This was wrong. No production LLM serving stack runs at literal batch=1 — vLLM, TGI, TRT-LLM-Backend, SGLang, and Triton all use **continuous batching** (iteration-level batching), packing 8–64 concurrent user sessions into every forward pass. The user called this out directly: "honestly no chat app works at batch at 1, they configure these batches and process right? how come we missed this?"

I anchored on the literal phrasing "batch=1 is the biggest market" from an earlier user message and never sanity-checked against what real serving looks like. The user meant "per-user interactive latency is the dominant market" (each user's session experienced as a stream), which is satisfied by continuous batching at batch=16–64 — not by literal server-side batch=1.

**The physics consequence of the miss**:
- At literal batch=1: HBM is the binding constraint, MFU 85% is physically impossible (everyone including TRT-LLM tops out around 5-10% MFU)
- At batch=32-64: arithmetic intensity rises 32-64×, MFU 85% becomes physically reachable with INT4 weight-only quant on H100 for 7B-class models
- I spent days engineering around an impossible target

**How to apply**:
- Before designing any inference-performance work, ask: "what does the user's production serving stack actually look like? Continuous batching? Static batching? Single-user local?"
- If the answer isn't explicit, CHECK against what vLLM/TGI/TRT-LLM do by default. Their defaults are the de-facto definition of "production serving".
- "batch=1" from a user likely means "per-session latency matters", NOT "the server processes one sequence at a time". Clarify.
- Run a quick arithmetic check early: at the stated batch size, what does the memory-compute balance look like on the target hardware? If MFU 85% is physically unreachable, raise it immediately — do not accept the goal and then fail to deliver.
- For interactive LLM workloads, the realistic envelope is continuous batching at batch=16–64 per GPU, tensor-parallel across 1-8 GPUs per replica, multiple replicas per cluster. Memorize this as the default.
- Run `nvidia-smi dmon` or equivalent on an actual production inference workload if available — see what batch sizes the GEMMs actually have.

**Red flag to watch for**: anchoring a multi-phase plan on a physics-impossible target without the user having explicitly verified the envelope. When the arithmetic says "no one in the industry achieves this number", the envelope is wrong, not the product.
