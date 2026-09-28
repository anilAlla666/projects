# CP 2.4 sub-task (iii) — Llama-arm hang: root cause

**Date:** 2026-05-16. The Llama-arm GPU smoke (`spec_llama_smoke.py`) hung
~indefinitely (killed after 1 h 01 m; GPU pegged 100 %, no progress past
check 2). Root-caused by gdb stack + a 3-point bisection. **Not the same
mechanism as the earlier Marlin hang.**

---

## 1. The stack (gdb via sudo — ptrace_scope=1)

Main thread, state `R` (spinning), all ~95 other threads idle in
`futex_wait`:

```
#0–#12  libcuda.so.1  ...  ending at  cuKernelSetAttribute()
#13–#14 libcudnn_engines_runtime_compiled.so.9
#15–#17 cudnnBackendExecute  (libcudnn_graph.so.9)
#21     at::native::run_cudnn_SDP_fprop
#22     at::native::_cudnn_attention_forward
#26     at::native::_scaled_dot_product_cudnn_attention_cuda
#32     at::_ops::_scaled_dot_product_cudnn_attention::call   <- libcipher_rt.so
#33     at::native::scaled_dot_product_attention
#36     THPVariable_scaled_dot_product_attention  (PyTorch)
```

The hang is inside the CUDA driver, spinning in **`cuKernelSetAttribute`** —
a kernel-*setup* call (cuDNN's runtime-compiled attention engine setting a
JIT'd kernel attribute, almost certainly `MAX_DYNAMIC_SHARED_SIZE_BYTES`).
The SDP attention call passes through CIPHER's T4.6.1 attention substrate
(frame #32, `libcipher_rt.so`) — which here is a near-passthrough to real
cuDNN.

**This is NOT a kernel-residency deadlock** (the original Marlin hang was:
grid=132 persistent kernel cannot schedule all CTAs in an 8-SM green context).
This is the main thread spinning on a driver *setup* call. Same component
area (Marlin context handling), **different surface.**

## 2. Bisection — 6 runs (Llama-3.1-8B unless noted), all 64-token, single prompt

| # | config | result |
|---|---|---|
| A  | Marlin ON + substrate ON + **model-draft spec** (own stream) | **HANG** — cuDNN SDPA `cuKernelSetAttribute` |
| R1 | model-draft spec, **no v2 substrate** | PASS — 20 rounds, **accept 0.490, byte-identical to stock** |
| R2 | substrate ON + **Marlin OFF** + model-draft spec | PASS — identical clean result |
| T1 | Marlin ON + substrate ON + **no spec** (stock generate) | PASS — 64 tok, 36.2 s |
| T2 | Marlin ON + substrate ON + **n-gram spec** (no 2nd stream) | PASS — 39 rounds, 35.0 s, accept 1.000 |
| T3 | Marlin ON + substrate ON + model-draft spec, **draft on default stream** (`CIPHER_SPEC_DRAFT_STREAM=0`) | **PASS (no hang)** — 51 rounds, 53.9 s, **accept 0.036** |
| — | Mistral-7B + Marlin ON + substrate ON | PASS (50 clean runs — varied-prompt measurement) |

## 3. Conclusion — TWO distinct problems

### 3a. The hang = the draft's separate CUDA stream

A → T3 isolates it cleanly. The only delta between the hanging run (A) and
the clean run (T3) is whether the model draft runs on its **own
`torch.cuda.Stream()`** or the **default stream**. T1 (no spec) and T2
(n-gram spec, no 2nd stream) both pass — so it is **not** a broad
Marlin × cuDNN-attention problem, and **not** Marlin's kernels. It is a
**non-default CUDA stream + Marlin + substrate** interaction that hangs
cuDNN's runtime-compiled attention engine at `cuKernelSetAttribute` (§1
stack) on full-causal models. Mistral-7B never exposed it — its sliding-
window attention uses a different (precompiled) cuDNN engine.

**Fix is one line:** run the model draft on the default stream. The draft's
own stream gave *zero* benefit anyway — the draft↔target handoff is fully
synchronous (`int()` / `.tolist()` force syncs, draft and target never
overlap). Knob added: `CIPHER_SPEC_DRAFT_STREAM` (default 1; `0` → default
stream). Making `0` the default closes the hang.

### 3b. SEPARATE problem — model-draft acceptance collapses under the substrate

T3 does not hang, but `accept_rate` is **0.036** — vs **0.490** for the
identical model draft + prompt + 64 tokens with the substrate OFF (R1). The
draft's proposals almost never match the target under the v2 substrate
(Marlin INT4). Consequence: T3 took **53.9 s** for 64 tokens — *slower* than
stock T1 (36.2 s). **The model-draft Llama arm under the full substrate
yields negative lift.** Root cause not yet diagnosed — likely Marlin INT4
quantizing the 1B draft and 8B target such that the draft no longer tracks
the target (a weight-cache / two-model interaction is the leading
hypothesis). This is **not** a one-line fix.

## 4. Impact

- The **model-draft Llama arm** (the primary 1.75× criterion's intended
  vehicle) is non-viable under the v2 substrate as it stands — even with the
  stream-hang fixed, accept 0.036 → negative lift.
- The **n-gram Llama arm works** (T2) — clean under the full substrate.
- **Composed gate (step 3):** clean on Mistral-7B (50 runs); on Llama it
  works for stock / n-gram, not model-draft spec.
- The smoke caught both problems before the n=5 gate — ~10× cheaper than
  discovering them mid-measurement.

## 5. Decision required (see session report)

- **Option 3 — n-gram Llama arm.** Viable now (T2). Close CP 2.4 with two
  n-gram arms (Mistral 1.64× + Llama n-gram) + composed gate. The model-draft
  1.75× primary criterion is **not met** — documented, deferred.
- **Option A — diagnose + fix 3b** (the acceptance collapse) so the
  model-draft arm becomes viable. Open-ended: 3b is not yet root-caused.
  The stream-hang (3a) fix is free either way.
- **Option B — scope Llama out of CP 2.4 entirely.**

Anchors held — nothing rebuilt: kmod 0.4.8 `e2f50452`, libcipher_v2
`86618c30`, libcipher_rt `5e304549`. (`cipher_spec_decode.py` gained the
`CIPHER_SPEC_DRAFT_STREAM` knob — Python module, not a lib anchor.)
