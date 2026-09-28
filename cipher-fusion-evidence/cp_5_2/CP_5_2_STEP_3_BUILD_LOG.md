# CP 5.2 — Step 3: vLLM v1 KV offload (Option A) — BUILD LOG

**Date:** 2026-05-17. Option A adjudicated **APPROVED** earlier this campaign
session (operator prompt: *"Option A APPROVED. Proceed to CP 5.2 Step 3"*),
ratifying the recommendation in the Step 1 scope memo
(`CP_5_2_STEP_1_OFFLOAD_PATH.md` §(d) — snapshot-on-preempt / restore-on-resume,
two-tier HBM↔host-DRAM, NVMe cold tier deferred, dedup held as a separable
"offload narrow first" axis). This is the build/measure log — not a design
memo. Findings are recorded as engineering calls.

This log is written retroactively: the prior session built and gated Step 3
but the SSH connection dropped before it could be authored. The build
artifacts and gate runlogs on disk are the primary evidence; this document
narrates them.

Anchors held — none rotated by CP 5.2:
kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`,
cipher_kv_bridge `8d6ffe3f`. CP 5.2 touches no kmod ABI (Step 1 §F: residency
is a vLLM-scheduler userspace concern; the kmod observes nothing about it).

---

## What was built

`cipher_vllm_plugin/cipher_kv_offload.py` — installed from
`cipher_vllm_kv.register()` so the whole CIPHER substrate (CP 5.1 buffer
ownership + CP 5.2 offload) composes under one `vllm.general_plugins` entry
point. Env gate: `CIPHER_KV_OFFLOAD` (`1`|`0`, default `1`).

vLLM v1 preemption is recompute-only: `Scheduler._preempt_request` frees the
request's KV blocks and zeroes `num_computed_tokens`, so a resumed request
re-runs prefill from scratch (Step 1 memo §(a) F5). CP 5.2 turns that discard
into a two-tier HBM↔host-DRAM offload.

### Hook architecture — 3 hooks

1. **KV capture at engine init** — `GPUModelRunner.initialize_kv_cache`.
   After the original runs, bind the live per-layer KV tensors and
   `num_blocks` into the process-wide `_OffloadManager`. The profiling-cache
   pass (`is_profiling=True`) is skipped — only the real cache is bound.

2. **Snapshot on `_preempt_request`** — before the original frees the blocks,
   gather-copy the request's KV blocks D2H into a pinned host-DRAM store keyed
   by `request_id`. After the original runs (blocks freed, `num_computed_tokens`
   zeroed), **restore `num_computed_tokens`** so the resumed request rides
   vLLM's existing `num_computed_tokens > 0` waiting-loop branch
   (`scheduler.py` ~line 657, the KV-connector path) and skips recompute.

3. **Restore on resume** — `Scheduler.schedule`. After the original schedules,
   any parked request that is now in `self.running` has had fresh blocks
   allocated; scatter-copy its KV H2D into those blocks before `execute_model`.

Gather/scatter uses `torch.index_select` / `Tensor.index_copy_` along the
detected block dimension over the strided `[2, num_blocks, …]` K/V layout —
backend-agnostic and correct-by-construction. **No new `cipher_kv_bridge`
pybind gather method was added**: the torch ops cover gather/scatter directly;
the bridge is reused only for `page_info` (the tenant-tag check, gate
criterion C). Adding a pybind path would have duplicated working torch
functionality and rotated a non-anchor artifact for no gain.

---

## Findings (engineering calls)

### F1 — block-dim detection: `num_blocks` collided with `head_size`

With a tiny `num_gpu_blocks_override` the detected `num_blocks` (64) coincided
with `head_size` (64), so a naive "find the dim whose size == num_blocks"
search returned two candidate dimensions and could have gather-copied along
the wrong axis. **Fix:** the innermost dim of a paged KV tensor is always
`head_size` and is never the block dim, so it is excluded from the candidate
set; the bind then requires exactly one match and every layer must agree on
it, else offload binds INACTIVE rather than guessing. Correct-by-construction
over the collision.

### F2 — pool sizing to force preemption

The gate must actually trigger preemption, which means a KV pool too small for
all concurrent requests. vLLM refuses to start if the pool cannot hold even
one `max_model_len` request, so `max_model_len` was tuned down across the
build: **2048 → 1024 → 512 → 480**. With `num_gpu_blocks_override=32` the pool
is 31 usable blocks (vLLM reserves a null block) × 16 = 496 token slots; 480
clears that for **both** TinyLlama and Mistral-7B. Six concurrent requests
generating ~268 tokens each contend for the 32-block pool and force
preemption — confirmed by the snapshot/restore counts below.

### F3 — gate verdict logic correction: control is a diagnostic, not a criterion

An early verdict draft treated "control (vanilla recompute) == baseline" as a
pass criterion. **This is wrong.** vLLM v1 recompute re-runs prefill; prefill
and incremental-decode kernels differ in floating-point reduction order, so
recompute is *numerically equivalent* but **not guaranteed byte-identical**. A
control divergence is expected and is in fact corroborating evidence — it
proves the tiny pool forced destructive preemption. The verdict was corrected:
control-vs-baseline is a **diagnostic only**; the three pass criteria are A
(offload == baseline token identity), B (hooks fired, snapshots == restores),
C (tenant tag preserved).

---

## Gate results — `cp52_offload_gate.py`

The gate runs the worker three times in separate processes: BASELINE (large
pool, no preemption), OFFLOAD (32-block pool, offload ON), CONTROL (32-block
pool, offload OFF). Runlogs: `cp52_offload_gate.runlog`,
`cp52_offload_gate_mistral.runlog` (2026-05-17 17:37–17:41).

### TinyLlama-1.1B — MHA, 22 KV layer tensors

| Check | Result |
|---|---|
| A — token-ID identity (offload vs baseline), 6 prompts | **PASS** — all 6 byte-identical |
| B — hooks fired | **PASS** — installed; 6 snapshots / 6 restores, balanced (no parked-sequence leak) |
| C — tenant tag preserved | **PASS** — 6 PRESERVED, 0 MISMATCH |
| Diagnostic — control vs baseline | recompute diverged on prompt 3 at token 222 (expected) |

### Mistral-7B-v0.1 — GQA 32q / 8kv, 32 KV layer tensors

| Check | Result |
|---|---|
| A — token-ID identity (offload vs baseline), 6 prompts | **PASS** — all 6 byte-identical |
| B — hooks fired | **PASS** — installed; 7 snapshots / 7 restores, balanced |
| C — tenant tag preserved | **PASS** — 7 PRESERVED, 0 MISMATCH |
| Diagnostic — control vs baseline | recompute diverged on prompt 4 at token 104 (expected) |

**`CP 5.2 STEP 3 OFFLOAD CORRECTNESS GATE: PASS`** on both models.

### F4 — diagnostic finding: offload is strictly more correct than recompute

In both runs the control (vanilla recompute) diverged from baseline while the
CIPHER offload stayed byte-exact. This is the substantive result, not just a
sanity check: vLLM v1's native preemption recovery is **lossy** — re-prefill
produces a numerically different KV than the sequence had before preemption.
CIPHER's snapshot/restore copies the exact KV bytes, so a preempted-and-resumed
request decodes **identically** to one never preempted. CIPHER offload is
strictly more correct than the path it replaces.

---

## Scope of this build

This gate certifies CP 5.2 design-memo **§4 criterion #1 (correctness)** only:
live KV blocks route through the substrate with output correctness preserved.

**Not asserted by this build:**
- §4 #2 — measurable KV-capacity extension (effective context length, or
  tenant count at fixed context) beyond the HBM-only budget, with
  per-decode-step latency overhead within a *pre-registered* bound. No build
  memo with a pre-registered latency bound was written; capacity extension was
  not measured.
- §4 #3 — dedup hit-rate on shared-prefix traffic at 32K+ context. Deferred
  per the Step 1 memo's "offload narrow first" recommendation; dedup is a
  separable axis for a later CP.

See `CP_5_2_CLOSE_MEMO.md` for the close decision.
