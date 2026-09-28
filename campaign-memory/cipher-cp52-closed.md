---
name: cipher-cp52-closed
description: "CIPHER CP 5.2 (KV offload hierarchy) CLOSED 2026-05-17 — Option A snapshot-on-preempt; correctness gate PASS on TinyLlama + Mistral-7B; §4 #1 only"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7040044-d7bf-4078-b5f6-ce5458786ce4
---

CP 5.2 (KV offload hierarchy, Phase 5 — see [[cipher-phase5-scoped]]) **CLOSED 2026-05-17**.

**Campaign trail:** Step 1 scope memo → Step 2 adjudication (operator approved Option A earlier this campaign session: *"Option A APPROVED. Proceed to CP 5.2 Step 3"*) → Step 3 build + gate. The prior session built Step 3 legitimately but SSH dropped before paperwork; Step 3 build log + close memo were authored retroactively from on-disk artifacts. Evidence: `cipher-fusion-evidence/cp_5_2/` — `CP_5_2_STEP_1_OFFLOAD_PATH.md`, `CP_5_2_STEP_3_BUILD_LOG.md`, `CP_5_2_CLOSE_MEMO.md`, `cp52_offload_gate.py` + two PASS runlogs.

**What shipped:** `cipher_vllm_plugin/cipher_kv_offload.py` — vLLM v1 KV offload, Option A snapshot-on-preempt / restore-on-resume, two-tier HBM↔host-DRAM (NVMe deferred). 3 hooks: KV capture at `GPUModelRunner.initialize_kv_cache`; snapshot D2H on `Scheduler._preempt_request` + preserve `num_computed_tokens` so the resumed request rides vLLM's `num_computed_tokens>0` waiting branch and skips recompute; restore H2D on `Scheduler.schedule`. Gather/scatter via `torch.index_select`/`index_copy_` (no new pybind). Installed under the CP 5.1 `vllm.general_plugins` entry point; env gate `CIPHER_KV_OFFLOAD`.

**Gate PASS on both models:** TinyLlama-1.1B (MHA, 22 KV tensors) 6/6 snapshots/restores; Mistral-7B-v0.1 (GQA, 32 KV tensors) 7/7. Checks A (token-ID identity offload-vs-baseline), B (hooks fired, balanced), C (tenant tag preserved) all PASS. Diagnostic: vanilla recompute (control) diverged on both — **CIPHER offload is strictly MORE correct than vLLM's lossy recompute preemption** (exact-byte KV restore vs re-prefill with different FP reduction order).

**Close certifies §4 criterion #1 (correctness) ONLY** — same partial-criterion pattern as [[cipher-cp51-closed]]. NOT asserted: §4 #2 (KV-capacity extension + a *pre-registered* latency bound — no build memo, unmeasured; §6 PCIe-bandwidth risk untested) and §4 #3 (dedup hit-rate at 32K+ — **deliberately deferred** per Step 1's "offload narrow first", a separable axis for a later CP).

**Why:** campaign runs design-memo→approve→build per atomic step ([[cipher-fusion-campaign]]).

**How to apply:** Anchors all held, none rotated — CP 5.2 is pure vLLM-userspace Python, no kmod ABI: kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`, cipher_kv_bridge `8d6ffe3f`. Staged `cipher_proc.c` `0.4.5`→`0.4.8` banner edit still NOT built (rides next legitimate kmod rebuild). Phase 5: 2/5 CPs closed; next GPU work is CP 5.3 partition-aware Marlin.
