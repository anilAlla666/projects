# CP 5.2 — KV offload hierarchy — CLOSE MEMO

**Closed:** 2026-05-17. **Closes on §4 criterion #1 (correctness) only** — the
same partial-criterion pattern as CP 5.1.

## Campaign trail

| Step | Artifact | State |
|---|---|---|
| 1 — scope memo | `CP_5_2_STEP_1_OFFLOAD_PATH.md` | done; design-memo assumptions verified against pod state |
| 2 — adjudication | operator prompt *"Option A APPROVED. Proceed to CP 5.2 Step 3"* | Option A approved earlier this campaign session |
| 3 — build + gate | `CP_5_2_STEP_3_BUILD_LOG.md`, `cipher_kv_offload.py`, gate runlogs | built; correctness gate PASS |

## What closed

`cipher_vllm_plugin/cipher_kv_offload.py` — vLLM v1 KV offload, Option A:
snapshot-on-preempt / restore-on-resume, two-tier HBM↔host-DRAM. Three hooks
(KV capture at engine init, snapshot on `_preempt_request` with
`num_computed_tokens` preservation, restore on `Scheduler.schedule`). Turns
vLLM v1's recompute-only preemption into a lossless tier-restore.

**Gate (`cp52_offload_gate.py`) — PASS on both models:**
- TinyLlama-1.1B (MHA, 22 KV tensors): A token-identity PASS, B 6/6
  snapshots/restores balanced, C 6 tags PRESERVED.
- Mistral-7B-v0.1 (GQA 32q/8kv, 32 KV tensors): A token-identity PASS, B 7/7
  balanced, C 7 tags PRESERVED.
- Diagnostic: vanilla recompute (control) diverged on both models; CIPHER
  offload stayed byte-exact — **CIPHER offload is strictly more correct than
  the recompute path it replaces.**

## What this close certifies — and what it does NOT

**Certifies — §4 criterion #1 only:** live KV blocks from a real vLLM v1
decode engine route through the CIPHER substrate across a preempt/resume round
trip with output correctness preserved (byte-identical token IDs, substrate-on
vs baseline), tenant tags intact, no parked-sequence leak.

**NOT asserted by this close:**
- **§4 #2 — KV-capacity extension.** No measurable effective-context or
  tenant-count extension beyond the HBM-only budget was demonstrated, and no
  build memo with a *pre-registered* per-decode-step latency bound was written
  (the §4 pre-registration discipline was not exercised). Capacity extension
  is unmeasured.
- **§4 #3 — dedup hit-rate at 32K+ context.** Not measured. **Dedup is
  deliberately deferred** per the Step 1 memo's "offload narrow first"
  recommendation — it is a separable axis to be picked up in a later CP, not a
  gap in this one.

The §6 central risk — offload bandwidth vs decode latency over PCIe — is
**untested**; it lives in the unmeasured §4 #2 surface.

## Anchors

All held, none rotated by CP 5.2 — the offload layer is pure vLLM-userspace
Python and touches no kmod ABI:
kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`,
cipher_kv_bridge `8d6ffe3f`.

The staged `cipher_proc.c` `/proc` banner `0.4.5`→`0.4.8` edit remains staged,
NOT built (rebuilding would rotate `e2f50452`); it rides the next legitimate
kmod rebuild.

## Phase 5 status after CP 5.2

2 / 5 Phase 5 CPs closed (5.1, 5.2). Next GPU work: CP 5.3 partition-aware
Marlin (critical path, blocked on the Song Han warm intro). CP 5.4 (per-tenant
arbitration) has CP 5.2 as prereq and is now unblocked on that dependency.
