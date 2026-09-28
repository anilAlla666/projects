# W.4b — cross-tenant POOL transport — ENTRY NOTE

**Date:** 2026-05-28
**Pre-state anchors (UNCHANGED at entry):** cipher_rt_phase4 `8b5e928` tag
`w4a-pool-eligibility`, libcipher_rt.so md5 `9fe23143b12e67355e7acc0b606d0c25`;
cipher_kmod `02fc2d1` (0.6.6); cipher-platform rev8 unchanged.
**Predecessor:** W.4a (POOL eligibility + correctness substrate, decision-only)
CLOSED 2026-05-28 09:40 (`W4A_POOL_ELIGIBILITY_CLOSE_REPORT.md`). W.4a §F #1
forecasts W.4b; this note discharges the mandated W.4b entry verification.

## Section F substep declaration

```
WORKLOAD CLASS:  agent inference multi-tenant (same-model, B=1 decode, cross-process)
CAPABILITY:      #? POOL executor batching (the transport half of D.6 row "W.4")
GOAL × WORKLOAD: Goal 1 (concurrent same-model agents/H100) + Goal 2/3 density
                 on multi-tenant same-model
ENGAGEMENT GATE: substrate-attributable tok/W on a reference config —
                 N=4 Mistral-7B 3.26× and N=8 TinyLlama 3.69× (CP 5.6 targets),
                 each correctness-gated (teacher-forced per-row logit-KL).
                 W.4a was substrate-only/decision-only; W.4b is the substep
                 that makes POOL move the customer-visible needle (Memory #25).
```

## Entry verification (the three things W.4a §F #1 said to check)

1. **"re-port" is real, not "re-implement" — source LOCATED.** The CP 5.6
   cross-tenant batching executor exists on disk at
   `cp_5_6/phase_b/session1/`: `cipher_batch_executor.py` (one process holds
   the model, runs the B=N continuous-batch decode loop + teacher-forced KL
   gate), `batch_client.py` (tenant client), `batch_ipc.py` (length-prefixed
   JSON over a Unix socket), `cp54_pool.py`, and the run harness
   (`run_batch_n.sh`, `run_mistral_verify.sh`). These produced the verified
   CP 5.6 numbers (N=8 3.69× / N=4 Mistral 3.26×, `CP_5_6_CLOSEOUT.md` §7).
   **No surface-to-Anil on source loss.**

2. **Discipline (j) — barrier stays substrate-side: VERIFIED.** The decode-step
   barrier is the executor's Unix-socket gather/scatter + per-round batch loop
   (`cipher_batch_executor.py:106-162`, `batch_ipc.py`). It is a standalone
   substrate orchestrator that transparently intercepts `model.forward` at the
   dispatch boundary (CP 5.6 design §2.3, same mechanism as
   `cipher_spec_decode.install()`). It is **NOT** a vLLM scheduler hook. The
   re-port preserves this; W.4b must keep the barrier in libcipher_rt-side IPC.

3. **Engagement gate + harness identified.** `N=8 bash run_batch_n.sh` (WL01
   TinyLlama) and `run_mistral_verify.sh` (Mistral N=4); tok/W is windowed via
   the `DECODE_START`/`DECODE_END` power sentinel against `nvidia-smi` power.
   Denominator is naive N-concurrent (substrate-attributable), per CP 5.6
   design §1. (The modified `phase_c/sc6_*` files in the working tree are
   Track 2 SC6 weight-sharing outputs — adjacent, not the W.4b batching gate.)

## W.4a substrate this builds on

Pool API on the current anchor (`cipher_rt_pool.h`): `cipher_rt_pool_partition`
/ `_group_get` (eligibility decision: same-fingerprint cohort peers only),
`_correctness_check` (per-tenant max-rel-diff backstop, tol 0.01),
`_mark_blocked` / `_is_blocked` / `_blocked_by_correctness` (block table),
counters `_eligible_groups` / `_solo` / `_distinct_fp_rejected` /
`_current_group_size`. W.4a delivered the decision + block primitive + check;
W.4b is the transport that calls them with real coalesced outputs.

## Sub-step plan (execute → close report; W.x cadence, no per-step scope-lock)

- **W.4b.1** re-port executor/IPC/client to a W.4b working dir; smoke-reproduce
  CP 5.6 N=4 TinyLlama batched tok/W at anchor `9fe23143` (confirm lever
  survives on the W.4a substrate before gating).
- **W.4b.2** gate batch formation through `cipher_rt_pool` eligibility:
  executor coalesces only same-fp client tenants; distinct-fp run solo
  (Memory #11 catastrophic-corruption guard at the real-coalescing site).
- **W.4b.3** wire failure→block→disable: `mark_blocked` on first-coalesce
  correctness fail + ratio auto-disable (>25% per-tenant / >50% session-wide).
- **W.4b.4** resolve self host-tgid (W.4a debt #2) for executor↔client cohort
  coordination (kmod QUERY caller tgid, or `/proc/self/status` NSpid).
- **W.4b.5** engagement gate on the GATED substrate (N=4 Mistral 3.26× / N=8
  TinyLlama 3.69×, correctness-gated) + `W4B_TRANSPORT_CLOSE_REPORT.md`.

## Risks at entry

- **R-W.4b.1 (HIGH — Memory #11 catastrophic surface).** W.4b is where real
  coalescing first happens; a fingerprint-collision or gate-bypass would
  silently corrupt a tenant. Mitigation: the W.4a correctness backstop runs
  first-coalesce per group; `mark_blocked` on fail; no scatter on a blocked
  group. The engagement gate is correctness-first (teacher-forced KL), not
  throughput-first.
- **R-W.4b.2 (executor form).** CP 5.6's verified executor is **Form A**
  (dedicated process owns the model; design §2.2). The W.4a eligibility layer
  is per-process-tenant (Form B-leaning). Form A is the lowest-risk re-port
  (it reproduces the verified result; the executor registers as a cohort
  tenant and coalesces same-fp clients). Form B (elected-tenant) preserves
  green-partition-per-tenant + composes with Track 2 capacity but is unbuilt.
  Default: re-port Form A; revisit at W.4b.2 if cohort coordination forces it.
- **R-W.4b.3 (Mistral env).** Prior sessions logged a Mistral env-block
  (E.7). N=4 Mistral is half the gate; TinyLlama N=8 is the other half.
  Surface if the env-block recurs.

## Status

Entry verification COMPLETE; no HARD STOP. Source clean, discipline-(j) PASS,
gate identified. W.4b is a multi-day transport build at the campaign's single
highest-risk substrate surface (Memory #11). Per "SURFACE BEFORE substrate
engineering" discipline, this note is the entry checkpoint before W.4b.1.
