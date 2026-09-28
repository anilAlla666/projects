# CP 5.4 — Step 1.6B-1 (partition-tenant workload runner) — CHECKPOINT

**Date:** 2026-05-19. **Status: COMPLETE.** Sub-phase checkpoint (Step 1.6 is
multi-session). Anchors unchanged — throwaway harness only.

---

## Built

`cp54_s16_partition_tenant.py` — a sustained-decode PARTITION-class tenant.
Runs under `CUDA_INJECTION64_PATH=<libcipher_rt.so>` with
`CIPHER_QOS_CLASS=partition` + `CIPHER_SM_COUNT=<n>`; loads TinyLlama-1.1B, runs
the WL01 5-prompt workload as a sustained KV-cached greedy decode. Captures:

- **per-decode-step wall latency** (sync-bracketed; the prefill step is
  recorded separately and excluded from the decode population — memo §11.2);
- **per-prompt teacher-forced logit-KL** vs WL01 gold (correctness gate ≤ 0.1);
- a **per-round `%smid` probe** on the green-ctx stream → observed physical SM
  set, with a **clause-1 self-check** (`observed ⊆ allocated`).

Coordinates with the orchestrator over stdin/stdout (`READY` / wait `START` /
`ROUND r SMS …` / `DONE`), Phase B pattern. Allocation is read authoritatively
from the kmod ledger (`CIPHER_CP54_QUERY`).

## Two findings during build

1. **Loader: `CUDA_INJECTION64_PATH`, not `LD_PRELOAD`.** libcipher_rt's
   green-context init runs in `InitializeInjection` (`cipher_inject.c`), which
   the CUDA driver fires *only* for the injection path — CP 2.5 made the
   substrate LD_PRELOAD-free + GOT-patched. A first sanity run with plain
   `LD_PRELOAD` produced **no green context** (`%smid` probe showed all 132
   SMs — partition not enforced). Re-run with `CUDA_INJECTION64_PATH` →
   green context activates, `%smid` shows exactly the allocated 16 SMs.
   **The 1.6B-2 orchestrator must launch PARTITION tenants via
   `CUDA_INJECTION64_PATH`.** Harness lesson, not a substrate defect.
2. **`cipher_rt_green_ctx_sm_count()` is not a reliable `ctypes` read** under
   the injection load (the symbol's static state is not visible to a
   `CDLL(None)` reader). Resolved by reading the allocation from the kmod
   ledger via `CIPHER_CP54_QUERY` instead — authoritative, and exactly the
   source memo §6 specifies. The `%smid` probe is the ground truth for the
   *observed* SM set; both work.

## Standalone sanity (qos=partition, 16 SM, 4 rounds + 1 warmup)

`b1_sanity_p0.json`. kmod ledger: `grp_mask=0x0003` → 2 groups / 16 SMs,
`n_part=1 free=13`. TFGATE: all 5 prompts PASS, `kl_max=5.5e-5`. 396 decode
steps; mean 15.0 ms, p50 14.0, p95 14.5, **p99 45.7** (the per-round first
decode step is a ~45 ms shape-warmup outlier — ≈1 % of steps; the 1.6B-4
analysis should exclude step 1 of each round, or report with/without).
`%smid` probe = the 16 allocated SMs every round; **clause-1 0/4 fails**. A
warmup round is run before the timed rounds (round 0 was otherwise a 45 ms→14 ms
cold outlier).

## Next — Phase 1.6B-2 (orchestrator)

Launch the POOL executor + N PARTITION tenants concurrently (PARTITION tenants
via `CUDA_INJECTION64_PATH`); per-round barrier; collect per-tenant latency +
correctness; run the central two-clause disjointness check
(`CIPHER_CP54_QUERY` for every tenant's mask + each tenant's reported observed
SMs). Then 1.6B-3 naive baseline, 1.6B-4 sweep.

**OP-2 arithmetic still open** (memo §11.1) — `2×16 SM + pool` is 4+11 groups
(88-SM pool), not the "13 groups/104 SM" in the adjudication text; needs user
confirm before 1.6B-4. Does not block 1.6B-2/-3.
