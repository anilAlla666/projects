# Phase C / Track 2 — SC4 CLOSEOUT — model-identity fingerprinting

**Date:** 2026-05-19. **Verdict: SC4 COMPLETE — cross-tenant weight-sharing is
now safe against wrong-model import.** **No anchor rotation** — SC4 is pure
Python; the substrate is byte-identical. Stopping for adjudication before SC5.

---

## §1 — What SC4 delivered

SC3 verified weight-sharing *assuming* a matching model. SC4 makes that
assumption **checked** — the safety primitive that turns a silent
wrong-model import (consumer rebinds onto wrong bytes, serves wrong tokens,
no error) into a clean, logged rejection + fallback.

- `cipher_model_fingerprint.py` — a **two-tier SHA256** model fingerprint:
  tier 1 structural (config + index + shard sizes + dtype), tier 2 sampled
  weight bytes (1 MiB head/mid/tail per shard; **full-hash for shards
  < 3 MiB** — Item-1 PUSH). `verify_fingerprint` raises `FingerprintMismatch`
  with tier attribution + operator-diagnosable detail (Item-3 PUSH).
- The consumer verifies **before** `weight_arena_import` (pre-resource-commit);
  on mismatch it logs the full detail and **falls back to an independent
  load** (no regression, no silent wrong output).

## §2 — Verification

| gate | result |
|---|---|
| unit tests | **11/11 PASS** — tier-1/tier-2 discrimination, large-shard sampling, the <3 MiB full-hash guard, `verify` tier attribution |
| integration — matching model | **PASS** — fingerprint match → shared import, bit-identical forward (max-abs-diff 0.0), 0 MiB added |
| integration — mismatched model | **PASS** — mismatch detected at tier 1 *before* import, logged with detail, clean fallback to an independent Llama-3.2-1B load that runs correctly |
| regression | **PASS** — all 4 anchor md5s byte-identical, W3 `SC2_PASS`, isolation 15/15, dmesg clean |

## §3 — Findings (honest)

- **Lifetime bug found and fixed (build).** SC4's factoring of the consumer
  into `run_shared()` exposed a latent lifetime coupling: the `WeightArena`
  must outlive the model's `from_blob` parameter views. As a function local
  it was GC'd on return → the arena unmapped → the forward faulted. Fixed by
  pinning the arena to the model (`model._cipher_weight_arena = arena`). This
  is precisely the lifetime concern **SC5 (kmod-owned arena) makes robust** —
  the SC4 fix is the interim user-space pin. Surfaced, not papered over.
- **SC4 is anchor-stable** — the SC4-1 design-memo finding held: model-identity
  is a verification-layer concern (hash of disk files + a manifest field + a
  compare), pure Python, no `cipher_kv_bridge` change. Confirmed: all 4 anchor
  md5s byte-identical.
- **v1 boundary (carried from SC4-1):** the fingerprint is operator-error-proof,
  not adversary-proof — large-shard sampling leaves unsampled regions (unit
  test 7 demonstrates this directly). A full weight hash is the v2 hardening.

## §4 — Anchors

kmod `285d102e`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `86618c30` — **all unchanged through SC4** (md5-verified). No
rotation, no `.pre_sc4` binary fallback needed.

## §5 — SC4 deliverables

`TRACK_2_SC4_DESIGN_MEMO.md`, `TRACK_2_SC4_PRE_BASELINE.md`,
`TRACK_2_SC4_BUILD_LOG.md`, `cipher_model_fingerprint.py`,
`sc4_{producer,consumer,run,unit}.py`, `TRACK_2_SC4_UNIT_TESTS.md`,
`TRACK_2_SC4_INTEGRATION.md`, `TRACK_2_SC4_REGRESSION.md`, this closeout.

## §6 — Adjudication ask

**STOPPING — SC4 complete, all gates PASS, no anchor rotation.** Confirm:
1. **SC4 = PASS** — model-identity fingerprinting works; matching → safe
   shared import, mismatched → clean pre-commit rejection + fallback.
2. The §3 findings — the lifetime bug + fix; SC4 anchor-stable.

On adjudication, proceed to **SC5** — kmod-owned arena lifetime (refcount +
`do_exit` reaper, so the shared arena survives the producer crashing/exiting).
SC5 is the **heavier** Track 2 SC: it touches the kmod and rotates an anchor.
The SC4 user-space arena pin (§3) is the interim that SC5 supersedes.
