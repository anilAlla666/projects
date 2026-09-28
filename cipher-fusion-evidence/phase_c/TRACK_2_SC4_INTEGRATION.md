# Track 2 SC4-2 — INTEGRATION TESTS

**Date:** 2026-05-19. `sc4_run.py` — one producer (TinyLlama), two consumer
scenarios. **Verdict: PASS** (both scenarios).

---

## Scenario 1 — matching model → shared import

Consumer model == producer model (TinyLlama-1.1B).

| check | result |
|---|---|
| fingerprint verify | **MATCH** — consumer proceeds with the shared import |
| consumer mode | `shared` |
| **bit-identical forward** (Y_producer == Y_consumer) | **true, max-abs-diff 0.0** |
| consumer GPU FB added | **0 MiB** |

A matching fingerprint → the SC3 weight-sharing path runs unchanged:
bit-identical forward, zero added weight memory. SC4 adds the safety gate
without disturbing the SC3 behaviour.

## Scenario 2 — mismatched model → clean fallback

Producer = TinyLlama-1.1B; consumer model = **Llama-3.2-1B-Instruct** (a
genuinely different model).

| check | result |
|---|---|
| fingerprint verify | **MISMATCH — detected at tier 1** (different config/architecture) |
| mismatch logged | producer_hash `72b3d0ab…`, consumer_hash `641a30ef…`, both model paths, `differing_tier=1` — the full Item-3-PUSH detail |
| arena import | **not attempted** — verification fired *before* `weight_arena_import` (zero VMM resource committed) |
| consumer mode | `fallback_independent` |
| fallback forward | **`forward_ok=true`** — the consumer loaded Llama-3.2-1B independently and ran (logits `[1,14,128256]` — Llama-3.2's vocab) |

The mismatch is caught **before** any resource commitment, logged with
operator-diagnosable detail, and the consumer falls back to an independent
load that runs correctly. The silent-wrong-output failure mode is closed.

## The lifetime fix (see `TRACK_2_SC4_BUILD_LOG.md`)

Scenario 1 initially failed — the matching consumer crashed in the forward
because `run_shared()` let the `WeightArena` local be GC'd (unmapping the
arena under the model's tensors). Fixed by pinning the arena to the model
(`model._cipher_weight_arena = arena`); re-run PASS. Surfaced honestly — it is
the lifetime coupling SC5 (kmod-owned arena) makes robust.

## Verdict

**PASS** — matching → safe shared import (bit-identical, 0 MiB); mismatched →
clean pre-commit rejection + logged + correct fallback. Evidence:
`sc4_integration_result.json`, `sc4_consumer_*.log`, `sc4_producer.log`.
