# Track 2 SC4-2 — UNIT TESTS

**Date:** 2026-05-19. `sc4_unit.py` — `cipher_model_fingerprint` on synthetic
model directories (valid `config.json` + arbitrary-byte `*.safetensors`).
**Verdict: 11/11 PASS.**

| # | check | result |
|---|---|---|
| 1 | deterministic — same dir → identical fingerprint | PASS |
| 2 | tier-1 differs on a config change (arch/layers) | PASS |
| 3 | tier-2 unchanged when only config differs | PASS |
| 4 | combined differs (tier-1 mismatch) | PASS |
| 5 | tier-1 unchanged when only weights differ | PASS |
| 6 | tier-2 differs on a weight change | PASS |
| 7 | **large shard — a byte in an unsampled gap → tier-2 unchanged** (the head/mid/tail sampling works as designed) | PASS |
| 8 | **small shard < 3 MiB — full-hash → a mid-shard byte change IS detected** (Item-1 PUSH guard) | PASS |
| 9 | `verify_fingerprint` — matching model → `True` | PASS |
| 10 | `verify_fingerprint` — tier-1 mismatch → raises `FingerprintMismatch(differing_tier="1")` | PASS |
| 11 | `verify_fingerprint` — tier-2 mismatch → raises `FingerprintMismatch(differing_tier="2")` | PASS |

Tests 7 + 8 jointly verify the two `_shard_tier2_hash` branches: a ≥ 3 MiB
shard is *sampled* (an unsampled-gap byte is invisible — by design, the
documented v1 boundary) and a < 3 MiB shard is *full-hashed* (every byte
matters — the Item-1 PUSH guard, so small shards are not under-covered).

Tests 10 + 11 verify the differing-tier attribution that the mismatch log
(Item-3 PUSH) reports to the operator.
