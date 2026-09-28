# Track 2 SC4 — PRE-BASELINE

**Date:** 2026-05-19. Light baseline — SC4 is pure Python (no anchor rotation).

**Anchors (md5-verified, all unchanged from SC3 close):**
kmod `285d102e`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `86618c30`. SC4 modifies no binary — these stay byte-identical
through SC4 (no `.pre_sc4` binary fallback needed).

| check | pre-SC4 | SC4 post-build gate |
|---|---|---|
| W3 — Track 2 SC2 weight arena (`sc2_verify.py`) | **`SC2_PASS=True`, KL 0.0** | `SC2_PASS=True`, KL 0.0 |
| SC3 integration — `sc3_run.py` (unmodified SC3 scripts) | **PASS** — bit-identical both runs (same-VA + offset-relative) | PASS (SC3 behavior must not regress) |
| isolation — `cp54_isolation_test` | 15/15 (Track 3 substrate, untouched) | 15/15 |
| dmesg | clean | clean |

The SC3 producer/consumer scripts (`sc3_*.py`) are **not modified** by SC4 —
SC4 adds new `sc4_*.py` (= sc3 + fingerprinting). So "SC3 integration still
PASS" is a literal re-run of the unchanged `sc3_run.py`.

## Models for SC4 testing

`models/`: TinyLlama-1.1B (single-file `model.safetensors`, 2.20 GB — no
shard index), Llama-3.2-1B-Instruct (the realistic *different-model* for the
mismatch integration test), plus Mistral-7B-v0.1, Llama-3.1-8B, etc.

## Verdict

Pre-SC4 baseline established. SC4 post-build smoke must reproduce: W3
`SC2_PASS`, SC3 integration PASS, isolation 15/15, dmesg clean — and all four
anchor md5s byte-identical (SC4 changes no binary).
