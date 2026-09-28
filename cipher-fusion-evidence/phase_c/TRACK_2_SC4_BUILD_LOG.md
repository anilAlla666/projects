# Track 2 SC4-2 — BUILD LOG

**Date:** 2026-05-19. **Scope:** model-identity fingerprinting. **Pure Python
— no binary changed, NO anchor rotation.** kmod `285d102e`, libcipher_rt
`83afd1ca`, cipher_kv_bridge `c04b0c39`, libcipher_v2 `86618c30` — all
md5-verified byte-identical before and after SC4.

## New files (all `cipher-fusion-evidence/phase_c/`)

| file | role |
|---|---|
| `cipher_model_fingerprint.py` | the fingerprint helper — `compute_fingerprint(model_path)` (two-tier SHA256), `verify_fingerprint(manifest, local_model)` raising `FingerprintMismatch` with tier attribution |
| `sc4_producer.py` | = `sc3_producer.py` + the manifest carries a `fingerprint` field |
| `sc4_consumer.py` | = `sc3_consumer.py` + verify-before-import; on mismatch: explicit detailed log + fallback to independent load |
| `sc4_run.py` | integration driver — matching + mismatch scenarios |
| `sc4_unit.py` | fingerprint unit tests |

The SC3 scripts (`sc3_*.py`) are **not modified** — SC4 adds parallel `sc4_*`
files. So "SC3 integration still passes" is a literal re-run of unchanged code.

## Fingerprint design (as built)

`compute_fingerprint` → `{tier1, tier2, combined, shard_metadata, n_shards}`:
- **tier1** — SHA256 of `config.json` + `torch_dtype` + the safetensors index
  (if present) + per-shard `{name, size}`.
- **tier2** — SHA256 over per-shard sampled weight bytes: 1 MiB head/mid/tail
  for shards ≥ 3 MiB; **full-shard hash for shards < 3 MiB** (Item-1 PUSH —
  the three windows would overlap/exceed a small shard).
- **combined** — `SHA256(tier1 ‖ tier2)`.

`verify_fingerprint` compares the consumer's recomputed fingerprint to the
producer's (from the manifest); on mismatch raises `FingerprintMismatch`
carrying `producer_hash, consumer_hash, manifest_path, consumer_model_path,
differing_tier` — the Item-3 PUSH detail an operator needs to diagnose.

## BUILD FINDING — a lifetime bug SC4's refactor surfaced (fixed)

The first SC4 integration run **failed**: the matching consumer crashed in the
forward (`CUBLAS_STATUS_EXECUTION_FAILED` at `lm_head`, after all 22 transformer
layers had run). Root cause: `sc4_consumer.run_shared()` held the `WeightArena`
as a **function local**; on `return model` the local went out of scope →
`WeightArena.__del__` → `cipher_rt_weight_arena_free` **unmapped the arena** →
the model's parameter tensors (from_blob views, no-op deleter) dangled → the
forward read freed VA (the async fault surfaced at the last cuBLAS call).

SC3 never hit this because SC3 did import + rebind + forward **inline in
`main`** — `arena` stayed in scope. SC4's factoring into `run_shared()`
exposed the latent lifetime coupling the SC2 code comment already warned of
("hold the WeightArena alive as long as any tensor from it is in use").

**Fix:** `run_shared()` pins the arena to the model — `model._cipher_weight_arena
= arena` — so the arena's lifetime tracks the model's. Re-run: PASS. This is
exactly the lifetime concern **SC5 (kmod-owned arena lifetime)** makes robust;
the SC4 fix is the interim user-space pin.

## Verification

unit 11/11 PASS (`TRACK_2_SC4_UNIT_TESTS.md`); integration PASS — matching +
mismatch (`TRACK_2_SC4_INTEGRATION.md`); regression clean
(`TRACK_2_SC4_REGRESSION.md`). No `.pre_sc4` binary fallback — SC4 changed no
binary; the SC3 scripts are preserved simply by being unmodified.
