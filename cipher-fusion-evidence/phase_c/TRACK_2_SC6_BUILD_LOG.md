# Track 2 SC6-2 — BUILD LOG

**Date:** 2026-05-19. **Phase SC6-2** — the N=4 end-to-end harness. SC6 is
**pure harness work**: new Python test scripts, no substrate change. All four
anchors are byte-identical to the SC5 close (`TRACK_2_SC5_CLOSEOUT.md`) —
verified, no rotation.

## Pre-SC6 baseline

| anchor | md5 | state |
|---|---|---|
| `cipher_kmod.ko` | `008b3c66` | unchanged (SC5 close) |
| `libcipher_rt.so` | `83afd1ca` | unchanged |
| `cipher_kv_bridge.so` | `c04b0c39` | unchanged |
| `libcipher_v2.so` | `cc0479b8` | unchanged |

SC5 still works on this substrate — the SC5 integration passed at the SC5
close on this md5-identical kmod (`008b3c66`); the equivalence argument is
the anchor identity, so the 4-minute SC5 re-run was not re-burned (advisor
concurrence). W3 re-verified post-SC6 — see `TRACK_2_SC6_REGRESSION.md`.

## Files added (all harness — no substrate file touched)

| file | role |
|---|---|
| `sc6_models.py` | model registry (TinyLlama fp16 / Mistral-7B-v0.1 bf16), canonical prompt, `gpu_fb_mib()` |
| `sc6_producer.py` | producer — pack arena, `empty_cache` honesty correction, REGISTER; parameterised by model |
| `sc6_consumer.py` | consumer with a consumer-index (4 concurrent instances, namespaced output files); forward #1 (producer alive) + forward #2 (after producer death) |
| `sc6_independent_tenant.py` | independent-load CONTROL tenant — private weights, no arena, OOM-aware |
| `sc6_run.py` | N=4 integration driver; one `(model, phase)` per invocation |
| `sc6_aggregate.py` | substrate-value aggregator — savings %, W/C decomposition, formula |

Reused unchanged: `sc5_arena_ioctl.py` (the `/dev/cipher` ioctl helper),
`cipher_model_fingerprint.py` (SC4), `cipher_kv_bridge` (SC2/SC3).

## Design points carried from the SC6 design memo

- **`empty_cache` honesty correction (§4).** The producer holds two copies
  transiently — the `from_pretrained` load and the arena it is packed into.
  After `p.data = vt` for every parameter the original storages are
  unreferenced; `gc.collect()` + `torch.cuda.empty_cache()` returns them to
  the driver. Verified by framebuffer checkpoints in the producer log:

  | checkpoint | TinyLlama fb | Mistral-7B fb |
  |---|---|---|
  | after_load | 2717 MiB | 14469 MiB |
  | after_pack | 4833 MiB | 28299 MiB |
  | **after_empty_cache** | **2645 MiB** | **14359 MiB** |

  The pack→empty_cache drop is **2188 MiB** (TinyLlama) / **13940 MiB**
  (Mistral) — i.e. ~one weight copy is genuinely released. The producer's
  steady-state footprint is `arena + context`, not `arena + ghost copy`.

- **Consumer-index namespacing.** `sc6_consumer.py` takes an index 0-3 so the
  four concurrent consumers write `sc6_<model>_consumer_<i>_*` without
  collision (the SC5 single-consumer harness used fixed names).

- **dtype per model.** TinyLlama loads fp16 (continuity with SC3-SC5);
  Mistral-7B-v0.1 loads bf16 (its native dtype). `cipher_kv_bridge`'s
  `dtype_from_string` already maps `"bfloat16" → at::kBFloat16` — verified in
  the bridge source, no change needed.

- **Mistral blob fits.** The Mistral-7B manifest (291 tensors + fingerprint)
  serialises to **44662 bytes** — under `CIPHER_WA_BLOB_MAX` (65536). The
  producer asserts and exits cleanly on overflow; it did not trigger.

## Anchor verification — post-SC6

All four anchors **byte-identical** to the pre-SC6 baseline above. SC6
rotated nothing — confirmed by `md5sum` after the full TinyLlama + Mistral-7B
run. No fallback artifact is required (no binary changed).
