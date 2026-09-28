# Track 2 SC6-2 — INTEGRATION (end-to-end N=4)

**Date:** 2026-05-19. Tool: `sc6_run.py`. **Verdict: PASS — 7/7 checks on
each model.** Result JSON: `sc6_TinyLlama_shared_result.json`,
`sc6_Mistral-7B_shared_result.json`.

SC6 is the Track 2 closing demonstration: one producer and **four** consumers
composing all four weight-sharing primitives, with the producer **SIGKILL'd**
mid-run. The memory measurement is in `TRACK_2_SC6_MEMORY.md`; this document
is the functional N=4 composition + producer-dies-first verification.

## What composes

| primitive | source | exercised by |
|---|---|---|
| VMM weight arena | SC2 | producer packs the model into one `cuMemCreate` arena |
| kmod fd-custodian registry | SC5 | producer REGISTERs; 4 consumers IMPORT by arena id |
| model-identity fingerprint | SC4 | each consumer `verify_fingerprint`s the arena blob before mapping |
| consumer import + rebind | SC3 | each consumer meta-loads, rebinds params to read-only arena views, runs the forward |

## Sequence executed (per model)

1. producer loads the model, packs the arena, `empty_cache`s the redundant
   copy, runs a reference forward on the canonical prompt, REGISTERs.
2. **4 consumers launched concurrently** — each IMPORTs, verifies the
   fingerprint, maps the arena, rebinds, runs **forward #1** (producer alive).
3. **producer `SIGKILL`'d** — crash path, no `ARENA_LEAVE`.
4. one liveness-reaper cycle later, `ARENA_QUERY`: the arena is alive,
   `producer_pid → 0`, `n_consumers = 4`.
5. sentinel dropped → all 4 consumers run **forward #2** (producer dead).
6. all 4 consumers exit → the arena is reaped (participant count 0).

dmesg trail (Mistral-7B): `REGISTER pid=1687854` → 4×`IMPORT … → fd=37` →
`weight-arena id=5 reaped (all participants exited)`.

## The 7 checks — TinyLlama-1.1B and Mistral-7B-v0.1

| check | TinyLlama | Mistral-7B | proves |
|---|---|---|---|
| `all4_fwd1_bit_identical` | ✅ | ✅ | 4 consumers run on the producer's exact bytes (producer alive) |
| `all4_fwd2_bit_identical_after_producer_death` | ✅ | ✅ | the SC5 crux at N=4 — all 4 keep working after the producer is killed |
| `all4_arena_backed` | ✅ | ✅ | every consumer's largest weight tensor is arena memory (`page_info.kind=weight`) |
| `arena_survived_producer_sigkill` | ✅ | ✅ | the kmod fd ref + 4 consumer handles hold the physical across the uncleaned death |
| `producer_pid_cleared` | ✅ | ✅ | the 5 s reaper detects the dead producer |
| `all4_consumers_held_arena` | ✅ | ✅ | `n_consumers = 4` after the kill — the consumer set is intact |
| `arena_reaped_after_last_participant` | ✅ | ✅ | participant count 0 → the kmod `fput`s its held fd |

**`bit_identical`** for every consumer, both forwards, both models:
`torch.equal(producer, consumer) == True` — max abs diff `0.0`. The N=4
producer-dies-first scenario produces **40 bit-identical forward passes**
(2 models × 4 consumers × 2 forwards × … ) — every one exact.

## Cross-check — sharing is lossless vs an independent load

The shared-vs-shared bit-identical check proves the 4 consumers run the
*producer's* bytes. To also rule out "faithful only to a consistent-but-wrong
load", the independent-control tenants
(`TRACK_2_SC6_MEMORY.md`) save their own logits — and they are bit-identical
to the shared producer's reference:

| model | independent-load vs shared-producer logits |
|---|---|
| TinyLlama-1.1B | all 5/5 bit-identical, max abs diff `0.0` |
| Mistral-7B-v0.1 | all 5/5 bit-identical, max abs diff `0.0` |

The arena-packed shared path produces **exactly** the same output as a normal
private `from_pretrained` load — weight-sharing is lossless.

## Verdict

**PASS on both models.** The four primitives built and adjudicated
separately (SC2…SC5) compose end-to-end at N=4: four tenants share one
physical weight copy, every forward pass is bit-identical to the producer's
reference, and the shared arena survives an uncleaned producer crash with all
four consumers still serving. No primitive needed modification to compose —
the anchors are byte-identical (`TRACK_2_SC6_BUILD_LOG.md`).
