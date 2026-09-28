# Track 2 SC3-2 — UNIT + INTEGRATION TESTS

**Date:** 2026-05-19. cipher_kv_bridge `c04b0c39`. Driver `sc3_run.py` —
producer + consumer ×2 (same-VA, offset-relative). **Verdict: PASS.**

---

## Integration — weight-sharing end to end

The producer loads TinyLlama, packs all 201 weight tensors into a CIPHER VMM
weight arena, exports the arena fd + a JSON layout manifest over a UNIX socket
(`SCM_RIGHTS`). The consumer imports the arena, meta-loads TinyLlama
(`init_empty_weights(include_buffers=False)` — **the safetensors weight read
is skipped**), rebinds every parameter onto a READ-only arena view, and runs
the forward pass.

| check | same-VA run | offset-relative run |
|---|---|---|
| consumer rc | 0 | 0 |
| **bit-identical forward** (Y_producer == Y_consumer) | **true** | **true** |
| logits max-abs-diff | **0.0** | **0.0** |
| weight tensor arena-backed (`page_info kind='weight'`) | true | true |
| **consumer GPU FB added** | **0 MiB** | **0 MiB** |

**Bit-identical forward (Item-4 PUSH).** `torch.equal(Y_producer,
Y_consumer)` is **true**, max-abs-diff **0.0** — the consumer runs the model
on the producer's *exact physical weight bytes*. This is the load-bearing
correctness proof: had the rebind been wrong anywhere (wrong offset, stale
storage, gradient flag), the forward would diverge.

**Memory — sharing actually happened.** Producer-loaded GPU framebuffer:
5001 MiB (1×W + activations). After the consumer imports + rebinds + runs a
full forward: total FB **still 5001 MiB — the consumer added 0 MiB**. Two
tenants run TinyLlama on **one** physical weight copy. This is the Track 2
substrate-value claim, demonstrated.

## Unit-level coverage (within the integration runs)

- **Import primitive** — `cipher_rt_weight_arena_import` /
  `weight_arena_import`: the consumer imported the producer's exported fd and
  mapped a usable arena (both runs). ✓
- **Same-VA path** — same-VA run: `cuMemAddressReserve(want_base)` succeeded,
  the consumer mapped at the producer's base; `same_va=true`. ✓
- **Offset-relative path** — offset-relative run: `SC3_FORCE_OFFSET=1` passes
  `want_base=0`, so `weight_arena_import` takes the **reserve-without-hint
  branch**. That branch ran and the import succeeded bit-identically. **Note
  (honest):** the no-hint reserve *coincidentally* returned the producer's
  base VA — fresh-process VA allocation is deterministic, so two identical
  fresh processes reserve the same address. A genuinely-different base could
  not be *forced* in this harness. It is immaterial: `WeightArena.view`
  computes `arena.base + offset` from the *imported* arena's actual base, so
  the rebind is correct for **any** base by construction — exactly the
  property the SC3-1 adjudication accepted ("offset-relative rebind is correct
  either way; same-VA is an optimization, not a correctness requirement").
- **Manifest round-trip** — the JSON manifest (201 tensor entries: name,
  shape, dtype, itemsize, offset) was sent via `SCM_RIGHTS`, parsed by the
  consumer, and drove a correct rebind (the bit-identical forward proves every
  offset/shape/dtype was right). ✓

## Verdict

**PASS.** Cross-tenant weight-sharing works end to end: a consumer imports a
peer's VMM weight arena, runs the model on the shared physical bytes, the
forward is bit-identical, and the consumer adds **zero** weight memory.
Evidence: `sc3_integration_result.json`, `sc3_consumer_result.json`,
`sc3_producer.log`, `sc3_consumer_*.log`.
