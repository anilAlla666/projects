# Track 2 SC6-2 — MEMORY (substrate-value measurement)

**Date:** 2026-05-19. Tools: `sc6_run.py` (shared + independent phases),
`sc6_aggregate.py`. Summary JSON: `sc6_memory_summary.json`.

This is the Track 2 headline measurement: **how much GPU memory cross-tenant
weight-sharing saves**, measured — not asserted — against a real
independent-load control.

## Methodology

For each model, two phases, each a separate `sc6_run.py` process tree
(an OOM-killed control child cannot pollute a shared-phase number):

- **shared** — 1 producer + 4 consumers, one kmod weight arena. 5 processes,
  **one** physical weight copy.
- **independent (control)** — up to 5 tenants each loading the model
  *privately* (`from_pretrained().cuda()`), no arena, no kmod. 5 processes,
  **five** physical weight copies. Tenants are launched staggered (each fully
  resident before the next) and an OOM is recorded as a finding.

**Three framebuffer measurements per phase** (`nvidia-smi memory.used`),
per the decision-3 PUSH:

1. **idle** — before any tenant.
2. **loaded** — steady state, all tenants resident.
3. **exited** — after every process exits — the teardown / leak check.

**Caching-allocator note.** `nvidia-smi memory.used` reports the CUDA caching
allocator's *reservation*, not the live-tensor high-water mark. Each process
has its own allocator, so the independent control has 5 separate reservations
and the shared phase has 5 too — the comparison is **symmetric**, the
allocator behaviour is not a thumb on the scale. The producer additionally
`empty_cache`s its transient `from_pretrained` copy so its steady state is
`arena + context` (`TRACK_2_SC6_BUILD_LOG.md` verifies the drop).

## Results

### Three-point framebuffer (MiB)

| model | phase | idle | loaded | exited |
|---|---|---|---|---|
| TinyLlama | shared | 0 | **5848** | 0 |
| TinyLlama | independent (5/5) | 0 | **14572** | 0 |
| Mistral-7B | shared | 0 | **17572** | 0 |
| Mistral-7B | independent (5/5) | 0 | **73332** | 0 |

**Teardown is clean** — `exited == idle (0)` for all four runs. No leak: the
shared phase's arena is fully reclaimed once the last participant exits (the
third measurement would have exposed an arena leaked past teardown — it did
not).

### Savings — measured

| model | shared loaded | independent loaded (5 tenants) | **savings** |
|---|---|---|---|
| TinyLlama-1.1B | 5848 MiB | 14572 MiB | **8724 MiB — 59.9 %** |
| Mistral-7B-v0.1 | 17572 MiB | 73332 MiB | **55760 MiB — 76.0 %** |

Both independent controls fit all **5/5** tenants (Mistral-7B at 73332 MiB of
the 81559 MiB H100 — ~8 GiB headroom; no OOM, the savings is fully measured,
not projected).

### Weights / context decomposition

5 co-resident processes, idle subtracted: `shared = W + 5C`,
`independent per-tenant increment = W + C` ⇒ `C = (S−P)/4`, `W = P−C`.

| model | est. weights `W` | est. context `C` | check |
|---|---|---|---|
| TinyLlama-1.1B | 2181 MiB | 733 MiB | `W` ≈ 2.06 GiB fp16 weights + 16 MiB arena pad ✓ |
| Mistral-7B-v0.1 | 13940 MiB | 726 MiB | `W` ≈ 13.5 GiB bf16 weights ✓ (7.24 B × 2 B) |
| | | | `C` ≈ 0.7 GiB tracks the SC5 consumer-context figure ✓ |

## The structural claim — and its honest framing

Weight-sharing makes model weights a **per-GPU constant**, not a per-tenant
cost. With per-process context `C` and weights `W`, for N consumers + 1
producer:

```
savings(N) =  N·W / (N·W + (N+1)·C)        asymptote (large N) = W / (W+C)
```

| model | N=4 savings (measured) | large-N asymptote |
|---|---|---|
| TinyLlama-1.1B | 59.9 % | 74.8 % |
| Mistral-7B-v0.1 | 76.0 % | 95.0 % |

**The savings fraction is model-size-dependent — stated honestly:**

- TinyLlama's weights (2.06 GiB) are a *small* multiple of a CUDA-13 context
  (~0.7 GiB), so it **understates** the production case — kept only for
  SC3-SC5 continuity.
- Mistral-7B is production-representative: `W` dominates `C`, the asymptote
  is **95 %**. CP 5.5's Nemotron Nano (~4 GiB) sits between the two —
  reviewers can place it on the curve with the formula and the measured
  `(W, C)` rather than trusting an extrapolation.
- The N=4 points are **empirical anchors**; the formula projects. At N=100
  Mistral-7B the formula gives `100·13940 / (100·13940 + 101·726)` = **95.0 %**
  — and the *absolute* statement is starker: 100 independent Mistral-7B
  copies ≈ 1.36 TiB of weights ≫ any single GPU, whereas shared it is one
  13.6 GiB copy. Weight-sharing is what makes a 100-tenant single-GPU
  deployment physically possible at all.

## Verdict

The substrate-value claim is **measured on a production-representative model**:
at N=4, Mistral-7B weight-sharing saves **76 %** of GPU framebuffer vs. five
independent loads, with bit-identical output (`TRACK_2_SC6_INTEGRATION.md`)
and clean teardown. The mechanism is structural and the formula lets it be
projected, with the measured `(W, C)`, to the CP 5.5 100-tenant regime.
