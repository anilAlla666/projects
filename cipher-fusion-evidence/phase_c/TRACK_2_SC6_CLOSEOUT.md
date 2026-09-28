# Track 2 SC6 — CLOSEOUT

**Date:** 2026-05-19. **Status: SC6-2 COMPLETE — STOP for adjudication before
`TRACK_2_CLOSEOUT.md` (SC6-3).** Predecessor: SC5 closed
(`TRACK_2_SC5_CLOSEOUT.md`).

SC6 is the **Track 2 closing milestone** — the end-to-end N=4 demonstration
that the four weight-sharing primitives (SC2 arena / SC3 import+rebind / SC4
fingerprint / SC5 kmod fd-custodian) **compose**, plus the substrate-value
memory measurement that justifies the track.

## What SC6-2 delivered

| phase | deliverable | result |
|---|---|---|
| build | `sc6_*.py` harness — N=4, 2 models, OOM-aware control | pure harness, no anchor rotation — `TRACK_2_SC6_BUILD_LOG.md` |
| integration | N=4 composition + producer-dies-first, both models | **7/7 checks each** — `TRACK_2_SC6_INTEGRATION.md` |
| memory | shared vs 5× independent control, 3-point FB | **measured** — `TRACK_2_SC6_MEMORY.md` |
| regression | anchors / isolation / W3 / dmesg / teardown | **PASS** — `TRACK_2_SC6_REGRESSION.md` |

## What is proven

1. **The primitives compose.** One producer + four consumers, all four
   primitives in one run, on both TinyLlama-1.1B and Mistral-7B-v0.1 — 7/7
   functional checks each. No primitive needed modification to compose; all
   four anchors are byte-identical.
2. **Producer-dies-first at N=4.** The producer is `SIGKILL`'d while 4
   consumers hold the arena; the arena survives (`producer_pid→0`,
   `n_consumers=4`), all 4 consumers run a forward pass *after* the producer's
   death bit-identical to its reference, and the arena is reaped once the
   last consumer exits. The SC5 crux holds at N=4.
3. **Bit-identical at scale.** Every consumer, both forwards (producer alive
   and producer dead), both models — `torch.equal` true, max abs diff `0.0`.
4. **Substrate value — measured on a production model.** At N=4, Mistral-7B
   weight-sharing saves **76.0 %** of GPU framebuffer (17572 MiB shared vs
   73332 MiB for five independent loads); TinyLlama saves 59.9 %. The
   structural formula `savings(N) = N·W/(N·W+(N+1)·C)` (measured `W`, `C`)
   projects to a **95 %** asymptote for Mistral-7B and lets a reviewer place
   CP 5.5's Nemotron Nano on the curve.
5. **Clean teardown.** The 3-point framebuffer measurement's `exited` reading
   returned to `idle` (0 MiB) on every run — no arena or context leaked past
   process teardown.

## Notes / honest boundaries

- **Model-size dependence is stated, not hidden.** TinyLlama understates the
  substrate value (small weights vs context); Mistral-7B is the
  production-representative figure. Both are reported with the formula and
  the measured `(W, C)` so the projection is auditable
  (`TRACK_2_SC6_MEMORY.md`).
- **Both independent controls fit 5/5** — the anticipated Mistral-7B OOM did
  not occur (73.3 GiB of 81.5 GiB, ~8 GiB headroom). The savings is therefore
  fully measured, not projected; the OOM-as-finding path in the harness was
  not exercised.
- SC6 inherits the SC5 v1 boundaries unchanged (16 arena slots, 64 KiB blob,
  5 s reaper, single-GPU, same-model-per-arena, no registration grace
  window). The Mistral-7B manifest blob is 44662 B — within the 64 KiB cap.

## Anchors — none rotated

kmod `008b3c66`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `cc0479b8` — all byte-identical to the SC5 close. SC6 is pure
harness work; no fallback artifact required.

## Deliverables

`TRACK_2_SC6_DESIGN_MEMO.md`, `TRACK_2_SC6_BUILD_LOG.md`,
`TRACK_2_SC6_INTEGRATION.md`, `TRACK_2_SC6_MEMORY.md`,
`TRACK_2_SC6_REGRESSION.md`, this closeout. Code: `sc6_models.py`,
`sc6_producer.py`, `sc6_consumer.py`, `sc6_independent_tenant.py`,
`sc6_run.py`, `sc6_aggregate.py`. Result JSON: `sc6_*_result.json`,
`sc6_memory_summary.json`.

## Adjudication ask — STOP before `TRACK_2_CLOSEOUT.md`

SC6-2 is complete: the N=4 composition is verified on two models (7/7 each),
producer-dies-first holds at N=4, the substrate-value measurement gives a
production-credible **76 % saving on Mistral-7B**, and the regression smoke
is clean with every anchor byte-identical.

**Decisions requested:**

1. **Accept SC6-2** — the N=4 end-to-end composition + producer-dies-first +
   the measured substrate-value numbers.
2. **Authorize SC6-3** — finalize `TRACK_2_CLOSEOUT.md`, the Track 2 closing
   document (headline, measurement-provenance table, v1 boundaries, v2
   deferrals, production-deployment guidance, CP 5.5 integration plan,
   SC1-SC6 sub-component summary, anchor lineage). This **closes Track 2**.

**STOPPING for adjudication.** No further source modified pending the
decision; `TRACK_2_CLOSEOUT.md` is not written until SC6-3 is authorized.
