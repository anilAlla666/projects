# Phase C / Track 2 — Weight-Sharing — SC6 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU. STOP for adjudication before SC6-2. **Predecessor:** SC5
closed (`TRACK_2_SC5_CLOSEOUT.md`, kmod anchor `008b3c66`).

SC6 is the **Track 2 closing milestone**: an end-to-end N=4 demonstration
that composes all four weight-sharing primitives, plus the substrate-value
memory measurement that justifies the track. Per the SC1 design memo, SC6 is
"end-to-end N-tenant verification" — the proof that the pieces built and
adjudicated separately (SC2…SC5) actually compose into the capacity primitive
CP 5.5 needs.

---

## §1 — What SC6 composes

One producer, four consumers, all four primitives in one run:

| primitive | from | role in SC6 |
|---|---|---|
| VMM weight arena | SC2 | the producer packs TinyLlama into one `cuMemCreate` arena |
| kmod fd-custodian registry | SC5 | the producer REGISTERs; the 4 consumers IMPORT by arena id |
| model-identity fingerprint | SC4 | each consumer `verify_fingerprint`s the arena's metadata blob before mapping |
| consumer import + rebind | SC3 | each consumer meta-loads, rebinds params onto read-only arena views, runs a bit-identical forward |

SC3/SC4/SC5 each verified one primitive in isolation (and SC5 was already
2-consumer-capable). SC6 verifies the **composition** at N=4 and produces the
**aggregate memory number** that is the track's headline.

## §2 — Orchestrator design — `sc6_run.py`, `sc6_consumer.py`

`sc6_run.py` extends `sc5_run.py` from one-consumer-plus-one to **N=4
concurrent consumers**. `sc5_consumer.py` writes fixed-name artifacts
(`sc5_consumer_A_logits.pt`) — four concurrent instances would collide — so
SC6 adds `sc6_consumer.py`, identical in logic but taking a **consumer index
0-3** that namespaces its output files (`sc6_consumer_{i}_logits.pt`, etc.).
This is harness-only Python; no substrate change (§6).

Flow:
1. producer — load TinyLlama, pack the arena, save reference logits,
   `torch.cuda.empty_cache()` to drop the transient non-arena copy (§4),
   REGISTER, stay alive.
2. launch **4 consumers concurrently**; each IMPORTs, verifies the
   fingerprint, maps the arena, rebinds, and blocks at a barrier
   (the SC5 mode-A sentinel pattern) — all 4 alive simultaneously so the
   N=4 memory snapshot is real.
3. memory snapshot with all 5 processes resident (§4).
4. producer-dies-first at N=4 (§3).
5. release the barrier — all 4 consumers run their forward pass.
6. verify (§5), then tear down.

## §3 — Producer-dies-first at N=4

The SC5 crux, scaled. Producer REGISTERs; **all 4 consumers IMPORT** (arena
participant count = 5: producer + 4); producer is `SIGKILL`'d (crash path,
no `ARENA_LEAVE`); `ARENA_QUERY` must show the arena alive with
`producer_pid → 0` and `n_consumers = 4`; all 4 consumers then run forward
passes that are bit-identical to the producer's pre-death reference. After
all 4 exit, the arena is reaped (count 0 → `fput`). This stays inside the
SC5 v1 boundary — the 4 consumers join while the producer is alive, so the
participant count never hits 0.

## §4 — Memory measurement — the substrate-value claim

**The headline number, and the part that needs the most honesty.**

The claim is structural: **shared weights are a per-GPU constant, not a
per-tenant cost.** SC6 measures it two ways in the same run:

- **Shared (N=4):** 1 producer arena + 5 process contexts. GPU framebuffer
  with all 5 processes resident.
- **Independent control:** 5 processes each loading TinyLlama *normally*
  (private weights, no sharing). GPU framebuffer with all 5 resident.

Savings = `control − shared`, measured — **not asserted**.

**Honesty notes (campaign discipline, [[cipher-lift-framing]]):**

1. The producer transiently holds *two* copies — the `from_pretrained().cuda()`
   load and the arena it is packed into. That redundant copy is a harness
   artifact, not the substrate story. The SC6 producer will
   `torch.cuda.empty_cache()` after packing so its steady-state footprint is
   `arena + context`. The measurement is taken at steady state.
2. **The savings fraction is model-size-dependent.** With per-process context
   `C` and weights `W`, shared-vs-independent savings →
   `N·W / (N·W + (N+1)·C)`. TinyLlama's `W ≈ 2.06 GiB` is *small* relative to
   a CUDA-13 + cuBLAS/cuDNN context (~0.5–1 GiB measured in SC5), so TinyLlama
   **understates** the production case. A 7B-class model (`W ≈ 13 GiB`) pushes
   the asymptote toward ~95%. SC6's TinyLlama number is therefore a
   *conservative* demonstration of the *mechanism*; the headline % is honest
   only when stated with the model and N. The expected TinyLlama N=4 figure is
   ~50–60% — the exact value comes from the build, and SC6 reports the formula
   alongside it so the scaling to N=100 / 7B is auditable, not extrapolated by
   assertion.
3. **Open decision (§7):** whether SC6 also runs a Mistral-7B confirmation so
   the headline is stated on a production-representative model.

`page_info` of each consumer's largest weight tensor is recorded (per SC3) —
`kind=weight` confirms each consumer is arena-backed; combined with the
0-extra-weight-copy memory delta, that *is* the physical-page-identity
evidence (cross-process VAs differ; physical identity shows up as the
framebuffer counting the weights once).

## §5 — Verification gates

| gate | pass condition |
|---|---|
| all-4-fingerprints-verified | every consumer's `verify_fingerprint` returns True |
| N=4 bit-identical | `torch.equal(producer, consumer_i)` for i=0..3, max-abs-diff 0.0 |
| N=4 arena-backed | `page_info.kind == "weight"` for all 4 consumers |
| producer-dies-first @ N=4 | arena survives `SIGKILL`; `producer_pid→0`; all 4 forward passes complete bit-identical after producer death |
| arena reaped | `ARENA_QUERY` empty after all 5 processes exit |
| memory — sharing real | shared N=4 framebuffer `<` independent-control framebuffer by ≈ `4 × W` |
| regression smoke | isolation 15/15, W3 `SC2_PASS`, dmesg clean |

## §6 — Anchors — none rotate

SC6 is **pure harness work**: `sc6_run.py`, `sc6_consumer.py`, reusing
`sc5_arena_ioctl.py` / `cipher_model_fingerprint.py` / `cipher_kv_bridge`
unchanged. **No kmod, libcipher_rt, cipher_kv_bridge, or libcipher_v2
modification is expected.** Expected post-SC6 anchors — all byte-identical:
kmod `008b3c66`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `cc0479b8`. **If any source modification surfaces during the
build → STOP and surface for adjudication** (scope expansion needs explicit
approval — it would mean a primitive does not actually compose, which is
itself the finding).

## §7 — Open design decisions for adjudication

1. **Model coverage.** Primary run on **TinyLlama-1.1B** (continuity with the
   SC3/SC4/SC5 harness, fast). *Recommend* additionally a **single Mistral-7B
   N=4 confirmation** so the memory headline is stated on a
   production-representative model where the savings asymptote is meaningful
   — SC5's substrate is model-agnostic and Mistral-7B was the SC1 reference
   model. Adjudicator: TinyLlama-only, or TinyLlama + Mistral-7B?
2. **Prompt set.** *Recommend* the single canonical prompt used through
   SC3-SC5 (bit-identical is a binary property — one prompt suffices to prove
   it; more prompts add runtime, not assurance). Adjudicator may direct a
   2-3 prompt set if broader coverage is wanted.
3. **Independent-load control.** *Recommend* SC6 actually runs the 5×
   independent-load control (it fits comfortably in 80 GiB and makes the
   savings *measured*, not computed). Adjudicator may down-scope to an
   analytic counterfactual if GPU time is constrained.

## §8 — SC6-2 / SC6-3 plan & deliverables

| phase | scope | est. |
|---|---|---|
| SC6-1 | this design memo | done |
| SC6-2 | build `sc6_run.py` + `sc6_consumer.py`; N=4 composition run; producer-dies-first @ N=4; memory measurement + independent-load control; regression smoke | ~1 d |
| SC6-3 | `TRACK_2_SC6_CLOSEOUT.md` + **`TRACK_2_CLOSEOUT.md`** (the track closing document) | ~0.5 d |

Deliverables: `TRACK_2_SC6_DESIGN_MEMO.md` (this), `TRACK_2_SC6_BUILD_LOG.md`,
`TRACK_2_SC6_INTEGRATION.md` (N=4 + producer-dies), `TRACK_2_SC6_MEMORY.md`
(substrate-value measurement), `TRACK_2_SC6_REGRESSION.md`,
`TRACK_2_SC6_CLOSEOUT.md`, `TRACK_2_CLOSEOUT.md` (headline, measurement
provenance table, v1 boundaries, v2 deferrals, CP 5.5 integration plan,
sub-component summary SC1-SC6, anchor lineage).

## §9 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **Composition scope (§1-§3)** — accept the N=4 end-to-end run (4 consumers
   composing SC2+SC3+SC4+SC5) plus producer-dies-first at N=4.
2. **Memory measurement (§4)** — accept the shared-vs-independent-control
   methodology, the producer `empty_cache` steady-state correction, and the
   honesty framing (model-size-dependent %, formula reported alongside the
   number).
3. **Open decisions (§7)** — rule on: (a) TinyLlama-only vs + Mistral-7B
   confirmation; (b) single vs multi-prompt; (c) run the independent-load
   control vs analytic counterfactual.
4. **Anchors (§6)** — accept "no rotation expected; STOP if source surfaces."

On adjudication: proceed to **SC6-2** — build the N=4 harness and run.
