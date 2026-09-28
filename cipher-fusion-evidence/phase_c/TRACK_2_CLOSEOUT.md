# TRACK 2 — CROSS-TENANT WEIGHT SHARING — CLOSEOUT

**Date:** 2026-05-19. **The Track 2 closing document.** Documentation only —
no source modified, no anchor rotation, no GPU. Predecessors: SC1-SC6
closeouts. Companion track: [[cipher-track3-dsm]] (Dynamic SM Migration,
closed 2026-05-19).

---

## 1 — HEADLINE

**Track 2 (Cross-Tenant Weight Sharing) delivers substrate-layer,
kernel-level weight sharing across multiple processes with crash-safe
lifetime semantics, transparent to application code via a kmod fd custodian +
`cipher_kv_bridge`, measured at ~76 % GPU-memory savings on Mistral-7B at N=4
with verified bit-identical forward correctness.** N same-model tenants share
one physical copy of the model weights in HBM; per-tenant residual cost
collapses to activations + context only. The shared arena's lifetime is
independent of the producer process — a producer may crash and every consumer
keeps serving. This is the capacity primitive CP 5.5 requires: it converts
model weights from a per-tenant cost into a per-GPU constant.

---

## 2 — MEASUREMENT PROVENANCE

Every claim Track 2 makes, mapped to the measurement that substantiates it
and the document that records it. No buried caveats.

| claim | measurement | document |
|---|---|---|
| Cross-tenant weight sharing works (consumer imports producer's physical bytes) | SC3 integration — bit-identical forward, consumer adds 0 MiB weight FB | `TRACK_2_SC3_CLOSEOUT.md` |
| Model-identity fingerprinting prevents wrong-model imports | SC4 unit (11/11) + integration — tier-1 mismatch detected pre-commit, clean fallback | `TRACK_2_SC4_CLOSEOUT.md` |
| Arena lifetime survives producer crash | SC5 producer-dies-first — `SIGKILL`, arena survives, consumer keeps working | `TRACK_2_SC5_INTEGRATION.md` |
| All 4 primitives compose at N=4 with bit-identical forward | SC6 N=4 integration — 7/7 checks, both models, 40 bit-identical forwards | `TRACK_2_SC6_INTEGRATION.md` |
| Memory savings 76.0 % on Mistral-7B at N=4 | SC6 memory — 17572 MiB shared vs 73332 MiB independent (5×), 3-point FB | `TRACK_2_SC6_MEMORY.md` |
| Weight sharing is lossless vs an independent load | SC6 cross-check — independent-load logits bit-identical to shared producer | `TRACK_2_SC6_INTEGRATION.md` |
| Structural formula projects ~95 % asymptote (Mistral-7B, large N) | SC6 aggregate — `savings(N)=N·W/(N·W+(N+1)·C)`, measured `W`,`C` | `TRACK_2_SC6_MEMORY.md` |
| Independent control barely fits 5 Mistral-7B tenants on an H100 | SC6 independent scenario — 73332 / 81559 MiB, ~8 GiB headroom | `TRACK_2_SC6_MEMORY.md` |
| The probe crux — VMM POSIX-fd is cross-process refcounted | SC5 Phase A probe — consumer mapping survives producer `SIGKILL` | `TRACK_2_SC5_PROBE_LOG.md` |

**Two measurement notes (advisor-flagged, accepted for inclusion):**

- **The `W`-vs-theoretical gap.** SC6's decomposition estimates Mistral-7B
  `W = 13940 MiB` against a theoretical 14480 MiB (7.24 B × 2 B). The ~540 MiB
  difference is the producer-vs-consumer **context asymmetry**: the
  decomposition `S = W + 5C` assumes all five process contexts are equal, but
  the producer ran the arena-pack and a forward pass, so its context is
  slightly larger than a fresh consumer's. The gap is an artifact of the
  estimator, not lost memory; the *measured* savings (76.0 %, the difference
  of two directly-observed framebuffer readings) does not depend on the
  decomposition.
- **The OOM-margin substrate case.** The Mistral-7B independent control fit
  5/5 tenants at 73332 of 81559 MiB — **the narrowest possible margin** for
  the comparison to be measurable at all. On any smaller-VRAM GPU (an 80 GiB
  A100 has a tighter usable margin; a 40 GiB card cannot hold even three),
  five independent Mistral-7B tenants is at or past the OOM cliff. The
  architectural case is therefore not "76 % is a nice saving" — it is
  **"without weight sharing this multi-tenant configuration is structurally
  impossible to run."** Weight sharing is what makes the deployment exist,
  not merely what makes it cheaper.

---

## 3 — V1 BOUNDARIES

What Track 2 v1 does **not** do — stated up front, not buried.

- **Single-GPU.** The kmod fd custodian is single-device scope. Weights are
  shared among processes on one GPU; cross-GPU sharing is v2.
- **16 arena slots maximum** (`CIPHER_WA_MAX_ARENAS`). 16 distinct shared
  arenas concurrently; the 17th REGISTER returns `-ENOSPC`.
- **64 KiB max opaque metadata blob per arena** (`CIPHER_WA_BLOB_MAX`). The
  layout manifest + fingerprint must serialise within 64 KiB — Mistral-7B's
  291-tensor manifest used 44662 B, comfortably inside.
- **5 s liveness-reaper cadence.** Arena reclamation is workqueue-driven, not
  `do_exit`-immediate (deliberate — keeps SC5 off the delicate `do_exit`
  path; the fd reference holds the physical safely meanwhile).
- **Same-model assumption per arena.** One arena carries one model; the SC4
  fingerprint *enforces* this — a consumer importing with a different local
  model is rejected (`FingerprintMismatch`) before mapping.
- **Registration grace window NOT implemented (v1.5).** An arena whose
  producer dies with **zero consumers ever joined** is reaped on the next 5 s
  cycle. `IMPORT`-after-producer-death requires ≥1 consumer to have joined
  while the producer was alive. Accepted v1 boundary (SC5 adjudication);
  named v1.5 work.
- **Fingerprint is operator-error-proof, not adversary-proof.** Two-tier
  SHA256 — tier 1 structural (config + index + shard sizes + dtype), tier 2
  sampled weight bytes (1 MiB head/mid/tail per shard; < 3 MiB shards
  full-hashed). The unsampled regions of large shards are not covered; a
  full weight hash is the v2 hardening.

---

## 4 — V2 DEFERRALS

Named explicitly so they are scope, not surprises.

- **Multi-GPU weight sharing** — arenas shared across devices / nodes.
- **Registration grace window (v1.5)** — a `never_imported` sticky bit or a
  `registered_at` + module-param grace, so a producer can register and exit
  before any consumer joins.
- **Arena migration across devices** — move a live arena GPU→GPU (the Track 3
  DSM analogue for weights).
- **Per-arena quota / billing infrastructure** — operator accounting of
  shared-vs-attributed memory per tenant.
- **Adversary-proof fingerprint** — full weight hash instead of sampled.
- **Heterogeneous-model arena** — different models sharing partial weights
  (e.g. a shared base model + per-tenant LoRA deltas).

---

## 5 — PRODUCTION DEPLOYMENT GUIDANCE

- **The 100-tenant POOL-served pathway is now substrate-supported.** A
  producer-of-a-popular-model registers one shared arena; POOL-served
  consumer tenants IMPORT it. Weights cost one copy regardless of tenant
  count.
- **Memory math at production scale** — `savings(N) = N·W/(N·W+(N+1)·C)`,
  with `W` = weight bytes, `C` = per-process context (measured `C ≈ 0.7 GiB`
  on this H100 / CUDA 13):

  | model class | N=4 savings (measured) | large-N asymptote |
  |---|---|---|
  | TinyLlama-1.1B (W ≈ 2.1 GiB) | 59.9 % | 74.8 % |
  | Mistral-7B (W ≈ 13.6 GiB) | 76.0 % | 95.0 % |
  | Nemotron-Nano-class (W ≈ 4 GiB) at N=100 | — | ~80–85 % (formula projection; exact value depends on `C`) |

  The N=4 points are **empirical anchors**; the formula projects. The
  Nemotron figure is an explicit projection — a reviewer recomputes it from
  the formula and the measured `(W, C)`, not from assertion.
- **Operator visibility** — `/proc/cipher/arenas` is a live read-only view
  of the registry (arena ids, producer pid, consumer counts, sizes);
  `CIPHER_ARENA_QUERY` (ioctl NR 24) is the programmatic equivalent.
- **Fault tolerance is substrate-layer, not application-layer.** A producer
  crash does not strand the arena and does not interrupt consumers — verified
  to N=4. Applications need no crash-recovery code for the shared-weights
  case.
- **Recommended pattern** — *producer-of-popular-model*: one tenant (or a
  dedicated loader) produces the shared base-model arena; all consumers join.
  Keep ≥1 consumer joined before the producer can exit (the v1 boundary).

---

## 6 — CP 5.5 INTEGRATION PLAN

- Track 2 (weight sharing) and Track 3 (Dynamic SM Migration) are both
  **active substrate** — together they are the foundation for the 100-tenant
  POOL-served Nemotron-Nano architecture: Track 2 makes weights a per-GPU
  constant; Track 3 keeps SM partitions packed under churn.
- **Step 1.6P** (POOL-scaling at B=20/50/100) is now measurable — weight
  sharing removes the per-tenant weight cost that previously made high-N POOL
  scaling a memory-capacity question rather than a scheduling one.
- The aggregate tok/W benchmark against the Clarifai ~400-per-watt reference
  becomes substrate-feasible: 100 tenants can co-reside, so an aggregate
  measurement is physically possible.
- Sequence after Track 2 close: CP 5.4 closure (1.6B-3, 1.6B-4, 1.6P, 1.7,
  1.8) → FUTURE_SCOPE/A (CIPHER + vLLM composed) → CP 5.5 (100-tenant
  integration soak), which closes Phase 5.

---

## 7 — SUB-COMPONENT SUMMARY

| SC | scope | outcome | anchor effect |
|---|---|---|---|
| SC1 | design memo — interception at model-load via the VMM bridge | paperwork | none |
| SC2 | producer-side weight arena (`cipher_rt_weight_arena_*`, `WeightArena` pybind) | 5/5 PASS; one `cuMemCreate`, TinyLlama 201 tensors, KL 0.0 | cipher_kv_bridge `fca6843d` |
| SC3 | consumer-side import (`weight_arena_import`, `WeightArena.view`); meta-load + rebind | bit-identical forward, consumer 0 MiB weight FB | cipher_kv_bridge `fca6843d → c04b0c39` |
| SC4 | model-identity fingerprint (`cipher_model_fingerprint.py`, two-tier SHA256) | 11/11 unit, mismatch → clean fallback | none (pure Python) |
| SC5 | kmod-owned arena lifetime — fd-custodian registry, ioctls 21-24, 5 s reaper | 15/15 unit, 12/12 producer-dies-first | kmod `285d102e → 008b3c66` |
| SC6 | end-to-end N=4 verification, two models, substrate-value measurement | 7/7 each, 76 % Mistral-7B saving | none (pure harness) |

The crux that shaped the track: SC5's Phase A probe confirmed CUDA VMM
POSIX-fd handles are already cross-process refcounted by the driver — so the
kmod is an **fd custodian** (`fget`/`fd_install`/`fput`), never calls the CUDA
driver, and physical-memory lifetime is the driver's own refcount.

---

## 8 — ANCHORS AT TRACK 2 CLOSE

| artifact | md5 | state | fallback |
|---|---|---|---|
| `cipher_kmod.ko` | `008b3c66` | loaded | `cipher_kmod.ko.track2_sc5`; pre: `cipher_kmod.ko.pre_track2_sc5` (`285d102e`); src `cipher_kmod_src_track2_sc5.tar.gz` |
| `cipher_kv_bridge.so` | `c04b0c39` | loaded | `cipher_kv_bridge.so.track2_sc3`; pre: `cipher_kv_bridge.so.pre_track2_sc3` |
| `libcipher_rt.so` | `83afd1ca` | unchanged through Track 2 | Track 3 close anchor — still active |
| `libcipher_v2.so` | `cc0479b8` | unchanged through Track 2 | (= the `v0.3.0-cupti.unpromoted` build) |

Track 2 rotated **two** anchors over six sub-components: cipher_kv_bridge
(SC3, the consumer-import path) and the kmod (SC5, the fd custodian).
libcipher_rt and libcipher_v2 were untouched end-to-end. All fallbacks are
preserved with md5 verification ([[cipher-kbuild-clean-wipes-ko]] — kmod
`.ko` fallbacks live outside the kbuild tree). ABI additions documented in
`TRACK_2_ABI.md` (ioctl NRs 21-24, additive, [[cipher-abi-rule]] honored).

---

## VERDICT — TRACK 2 CLOSED

Track 2 delivers cross-tenant weight sharing as a substrate primitive:
correct (bit-identical, fingerprint-guarded), crash-safe (producer-death
survivable, verified to N=4), and measured (76 % memory saving on a
production-representative model, with a formula that projects to the CP 5.5
regime). The capacity ceiling that was weight-bound is now context-bound;
weights are a per-GPU constant.

**Awaiting final adjudication of Track 2 closure.**
