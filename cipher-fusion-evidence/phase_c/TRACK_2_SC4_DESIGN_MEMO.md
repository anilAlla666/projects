# Phase C / Track 2 — Weight-Sharing — SC4 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU. STOP for adjudication before SC4-2. **Predecessor:** Track 2
SC3 closed (`TRACK_2_SC3_CLOSEOUT.md`, cipher_kv_bridge `c04b0c39`).

SC4 is the **safety** primitive: a consumer must **never** import a producer's
arena for a *different* model and silently produce wrong results. SC3 verified
sharing under the *assumption* of a matching model; SC4 makes that assumption
*checked* — the precondition for production deployability.

---

## §0 — Threat model (what SC4 defends against)

**Operator error**, not an adversary: a wrong model launched on the producer,
a mismatched consumer config, a stale manifest. The failure mode without SC4
is **silent**: the consumer rebinds onto bytes that are the wrong weights and
serves wrong tokens with no error. SC4 turns that into a clean, logged
rejection + fallback. (Adversarial collision resistance is a v2 concern — §a.)

## §a — Fingerprint mechanism — **recommend a two-tier disk-file fingerprint**

Weights must be **byte-identical** to share (`Mistral-7B-v0.1` vs `-Instruct`:
same architecture, same shapes, *different* weights — a config hash alone is
insufficient, Track 2 design §5). A full cryptographic hash of 4 GB of weights
is ~2–4 s — over the "< few hundred ms" bar (§f). Recommended instead:

**`fingerprint = SHA256( tier1 ‖ tier2 )`**, computed from the on-disk model:
- **tier 1 (structural, fast-fail)** — `config.json` (architecture, layer
  count, hidden/intermediate dims) + the `safetensors` index (tensor names,
  shapes, dtypes, offsets) + each shard's filename + exact byte size +
  `torch_dtype`. Sub-millisecond. Catches a different architecture / shape /
  shard layout outright.
- **tier 2 (weight-content)** — SHA256 over **sampled** weight bytes: a fixed
  1 MiB chunk at the head, middle, and tail of every `safetensors` shard.
  ~tens of ms (reads ~3 MiB/shard). Catches *same-architecture, different
  weights* (v0.1 vs Instruct, different fine-tune) with overwhelming
  probability — fine-tuning perturbs essentially every weight, so any sampled
  chunk differs.

**v1 boundary (documented):** this is **operator-error-proof, not
adversary-proof** — a hostile party could craft a collision in the unsampled
regions. The threat model (§0) is operator error; a full weight hash is the
v2 hardening if a hostile multi-tenant model emerges. Tokenizer hash is **not**
included — weight bytes are what must match for *sharing*; the tokenizer is
per-tenant and not shared.

## §b — Computation site — **recommend both (producer stores, consumer recomputes)**

The producer computes the fingerprint from **its** `model_path` at arena
creation and writes it into the JSON manifest. The consumer computes the
fingerprint from **its** `model_path` at import and **compares** to the
manifest's. This is not "trust the producer" vs "don't" — it is inherently
two-sided: the safety question is *do the producer's and consumer's models
match*, which only a both-sides computation answers. Each side hashes its own
files; equality ⇒ safe to share.

## §c — When verification fires — **recommend at import, before mapping**

The consumer compares fingerprints **before calling `weight_arena_import`** —
before any VMM resource is committed. Cheapest, and a mismatch fails with zero
cleanup. Sequence: receive manifest → compute consumer fingerprint → compare →
*only on match* call `weight_arena_import` and rebind.

## §d — On mismatch — **recommend consumer-side hard-fail + fallback + log**

A fingerprint mismatch → the consumer does **not** import; it falls back to an
independent `from_pretrained` load (today's behaviour — no regression) and
logs the mismatch (`cipher_log` / stderr, with both fingerprints).
**Producer-side enforcement is NOT in v1**: the producer cannot know the
consumer's model at export time without extra protocol, and it adds nothing
for *correctness* — the consumer-side check already fully prevents the
silent-wrong-output failure. Producer-side enforcement is *access control*
(don't even hand the fd to a wrong consumer), a security concern → v2.

## §e — Transport — **recommend the JSON manifest**

The producer's fingerprint is one more field in the SC3 JSON layout manifest
(`fingerprint`, plus `fingerprint_tiers` for diagnosability). No new channel.
An arena-embedded header is rejected — same reasoning as SC3-1 §3 (chicken/egg
on sizing; tamper-resistance is not the v1 concern, §0).

## §f — Cost — **target < 100 ms each side**

Sampled-chunk hashing reads ~3 MiB/shard → producer-side at export and
consumer-side at import each ≈ tens of ms (well under the "< few hundred ms"
bar). SC4-3 measures both empirically; if a model's shard count makes it
exceed ~100 ms, that is a finding, not a failure. (The design *chose* sampling
precisely to stay under the bar — a full 4 GB hash would not.)

## §g — Failure-injection tests (SC4-3)

- **Unit — manifest-fingerprint tamper:** flip a byte of the producer
  fingerprint in the manifest; the consumer must reject the import cleanly and
  fall back. Deterministic, needs no second model.
- **Unit — fingerprint discrimination:** the helper computes *distinct*
  fingerprints for two genuinely different models / a config-perturbed variant.
- **Integration — mismatch fallback:** a consumer pointed at a different model
  than the producer attempts the share; gets a clean rejection; **falls back
  to an independent load and runs correctly** (verified by a real forward).
  SC4-2 selects the second model from `models/` (or a config-perturbed copy if
  no second same-family model is present).

## §1 — SC4-2 / SC4-3 / SC4-4 plan — and the NO-ANCHOR-ROTATION finding

**Finding (surfaced for adjudication):** the fingerprint is a hash of on-disk
files + a manifest string field + a consumer-side string compare — **pure
Python**. It does **not** belong in `cipher_kv_bridge` (C/C++): hashing
safetensors shards and comparing strings is not a VMM/substrate operation.
SC4 is therefore implemented as a **shared Python helper**
`cipher_model_fingerprint.py` + producer/consumer harness changes — **no
C/C++ change, no `cipher_kv_bridge` rebuild, NO ANCHOR ROTATION.** The user's
authorization noted "*likely* cipher_kv_bridge → SC4 anchor *if* the primitive
lives in kv_bridge" — it does not need to. SC4 is a thin, anchor-stable SC.

| phase | scope | est. |
|---|---|---|
| SC4-1 | this design memo | done |
| SC4-2 | `cipher_model_fingerprint.py` (two-tier hash, §a); producer writes `fingerprint` into the manifest; consumer computes + compares before import (§c), hard-fail + fallback (§d) | ~0.5 d |
| SC4-3 | unit + integration failure-injection (§g); cost measurement (§f); regression smoke (W3 + isolation + dmesg — substrate unchanged, so this is a formality) | ~0.5 d |
| SC4-4 | `TRACK_2_SC4_CLOSEOUT.md`; update memory | ~0.25 d |

**Anchors:** kmod `285d102e`, libcipher_rt `83afd1ca`, libcipher_v2
`86618c30`, **cipher_kv_bridge `c04b0c39` — all unchanged through SC4.** No
`.pre_sc4` binary fallback is needed (no binary changes); the SC3
producer/consumer harness scripts are preserved as `.pre_sc4` copies before
SC4-2 edits.

## §2 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **Mechanism (§a)** — accept the two-tier (structural + sampled-weight-byte)
   SHA256 fingerprint, with the documented v1 boundary (operator-error-proof,
   not adversary-proof; full weight hash is v2).
2. **Site (§b) / timing (§c)** — accept producer-stores + consumer-recomputes;
   compare before `weight_arena_import` (pre-resource-commit).
3. **Mismatch (§d)** — accept consumer-side hard-fail + fallback + log; no
   producer-side enforcement in v1.
4. **Transport (§e)** — accept the JSON-manifest `fingerprint` field.
5. **The §1 finding** — accept that SC4 is **pure Python, no anchor rotation**
   (`cipher_model_fingerprint.py` + harness), not a `cipher_kv_bridge` change.

On adjudication: proceed to **SC4-2** — build `cipher_model_fingerprint.py`
and wire it into the producer/consumer path.
