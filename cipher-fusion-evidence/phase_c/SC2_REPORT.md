# Phase C / Track 2 — SC2: producer-side VMM weight arena — REPORT

**Date:** 2026-05-18. **Status:** built + verified — **5/6 SC2 criteria PASS**;
criterion 6 (kmod ownership) is a scope item, see §4. **Adjudication
checkpoint** per DECISION 1 before SC3. Anchor `a7ac8e97` **unchanged**.

---

## §1 — What SC2 built

The producer side of cross-tenant weight-sharing: a tenant allocates its model
weights into a CIPHER-owned VMM allocation that is exportable as a POSIX file
descriptor (the handle a peer process imports in SC3).

| file | change | nature |
|---|---|---|
| `cipher_rt_kv_alloc.h` | weight-arena API: `cipher_rt_weight_arena_{create,export,info,free}` | additive |
| `cipher_rt_kv_alloc.c` | weight-arena implementation (~150 LOC) | additive — KV slab/dedup paths untouched |
| `cipher_kv_bridge.cpp` | `WeightArena` pybind class (`weight_arena_create` / `.alloc` / `.export_fd`); `page_info` extended with the weight branch | additive |

`cipher_kv_bridge.so` rebuilt → md5 `fca6843d`. Prior preserved as
`cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so.pre_phase_c_sc2` (`8d6ffe3f`).
`cipher_rt_kv_alloc.{c,h}` prior preserved as `.pre_phase_c_sc2`.

**`libcipher_rt.so` is untouched — anchor `a7ac8e97` unchanged** (md5 verified).
`cipher_rt_kv_alloc.o` links only into `cipher_kv_bridge.so`, never into
`libcipher_rt.so` (Makefile `OBJS` confirmed) — so no substrate anchor rotates.

## §2 — Design decision: one allocation, not a WEIGHT-tagged KV slab

The design memo §4 said "add a weight-arena slab tag" — i.e. route the arena
through the KV slab machinery with a new tag. **SC2 instead implements the
weight arena as one dedicated `cuMemCreate` allocation.** Rationale:

- KV slabs grow page-by-page (2 MiB at a time) as a sequence decodes — per-page
  `cuMemCreate`/`cuMemMap` is correct *for KV*. Model weights are **load-once,
  fixed-size**: one `cuMemCreate` of the whole arena is the natural shape.
- **The decisive reason — export.** A 2.2 GB arena is ~1100 × 2 MiB pages. As a
  tagged KV slab each page is a separate physical handle → ~1100 export fds →
  blows the default 1024 fd limit. As one allocation it is **one handle, one
  export fd.** This is also what the memo's own §3b ("the shared weight arena
  mapped at the same VA") implicitly wants.

The arena path is fully additive — it shares only `g.mu` and the device/primary
context from `cipher_rt_kv_alloc_init`; the KV slab and dedup paths are byte-
unchanged. This divergence from the memo wording is flagged for adjudication
(§5) — it is an implementation improvement, not a scope change.

## §3 — Verification (TinyLlama-1.1B, `phase_c/sc2_verify.py`)

TinyLlama-1.1B loaded; all **201 weight tensors (2.200 GB)** rebound into one
weight arena (`weight_arena.alloc` + `copy_` + `param.data = vmm_tensor`).
Teacher-forced over a fixed 15-token gold prefix; stock-weights forward vs
VMM-weights forward.

| # | SC2 criterion | result | gate |
|---|---|---|---|
| 1 | weights allocated via VMM-backed path | `page_info` → `kind='weight'`; arena 2.07 GiB at va `0x306000000` | **PASS** |
| 2 | `page_info` reports VMM-handle presence | weight tensor → `vmm_handle=0x58af5efad330`; non-CIPHER ptr → `None` | **PASS** |
| 3 | DtoH memcmp bytes-identical to stock | `model.embed_tokens.weight` (largest, 131 MB) byte-identical; all 201 tensors `torch.equal` | **PASS** |
| 4 | real forward pass reads VMM weights — teacher-forced KL ≤ 0.1 | **KL max = 0.000e+00**, logits max-abs-diff = 0.0 | **PASS** |
| 5 | producer creates exportable VMM handle | `export_fd` = 44, `fstat` ok | **PASS** |

KL is *exactly* 0.0 (not merely ≤ 0.1): byte-identical weights through
deterministic kernels produce bit-identical logits. KL=0.0 is the *consequence*
of an evidence chain, not the sole evidence: (i) `page_info(data_ptr())` →
`kind='weight'` confirms the param's storage is in the arena after rebind;
(ii) `torch.equal(vt, p.data)` after `copy_` confirms bytes actually moved (a
no-op copy would leave `vt` at the `cuMemsetD8` zero-init and fail the check on
non-zero weights); (iii) DtoH memcmp confirms byte round-trip; (iv) `p.data =
vt` replaces `nn.Parameter` storage, and LLaMA modules re-fetch `self.weight`
each forward. Arena freed cleanly on process exit (GPU back to 0 MiB).

### §3.1 — Negative control + tied-weights (`phase_c/sc2_negctl.py`)

- **Negative control — the forward provably reads arena memory.** After rebind,
  corrupting one arena-resident weight in place (`model.layers.0.mlp.gate_proj.
  weight += 1.0`) shifts the logits by **max-abs-diff 18.39**. The forward
  consumes the VMM arena directly — KL=0.0 in §3 is not a stale-pre-rebind-copy
  artifact.
- **Tied weights:** TinyLlama-1.1B has **untied** embeddings here
  (`embed_tokens.weight.data_ptr() != lm_head.weight.data_ptr()`) — both are
  rebound as separate tensors (the 201-tensor / 2.20 GB count reflects this).

## §4 — Criterion 6 (kmod pool ownership) — SCOPE ITEM, not done

DECISION 1's SC2 criteria list "kmod pool ownership extension working (T4.6.4
refcount + do_exit reaper for weight arenas)." This was **not** done, by
deliberate scope judgment:

- The design memo §7 sub-component table scopes kmod-owned arena lifetime as
  **SC5** ("Lifetime: kmod-owned weight arena, refcount, crash teardown —
  ~0.5–1 day"), separate from SC2 ("Weight-arena VMM allocation + export/
  import"). DECISION 1's restated criteria pulled it into SC2 — an internal
  inconsistency with the memo that DECISION 1 also approved.
- A kmod extension is a kernel-module change: a new ioctl (additive ABI nr per
  the ABI rule), `cipher_kmod` rebuild, and a **live `.ko` reload** on the
  pod. That is a genuine, separate risk surface — not something to fold into
  SC2 unannounced.
- SC2's actual deliverable — the producer-side allocator + export — **does not
  need the kmod.** Kmod ownership matters for *crash-teardown lifetime* (the
  arena outliving a crashed tenant), which only becomes load-bearing once peers
  **import** the arena (SC3) and many tenants share it (SC5).

**Recommendation:** keep kmod-owned arena lifetime as SC5 per the design memo;
do not reframe SC2 to require it. SC2 is reported as 5/5 of its memo-scoped
criteria PASS, with criterion 6 surfaced as a scope correction for adjudication.
(This is the same SC2-vs-SC5 / memo-vs-brief reconciliation as the earlier
memcmp-vs-KL flag.)

## §5 — Adjudication ask (DECISION 1 checkpoint before SC3)

1. **Adjudicate SC2 = PASS** on its 5 memo-scoped criteria (§3).
2. **Accept the §2 design divergence** — one `cuMemCreate` allocation instead of
   a WEIGHT-tagged KV slab (avoids the ~1100-fd export explosion).
3. **Confirm criterion 6 (kmod ownership) belongs to SC5**, not SC2 (§4) — or
   direct otherwise.
4. On adjudication, proceed to **SC3** (consumer-side import: a peer process
   imports the exported fd and wraps the arena as its own parameter storage).

## §6 — Anchors

- `libcipher_rt.so` `a7ac8e97` — **unchanged** (SC2 does not touch it).
- `cipher_kv_bridge.so` rebuilt `8d6ffe3f` → `fca6843d`; `.pre_phase_c_sc2`
  preserved. (The bridge `.so` is a build artifact, not a campaign anchor.)
- kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30` — unchanged.

## §7 — Artifacts

`phase_c/sc2_verify.py`, `phase_c/sc2_result.json`. Sources +
`.pre_phase_c_sc2` preserved copies in `cipher_rt_phase4/`.
