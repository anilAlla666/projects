# Phase C / Track 2 — Cross-Tenant Weight-Sharing — SC3 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU, no measurement. STOP for adjudication before SC3-2 build.
**Predecessors:** Track 2 SC1 (`PHASE_C_TRACK_2_DESIGN.md`) + SC2
(`SC2_REPORT.md`, 5/5 PASS) closed. Track 3 closed (`TRACK_3_CLOSEOUT.md`).

SC3 builds the **consumer side** of weight-sharing: a peer process imports
tenant 0's exported VMM weight arena and wraps it as its own model parameter
storage — **skipping the safetensors read entirely**. SC2 proved the producer
side (create / export / `page_info` / a peer can DtoH-`memcmp` identical
bytes). SC3 turns "a peer can read the bytes" into "a peer *runs the model* on
the shared bytes." It is the load-bearing fiddly step the Track 2 design memo
§7 flagged at ~2 days.

---

## §0 — Where SC3 plugs in

SC2 (`cipher_rt_kv_alloc.c` / `cipher_kv_bridge.cpp`, anchor `fca6843d`)
gives the producer: `cipher_rt_weight_arena_{create,export,info,free}` and the
`WeightArena` pybind (`weight_arena_create` / `.alloc(shape,esize,dtype)` /
`.export_fd()`). Tenant 0 allocs every weight tensor into one `cuMemCreate`
arena, `copy_`s the safetensors bytes in, and `export_fd()`s a POSIX fd.

**SC3 adds the symmetric consumer half:** import that fd, map the arena, and
rebind a meta-loaded model's parameters onto it. SC4 (model-identity
fingerprint) and SC5 (kmod-owned arena lifetime) are explicitly **out of SC3
scope** — SC3 assumes a matching model and tenant-0-owned arena lifetime;
those become load-bearing only at N>2 / crash-teardown, which SC4/SC5 cover.

## §1 — The import primitive

New, symmetric to `cipher_rt_weight_arena_export`:

```c
/* Import a weight arena exported by a peer. Reserves VA at `want_base`
 * (the producer's arena base — same-VA, §2), imports the POSIX-fd VMM
 * handle, maps it. Returns 0 + the mapped base; <0 on any failure (caller
 * falls back to independent load). */
int cipher_rt_weight_arena_import(int fd, unsigned long long want_base,
                                  size_t bytes,
                                  unsigned long long *out_base);
```

Sequence (the verified T4.6.4 path, consumer direction):
`cuMemImportFromShareableHandle(fd, POSIX_FILE_DESCRIPTOR)` →
`cuMemAddressReserve(bytes, align, fixedAddr=want_base)` →
`cuMemMap` → `cuMemSetAccess(READ — weights are read-only in the consumer)`.
A `WeightArena.import_fd(fd, base, bytes)` pybind wraps it (mirrors
`export_fd`); a consumer-side `.view(name, shape, dtype, offset)` returns a
`torch.Tensor` aliasing the mapped arena (no copy — unlike the producer's
`.alloc`).

## §2 — Same-VA mapping — **recommend, with a fallback**

The Track 2 design §3b chose VMM specifically so the arena can be mapped at
the **same virtual address** in every tenant — then the producer's tensor
`data_ptr()`s are valid verbatim in the consumer, and no per-tensor pointer
fix-up is needed.

- **Primary:** the consumer `cuMemAddressReserve`s the producer's exact base
  VA (`fixedAddr`, from the manifest §3). If that VA is free in the consumer
  (it normally is — a fresh process, the arena reserved early), the mapping is
  same-VA and the manifest's offsets *are* absolute pointers.
- **Fallback — VA collision:** if `want_base` is unavailable in the consumer,
  reserve anywhere and map there; tensor pointers are then
  `consumer_base + offset` (offset from the manifest). The consumer rebuilds
  tensors from `(offset, shape, dtype)` regardless — §4 — so a different base
  is **not fatal**, only less elegant. SC3 builds both; same-VA is the fast
  path, offset-relative is the always-correct path.

Recommendation: attempt same-VA, fall back to offset-relative; the consumer's
rebind (§4) is written offset-relative so it is correct either way.

## §3 — The layout manifest

The consumer must know what is *in* the arena. SC2's producer packs tensors by
`.alloc()` in `model.named_parameters()` order; SC3 makes that layout explicit:

**Manifest content** — a JSON sidecar: arena `base` VA, `bytes`, and an
ordered list of `{name, shape, dtype, offset, nbytes}` per weight tensor,
plus a `producer_pid` and a coarse `model_path`. (The byte-exact model
*fingerprint* is SC4 — SC3's manifest carries `model_path` only as a sanity
string.)

**Transport — adjudication item.** Three options:

| option | assessment |
|---|---|
| **(a) JSON sidecar file** at a well-known path *(recommend)* | the producer writes `…/cipher_weight_arena_<tenant>.json`; the consumer reads it. Simple, debuggable, no new IPC. The fd itself still passes by the SC5 kmod-owned mechanism / a UNIX-socket `SCM_RIGHTS` in the SC3 test harness. |
| (b) prepend a header *inside* the arena | self-describing, but the consumer must map before it can read the layout — chicken/egg for sizing; rejected |
| (c) a new kmod ioctl carrying the manifest | over-engineered for SC3; the kmod's role is lifetime (SC5), not layout |

Recommendation **(a)** — JSON sidecar. The fd transport for the SC3 *test* is
a UNIX-domain socket `SCM_RIGHTS` pass (standard, self-contained); the
production fd hand-off is SC5's kmod-owned path.

## §4 — Consumer storage rebind (the fiddly HF/PyTorch step)

The consumer must run a real model whose parameters *are* the imported arena:

1. **Load on `meta`** — `AutoModelForCausalLM.from_pretrained(model_path,
   torch_dtype=…, device_map=None)` then `.to_empty(device='meta')`, or load
   directly with `init_empty_weights()` — no weight bytes read, no HBM for
   weights.
2. **Rebind each parameter** — for each `name, p` in `named_parameters()`,
   look up the manifest entry, build a `torch.Tensor` aliasing the imported
   arena at `base+offset` with `(shape, dtype)` (the `WeightArena.view`
   pybind, §1), and assign `p.data = view` — the exact storage-rebind SC2's
   `sc2_verify.py` already does for the *producer* (`p.data = vt`), here with
   an aliasing view instead of a fresh `.alloc`.
3. **Read-only** — the consumer never writes weights; the arena is mapped
   `cuMemSetAccess(READ)` for the consumer (write would fault — a hard guard
   against a consumer corrupting shared weights).
4. **Tied weights / non-parameter buffers** — TinyLlama has untied embeddings
   (SC2 §3.1); the rebind walks `named_parameters()`, and `named_buffers()`
   (e.g. RoPE inv_freq) are *recomputed locally*, not shared — they are tiny
   and not in the arena. The manifest lists only parameters.

The "fiddly" risk is HF/PyTorch internals: `meta`-load coverage of all
submodules, `register_parameter` vs direct `.data` assignment, dtype/stride
exactness. SC3-2 builds against TinyLlama-1.1B (the SC2 model) first.

## §5 — Failure modes & fallback

| failure | handling |
|---|---|
| `cuMemImportFromShareableHandle` fails (bad fd / VMM error) | import returns <0 → consumer falls back to independent `from_pretrained` (today's behaviour, no regression) + logs |
| same-VA reserve collision | §2 fallback — offset-relative mapping; not fatal |
| manifest missing / unreadable | fallback to independent load |
| manifest tensor set ≠ consumer's `named_parameters()` (arch/variant mismatch) | SC3 detects the mismatch (name/shape/dtype set compare) and falls back; the *byte-exact* fingerprint guard is SC4 |
| producer (tenant 0) exits while a consumer is mapped | **SC3 scope assumes tenant-0-owned lifetime** — a documented SC3 limitation; the kmod-owned arena that survives the producer is **SC5**. SC3's N=2 test keeps the producer alive for the consumer's run. |
| consumer writes a weight | precluded — arena mapped READ-only in the consumer (§4.3) |

## §6 — SC3-2 / SC3-3 / SC3-4 plan

| phase | scope | est. |
|---|---|---|
| **SC3-1** | this design memo | done |
| **SC3-2** | build: `cipher_rt_weight_arena_import` + `WeightArena.import_fd`/`.view` (`cipher_rt_kv_alloc.c` / `cipher_kv_bridge.cpp`); the JSON manifest writer (producer) + reader (consumer); the meta-load + storage-rebind consumer path; an `SCM_RIGHTS` fd-pass test harness. Anchor `fca6843d` → `.pre_track2_sc3` **before any edit**. | ~2 d |
| **SC3-3** | verify: a consumer imports tenant 0's TinyLlama arena, rebinds, and runs decode — **teacher-forced KL must be ~0** (bit-identical — the SC2 §6c gate); `page_info` shows the consumer's weight tensors on the **same physical pages** as the producer's (the §6b structural proof); producer + consumer total FB ≈ 1×W + 2×ε. Regression: Track 3 substrate intact (kmod `285d102e` + libcipher_rt `83afd1ca` — W1/W2/W3 + isolation + SC3-DSM e2e). | ~1–2 d |
| **SC3-4** | closeout — `TRACK_2_SC3_CLOSEOUT.md`; anchor rotation if `cipher_kv_bridge` changed; update anchors + memory. | ~0.5 d |

**Anchors:** SC3 touches `cipher_rt_kv_alloc.{c,h}` + `cipher_kv_bridge.cpp` —
which build into **`cipher_kv_bridge.so`** only (SC2 report: never into
libcipher_rt). So **SC3 rotates `cipher_kv_bridge` `fca6843d`**; libcipher_rt
`83afd1ca`, kmod `285d102e`, libcipher_v2 `86618c30` are **unchanged**.
Preserve `cipher_kv_bridge.*.pre_track2_sc3` before any edit. Pre-SC3 baseline
(W1/W2/W3 + isolation + DSM SC3 e2e) confirms the Track 3 substrate before the
SC3 build.

## §7 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **Import primitive (§1)** — accept `cipher_rt_weight_arena_import` +
   `WeightArena.import_fd`/`.view`, symmetric to SC2's export, the verified
   T4.6.4 VMM POSIX-fd path in the consumer direction.
2. **Same-VA (§2)** — accept attempt-same-VA / fall-back-offset-relative, with
   the rebind written offset-relative so it is correct either way.
3. **Manifest (§3)** — accept the JSON sidecar (option a); fd transport for
   the SC3 test = `SCM_RIGHTS`, production fd hand-off deferred to SC5.
4. **Rebind (§4)** — accept meta-load + per-parameter `.data` rebind to
   READ-only aliasing views; buffers recomputed locally.
5. **SC3 scope line (§5)** — accept that SC3 assumes a matching model
   (fingerprint = SC4) and tenant-0-owned lifetime (kmod-owned = SC5); SC3's
   test keeps the producer alive.

On adjudication: proceed to **SC3-2** (build), closing with the §6 regression
discipline and the `cipher_kv_bridge` anchor rotation.
