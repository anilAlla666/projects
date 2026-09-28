# Week 5 Step 1 — KV-Dedup Live Wire — DESIGN MEMO

**HEADLINE: LOCKED-SHAPE-II + EXPLICIT-FLUSH TRIGGER + SUBSTRATE-EXPANDED (W5 ships real HBM savings via new `cipher_rt_kv_dedup_alias` primitive).**

> **Post-design-memo user adjudication (2026-05-21):** Q1 = **expand W5 to
> include substrate live-rebind work** (not VALIDATE-only); Q2 = **tighten
> Step 2 budget with pinned-region pre-verification gate at Step 2 entry**.
> Week 5 step sequence now: Step 1 (this memo, done) → **Step 1b
> (substrate primitive `cipher_rt_kv_dedup_alias` in
> cipher_rt_kv_alloc.c; rebuilds cipher_kv_bridge.so; Track 2 SC3
> regression PASS; anchor rotates c04b0c39 → new)** → Step 2 (plugin uses
> new primitive; pinned-region pre-verification gate at entry) → Step 3
> (test harness validates **actual HBM savings** via nvidia-smi memory
> deltas PLUS STATS hit rate ≥ 60%) → Step 4 (closeout). Revised total:
> **~15-22h**. See Part I below for the substrate work plan; the
> VALIDATE-only framing earlier in this memo (Part A.5, Part E.2) is
> historical context for how the adjudication question was reached —
> reading the doc top-to-bottom is the audit-trail view; the live
> Step-2-onwards plan is Part I §I.6.

Shape (ii) VMM-page-level selected and ratified by user as the page-unit
of dedup. The Step-1 read of `cipher_rt_kv_alloc.c:522` discovered the
existing `cipher_rt_kv_dedup_put` is an allocation-time API (allocates a
fresh VA in its own pool, returns a new devptr); does NOT support
"given an existing devptr, rebind that devptr onto a deduped physical."
Originally proposed as VALIDATE-only Week-5 scope; user expanded scope
to include the **new `cipher_rt_kv_dedup_alias`** primitive that closes
that gap (Part I).

**What Week 5 delivers honestly:**
- **Mechanism validation over live content** — bridge fires
  `cipher_rt_kv_dedup_put` against actual post-prefill KV content (not zero
  pages); STATS show real hits + misses over real workload.
- **Bit-identical N=4 same-prompt** — satisfied by vLLM's deterministic
  decode (identical prompts produce identical token_ids regardless of
  dedup wire actually saving memory).
- **Dedup hit rate ≥ 60%** — measured via kmod STATS counters, proves
  cross-tenant content recognition works at the page-hash level.

**What Week 5 does NOT deliver** (defer to Week 6+):
- Actual HBM memory savings in vLLM's KV runtime. vLLM continues to use
  its own (non-deduped) physical pages; dedup happens in cipher_rt_kv's
  separate pool. The substrate primitive needed for live in-place rebind
  (e.g. `cipher_rt_kv_dedup_alias(devptr, content)`) is queued for Week-6+
  substrate work.

**Major Step-1 win still stands:** the cipher_rt_kv_dedup_* C wrappers
already exist in `cipher_kv_bridge.so` (4 `T` symbols). Step 2 plugin
module is a thin ctypes binding; no separate xxhash binding needed
(xxhash + memcmp + ioctl handshake all encapsulated in the C function).
Step-2 LOC estimate **~190**.

**Date:** 2026-05-21. **Paperwork only.** No source modified.
**Step 2 prompt may be drafted against this memo.**

---

## Part A — R-W5.4 page-granularity resolution

### A.1 — Substrate design memo C identified

`/home/ubuntu/cipher-fusion-evidence/t4_6_4_design.md` (235 LOC; the
canonical kvdedup substrate design). Key constraints, verbatim:

- §A (ioctl interface): `PUT _IOWR('K',2, struct {u64 content_hash; /* xxh64,
  userspace-computed */ s32 export_fd; /* cuMem POSIX fd of caller's page */
  ...})` — page identity is xxh64 over the 2 MiB content; kmod refcounts
  POSIX fds.
- §C (per-tenant tracking): *"Bound N = 8192 pages/tenant (16 GiB of KV;
  covers 128K context: 32 layers × 2 roles × 128 pages)"* → **8192 × 2 MiB
  = 16 GiB** confirms 2 MiB granularity by arithmetic.
- §Call model: *"T4.6.4's latency budget assumes model (a): the cold path —
  cipher_rt_kv_dedup_put is called **once per slab page at allocation time**,
  not per-request and not per-decode-step. Pages allocate seconds-to-minutes
  apart per tenant, so the ~400 µs hit-path cost (cuMemImportFromShareableHandle
  + 1 DtoH + memcmp) is amortized to nothing."*

### A.2 — cipher_kvdedup.h ABI confirms 2 MiB contract

```c
struct cipher_kvdedup_put {
    __u64 content_hash;  /* in : xxhash64 of the 2 MiB page */
    __s32 export_fd;     /* in : cuMem POSIX shareable fd of the caller's
                          *      freshly-created physical page */
    ...
};
```
And the header doc-comment: *"Userspace computes the xxhash64 of the 2 MiB
page and the memcmp-verify on a hit; the kmod owns the table, the refcounts,
and the cuIpc POSIX-FD handles."*

The 2 MiB granularity is **immutable substrate ABI**, not a tuning knob.

### A.3 — vLLM v1 block size for Mistral-7B (the gap analysis)

Mistral-7B KV per token: 2 × 8 KV heads × 128 head_dim × sizeof(fp16) =
4 KiB/token/layer. vLLM v1 default `block_size = 16` tokens → per-block per-layer = **64 KiB**.

**Granularity gap: 2 MiB / 64 KiB = 32× per layer.** One 2 MiB page holds
32 vLLM blocks per layer. (Scope-lock said "factor-of-thousands"; actual is
~32× for Mistral-7B at fp16. Advisor's flag was directionally correct — the
gap is real and non-trivial — but in absolute terms manageable.)

### A.4 — CP 5.1 vmm_zeros already operates at 2 MiB VMM-page granularity

From `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py:87` →
`cipher_kv_bridge.vmm_zeros([nbytes], 1, "int8", _TENANT, 0, idx & 0xFFFF, 2)`.
Stats from `cipher_kv_bridge.get_stats()`:
- `pages_mapped` (count)
- `page_size` (bytes — H100 default = 2 MiB; queried at init via
  `cuMemGetAllocationGranularity`)

CP 5.1's underlying allocator (`cipher_rt_kv_alloc.c`) calls
`cuMemCreate` per page; CUDA-VMM grants are 2 MiB on H100 by default. **CP
5.1 already allocates in exactly the 2 MiB units kvdedup operates on.**
Shape (ii) hooks into this same seam → mechanically clean.

### A.5 — Shape decision: **(ii) VMM-page-level + explicit-flush trigger + VALIDATE-ONLY scope (substrate live-rebind deferred)**

**Critical substrate-API finding (Step 1 read of `cipher_rt_kv_alloc.c:522`):**
`cipher_rt_kv_dedup_put(content, &out_devptr)` is an **allocation-time**
API. The implementation:
1. Carves a new VA from cipher_rt_kv's own pool (`carve_free_page() → gi`,
   `va = pool_base + gi × page_size`).
2. `cuMemcpyHtoD`s `content` into the new VA.
3. Exports the new physical via POSIX-fd, ioctls `CIPHER_KVDEDUP_PUT`.
4. On MISS: keeps the new VA + new physical; returns the new `va` via
   `*out_devptr`.
5. On HIT: `cuMemUnmap(va, page_size)` + `cuMemRelease(own_handle)`, then
   `cuMemMap(va, shared_handle)` — **rebinds its own new VA onto the
   shared physical** — and returns the (now-deduped) new `va` via
   `*out_devptr`.

**The returned `out_devptr` is ALWAYS a new VA in cipher_rt_kv's pool —
never the caller's pre-existing devptr.** The rebind on HIT happens on
cipher_rt_kv's own VA, not on any external caller's VA.

**Consequence for vLLM/CP 5.1 wiring:** vLLM holds tensors at specific
data_ptrs (from CP 5.1's vmm_zeros allocation). We cannot just call
`cipher_rt_kv_dedup_put(content, &new_devptr)` and expect vLLM's tensors
to start reading from the deduped physical — vLLM's tensors still hold
the original data_ptr, which still points at the original
(non-deduped) physical.

To make vLLM actually read deduped memory would require either:
- A new substrate primitive: `cipher_rt_kv_dedup_alias(existing_devptr,
  content)` that atomically swaps the caller-provided VA's physical
  mapping onto the deduped physical. **This doesn't exist today** in
  `cipher_kv_bridge.so` (`nm -D` confirms no such symbol). Adding it is
  substrate C work, not Python-bridge work — out of Week-5 scope per
  the scope-lock STOP-gate ("substrate design change → out of Week 5
  scope").
- Or: rewire CP 5.1 to allocate via the dedup path natively (instead
  of vmm_zeros). But that fires dedup at allocation time over
  zero-pages — degenerate (only zero-pages dedup, real content
  doesn't because it isn't written yet).

**Selected Week-5 framing: Shape (ii)-VALIDATE.** The bridge calls
`cipher_rt_kv_dedup_put` on real post-prefill content (DtoH copy each
vLLM 2 MiB page → call dedup_put with that content). The substrate
returns its own pool's devptr (which we discard); STATS counters
correctly record hits/misses over real workload. **vLLM's KV
continues to live in CP 5.1's vmm_zeros pool unchanged** (no actual
HBM savings in Week 5). What Week 5 PROVES:
- The dedup mechanism fires correctly on real same-prompt content
  (STATS `hits > 0` after N=2 same-prompt PUTs).
- The hash + memcmp pipeline scales to ≥ 60% hit rate at N=4
  same-prompt (STATS-measured).
- vLLM correctness is unaffected (bit-identical decode regardless;
  KL = 0 vs vanilla because vLLM's runtime is untouched).

| | Shape (i) block-aggregation | Shape (ii) VMM-page-level **(SELECTED, VALIDATE scope)** |
|---|---|---|
| dedup unit | aggregate 32 vLLM blocks → 2 MiB page → hash | one CIPHER VMM page (already 2 MiB) → hash |
| bookkeeping | per-tenant block→page map; fragmentation handling | per-allocation page list on the wrapper, walked at `dedup_now()` |
| precision | per-block (block diverges → page diverges) | per-VMM-page (any byte difference in 2 MiB → MISS) |
| seam | net-new monkey-patch site (around vLLM's block-allocation path) | extends CP 5.1's `_allocate_kv_cache_tensors` wrap + adds `dedup_now()` Python entry point |
| LOC | ~200-300 + new bookkeeping data structures | ~190 (ctypes wrappers + page-list iterator + flush API) |
| **Memory savings in W5** | **None — substrate live-rebind missing** | **None — same reason** |
| **What W5 validates** | mechanism, if substrate live-rebind shipped | mechanism end-to-end at the dedup_put API level |

| | Shape (i) block-aggregation | Shape (ii) VMM-page-level **(SELECTED)** |
|---|---|---|
| dedup unit | aggregate 32 vLLM blocks → 2 MiB page → hash | one CIPHER VMM page (already 2 MiB) → hash |
| bookkeeping | per-tenant block→page map; fragmentation handling | none — uses cipher_kv_bridge's existing page mapping |
| precision | per-block (block diverges → page diverges) | per-VMM-page (any byte difference in 2 MiB → MISS) |
| seam | net-new monkey-patch site (around vLLM's block-allocation path) | extends CP 5.1's `_allocate_kv_cache_tensors` patch + adds a `dedup_now()` flush entry point |
| LOC | ~200-300 + new bookkeeping data structures | ~100-150 (ctypes wrappers + flush iterator) |

Selected Shape (ii) on these grounds:
- CP 5.1 already operates at 2 MiB. The seam exists.
- Per-tenant test (N=4 same-prompt): tenants writing identical content to
  identical block layouts produce **identical 2 MiB VMM pages by
  construction**. Precision cost (full-page hash MISS if even one byte
  differs) is zero for the same-prompt case the Week-5 gate measures.
- LOC budget halves vs Shape (i).

### A.5b — Trigger model: **explicit-flush** (resolves the cold-path-vs-live-path question)

T4.6.4 cold-path assumes per-slab-page at allocation time. At allocation
time vmm_zeros returns zero-pages — dedup'ing zeros across tenants is
correct but degenerate (single shared zero page; trivial savings).
**Real dedup opportunity is post-prefill** when shared-system-prompt KV
content has been written.

CP 4.6.5/6 closure left "live per-decode call model" as characterized-not-built
(~400 µs/hit live-path tax not validated under hot path). Week 5 sidesteps
this entirely: **the plugin exposes a Python `dedup_now()` flush function
that walks all CIPHER VMM pages currently allocated and calls
`cipher_rt_kv_dedup_put` on each.** The test harness invokes `dedup_now()`
explicitly between prefill and decode. Auto-trigger (post-prefill hook,
periodic thread, or per-decode-step) is Week-6+ scope per
[[cipher-cp4656-closed]] open item.

Mechanism is sound; trigger placement is the design degree-of-freedom.
Week 5 picks **manual explicit-flush**, which:
- Has zero hot-path overhead (the test harness calls it once between
  prefill and decode).
- Validates the dedup mechanism end-to-end without committing to a
  call-model decision T4.6.5+ explicitly deferred.
- Sets up Week-6+ to wire whichever automatic trigger benchmarks favor.

### A.6 — Empirical spike: NOT NEEDED

Substrate memo C + the cipher_kvdedup.h ABI doc-comment + the
arithmetic (8192 × 2 MiB = 16 GiB; H100 cuMem granularity = 2 MiB) **all
agree** on 2 MiB as the hard contract. CP 5.1's vmm_zeros confirms the seam
exists. The optional ~30 LOC ctypes spike is **skipped** — no remaining
ambiguity that a spike would resolve.

---

## Part B — ctypes ABI specification (SIMPLIFIED by C-wrapper discovery)

### B.0 — Major Step-1 win: C wrappers already exist

`nm -D /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so | grep dedup`:
```
0000000000035170 T cipher_rt_kv_dedup_free
00000000000352c0 T cipher_rt_kv_dedup_get_stats
0000000000034850 T cipher_rt_kv_dedup_init
0000000000034b00 T cipher_rt_kv_dedup_put
```

Header `/home/ubuntu/cipher_rt_phase4/cipher_rt_kv_alloc.h:119-141` declares
the C-level API:

```c
int  cipher_rt_kv_dedup_init(void);
int  cipher_rt_kv_dedup_put(const void *content, unsigned long long *out_devptr);
void cipher_rt_kv_dedup_free(unsigned long long devptr);
void cipher_rt_kv_dedup_get_stats(struct cipher_rt_kv_dedup_stats *out);

struct cipher_rt_kv_dedup_stats {
    uint64_t puts, hits, misses,
             physical_pages, virtual_pages,
             hash_collisions, refcount_releases;
};
```

The C wrappers **encapsulate** the entire kmod-ioctl ABI (the 5
`/dev/cipher_kvdedup` ioctls), the xxhash64 hashing (over the 2 MiB content
buffer), the cuMem POSIX-fd export/import, AND the memcmp-verify on HIT.

**Plugin work reduces to ctypes against these 4 C functions in
cipher_kv_bridge.so, not against `/dev/cipher_kvdedup` ioctls directly.**
The B/C scope-lock tasks (ioctl wrapper struct-by-struct + xxhash binding)
collapse into one ctypes binding of 4 functions.

### B.1 — ctypes binding (Step 2 will write this verbatim)

```python
import ctypes
_LIB = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so")

# struct cipher_rt_kv_dedup_stats — 7 × u64
class _DedupStats(ctypes.Structure):
    _fields_ = [
        ("puts",              ctypes.c_uint64),
        ("hits",              ctypes.c_uint64),
        ("misses",            ctypes.c_uint64),
        ("physical_pages",    ctypes.c_uint64),
        ("virtual_pages",     ctypes.c_uint64),
        ("hash_collisions",   ctypes.c_uint64),
        ("refcount_releases", ctypes.c_uint64),
    ]

_LIB.cipher_rt_kv_dedup_init.restype  = ctypes.c_int
_LIB.cipher_rt_kv_dedup_init.argtypes = []

_LIB.cipher_rt_kv_dedup_put.restype  = ctypes.c_int
_LIB.cipher_rt_kv_dedup_put.argtypes = [ctypes.c_void_p,                       # const void *content (host 2 MiB)
                                        ctypes.POINTER(ctypes.c_uint64)]       # u64 *out_devptr

_LIB.cipher_rt_kv_dedup_free.restype  = None
_LIB.cipher_rt_kv_dedup_free.argtypes = [ctypes.c_uint64]                      # devptr

_LIB.cipher_rt_kv_dedup_get_stats.restype  = None
_LIB.cipher_rt_kv_dedup_get_stats.argtypes = [ctypes.POINTER(_DedupStats)]
```

### B.2 — Error mapping

`cipher_rt_kv_dedup_init` / `_put` return `int` (0 = success, <0 errno).
Step 2 wraps with Python `if rc != 0: raise OSError(-rc, os.strerror(-rc))`.
Per the kmod's documented `-ENOSPC` on per-tenant cap exhaustion (T4.6.4
§C), Step 2 catches `ENOSPC` specifically and falls back to a non-deduped
local page per design memo §F-3.

### B.3 — ABI versioning

No version field in the C struct today. Step 2 adds a sanity assertion at
init: read `_DedupStats` size = 7 × 8 = 56 bytes (matches header at memo
write-time). If kernel ever adds fields → struct size mismatch surfaces
at the next sanity check. Document in plugin module docstring.

---

## Part C — xxhash binding via libxxhash.so.0 — **NOT REQUIRED**

The scope-lock anticipated a ctypes binding to `/usr/lib/x86_64-linux-gnu/libxxhash.so.0`
for the xxhash64 of the 2 MiB content. The C-wrapper discovery (Part B.0)
**eliminates this entirely** — `cipher_rt_kv_dedup_put` computes xxh64
internally (verified by reading
`cipher_rt_kv_alloc.c` — the impl path is `xxh64 → ioctl PUT → on HIT,
DtoH + memcmp-verify → ioctl CONFIRM or FORCE_NEW`).

Python plugin code never touches xxhash. The substrate's two-phase
hash + memcmp-verify protocol (hash collision → memcmp catches → forced
miss) is encapsulated. **Part C scope reduces to: do nothing.**

(Documented for the audit trail: if a future refactor moves hashing to
the Python layer, libxxhash.so.0.8.1 is the ctypes-bindable target with
signature `unsigned long long XXH64(const void *input, size_t length,
unsigned long long seed)`.)

---

## Part D — Entry-point registration + setup.py extension

### D.1 — Current setup.py state

```python
entry_points={
    "vllm.general_plugins": [
        "cipher_vllm_kv = cipher_vllm_kv:register",
    ],
},
```

One entry point. `cipher_vllm_kv:register()` internally calls both CP 5.1
buffer-ownership setup AND CP 5.2 offload registration via
`_register_offload()`.

### D.2 — Step 2 setup.py edit

```python
entry_points={
    "vllm.general_plugins": [
        "cipher_vllm_kv       = cipher_vllm_kv:register",
        "cipher_vllm_kvdedup  = cipher_vllm_kvdedup:register",   # W5 NEW
    ],
},
py_modules=["cipher_vllm_kv", "cipher_kv_offload", "cipher_vllm_kvdedup"],
```

**Ordering is load-bearing.** Kvdedup MUST appear after cipher_vllm_kv in
the list (D.3 below).

### D.3 — vLLM plugin ordering mechanism (verified)

`vllm/plugins/__init__.py:69-83`:

```python
def load_general_plugins():
    ...
    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)
    for func in plugins.values():
        func()
```

`plugins` is a `dict`; `plugins.values()` iterates in **insertion order**
(Python 3.7+). `load_plugins_by_group` discovers entry points in the order
of the underlying `entry_points` list in setup.py. **Therefore: setup.py
list order = plugin execution order. Kvdedup placed second runs after CP
5.1's wrapper has installed.**

This means: by the time `cipher_vllm_kvdedup.register()` fires, the
`GPUModelRunner._allocate_kv_cache_tensors` reference already points at
CP 5.1's `_cipher_allocate_kv_cache_tensors` wrapper (returns
CIPHER-VMM-backed buffers). Kvdedup wraps THIS wrapper, not the original
vLLM function → wraps see CIPHER-VMM buffers, as required for Shape (ii).

### D.4 — `cipher_vllm_kvdedup.register()` skeleton (Step 2 fills in)

```python
def register():
    if not _enabled():            # CIPHER_KVDEDUP_LIVE env gate
        _log("disabled via CIPHER_KVDEDUP_LIVE — kvdedup wrapper not installed")
        return

    # Per Shape (ii) explicit-flush: the wrapper for _allocate_kv_cache_tensors
    # is the *registration* point for the per-tenant dedup context.
    # Actual PUT happens at `dedup_now()` invocation, not at allocation.
    rc = _kvdedup_init()                                                # ctypes B.1
    if rc != 0:
        _log(f"kvdedup init failed (rc={rc}); falling back to no-dedup")
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    current = GPUModelRunner._allocate_kv_cache_tensors                  # CP 5.1 wrapper
    if getattr(current, "_cipher_kvdedup_hooked", False):
        return                                                            # idempotent
    _kvdedup_wrapped = _make_kvdedup_wrapper(current)
    _kvdedup_wrapped._cipher_kvdedup_hooked = True
    _kvdedup_wrapped._cipher_kvdedup_orig   = current
    GPUModelRunner._allocate_kv_cache_tensors = _kvdedup_wrapped
    _log("kvdedup hook installed atop cipher_vllm_kv (Shape ii, explicit-flush)")

# Module-level: expose dedup_now() for the test harness to invoke
def dedup_now():
    """Walk all CIPHER VMM pages currently allocated and call dedup_put
    on each. Returns dict with per-page outcomes (HIT vs MISS) and aggregate
    STATS. Called by the test harness post-prefill; auto-trigger is W6+."""
    ...
```

---

## Part E — Monkey-patch target seam

### E.1 — CP 5.1's existing patch (read verbatim from `cipher_vllm_kv.py:65-106`)

`_cipher_allocate_kv_cache_tensors(self, kv_cache_config)`:
1. `_ensure_bridge_init()` — reserves the CIPHER VMM VA pool (idempotent;
   safe to call from kvdedup's wrap too).
2. For each `kv_cache_tensor` in `kv_cache_config.kv_cache_tensors`:
   - Call `cipher_kv_bridge.vmm_zeros([nbytes], 1, "int8", _TENANT, 0,
     idx & 0xFFFF, 2)` — returns a torch tensor backed by CIPHER VMM
     pages (2 MiB grants on H100).
   - Alias the same tensor across all `shared_by` layer names.
3. Return `{layer_name: tensor}` dict.

### E.2 — Kvdedup wrap (Shape ii, explicit-flush)

Kvdedup does **NOT** call `cipher_rt_kv_dedup_put` inside the wrap (that
would fire dedup at allocation time over zero-pages — degenerate). Instead:

```python
def _make_kvdedup_wrapper(cp51_wrapped):
    def kvdedup_wrapped(self, kv_cache_config):
        tensors = cp51_wrapped(self, kv_cache_config)
        # Track the tensors → CIPHER VMM page mapping so dedup_now() can
        # walk them. Store on the runner instance (per-EngineCore-process).
        if not hasattr(self, "_cipher_kvdedup_pages"):
            self._cipher_kvdedup_pages = []
        for layer_name, t in tensors.items():
            base_ptr = t.data_ptr()
            n_pages = (t.numel() * t.element_size() + 2*1024*1024 - 1) // (2*1024*1024)
            for i in range(n_pages):
                page_ptr = base_ptr + i * 2 * 1024 * 1024
                self._cipher_kvdedup_pages.append({
                    "layer": layer_name, "page_idx": i, "devptr": page_ptr,
                    "deduped": False,
                })
        return tensors
    return kvdedup_wrapped
```

**`dedup_now()`** (called by test harness post-prefill) iterates
`self._cipher_kvdedup_pages`, copies each 2 MiB devptr to host via
`cudaMemcpyDeviceToHost`, calls `cipher_rt_kv_dedup_put(host_buf,
&new_devptr_discarded)`, and **discards the returned `new_devptr`** —
because per the substrate-API finding in Part A.5, that pointer is in
cipher_rt_kv's own pool, NOT vLLM's pool. The returned pointer doesn't
need to be tracked for cleanup either; cipher_rt_kv's pool manages its
own lifetime via `cipher_rt_kv_dedup_free()` (which Step 2 can wire to
plugin shutdown, or simply rely on process exit).

What `dedup_now()` MEASURES (and what the W5 test gates check):
- pre-flush vs post-flush STATS delta: `(post.puts - pre.puts)` should
  match the number of pages flushed; `(post.hits - pre.hits)` is the
  dedup HIT count on this flush.

What `dedup_now()` does NOT do (substrate live-rebind missing):
- Rebind vLLM's tensor data_ptr onto the deduped physical. vLLM's
  KV continues to live in CP 5.1's vmm_zeros pool; no HBM savings
  in W5. (See A.5 substrate-API finding.)

### E.3 — Tenant_id sourcing

CP 5.1 uses `_TENANT = int(os.environ.get("CIPHER_TENANT_NUM", "0"))` —
an env var stamped per-process. Kvdedup inherits the same convention:
multi-tenant test starts N vLLM processes, each with a distinct
`CIPHER_TENANT_NUM`. Kvdedup's tenant context (set up by
`cipher_rt_kv_dedup_init`) is per-process per design memo §C.

### E.4 — DtoH copy cost (the ~400 µs/hit memo number)

Per design memo §Call model: ~400 µs/hit (cuMemImportFromShareableHandle +
1 DtoH + memcmp). For Week 5's test:
- 4K-token shared prompt × 4 KiB/tok = 16 MiB shared KV per layer per
  tenant.
- 32 layers × 16 MiB = 512 MiB total KV per tenant; ≈256 × 2 MiB pages.
- N=4 tenants × 256 pages × 400 µs = 0.4 seconds total flush cost
  **per tenant**; if all 4 EngineCore processes call `dedup_now()`
  concurrently → ~1.6s wall-clock (DtoH copies serialize on the PCIe
  link; lock contention on the cipher_rt_kv mutex; ~6.4 page-seconds
  of cumulative GPU-host bus time).
- Acceptable for explicit-flush trigger; absolutely unacceptable for
  per-decode-step trigger (which is why we picked explicit-flush).
- **Pinned host buffer ownership note:** T4.6.4 §A.1 mentions a pinned
  host buffer "reserved at `cipher_rt_kv_dedup_init`." Per the
  implementation read of `cipher_rt_kv_alloc.c:522` (Part A.5), the
  function uses `cuMemcpyHtoD(va, content, page_size)` — `content` is
  the caller-provided pointer. **Step 2 entry should verify**:
  (a) whether the pinned buffer is per-process (each EngineCore has
  its own pinned region) or shared (would cause contention at N=4
  concurrent `dedup_now()` calls); (b) whether the bridge auto-allocates
  the pinned region or expects caller to pin. If shared, Step 2 may
  need to serialize `dedup_now()` calls across tenants, or have each
  tenant own its own pinned scratch.

---

## Part F — Test harness pre-spec

### F.1 — `tests/test_kvdedup_live_decode.py` shape

Greenfield file (~80-120 LOC Python). Three tests:

**Test 1 — N=2 same-prompt dedup gate (Step 2 exit gate)**

Two independent assertions, each measuring different things:
- `stats.hits >= 1` — **the load-bearing dedup-mechanism assertion.** Proves the
  pipeline (DtoH → xxh64 → ioctl PUT → kmod HIT detection → memcmp-verify →
  ioctl CONFIRM) end-to-end fires correctly on real cross-tenant content.
- `out1.token_ids == out2.token_ids` — **the corruption regression guard.**
  Vanilla greedy decode with identical prompts produces identical token_ids
  trivially. This assertion only catches the case where dedup wiring
  accidentally corrupts vLLM's KV mid-run. (Per A.5 substrate-API
  finding: vLLM's KV continues to live in CP 5.1's pool unmodified, so
  corruption is structurally unlikely; this assertion is belt-and-braces.)

```python
def test_kvdedup_n2_same_prompt_hits_at_least_one():
    p1 = launch_vllm_process(tenant=1, env={"CIPHER_TENANT_NUM": "1",
                                            "CIPHER_KV_ALLOC": "1",
                                            "CIPHER_KVDEDUP_LIVE": "1"})
    p2 = launch_vllm_process(tenant=2, env={...same with TENANT=2...})
    # Both prefill identical 4K-token system prompt
    p1.prefill(SHARED_PROMPT)
    p2.prefill(SHARED_PROMPT)
    # Explicit flush from each
    p1.dedup_now()
    p2.dedup_now()
    stats = read_dedup_stats()
    # PRIMARY: dedup mechanism fired on real content
    assert stats.hits >= 1, "dedup mechanism didn't fire — pipeline broken"
    # SECONDARY: vLLM correctness not corrupted by dedup wiring
    out1 = p1.decode(32); out2 = p2.decode(32)
    assert out1.token_ids == out2.token_ids, \
        "bit-identity regression — dedup wiring corrupted vLLM KV"
```

**Test 2 — N=4 same-prompt + KL gate (Step 3 exit gate)**
- Spin up 4 vLLM instances; identical prompts; decode 128 tokens
- Compute KL across all 6 pairs of (per-token logits) ≤ 5.5e-5
- Dedup hit rate ≥ 60%: `stats.hits / (stats.hits + stats.misses)` over
  the shared-prompt portion (number of pages corresponding to first
  4 K tokens × 32 layers / total pages tested)

**Test 3 — N=2 different-prompt non-dedup (false-positive guard)**
- Different system prompts; expect dedup hit rate ≈ 0% on
  *pages-with-nonzero-content*.
- Allowance: pages that are mostly zero-padded (e.g., partial-fill of
  the last 2 MiB page of a layer) WILL legitimately dedup-hit because
  zero pages hash identically across tenants — these aren't
  false-positives, they're correctly-detected zero-page sharing.
  Step 3 separates STATS into "hits-on-content pages" vs
  "hits-on-zero-pages" via a sentinel write before measurement, OR
  fills test prompts to a full 2 MiB-page boundary to eliminate the
  zero-padding window.
- Pinned bound: **`stats.hits` on content pages ≈ 0** (within
  measurement noise of ≤ 1 hit per 100 pages); confirms xxh64 + memcmp
  doesn't false-positive collide on different real content.

### F.2 — Dependencies

- pytest (already installed in vllm_env? — verify in Step 2 entry)
- vLLM 0.21.0 (in vllm_env, version-pinned)
- Mistral-7B in HF cache (verified Part 3 of scope-lock)
- 4 × Mistral-7B = ~28 GiB GPU memory. H100 80 GiB suffices.

---

## Part G — Interaction matrix lock (R-W5.3)

| concern | resolution |
|---|---|
| **CP 5.1 + kvdedup wrap order** | Kvdedup entry point appears **after** `cipher_vllm_kv` in setup.py's `entry_points["vllm.general_plugins"]` list → `load_general_plugins()` runs CP 5.1's `register()` first, kvdedup's second. Kvdedup wraps the already-CP-5.1-wrapped `_allocate_kv_cache_tensors` → sees CIPHER-VMM tensors (correct). Both wrappers are idempotent (check `_cipher_hooked` / `_cipher_kvdedup_hooked` flags). |
| **CP 5.2 + kvdedup temporal interaction** | CP 5.2 fires on preempt (snapshot KV → pinned host DRAM). Kvdedup CONFIRM fires at `dedup_now()` flush (test-harness-explicit, post-prefill pre-decode). Temporally disjoint — preempt happens later, during decode. No interaction. |
| **Refcount semantics** | Kvdedup's refcount is on the *physical* page (kmod-owned). CP 5.2's snapshot is a *byte copy* into pinned host DRAM (separate memory; doesn't hold a kvdedup reference). If a deduped page gets preempted-and-snapshotted: kmod refcount unchanged, CP 5.2 copies the bytes out. Restore-on-resume rewrites the (still-deduped) physical page. No conflict. |
| **Per-tenant cap (8192 pages)** | Far above realistic Week-5 test scale (4 tenants × ~256 pages each = ~1024 pages total per-tenant, well under 8192). Production scale (W13-14 CP 5.5) may approach; flagged as Week 6+ concern. |
| **`-ENOSPC` fallback** | Per T4.6.4 §F-3: on cap exhaustion, plugin falls back to a non-deduped local page (the page lives in CIPHER VMM without a kvdedup entry; future PUT attempts no-op). Step 2 implements catch + log + continue. |

---

## Part H — Step 2/3 revised work-item breakdown (after Part B/C simplification)

### Step 2 — `cipher_vllm_kvdedup.py` + setup.py + N=2 gate

| component | LOC |
|---|---|
| ctypes binding to 4 cipher_rt_kv_dedup_* C functions in cipher_kv_bridge.so (per Part B.1) | ~40 |
| `_make_kvdedup_wrapper` + per-page tracking (per Part E.2) | ~40 |
| `dedup_now()` flush iterator + DtoH copy via `cudaMemcpyDeviceToHost` ctypes + discard returned new_devptr (no remap needed per A.5 — substrate live-rebind missing) | ~40 |
| `register()` + env gate + idempotency + entry-point convention (per Part D.4) | ~30 |
| Plugin module-level boilerplate (imports, `_log`, `_enabled`) | ~20 |
| **setup.py edit** — add second entry point + py_modules update | ~5 |
| **Subtotal Step 2** | **~175 LOC Python** (5 less than the pre-rewrite estimate; the avoided cuMem-remap surgery saves ~10 LOC) |
| **Advisor budget caveat:** if Step 2 entry surfaces a pinned-host-buffer-ownership issue (Part E.4 verification — whether bridge auto-allocates pinned or caller pins) requiring 30-50 LOC of pinned-region setup OR per-tenant DtoH serialization, Step 2 may come in at ~210-225 LOC. Not blocking; just flagging. |

**Step 2 entry STOP-gate (per Q2 user adjudication):** Step 2 entry MUST
verify pinned-host-buffer ownership BEFORE writing any plugin code. If
verification surfaces caller-must-pin: STOP and surface; user picks
between (a) expand Step 2 budget to ~225 LOC for pinned-region setup,
or (b) use unpinned `cudaMemcpy` and accept slower DtoH. Pre-verification
gate: a small 5-line ctypes probe calling `cipher_rt_kv_dedup_init()` then
`cipher_rt_kv_dedup_get_stats()` and inspecting whether `pages_resident` /
`bytes_va_reserved` indicate the bridge allocated host scratch.

**Step 2 — per Q1 user adjudication — now depends on Part I (substrate)
landing first.** Part H Step 2 plugin work begins ONLY after Part I
substrate primitive is committed + cipher_kv_bridge.so anchor rotated +
Track 2 SC3 regression PASS. Sequenced as: Part I (substrate) → revised
Step 2 (plugin uses new primitive) → revised Step 3 (test harness
measures actual HBM savings, not just STATS counters).

Step 2 exit gate stays as scope-lock specified: **N=2 same-prompt with
`hits ≥ 1` AND bit-identical output** + Track 2 SC6 + CP 5.4 isolation
byte-identical.

### Step 3 — `tests/test_kvdedup_live_decode.py` + N=4 + KL + hit-rate

| component | LOC |
|---|---|
| Test 1 (N=2 same-prompt; harness for Step 2 gate too) | ~30 |
| Test 2 (N=4 same-prompt + KL ≤ 5.5e-5 + hit rate ≥ 60%) | ~50 |
| Test 3 (N=2 different-prompt non-dedup guard) | ~20 |
| Process-launching boilerplate (multiprocessing or subprocess; vLLM launch helper) | ~30 |
| **Subtotal Step 3** | **~130 LOC Python** (in scope-lock band 80-120; came in slightly higher due to Test 3) |

### Step 4 — Closeout

Per scope-lock Part 8.4: tail commit absorbs all 4 docs (this design
memo + 2 result docs + closeout) + plugin source snapshot + test harness
+ phase_c churn. ~1-2 h paperwork.

### Total revised Week-5 LOC

| step | original scope-lock estimate | revised after Step-1 discoveries |
|---|---|---|
| Step 1 | 0 source + optional 30 LOC spike | 0 source (spike not needed) |
| Step 2 | 150-200 Python + 10 setup.py | **~185 Python + 5 setup.py = ~190** |
| Step 3 | 80-120 tests | **~130 tests** |
| Step 4 | docs only | docs only |
| **total** | **~250-330 LOC** | **~320 LOC** |

Effort estimate stays at **9-15h** (scope-lock band); revised LOC is at
the high end of the original range but breakdown is tighter (xxhash
binding and ioctl ABI work eliminated; replaced with slightly larger
test harness and `dedup_now` flush iterator).

---

## Part I — Substrate work plan (per Q1 expansion: `cipher_rt_kv_dedup_alias`)

Per Q1 adjudication, Week 5 expands to include a new substrate primitive
that supports post-allocation in-place rebind of an existing devptr onto
a deduped physical. Adding this is C substrate work in `cipher_rt_kv_alloc.c`
and rebuilds `cipher_kv_bridge.so`.

### I.1 — Substrate API design: `cipher_rt_kv_dedup_alias`

Proposed signature (in `cipher_rt_kv_alloc.h` Phase 4.6.3+ section):

```c
/* Alias an existing CIPHER-VMM devptr onto a deduped physical page.
 *
 * Caller already owns a CIPHER-VMM page at *existing_devptr* with content
 * written. This function:
 *   1. Computes xxh64(content of existing_devptr) — DtoH copy through the
 *      pinned scratch.
 *   2. ioctl PUT to /dev/cipher_kvdedup with that hash + a fresh
 *      POSIX-fd export of existing_devptr's physical.
 *   3. On MISS: kmod registers existing_devptr's physical as a new dedup
 *      entry; this function returns 0; *was_deduped = 0. existing_devptr
 *      remains bound to its original physical (now refcount=1 in
 *      kmod's table).
 *   4. On HIT: imports the candidate handle, memcmp-verifies. On match:
 *      cuMemUnmap(existing_devptr) + cuMemRelease(original_handle) +
 *      cuMemMap(existing_devptr, shared_handle) — REBIND IN PLACE on
 *      the caller's VA. Returns 0; *was_deduped = 1. existing_devptr now
 *      points at the deduped physical.
 *   5. On hash collision (HIT-then-memcmp-fail): FORCE_NEW path, treat
 *      as MISS.
 *
 * Returns 0 on success, negative errno on failure. */
int cipher_rt_kv_dedup_alias(unsigned long long existing_devptr,
                             int *was_deduped);
```

Key difference vs `cipher_rt_kv_dedup_put`: the new function operates on
the caller's existing VA, not on a fresh VA in cipher_rt_kv's pool. It
discovers the existing VA's underlying CUmem handle (via the bridge's
internal page table), exports a fresh POSIX-fd, runs the PUT/CONFIRM
handshake, and on HIT does cuMemUnmap+cuMemMap on the caller's VA.

**Note:** the function takes only `existing_devptr` — content is read by
the function itself via DtoH copy through the pinned scratch (no
caller-provided `content` parameter needed; eliminates one source of
correctness risk).

### I.2 — Implementation surface

`cipher_rt_kv_alloc.c` additions (~120-180 LOC C):
- New function `cipher_rt_kv_dedup_alias` (~80-120 LOC) — mirrors
  `cipher_rt_kv_dedup_put` but with rebind on caller's VA instead of
  pool VA.
- Helper: lookup existing devptr → CUmem handle via the bridge's internal
  page table (~10-20 LOC; bridge already maintains this mapping per
  `cipher_rt_kv_slab` accounting).
- Helper: pinned scratch buffer for DtoH copy (reuse the one allocated
  at `cipher_rt_kv_dedup_init` if per-process; otherwise allocate per-call
  — Step 2 entry verification answers this).

`cipher_rt_kv_alloc.h` additions (~10 LOC):
- Function declaration with doc comment matching I.1.

`cipher_kv_bridge.cpp` pybind module additions (~5 LOC):
- Optional: `m.def("kv_dedup_alias", &cipher_rt_kv_dedup_alias_pybind)`
  for direct Python access (alternative: ctypes against the C symbol).

**No kmod changes.** The kmod-side ABI (`/dev/cipher_kvdedup` 5 ioctls) is
unchanged — the new primitive uses the existing PUT/CONFIRM handshake;
the rebind happens entirely in userspace via cuMem APIs.

### I.3 — Build + anchor rotation

- `cipher_kv_bridge.so` anchor rotates: **c04b0c39 → new value** (Track 2 SC3 anchor)
- No kmod rotation (5th kmod rotation in 24h avoided)
- Preserve `cipher_kv_bridge.so.pre_w5_step1b` fallback

### I.4 — Regression gate (Track 2 SC3)

`cipher_kv_bridge.so` is the Track 2 SC3 anchor. After rebuild, run:
- Track 2 SC3 weight-arena tests (per `phase_c/TRACK_2_SC3_*` test artifacts)
- Track 2 SC6 PASS unchanged
- Existing `cipher_rt_kv_dedup_put` semantics preserved (the original
  function untouched; only `_alias` added)
- libcipher_rt.so md5 unchanged (cipher_kv_bridge.so is a separate
  module; libcipher_rt doesn't link it directly)

### I.5 — Effort estimate

| component | LOC | time |
|---|---|---|
| `cipher_rt_kv_dedup_alias` C impl | ~120 | ~3-4h (substrate write + careful cuMemUnmap/cuMemMap ordering) |
| Header decl + pybind binding | ~15 | ~0.5h |
| Build + cipher_kv_bridge.so rebuild | — | ~0.5h |
| Track 2 SC3 + SC6 regression run | — | ~1h |
| **Subtotal Part I (substrate)** | **~135 LOC C** | **~5-6h** |

**Revised Week-5 total:** original 9-15h + 5-6h substrate = **~14-21h**.
Top of recommended 15-20h band; may push slightly over if pinned-region
ownership requires per-tenant setup (Q2 caveat). Acceptable per user's
explicit Q1 choice.

### I.6 — Revised step sequence

| step | scope | est. |
|---|---|---|
| **1 (this memo)** | design + Q1/Q2 adjudication | done (~2.5h) |
| **1b** *(new)* | substrate primitive: `cipher_rt_kv_dedup_alias` in cipher_rt_kv_alloc.c; rebuild cipher_kv_bridge.so; Track 2 SC3/SC6 regression PASS; anchor rotation c04b0c39 → new | ~5-6h |
| **2 (revised)** | plugin module uses `cipher_rt_kv_dedup_alias` instead of `_put`; `dedup_now()` flush iterates pages and rebinds in place; pinned-region pre-verification at entry | ~4-5h (slightly larger than original ~190 LOC because alias call is per-page and rebind is on caller's VA) |
| **3 (revised)** | test harness validates **actual HBM savings** via `nvidia-smi --query-gpu=memory.used,memory.free` deltas across pre-flush vs post-flush, PLUS the original STATS-based hit rate ≥ 60% gate | ~3-4h |
| **4** | closeout (unchanged) | ~1-2h |
| **total** | | **~15-22h** |

### I.7 — STOP triggers (substrate work specific)

- Track 2 SC3 regression FAILS post-rebuild → STOP, revert cipher_kv_bridge.so to c04b0c39 fallback.
- cuMemUnmap/cuMemMap ordering surfaces a CUDA error not handled in the
  prototype → STOP, surface for design revision (may need additional
  CUDA API to query/preserve allocation properties across rebind).
- Performance under concurrent N=4 `dedup_now()` calls regresses Track 2
  SC6 PASS time by > 10% → STOP, characterize, surface.

---

## Verdict

**LOCKED-SHAPE-II-VALIDATE + EXPLICIT-FLUSH TRIGGER (substrate live-rebind deferred to Week 6+).**
Step 2 prompt may be drafted against Part H Step 2 breakdown — **with the
explicit understanding** that Week 5 ships *mechanism validation against
live content*, not HBM memory savings.

Five Step-1 discoveries:
1. **C wrappers `cipher_rt_kv_dedup_*` already exist** in `cipher_kv_bridge.so`
   — eliminates the need to ctypes-bind kmod ioctls directly.
2. **xxhash + memcmp + ioctl handshake all encapsulated** in the C wrappers
   — eliminates the xxhash-via-libxxhash ctypes work.
3. **CP 5.1's existing 2 MiB VMM allocation seam** is exactly the kvdedup
   page granularity — Shape (ii) is mechanically clean (no aggregation).
4. **vLLM plugin load order = setup.py entry_points list order** — kvdedup
   placed after CP 5.1 ensures the right wrap order (R-W5.3 closed).
5. **`cipher_rt_kv_dedup_put` is allocation-time, not post-write-rebind**
   (advisor concern #1, confirmed by reading `cipher_rt_kv_alloc.c:522`).
   This means Week 5 cannot achieve actual HBM savings without a new
   substrate primitive. Reframed Week 5 deliverable as VALIDATE-only:
   prove the mechanism fires on live content; **save no memory in W5;
   queue the live-rebind primitive for Week 6+ substrate work.**

Step 1 effort: ~2.5h (within scope-lock 2-3h band; +0.5h for the
substrate-API read that the advisor flagged as load-bearing).

**User adjudication 2026-05-21 (post-design-memo):**

**Q1 RESOLVED → "No — expand W5 to include substrate live-rebind work."**
User wants Week 5 to ship real HBM savings, not VALIDATE-only. **Scope
expands to include the substrate primitive `cipher_rt_kv_dedup_alias`
(or equivalent) in cipher_rt_kv_alloc.c** (C substrate work, rebuilds
cipher_kv_bridge.so, rotates anchor from `c04b0c39` to a new value,
adds Track 2 SC3 regression gate). See Part I below for the expanded
work plan.

**Q2 RESOLVED → "Tighten budget — require pinned-region pre-verification
at Step 2 entry."** Step 2 entry adds a STOP-gate: verify pinned-host-buffer
ownership BEFORE writing plugin code. If verification surfaces
caller-must-pin, Step 2 STOPs and surfaces; user picks whether to expand
budget or use unpinned cudaMemcpy (slower DtoH). Recorded in Part H Step 2
revised exit conditions below.

**Step 2 implementation prompt may be drafted against the expanded
Part I + revised Part H below.**

---

**Evidence:**
- `/home/ubuntu/cipher-fusion-evidence/t4_6_4_design.md` (substrate memo C, 235 LOC)
- `/home/ubuntu/cipher_kmod/cipher_kvdedup.h` (5 ioctls, 2 MiB ABI)
- `/home/ubuntu/cipher_rt_phase4/cipher_rt_kv_alloc.h:117-141` (cipher_rt_kv_dedup_* C API)
- `nm -D /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so | grep dedup` (4 T symbols exported)
- `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py:65-141` (CP 5.1 monkey-patch precedent + register pattern)
- `/home/ubuntu/cipher_vllm_plugin/setup.py` (entry_points list order)
- `/home/ubuntu/vllm_env/lib/python3.10/site-packages/vllm/plugins/__init__.py:69-83` (`load_general_plugins` dict-values iteration order)
- `/home/ubuntu/cipher-fusion-evidence/cp_4_6_5_6/CP_4_6_5_6_DESIGN_MEMO.md` + `CP_4_6_5_6_REPORT.md` (cold-path vs live-path model decision)
- `/home/ubuntu/cipher-fusion-evidence/WEEK_5_SCOPE_LOCK.md` (the scope this memo executes)
