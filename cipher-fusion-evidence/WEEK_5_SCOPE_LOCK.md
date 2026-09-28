# Week 5 Scope-Lock — KV-Dedup Live Wire

**HEADLINE: SCOPE-LOCKED. Step 1 design-memo prompt may be drafted.**

Substrate fully ready (5 ioctls verified live at `/dev/cipher_kvdedup` mode
666; libxxhash.so.0 present; vLLM 0.21.0 v1 KV offload backend interface
identified at `vllm/v1/kv_offload/base.py`). cipher_vllm_plugin/ has CP 5.1
+ CP 5.2 work but **zero kvdedup integration** — Week 5 is greenfield
plugin module + entry point + test harness. Substrate gap is structural
but bounded; sized at ~300-450 LOC Python total + ~9-15h effort across 4
sub-steps within 15-20h Week-5 budget.

**Date:** 2026-05-21. **Read-only scope-lock memo; no source modified.**

---

## Part 1 — kmod kvdedup substrate inventory

### 1.1 — Files

| file | LOC | role |
|---|---|---|
| `cipher_kmod/cipher_kvdedup.h` | 86 | ABI: structs + 5 ioctl macros + magic `'K'` + flags + result codes |
| `cipher_kmod/cipher_kvdedup.c` | 477 | impl: refcount table, xa_alloc, kvd_ioctl_* handlers, /dev/cipher_kvdedup miscdevice |

### 1.2 — 5 ioctls (verified at `/dev/cipher_kvdedup`, mode 666, major 510)

| nr | name | direction | struct | semantics |
|---|---|---|---|---|
| 1 | `CIPHER_KVDEDUP_INIT` | `_IOR` | `struct cipher_kvdedup_init { __u32 tenant_id; }` | per-tenant init; returns assigned tenant_id |
| 2 | `CIPHER_KVDEDUP_PUT` | `_IOWR` | `struct cipher_kvdedup_put { __u64 content_hash; __s32 export_fd; __u32 flags; __u32 result; __u64 pool_offset; __s32 candidate_fd; ... }` | two-phase: userspace passes xxhash64 + cuMem POSIX fd of fresh 2MiB page; kmod returns MISS (registered new) OR HIT (here's candidate_fd; userspace memcmp-verifies, then CONFIRMs). `CIPHER_KVDEDUP_FLAG_FORCE_NEW` skips bucket walk |
| 3 | `CIPHER_KVDEDUP_CONFIRM` | `_IOW` | `struct cipher_kvdedup_confirm { __u64 pool_offset; }` | userspace confirms memcmp matched after HIT_CANDIDATE; kmod bumps refcount on the existing entry |
| 4 | `CIPHER_KVDEDUP_FREE` | `_IOW` | `struct cipher_kvdedup_free { __u64 pool_offset; }` | userspace releases its reference; kmod decrements; physical page freed at refcount 0 |
| 5 | `CIPHER_KVDEDUP_STATS` | `_IOR` | `struct cipher_kvdedup_stats { __u64 entries, virtual_refs, puts, hits, misses, collisions_forced, refcount_releases; __u32 tenants_open; }` | aggregate diagnostics — entries/refs/hits/misses |

### 1.3 — Limits

- Per-tenant tracked-page cap: **8192** pages
- Global registered-entry cap: **1 << 20** (1 M)
- Page granularity: **2 MiB** (xxhash64 + memcmp at this granularity)

### 1.4 — Verification status (per plan §1.3 + R-A3)

Per `CIPHER_REENGINEERING_PLAN.md` §1.3 + R-A3 risk register:
*"the kmod substrate is COMPLETE (5 ioctls INIT/PUT/CONFIRM/FREE/STATS
wired through `cipher_rt_kv_alloc.c:490,563,602,650,669`; zero TODOs;
PUT/CONFIRM two-phase handshake is correctness-gated)."* No
`EXPORT_SYMBOL_GPL` for kvdedup (it's an ioctl interface, not an
internal kmod API). Substrate ready for userspace bridge.

---

## Part 2 — cipher_vllm_plugin current state inventory

### 2.1 — Plugin files (not a git repo; on-disk only)

| file | LOC | scope |
|---|---|---|
| `cipher_vllm_plugin/cipher_vllm_kv.py` | 141 | **CP 5.1** — monkey-patches `GPUModelRunner._allocate_kv_cache_tensors` to source from `cipher_kv_bridge.vmm_zeros` (CIPHER's CUDA-VMM allocator); CIPHER owns the KV buffer end-to-end |
| `cipher_vllm_plugin/cipher_kv_offload.py` | 284 | **CP 5.2** — snapshot-on-preempt / restore-on-resume KV via `torch.index_select`/`index_copy_` against the CIPHER VMM slabs; pinned host-DRAM staging; preserves vLLM's `num_computed_tokens` so resume rides the connector path |
| `cipher_vllm_plugin/setup.py` | 26 | packages both modules; registers **one** `vllm.general_plugins` entry point → `cipher_vllm_kv:register` (which internally also calls `_register_offload()` for CP 5.2) |

### 2.2 — Critical gap: zero kvdedup integration today

`grep -nE 'CIPHER_KVDEDUP|kvdedup|/dev/cipher_kvdedup' cipher_vllm_plugin/*.py` →
**no matches in any file.** CP 5.1 + CP 5.2 reference `cipher_kv_bridge`
(Track 2 SC3 anchor `c04b0c39`) — that's the **weight-arena + per-tenant
KV allocation** library, **not** the cross-tenant `/dev/cipher_kvdedup`
deduplication substrate.

**Week 5 is greenfield for the kvdedup bridge**, not "extend
`cipher_vllm_kv.py`." The path is:

- **NEW** `cipher_vllm_plugin/cipher_vllm_kvdedup.py` (~200-300 LOC)
- **NEW** ctypes ioctl wrappers for the 5 kvdedup ioctls (could be in same module)
- **NEW** xxhash64 helper via ctypes against `/usr/lib/x86_64-linux-gnu/libxxhash.so.0`
- **EXTEND** `setup.py` to add a second entry point: `cipher_vllm_kvdedup = cipher_vllm_kvdedup:register`

### 2.3 — Auto-memory cross-reference

Auto-memory entries [[cipher-track2-weight-sharing]] (SC3 `c04b0c39`) +
[[cipher-cp4656-closed]] (CP 4.6.5/6 kvdedup substrate primitive closed
2026-05-16) document the substrate prior art:
- Track 2 SC3 = `cipher_kv_bridge` (weight-arena VMM allocator; what CP 5.1
  uses)
- CP 4.6 = the kmod-side `/dev/cipher_kvdedup` substrate that Week 5
  wires up; closed-with-stub bridge per the cp4656 memo
- v1.2.2 §1.3 + R-A3 say substrate complete; bridge layer is the v1 work

**These are consistent.** Week 5 closes the bridge-layer gap that was
explicitly deferred at CP 4.6.5/6 close.

---

## Part 3 — vLLM environment state

### 3.1 — Installation

```
vllm 0.21.0  /home/ubuntu/vllm_env/lib/python3.10/site-packages/vllm/__init__.py
License: Apache-2.0; v1 engine (EngineCore subprocess model)
```

Same vLLM version Week 4 used (FUTURE_SCOPE/A Phase 2/3 confirmed
transparency + parity). System python's vLLM is not installed (only the
isolated venv at `/home/ubuntu/vllm_env`).

### 3.2 — KV offload backend interface — **the R-W5.2 integration surface**

Located at `vllm_env/lib/python3.10/site-packages/vllm/v1/kv_offload/`:

| file | LOC | role |
|---|---|---|
| `base.py` | 406 | **abstract `OffloadingManager`** with 11 methods: `lookup, prepare_load, touch, complete_load, prepare_store, complete_store, take_events, shutdown` etc. Plus `OffloadKey`, `LoadStoreSpec` (ABC), `BlockIDsLoadStoreSpec`, `ReqContext`, `OffloadingEvent` types |
| `factory.py` | 58 | `OffloadingSpecFactory.register_spec(name, module_path, class_name)` → loadable by name |
| `reuse_manager.py` | 125 | reference re-use logic over `OffloadingManager` |
| `simple_kv_offload/__init__.py` + `cuda_mem_ops.py` | (small) | reference impl |

Plus the existing CP 5.1 monkey-patch site: `vllm/v1/worker/gpu_model_runner.py:6637`
`_allocate_kv_cache_tensors(self, kv_cache_config)` (also called at L6871).

**Two integration patterns are open** (Step 1 picks one):

**(a) OffloadingManager ABC implementation.** Implement all 11 methods
against `/dev/cipher_kvdedup`. Most idiomatic; vLLM-blessed. Heavy:
~500-800 LOC for full ABC compliance. Registers via
`OffloadingSpecFactory.register_spec(...)`.

**(b) Monkey-patch the block-allocation seam.** Mirror the CP 5.1/5.2
pattern: hook the kv_cache_manager (or
GPUModelRunner._allocate_kv_cache_tensors) and inject a deduplication
PUT/CONFIRM cycle when a freshly-allocated block's content hashes to an
existing entry. Lighter: ~200-300 LOC. Less idiomatic but matches the
established CP 5.1 + CP 5.2 plugin shape.

### 3.3 — Test workload availability

| model | local cache | use |
|---|---|---|
| `mistralai/Mistral-7B-v0.1` | yes | primary same-prompt N=4 test (Week 4 SC6 confirmed bit-identical decode) |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | yes | smaller N=4 stress; faster iteration during Step 3 harness work |
| `meta-llama/Llama-3.1-8B`, `Llama-3.2-1B-Instruct`, `llava`, etc. | yes | available if needed |

No new model acquisition required. KV-dedup is single-model
cross-tenant; existing models suffice.

### 3.4 — xxhash availability

| source | available? |
|---|---|
| `pip install xxhash` in vllm_env | **NO** (not listed in `pip list`) |
| system Python `xxhash` | **NO** |
| `/usr/lib/x86_64-linux-gnu/libxxhash.so.0` (0.8.1) | **YES** |

→ ctypes binding against system libxxhash is the **only path** (pip
install not used since the user's pattern is no-network for
runtime deps).

---

## Part 4 — bit-identical correctness gate definition

### 4.1 — Primary gate

Per reconciled `CIPHER_REENGINEERING_PLAN.md` §7 Week 5 + closeout §6
rewrite:

- **Bit-identical KV across N=4 same-prompt tenants.** Specifically:
  for the shared system-prompt portion of the prompt, all 4 tenants'
  K and V tensors at the matching block IDs must be byte-equal.
  Mechanism: after dedup CONFIRM, all 4 tenants' KV pages at the
  shared-prompt offsets resolve to the same physical pool_offset.
- **KL ≤ 5.5e-5** per-token logit divergence vs vanilla (non-dedup)
  baseline, integrated over the decoded sequence per tenant. This is
  the looser end-to-end gate; bit-identity at the shared-prefix is
  the strict slice.

### 4.2 — Test harness — to be built

`tests/test_kvdedup_live_decode.py` — **does NOT exist on disk today**
(`find /home/ubuntu -name 'test_kvdedup*'` empty). Step 3 creates it.

Expected shape (~100-150 LOC Python):
- launch 4 vLLM instances same model + same prompt; greedy decode 128 tokens
- enable `CIPHER_KVDEDUP_LIVE=1` and the new `cipher_vllm_kvdedup`
  plugin
- after generation: query `/dev/cipher_kvdedup` STATS ioctl → assert
  hits/misses ratio
- diff token_ids across all 4 tenants → must be byte-identical for the
  shared-prefix portion (bit-identity slice)
- compute KL between dedup'd-run and vanilla-run logits → must be ≤ 5.5e-5

### 4.3 — Dedup hit rate target

**≥ 60%** for the shared system-prompt portion. Measurement source:

- `CIPHER_KVDEDUP_STATS` ioctl returns `hits` + `misses` counters
- Compute `hit_rate = hits / (hits + misses)`
- Measure ONLY over the shared-prompt portion (not the diverging decode
  trajectories)
- The 60% target assumes a ~4K-token shared system prompt at 2 MiB/page
  granularity → roughly 8-16 pages shared per tenant pair × N=4 →
  reasonable to hit 60% if dedup wiring is correct

---

## Part 5 — Step breakdown

### Step 1 — Design memo + bridge architecture decision (paperwork)

| | |
|---|---|
| scope | **Resolve the page-granularity question first** (R-W5.4 below — *blocking, not deferrable*): is the 2 MiB kmod page contract a fixed substrate ABI, and how does vLLM v1's much-smaller block size aggregate into 2 MiB pages? Then **execute the (b) monkey-patch baseline** (per CP 5.1/5.2 precedent) and explicitly consider (a) ABC implementation only as a forward-compat alternative with its tradeoff. Specify: ctypes ABI layer for the 5 ioctls; xxhash binding plan via libxxhash.so.0; entry-point design + setup.py extension; **plugin-registration ordering discipline** (kvdedup register before-vs-after cipher_vllm_kv — see R-W5.3); test harness shape pre-spec. |
| LOC | 0 source; design memo ≈ 250-500 lines docs |
| time | ~2-3h paperwork |
| entry conditions | this scope-lock approved; user has answered the pre-Step-1 question (b-baseline OK, or want (a)?) |
| exit gate | design memo adjudicated; integration pattern locked; page-granularity contract verified against substrate memo C; ordering discipline specified |
| STOP triggers | page-granularity can't be resolved on paper (needs GPU spike — Step 1 spikes a small Python script calling KVDEDUP_PUT with a non-2MiB allocation to confirm the failure mode); vLLM ABC compliance gap discovered that forces (a); kmod substrate's 2 MiB is a hard ABI but vLLM blocks can't aggregate (would force substrate design change → out of Week 5 scope) |

### Step 2 — `cipher_vllm_kvdedup.py` implementation + single-tenant parity

| | |
|---|---|
| scope | new Python module: ctypes ioctl wrappers (5 functions) + xxhash64 binding (libxxhash) + vLLM integration (per Step 1 pick) + register() entry-point + setup.py extension to add second entry point |
| LOC | ~200-300 Python |
| time | ~3-5h |
| entry conditions | Step 1 design memo adjudicated; integration pattern picked |
| exit gate | **N=2 same-prompt with `hits ≥ 1` AND bit-identical output** — this proves the full round-trip (first PUT registers, second PUT returns HIT, CONFIRM bumps refcount, deduped page reuse doesn't corrupt). N=1 is insufficient (no second tenant to hit against; STATS would show `puts > 0 hits = 0` even if PUT silently misregisters). Plus: regression smoke = Track 2 SC6 + CP 5.4 isolation byte-identical. |
| STOP triggers | integration pattern doesn't compose with vLLM 0.21.0's actual hot path; xxhash binding fails; ctypes ABI mismatch with kmod structs; N=2 HIT count stays 0 (dedup isn't firing); N=2 produces non-identical output (correctness regression) |

### Step 3 — Multi-tenant N=4 same-prompt + bit-identical gate + dedup hit rate

| | |
|---|---|
| scope | new test harness `tests/test_kvdedup_live_decode.py`; orchestrate 4 vLLM instances; KL measurement vs vanilla; STATS-based hit-rate measurement; bit-identity slice over shared-prompt KV pages |
| LOC | ~100-150 Python harness |
| time | ~3-5h (includes debug iterations on dedup-page-boundary corner cases) |
| entry conditions | Step 2 PASS (single-tenant parity) |
| exit gate | KL ≤ 5.5e-5 vs vanilla in all 4 tenants; hit rate ≥ 60% over shared system-prompt portion; W1 regression PASS; Track 2 SC6 PASS unchanged; CP 5.4 isolation 15/15 byte-identical |
| STOP triggers | KL exceeds threshold; hit rate < 60% (could be plumbing bug OR page-size mismatch); SC6 regresses (CIPHER_KVDEDUP_LIVE side effect on other paths); CP 5.4 regresses |

### Step 4 — Closeout

| | |
|---|---|
| scope | `WEEK_5_CLOSEOUT.md`; `week-5-complete` tags on rt_phase4 + may13 (not on kmod since Week 5 doesn't touch kmod); Week 6 placeholder adjudication (absorb vs fold-forward per §7 Week 6 reserved slot); memory updates; cipher-fusion-evidence tail commit absorbing all Step result docs + plugin source snapshot + phase_c SC6 churn from Step 0 preflight Gate 8 |
| LOC | docs only |
| time | ~1-2h |
| entry conditions | Step 3 PASS |
| exit gate | tags placed; closeout doc adjudicated; Week 6 decision recorded; memory live pointer flipped |
| STOP triggers | (none — this is paperwork) |

### Total

**~9-15h across 4 sub-steps.** Comfortable in the 15-20h Week-5 budget
recommendation. Underruns the Week 4 actual of 22.5h since Week 5 is
narrower in scope (single-feature integration vs Week 4's 13 ports +
LP-8 retire + exporter + Sub-4 measurement).

---

## Part 6 — Risk register

### R-W5.1 [HIGH] — bit-identical correctness gate at live decode

**Source:** reconciled §7 Week 5; promoted from R-W6.1 pre-reconciliation.

**Surface:** the bit-identical gate at live decode is tighter than the
synthesized-page T4.6.3/T4.6.4 gates that validated the CP 4.6 substrate.
Live attention shapes may surface KV-block layout assumptions that the
synthesized harness didn't exercise (e.g., the page-boundary alignment
of the shared prompt vs vLLM's internal block-size choice).

**Mitigation strategy:**
- Step 2 single-tenant parity first (KL must be 0 for N=1; if it's not,
  the monkey-patch site is wrong).
- Step 3 N=4 only after Step 2 PASSes.
- If KL > 5.5e-5 at Step 3: bisect by page (which page diverged first?
  is it always page N? is it always the same byte offset?).

**STOP-gate:** KL > 5.5e-5 in any test arm.

### R-W5.2 [MEDIUM] — vLLM upstream API stability

**Surface:** vLLM is at 0.21.0; the `v1/kv_offload/base.py` ABC and the
`v1/worker/gpu_model_runner.py:_allocate_kv_cache_tensors` site are
both v1-engine internal APIs not covered by vLLM's public contract.
Future vLLM versions may rename/restructure these.

**Mitigation strategy:** **pin to vLLM 0.21.0** explicitly in the new
`setup.py` (add `install_requires=["vllm==0.21.0"]` or similar
constraint; or pin in the plugin's `register()` with a version-check +
warn-log). Document pin rationale in Step 1 design memo.

**STOP-gate:** vLLM 0.21.0 API change discovered mid-Step-2 that breaks
our monkey-patch site.

### R-W5.3 [MEDIUM] — paused-FUTURE_SCOPE/A integration risk

**Surface:** cipher_vllm_plugin/ was last touched 2026-05-17 during CP
5.1/5.2 work; FUTURE_SCOPE/A Phase 4 design memo (paused) referenced
extensions that never landed. Current plugin state vs the v1.2.2 Week-5
intent is the open question — partially answered by Part 2 above
(plugin has CP 5.1/5.2; kvdedup is greenfield), but interactions
between the new module and the existing CP 5.1/5.2 monkey-patches need
verification.

**Mitigation strategy:** Step 1 design memo explicitly enumerates the
interaction matrix AND **pins the plugin-registration ordering
discipline**:
- vLLM's `load_general_plugins()` runs entry points in declaration
  order. setup.py's `entry_points["vllm.general_plugins"]` list
  ordering determines wrap order. If CP 5.1's
  `_allocate_kv_cache_tensors` patch wraps the function and the new
  kvdedup module wraps it again, the wrap order matters: who's outer
  (sees others' patched output) vs inner (sees the original).
- **Step 1 must specify which order is correct.** Candidate: kvdedup
  registers AFTER `cipher_vllm_kv` (so kvdedup's wrap sees the
  CIPHER-VMM-backed buffers from CP 5.1, then dedupes against the
  underlying VMM grants per Shape (ii) above). The reverse order
  (kvdedup before CP 5.1) makes CP 5.1's monkey-patch wrap kvdedup's
  output — semantically wrong because CP 5.1 expects to be allocating,
  not receiving an already-deduped reference. One ordering is
  correct; the other corrupts.
- Does kvdedup PUT happen AFTER CP 5.1's `_allocate_kv_cache_tensors`
  monkey-patch returns the CIPHER-VMM-backed buffer? (Required for
  Shape (ii); confirmed by ordering above.)
- Does kvdedup CONFIRM happen BEFORE CP 5.2's snapshot-on-preempt
  could fire for a deduped page? (Sequenced via vLLM's preempt
  hook order; Step 1 verifies CP 5.2 fires post-COMMIT, kvdedup
  CONFIRM fires at allocation time → temporally disjoint.)
- Reference-count semantics: if CP 5.2 snapshots a page that kvdedup
  also references, who frees first? (Step 1 specifies: kvdedup
  refcount independent of CP 5.2 snapshot lifetime; CP 5.2's
  snapshot is a *copy* into pinned host DRAM, doesn't hold a kvdedup
  reference.)

**STOP-gate:** the interaction matrix surfaces a structural conflict
(e.g., CP 5.1's CIPHER-VMM buffer can't be cuMem-exported for kvdedup's
POSIX-fd handle; or wrap ordering can't satisfy both modules
simultaneously).

### R-W5.4 [HIGH — blocking, drives integration architecture] — 2 MiB page granularity vs vLLM block size

**Surface:** kvdedup operates at **fixed 2 MiB page granularity** (per the
substrate design memo C and the cipher_kvdedup.h ABI: *"userspace
computes the xxhash64 of the 2 MiB page and the memcmp-verify on a
hit"* — this is a hard kmod-side ABI contract, not a tuning knob).
vLLM v1's default block size is 16 tokens × num_heads × head_dim ×
sizeof(dtype) × 2 (K + V) — typically a **few hundred KB per block**,
3-4 orders of magnitude smaller than 2 MiB. The dedup wiring cannot
simply forward vLLM blocks 1:1 to kvdedup PUTs; it must **aggregate
multiple blocks into 2 MiB pages before hashing**, OR the bridge
operates at the underlying CUDA-VMM allocation level (which CP 5.1's
`vmm_zeros` allocates in granularity-of-VMM units, typically 2 MiB on
H100).

**This is the architecture-driving question** of Step 1, not a routine
mitigation. Two viable shapes:
- **Shape (i) — block-aggregation bridge.** Bridge accumulates vLLM
  blocks until a 2 MiB boundary, hashes the aggregated buffer, calls
  PUT. Bookkeeping per-tenant: which blocks belong to which 2 MiB
  page; how dedup HITs invalidate cached block→page mappings on
  fragmentation.
- **Shape (ii) — VMM-page-level bridge.** Bridge hooks at the
  `vmm_zeros` allocation point (where CP 5.1 already operates) and
  PUTs every CUDA-VMM grant — which IS 2 MiB by construction on H100.
  vLLM blocks live inside the granted page; dedup ops at allocation
  time, not at block-fill time.

Shape (ii) composes better with CP 5.1 (same allocation seam) but
costs precision (a whole 2 MiB VMM grant might contain only one or
two filled vLLM blocks; the rest is zero — hashes identical only
when block layout matches exactly). Shape (i) is more correct
per-block but requires non-trivial aggregation bookkeeping.

**Mitigation strategy:** Step 1 design memo answers the page-size
contract via the substrate memo C (verify 2 MiB is the immutable
ABI), then picks Shape (i) vs (ii). If unresolvable on paper, Step 1
includes a small Python spike (~30 LOC) calling KVDEDUP_PUT with a
non-2MiB allocation to observe the kmod's reject behavior — confirms
the contract empirically.

**STOP-gate:** neither shape works because vLLM's block layout
fundamentally prevents 2 MiB aggregation (very unlikely — CP 5.1
already operates at VMM granularity, so Shape (ii) is structurally
viable as a fallback). If hit: Week 5 re-scopes to substrate-level
change (out of Week-5 budget).

### R-W5.5 [LOW] — phase_c SC6 log churn carry-over

**Surface:** 21 modified phase_c/ files from WEEK_5_ENTRY_PREFLIGHT.md
Gate 8 runs. Out of Step 0 scope; flagged for Week 5 absorption.

**Mitigation strategy:** Step 4 closeout commit absorbs.

**STOP-gate:** (none — this is housekeeping.)

---

## Part 7 — Estimated total Week-5 effort

| component | estimate |
|---|---|
| Step 1 (design memo, paperwork) | 2-3h |
| Step 2 (implementation + single-tenant parity) | 3-5h |
| Step 3 (multi-tenant N=4 + bit-identical + hit-rate) | 3-5h |
| Step 4 (closeout) | 1-2h |
| **total** | **9-15h** |

Comparison:
- **Weeks 1-4 actual:** 22.5h across multiple sessions (within 17-25h
  envelope).
- **v1.2.2 Week 5 budget (reconciled):** §7 doesn't give a precise
  LOC/time band for Week 5 (the old CP-5.5 measurement section that
  was relocated to W13-14 didn't either); the closeout §6 implicitly
  budgets Week 5 in the "next-up smaller week" pattern.
- **Recommended target:** 15-20h total Week-5 effort (per spec).
  9-15h estimate has ~5-10h slack for the R-W5.1/5.2/5.3 mitigations.

---

## Part 8 — Carry-over absorption

| item from Step 0 | resolution |
|---|---|
| **8.1** phase_c SC6 log churn (21 files from preflight Gate 8) | Absorb into Step 4 closeout tail commit (matches Week 4 closeout cadence) |
| **8.2** cipher_vllm_plugin packaging audit | Folded into Step 1 design memo (Part 2 above is a partial audit; Step 1 completes it) |
| **8.3** v1.2.2 L58 "W6 = KV-dedup" framing fix | Low-priority; not scheduled in Week 5. Add to Step 4 closeout `WEEK_5_CLOSEOUT.md` §Open-items as a Week-6-or-later doc-cleanup pass. |
| **8.4** Step 4 closeout absorbs ALL Week-5 audit artifacts | Per Week-4-closeout-tail-commit pattern (commit `9230b5c` absorbed 4 result docs + exporter snapshot + measurement probe). Week 5's tail commit must absorb: this `WEEK_5_SCOPE_LOCK.md` + Step 1 design memo + `WEEK_5_STEP_{2,3}_RESULT.md` + `WEEK_5_CLOSEOUT.md` + new `cipher_vllm_plugin/cipher_vllm_kvdedup.py` snapshot (cipher_vllm_plugin is not a git repo; same pattern as Week-4 exporter snapshot at `exporter_w4_step5/`) + the new `tests/test_kvdedup_live_decode.py` harness. |

---

## Verdict

**SCOPE-LOCKED.** Step 1 design-memo prompt may be drafted against the
Part 5 breakdown.

**Operating baseline:** **integration pattern (b) — monkey-patch the
block-allocation seam.** This matches the CP 5.1/5.2 precedent (same
plugin shape, same `vllm.general_plugins` entry-point registration),
is ~200-300 LOC vs ~500-800 for the ABC implementation, and the
ABC-compliance value (factory-loadable backend discoverable via vLLM
CLI) isn't a Week-5 deliverable.

Step 1 design memo **executes (b) by default**; (a) ABC implementation
is documented only as a forward-compat alternative with explicit
tradeoff analysis (forward-compat against future vLLM versions vs +300-500
LOC + tightened Step 2 budget). If user wants (a), they say so before
Step 1 drafts; otherwise Step 1 proceeds on (b).

**Step 1's first-resolve blocking question** is the page-granularity
contract (R-W5.4, HIGH). Shape (i) block-aggregation vs Shape (ii)
VMM-page-level — Step 1 must pick before any code is touched. If
unresolvable on paper, Step 1 includes a small Python KVDEDUP_PUT
spike (~30 LOC) to verify the kmod's 2 MiB contract empirically.

**Pre-Step-1 user question (single binary):** "(b) baseline OK, or
want (a) for ABC compliance?" — and "Step 1 may include the 30 LOC
ctypes spike if needed?" If yes to default + spike-allowed, Step 1
proceeds.

---

**Evidence:**
- `/home/ubuntu/cipher_kmod/cipher_kvdedup.{h,c}` (substrate)
- `/dev/cipher_kvdedup` (live, mode 666, major 510)
- `/home/ubuntu/cipher_vllm_plugin/{cipher_vllm_kv,cipher_kv_offload,setup}.py` (existing CP 5.1+5.2 plugin)
- `/home/ubuntu/vllm_env/lib/python3.10/site-packages/vllm/v1/kv_offload/{base,factory,reuse_manager}.py` (R-W5.2 surface; 589 LOC total)
- `/home/ubuntu/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py:6637` (CP 5.1 monkey-patch site)
- `/usr/lib/x86_64-linux-gnu/libxxhash.so.0.8.1` (ctypes-bindable xxhash64)
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md` §7 Week 5 (reconciled per Step 0) + §1.3 (substrate ready) + §8 R-A3 (substrate complete)
- Auto-memory: `[[cipher-track2-weight-sharing]]`, `[[cipher-cp4656-closed]]`, `[[week5-step0-doc-cleanup]]`
