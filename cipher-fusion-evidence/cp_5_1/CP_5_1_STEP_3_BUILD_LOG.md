# CP 5.1 — Step 3: vLLM KV buffer hook (Option A) — BUILD LOG

**Date:** 2026-05-17. Option A adjudicated APPROVED. This is the build/measure
log — not a design memo. Findings recorded as engineering calls.

Anchors held: kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`.

---

## Known-unknown #4 — vLLM memory profiler — RESOLVED: mitigation (d), none needed

**Question:** does Option A (CIPHER VMM owns the KV buffer, invisible to
torch's allocator) cause vLLM's init-time memory profiler to mis-size
`num_gpu_blocks` → overcommit / OOM?

**vLLM 0.20.2 source traced:**

1. **Allocation ordering is definitive.** `EngineCore` (`v1/engine/core.py`):
   `get_kv_cache_specs()` → `determine_available_memory()` (`:250`) →
   builds `kv_cache_configs` → `initialize_from_config()` (`:283`).
   CIPHER's hook point — `GPUModelRunner._allocate_kv_cache_tensors` — is
   reached only inside `initialize_from_config`. **CIPHER allocates KV strictly
   AFTER the profiler has computed the budget.** It cannot perturb the profile.

2. **The budget sums the four classes independently** (`utils/mem_utils.py`
   `memory_profiling`, `v1/worker/gpu_worker.py:391`):
   `non_kv_cache_memory = weights + torch_peak_increase + non_torch_increase`;
   `available_kv_cache_memory = requested_memory − non_kv_cache_memory −
   cudagraph_estimate`. `memory_profiling` calls `torch.accelerator.empty_cache()`
   on both entry and exit (`mem_utils.py:247,261`), so the activation arena is
   released and counted as a *separate* `torch_peak_increase` term — it is not
   conflated with the KV slice. Steady-state physical =
   `weights + activation_peak + non_torch + KV_budget = requested_memory ≤
   util × total`.

3. **KV being CIPHER-VMM vs `torch.zeros` is budget-neutral.** vLLM never
   relies on torch's caching allocator sharing the activation arena with the KV
   buffer — it budgets the two terms separately and sums them. CIPHER taking the
   KV slice via `cuMemCreate` instead of `torch.zeros` changes *which allocator*
   holds those bytes, not *how many* bytes are held. No overcommit.

4. **`_allocate_kv_cache_tensors` is NOT called during the profile run.**
   `profile_run()` → `_dummy_run(is_profile=True)` measures activation memory; it
   does not allocate the KV cache. The only profile-time KV allocation is the
   *minimal* cache in `profile_cudagraph_memory()` (`gpu_model_runner.py:5949`,
   `initialize_kv_cache(minimal_config, is_profiling=True)`), which is gated on
   `cudagraph_mode != CUDAGraphMode.NONE` (`gpu_worker.py:380`) — skipped under
   `enforce_eager`, and tiny + freed when not.

**Resolution:** mitigation **(d) — none required**. The profile→budget→allocate
ordering plus `empty_cache()` makes Option A budget-neutral. The advisor's
flagged risk (overcommit / double-allocate) assumed CIPHER memory existed during
profiling; the ordering rules that out.

**Documented invariant carried to the build (verified under known-unknown #5):**
`cipher_rt_kv_alloc_init` must reserve **VA address space only** (no
`cuMemCreate`) so CIPHER's physical footprint during the profile window is ~0.
If CIPHER held physical memory during profiling it would land in
`non_torch_increase` and *conservatively shrink* the KV budget — safe (never
OOM), but wasteful. VA-only init is native CUDA-VMM behaviour; confirm in
`cipher_rt_kv_alloc.c`.

**Mitigation rejected, with reason:** (b) scale `gpu_memory_utilization` down —
unnecessary and would waste KV capacity for a problem that does not exist.
(a)/(c) — no accounting to correct and no profile pass to intercept.

**Closeout checks (both pass):**
- *VA-only init verified.* `cipher_rt_kv_alloc_init` (`cipher_rt_kv_alloc.c:77`)
  does `cuInit` / `cuDevicePrimaryCtxRetain` / `cuMemAddressReserve` only — no
  `cuMemCreate`/`cuMemMap`. Physical mapping lives in a separate lazy page-map
  path (`:131+`). `bridge_init` is a thin wrapper over it. CIPHER profile-time
  physical footprint = 0. Invariant holds.
- *No post-alloc gating check.* `gpu_worker.py:608-664` builds a
  `--kv-cache-memory` suggestion string and `logger.debug`s it — diagnostic
  only, no assert/raise. No post-`initialize_from_config` memory check compares
  against torch accounting. CIPHER VMM (seen by `mem_get_info`, not by
  `memory_reserved`) trips nothing.

## Known-unknown #5 — VA pool sizing — RESOLVED: no allocator change needed

`cipher_rt_kv_alloc_init(size_t va_pool_bytes)` (`cipher_rt_kv_alloc.c:77`)
takes the pool size as a parameter, rounds up to CUDA-VMM granularity
(`:105`), and `cuMemAddressReserve`s it. **No hardcoded ceiling.** Page-tracking
array is `calloc(n_pages, …)` — for an 80 GiB pool at 2 MiB pages = 40960
entries, a ~MiB-scale host allocation. VA reservation costs no physical memory,
so over-reserving is free.

**Resolution:** the allocator already accepts an operator-sized pool. The Step-3
hook calls `bridge_init` with a generous pool (full device, ~80 GiB) once in the
EngineCore subprocess; physical pages map lazily per `vmm_zeros`. No edit to
`cipher_rt_kv_alloc.c`. (`bridge_init` is idempotent — `g.init_done` guard at
`:80` — so first caller fixes the pool size; the vLLM hook is the only caller in
the EngineCore process.)

## Hook build

Three artifacts, one campaign anchor untouched by any of them.

**1. `cipher_kv_bridge` rebuilt — `2cf82c06` → `8d6ffe3f`.** The Step-2 memo
§(b) listed the S2b bridge `.so` (`2cf82c06`) as "reusable as-is, no rebuild".
That line is now **stale**: CP 5.1 §(a) needs the raw KV buffer as a flat
**int8** tensor (`gpu_model_runner._allocate_kv_cache_tensors` →
`torch.zeros(size, dtype=torch.int8)`), and the bridge's `dtype_from_string`
did not accept `"int8"`. Added the `at::kChar` case (`cipher_kv_bridge.cpp:32`)
and rebuilt. Legitimate rebuild: `cipher_kv_bridge` is **not a campaign
anchor** (it is a build artifact; the three anchors are kmod `e2f50452`,
libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`, all re-verified unchanged
post-build). Prior `.so` preserved as `cipher_kv_bridge.so.pre_cp51`.

**2. `cipher_vllm_plugin/` — the Option A hook.** `cipher_vllm_kv.py` monkey-
patches `GPUModelRunner._allocate_kv_cache_tensors` with a faithful re-
implementation of vLLM 0.20.2's original (verified against
`inspect.getsource` of the installed method): same `kv_cache_tensors` loop,
same `shared_by` aliasing, same `runner_only_attn_layers` exclusion, same
layer-name assertion — the **only** change is the raw int8 buffer comes from
`cipher_kv_bridge.vmm_zeros(...)` instead of `torch.zeros`. Idempotent
(`_cipher_hooked` guard), env-gated (`CIPHER_KV_ALLOC`), original kept as
`_cipher_orig`.

**3. Packaging.** Installed as `cipher-vllm-kv 0.1.0` via `pip install -e`.
Note: the pod ships **setuptools 59.6.0**, which predates the PEP 621
`[project]` table (≥61) and PEP 660 editable installs (≥64) — a
`pyproject.toml`-only package installs as `UNKNOWN-0.0.0` with the entry point
silently dropped. Resolved by an explicit `setup.py` with the `entry_points`
dict; no network, no setuptools upgrade. Entry point verified registered
under `vllm.general_plugins`.

## Known-unknown #2 — EngineCore-subprocess install mechanism — RESOLVED

Mechanism chosen: the **`vllm.general_plugins` entry-point group**. vLLM's
`load_general_plugins()` runs every registered `register()` in *every*
process, so the hook installs in the EngineCore subprocess where
`_allocate_kv_cache_tensors` actually executes — without the design memo's
`CUDA_INJECTION64_PATH` (which injects C, not Python). **Empirically
confirmed** by the gate run: the `[cipher-vllm-kv] KV buffers owned by CIPHER
VMM: 22 tensor(s) ...` and `VMM allocator init` lines both carry the
`(EngineCore pid=1477712)` multiplex prefix — i.e. the patched method ran
*inside* EngineCore. (`register()` also runs once in the main process; that
"hook installed" line is unprefixed and harmless.)

## Known-unknown #1 — page-tag granularity — RESOLVED: tagged, deferred to dedup

vLLM gives one int8 buffer per layer with K and V interleaved by stride; there
is no single K-or-V slab. The hook therefore tags at **per-layer-buffer**
granularity: `role=2` (whole buffer, neither pure-K nor pure-V), `head_kv=
0xFFFF` (spans all heads), `layer=idx`. K-vs-V now lives *inside* one tagged
buffer. This shifts what T4.6.3 dedup keys on relative to S2b's per-(layer,
K/V-role) slabs. **Deferred to dedup wiring** — out of CP 5.1 scope (CP 5.1 is
buffer ownership + correctness, not dedup); flagged for whoever wires
T4.6.3-over-vLLM.

## Known-unknown #3 — uniform-kv path — RESOLVED: explicitly deferred

The hook patches `_allocate_kv_cache_tensors` only. The second allocation site,
`allocate_uniform_kv_caches` (fires only when a KV-connector / disaggregated
prefill is configured, `use_uniform_kv_cache=True`), is **not** patched —
under a connector config CIPHER would be bypassed and vLLM would use native
`torch.zeros` KV. This is **explicitly out of the CP 5.1 single-pod baseline**
(no connector). Documented as a known gap, not a defect; covering it is a
follow-on if disaggregated serving enters scope.

## Correctness gate — Option A — PASS

Driver `cp51_kv_gate.py`, worker `cp51_kv_gate_worker.py`, runlog
`cp51_kv_gate.runlog`. Two separate processes, identical Python, substrate
switched purely by `CIPHER_KV_ALLOC`: OFF (native `torch.zeros`) then ON
(CIPHER VMM). TinyLlama-1.1B, `enforce_eager=True` (removes the cudagraph
confound), greedy decode (`temperature=0.0`, `seed=1234`), 4 prompts ×
64 tokens.

- **Check A — token-ID identity (CP 5.1 §4 correctness bar):** PASS. All
  4 prompts, all 64 output token IDs identical OFF vs ON. Diffed token IDs,
  not decoded text (a tokenizer roundtrip can mask a 1-token divergence).
- **Check B — hook fired in EngineCore subprocess:** PASS. `hook installed`
  + `KV buffers owned by CIPHER VMM` present, the latter under
  `(EngineCore pid=)`.

Substrate-ON allocated **22 CIPHER-VMM KV buffers** (`slabs_created=22`,
`pages_mapped=10626`, `resident=20.75 GiB`) at `gpu_memory_utilization=0.30`,
and engine init did **not** OOM — the known-unknown #4 resolution
(profile→budget→allocate ordering makes Option A budget-neutral) is now
**empirically confirmed**, not just traced.

**CP 5.1 STEP 3 CORRECTNESS GATE: PASS.** Option A built, installed, and
verified coverage-immune (CIPHER *is* the buffer; identical tokens prove every
attention kernel read/wrote CIPHER VMM memory).

## Correctness gate — Mistral-7B (GQA) — PASS

Adjudication of the TinyLlama gate accepted, with one pre-close requirement:
re-run the *same* `cp51_kv_gate.py` harness on **Mistral-7B-v0.1** before CP 5.1
close. Rationale (adjudicator): TinyLlama-1.1B is multi-head attention (no GQA),
22 layers; Mistral-7B has **GQA — 32 query heads, 8 KV heads** — 32 layers, and
a different KV-head stride. The GQA stride exercises the per-layer-buffer
tagging path differently; the adjudicator wanted that exposed now, not at
CP 5.5.

Run: `CIPHER_GATE_MODEL=/home/ubuntu/models/Mistral-7B-v0.1 python3
cp51_kv_gate.py`, runlog `cp51_kv_gate_mistral.runlog`. Harness, decode params,
prompts identical to the TinyLlama gate (greedy, `temperature=0.0`, `seed=1234`,
`enforce_eager`, 4 prompts × 64 tokens; `gpu_memory_utilization=0.30` held
unchanged — Mistral-7B fits with headroom on the 80 GB H100, no KV-pressure
adjustment needed).

- **Check A — token-ID identity:** PASS. All 4 prompts, all 64 output token IDs
  identical OFF vs ON. The GQA KV layout (8 KV heads strided inside each
  per-layer int8 buffer) round-trips correctly through CIPHER VMM ownership.
- **Check B — hook in EngineCore subprocess:** PASS. `(EngineCore pid=1478520)
  [cipher-vllm-kv] KV buffers owned by CIPHER VMM: 32 tensor(s),
  slabs_created=32, pages_mapped=4256, resident=8.31 GiB` — **32** buffers
  (one per Mistral layer, vs 22 for TinyLlama), no OOM at `util=0.30`.

**MISTRAL-7B CORRECTNESS GATE: PASS.** Option A is now validated on both a
non-GQA model (TinyLlama-1.1B) and a GQA model (Mistral-7B-v0.1).

## CP 5.1 close

Per adjudication, CP 5.1 closes on:
1. **Both models validated** — TinyLlama-1.1B (MHA) and Mistral-7B (GQA),
   token-ID identity + hook-in-EngineCore, both PASS. ✓
2. **Build log updated** — this document. ✓
3. **Memo version-line reconciled** — `PHASE_5_CP_5_1_DESIGN_MEMO.md` §1 and §4
   "vLLM 0.5/0.6/0.7" → "vLLM 0.20.2" (version token only; surrounding scope
   text left as-is per adjudication). ✓

**Anchors held through CP 5.1:** kmod `e2f50452`, libcipher_rt `c2c5d313`,
libcipher_v2 `86618c30` — all re-verified unchanged. `cipher_kv_bridge` is a
non-anchor build artifact, now `8d6ffe3f` (int8 add), prior `.so` preserved as
`cipher_kv_bridge.so.pre_cp51`.

**Staged, NOT built:** the `cipher_kmod/cipher_proc.c` /proc banner
`0.4.5`→`0.4.8` edit remains in the working tree only. Rebuilding rotates
anchor `e2f50452`; the banner is cosmetic; it rides the next *legitimate* kmod
rebuild, not a banner-only one.

**Carried out of CP 5.1 scope (documented gaps, not defects):** KU#1 page-tag
granularity (per-layer-buffer, K/V interleaved — affects T4.6.3 dedup keying);
KU#3 uniform-kv path (`allocate_uniform_kv_caches`, connector-only, unpatched).

**Scope note for the adjudication trail:** the design memo §4 lists four gate
criteria. This CP 5.1 close, per adjudication, certifies criterion **#2**
(substrate-on vs -off correctness) on vLLM 0.20.2 across two models. Criteria
#1 (CP 4.8 env-fail workloads run clean), #3 (operator deployment recipe on a
reference stack), and #4 (MFU / tok-W on live continuous-batching decode) are
not covered by this gate and are not asserted here.
