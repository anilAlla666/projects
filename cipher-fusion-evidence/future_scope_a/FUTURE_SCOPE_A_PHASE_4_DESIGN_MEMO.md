# FUTURE_SCOPE / A — PHASE 4 — vLLM CROSS-INSTANCE WEIGHT SHARING — DESIGN MEMO

**Date:** 2026-05-20. **Type:** design/architecture — **paperwork only**, no
source modified, no GPU, no anchor rotation. **Phase 4 of FUTURE_SCOPE/A.**
STOP for adjudication before Phase 4 build.

Anchors at start (Track 2 + Track 3 close, unchanged through FUTURE_SCOPE/A
Phases 2/3/3.5):
- `cipher_kmod.ko` **`008b3c66`** (Track 2 SC5 fd-custodian)
- `cipher_kv_bridge.so` **`c04b0c39`** (Track 2 SC3 consumer-import)
- `libcipher_rt.so` **`83afd1ca`** (Track 3 SC3)
- `libcipher_v2.so` **`cc0479b8`**
- vLLM **0.21.0** at `/home/ubuntu/vllm_env`

**Predecessors:** FUTURE_SCOPE/A Phase 1 design memo, Phase 2 (transparency
verified), Phase 3 (parity within ~5%), Phase 3.5 (DVFS-under-vLLM = +14%,
NOT the headline lever). Track 2 closed end-to-end (SC1-SC6). The producer
half (arena create / export) + consumer half (import / `WeightArena.view`) +
kmod-owned fd lifetime (ioctls 21-24) are **already shipped** — Phase 4 wires
them up under vLLM with **zero new C/C++ code if the spec holds**.

---

## §0 — Where Phase 4 plugs in

The FUTURE_SCOPE/A design memo §2b flagged weight sharing as the **density
headline** — **76 % HBM saving measured on Mistral-7B N=4 (Track 2 SC6)**,
asymptote ~95 % at large N — and the **largest build item** in the composed
architecture, because vLLM loads weights through PyTorch's caching allocator
(`cudaMalloc`), not CIPHER's VMM arena. Risk **R2** (vLLM allocator path) is
the work this phase resolves.

Phase 3.5 measured DVFS under vLLM at +13.9 % tok/W and concluded:
> *the pitch's primary lift mechanism is multi-tenant density, not
> single-instance actuator acceleration.*

Phase 4 builds that density lever — N vLLM instances of the **same model**
sharing **one physical weight copy** — on top of Track 2's already-shipped
primitives, exposed to vLLM via its **public `register_model_loader`** seam
and the **`vllm.general_plugins` entry-point** (the CP 5.1 precedent).
Phase 5 (SM arbitration across N vLLM instances) and Phase 6 (composed
end-to-end with DVFS envelope) follow.

## §1 — The three integration mechanisms — and the verdict

The FUTURE_SCOPE/A §2b memo listed two candidates; the vLLM-0.21.0
read-only spike (2026-05-20) found a third, and rules out the first two.

| | option | mechanism | verdict |
|---|---|---|---|
| **A** | **PyTorch pluggable allocator** | `torch.cuda.memory.CUDAPluggableAllocator` + `change_current_allocator` to route weight allocs through the CIPHER arena | **REJECTED.** `torch/cuda/memory.py:1217` documents: *"If the current allocator has already been used/initialized, this function will error."* vLLM does CUDA work (device init, distributed setup) before the model loader runs at `base_loader.py:56`, so the change must happen in worker `__init__` before any CUDA op — a layering violation. The alloc/free callback signatures are fixed C; scoping the allocator to weight-load only and reverting before KV-cache alloc is not cleanly expressible. |
| **B** | **Post-load monkey-patch + rebind** | Let vLLM allocate normally, walk `named_parameters()` after `load_weights()`, copy into a fresh arena, assign `p.data = arena_view` | **REJECTED.** Peak HBM is 2×W (vLLM's cudaMalloc copy + arena copy), which negates the producer-side density argument momentarily and is wasteful. Also redundant: Track 2 SC3's `meta_load → WeightArena.alloc → safetensors copy_` path already does this in 1×W peak. |
| **C** | **Subclass `BaseModelLoader`, register as a load-format** *(recommended)* | A `CipherSharedModelLoader(BaseModelLoader)` overrides `load_model()` entirely — meta-load + arena-direct-alloc for the producer (the Track 2 SC3 path), meta-load + arena-import + view-rebind + skip `load_weights()` for the consumer. Registered via `register_model_loader("cipher_shared", CipherSharedModelLoader)` invoked from a `vllm.general_plugins` entry-point. | **ACCEPT.** Clean public-API seam — no monkey-patch. Peak 1×W on producer, ~0×W on consumer. Re-uses Track 2's shipped pybinds verbatim. Matches the CP 5.1 plugin pattern (`/home/ubuntu/cipher_vllm_plugin/setup.py`). |

**The spike found public API at `vllm/model_executor/model_loader/__init__.py:67-117`** — `register_model_loader(load_format)` is a documented decorator. We do not need to touch any vLLM source. User code dispatches by `LLM(model=..., load_format="cipher_shared")` (or sets `VLLM_LOAD_FORMAT=cipher_shared` in env).

## §2 — `CipherSharedModelLoader` — the two branches

The loader's `load_model()` selects branch from `CIPHER_WEIGHT_SHARING`:

| env value | role | sequence |
|---|---|---|
| `producer` *(or unset on tenant 0)* | producer | (1) read safetensors layout to size the arena; (2) `cipher_rt_weight_arena_create(bytes)`; (3) **meta-load** the model — `from_pretrained(..., torch_dtype, device_map=None)` with `init_empty_weights()`; (4) for each `(name, p)` in `named_parameters()`: `WeightArena.alloc(shape, p.element_size(), dtype)` → returns a view → assign `p.data = view`; (5) iterate the safetensors weight stream and `param.data.copy_(loaded)` per the upstream `default_weight_loader` (`weight_utils.py:1399-1417`); (6) **run `process_weights_after_loading()`** so kernel-format transforms (e.g. weight prepacking) write into the arena views, not into orphan tensors; (7) `WeightArena.export_fd()` → register fd with the **kmod fd custodian** (ioctls 21-24); (8) emit the manifest sidecar (SC3 §3 format). |
| `consumer:<fd>` *(or `consumer` + arena from `/proc/cipher/arenas`)* | consumer | (1) import arena: `cipher_rt_weight_arena_import(fd, want_base, bytes)`; (2) read manifest sidecar; (3) **meta-load** the model architecture only — `init_empty_weights()` — no allocation, no safetensors read; (4) `cipher_kv_bridge.consumer_fingerprint_check()` against producer's manifest (SC4 path); (5) for each `(name, p)` in `named_parameters()`: assign `p.data = WeightArena.view(name, shape, dtype, offset)`; (6) **skip** `load_weights()` entirely; (7) **skip** `process_weights_after_loading()` — see §3. |
| unset | passthrough | call the parent `DefaultModelLoader.load_model()` verbatim. Phase 4's loader is correctness-safe even when CIPHER is not wanted. |

Both branches end by switching the returned `nn.Module` to inference mode
(the upstream `.eval`-mode call at `base_loader.py:80`) and returning a real
module whose parameters live in the shared arena. From vLLM's perspective,
nothing else changes — graph capture, paged-attn, continuous batching all
run on the rebound parameters.

## §3 — `process_weights_after_loading()` — load-bearing subtlety

`base_loader.py:78` calls `process_weights_after_loading(model, model_config,
target_device)` after `load_weights()`. This is **not a no-op** — it runs
kernel-format transforms (weight prepacking, per-kernel re-layouts; for
quantized models also unpacks/repacks). Anything it writes goes into
`p.data`'s storage, which in our architecture **is the arena**.

| role | handling | reason |
|---|---|---|
| **producer** | run `process_weights_after_loading()` **after** rebind to arena views and **before** `export_fd()` | transforms must land in the arena, so the exported bytes are already in kernel-format. The fd custodian holds the post-transform arena. |
| **consumer** | **skip** `process_weights_after_loading()` entirely | the bytes in the arena are already kernel-format. Re-running the transform would re-mutate the shared arena — non-idempotent for many transforms (e.g. AWQ unpack-then-repack) — and corrupt every co-resident consumer including the producer. **The consumer's `load_model()` returns immediately after rebind.** |

This is the single most subtle correctness requirement of Phase 4 and the
gate that catches it is **teacher-forced KL = 0.0 vs vLLM-alone** (§6.b).

## §4 — vLLM process model & env propagation

The spike confirmed (`vllm/v1/executor/multiproc_executor.py:676-687`,
`vllm/utils/system_utils.py:175-181`):

- vLLM 0.21.0 defaults to **V1 (EngineCore subprocess model)**;
- The CUDA-owning worker process is a Python `multiprocessing.Process`
  (spawn or fork — `VLLM_WORKER_MULTIPROC_METHOD`);
- **Parent env propagates to workers** in both modes (spawn re-serialises
  arguments but preserves `os.environ`);
- Plugins registered under the `vllm.general_plugins` entry-point group are
  loaded by `load_general_plugins()` **in every process including EngineCore
  subprocesses** — CP 5.1's existing pattern (`cipher_vllm_plugin/`).

Therefore:
- `CUDA_INJECTION64_PATH=libcipher_rt.so`, `CIPHER_WEIGHT_SHARING`,
  `CIPHER_KV_VA_POOL_GIB` etc. can be set in the parent (shell or supervisor)
  and reach every worker;
- `register_model_loader("cipher_shared", CipherSharedModelLoader)` is
  invoked from the plugin's `register()` and applies in every worker;
- No vLLM source modification; no monkey-patch.

## §5 — Failure modes

| failure | handling | source |
|---|---|---|
| `cuMemImportFromShareableHandle` fails (bad fd / VMM error) | consumer falls back to **`load_format="safetensors"`** — independent load, no regression | SC3 §5 |
| same-VA reserve collision | offset-relative mapping; rebind is offset-relative regardless | SC3 §2 |
| manifest missing / unreadable | consumer falls back to safetensors | SC3 §5 |
| consumer model architecture ≠ producer's (name/shape/dtype mismatch) | **fingerprint check** (Track 2 SC4) → consumer falls back to safetensors | SC4 |
| **producer crashes while consumers are live** | **consumer's VA mapping is preserved** — Track 2 SC5 closeout line 62: *"a live consumer keeps its existing mapping, new IMPORTs fail."* The kmod fd custodian holds the cuMem fd ref; the consumer's `cuMemMap` already executed in the consumer's address space, so its CUDA graphs continue to replay against the same VA. Only **new** join attempts after producer death are rejected. | SC5 closeout |
| zero-participant survival (producer crashes before any consumer joined) | arena reclaimed by the 5 s workqueue reaper; documented v1 boundary; v2 hardening = `registered_at` grace window | SC5 closeout §v1 boundary |
| consumer writes a weight | precluded — arena mapped `cuMemSetAccess(READ)` in the consumer | SC3 §4 |
| kmod unload with active arenas | `cipher_wa_exit` `fput`s all held fds; live consumer mappings survive; new IMPORTs fail | SC5 |
| vLLM's `process_weights_after_loading` on consumer mutates arena | **the loader skips it** (§3); regression caught by §6.b KL = 0 gate | this memo |
| quantized model (online quant path) in v1 | **fail-closed** — Phase 4 v1 refuses `model_config.quantization is not None`; falls back to safetensors with a logged warning | §7 boundary |

## §6 — Gates for the Phase 4 build (mention only — paperwork phase)

The build phase will close on this gate matrix. Stated here so adjudication
can accept the **shape** of the eventual close.

| gate | target | how |
|---|---|---|
| **a. bit-identity** | `token_ids` byte-identical AND teacher-forced KL = 0.0, consumer vs vLLM-alone, on **TinyLlama-1.1B** and **Mistral-7B**, 5 reps | Phase-2/3 harness extended with a 2-arm consumer-vs-baseline producer |
| **b. density (the headline)** | `nvidia-smi --query-gpu=memory.used` per process: consumer-2..N add only ε on top of producer's 1×W; honest measurement at **N=4 Mistral-7B** (the Track 2 SC6 anchor point); record the per-N curve at N=2,4,8 | per-process polling under steady-state decode |
| **c. parity** | producer-mode tok/s within **±5 %** of vLLM-alone single-instance (Phase 3 baseline holds); consumer-mode tok/s within ±5 % of vLLM-alone single-instance | TinyLlama greedy 128-tok, 5 reps, graph mode (Phase 3 protocol) |
| **d. regression** | native W1/W2/W3 + isolation 15/15 + dmesg clean + FUTURE_SCOPE/A Phase 3 + Phase 3.5 reproduce within ±3 % | the existing smoke harness |
| **e. crash-survival** | `SIGKILL` the producer mid-serve while ≥1 consumer is mid-decode → consumer completes its decode with KL = 0 vs vLLM-alone | SC5 producer-dies-first 12/12 path extended to under-vLLM |

The build phase will follow the campaign discipline: **7-item plan → approve
→ execute → tarball; preserve prior anchor before any edit**
([[cipher-fusion-campaign]], [[cipher-regression-discipline]]).

## §7 — v1 boundaries

- **Single H100 / single node.** Multi-GPU is v2 (Phase 5/6 + FUTURE_SCOPE/A v2).
- **Same model across the participating set.** Heterogeneous-model arenas are
  v2; the SC4 fingerprint check is fail-closed.
- **No tensor parallel** — TP=1 per vLLM instance (FUTURE_SCOPE/A §7 R6).
- **No LoRA.** vLLM's LoRA path adds dynamic weight deltas the arena does not
  model in v1; fail-closed.
- **No online quantization.** `process_weights_after_loading` for quantized
  models is non-idempotent; fail-closed in v1 (§5). Pre-quantized AWQ/GPTQ
  models are eligible if their kernel-format transform is **idempotent on
  arena bytes** — Phase 4 ships v1 on FP16/BF16 only; quantized adds in v1.5.
- **Producer must outlive zero-participant** — at least one consumer must
  register before producer crashes for arena survival. (SC5 v1 boundary.)
- **fd hand-off uses the SC5 kmod custodian path** — `/proc/cipher/arenas`
  + ioctls 21-24, not `SCM_RIGHTS` (which was the SC3 *test harness*).

## §8 — Anchor expectation (key adjudication point)

| case | rotation | likelihood |
|---|---|---|
| **A — pure plugin** | New Python package `cipher_vllm_weight_share` (sibling of `cipher_vllm_kv`) using existing `WeightArena.alloc / .export_fd / import_fd / .view` + existing kmod ioctls 21-24 verbatim. **No anchor rotates.** | **Most likely** — the Track 2 SC2/SC3 pybinds already expose every primitive listed in §2. |
| **B — one convenience pybind** | If the consumer-side `view()` needs a small wrapper to accept a `nn.Parameter` and assign in place (sugar around `p.data = arena.view(...)`), it rotates `cipher_kv_bridge` `c04b0c39`. | Possible if §2 (5) is awkward in plain pybind. Estimate ≤10 LOC. |
| **C — kmod arena-discovery surface** | A new ioctl/proc node that lets a consumer enumerate available arenas without an out-of-band fd. v1 uses fd handoff so this is **explicitly deferred to v2**. | **Not in Phase 4 v1.** |

The decision: **expect case A; pre-authorize case B in adjudication so an
SC3-style ≤10-LOC sugar wrapper doesn't require a second adjudication round.**
Case C is v2.

## §9 — Build phases (the eventual sequence after this memo is approved)

| phase | scope | est. |
|---|---|---|
| **P4-1** | this design memo (paperwork only) | done |
| **P4-2** | build the plugin package `cipher_vllm_weight_share/` + `setup.py` with the `vllm.general_plugins` entry-point + `CipherSharedModelLoader` (producer/consumer branches) + manifest writer/reader + fingerprint helper; **preserve `cipher_kv_bridge.so` as `.pre_phase4` before any edit** even if no rotation is expected | ~2 d |
| **P4-3** | verify §6 gates (a-e) on TinyLlama-1.1B (N=2 then N=4) then Mistral-7B (N=2 then N=4). Producer-only smoke first (parity), consumer-only correctness second (KL=0), density third (the headline), crash-survival fourth | ~2 d |
| **P4-4** | closeout — `FUTURE_SCOPE_A_PHASE_4_RESULTS.md`; anchor rotation **only if** §8 case B fired; update anchors manifest + memory. | ~0.5 d |

## §10 — Adjudication ask

**STOPPING — no source modified, no build, no GPU, no anchor rotation.**
Decisions:

1. **Accept §1 verdict — option C (subclass + `register_model_loader`)**, with options A and B explicitly rejected for the reasons stated.
2. **Accept §2 producer/consumer branch architecture**, both built on Track 2's already-shipped pybinds (`WeightArena.alloc/.export_fd/.import_fd/.view`).
3. **Accept §3 — the consumer skips `process_weights_after_loading()`**, with the §6.b KL=0 gate as the catch.
4. **Accept §4 — `vllm.general_plugins` entry-point dispatch** (CP 5.1 precedent), env-driven producer/consumer role selection.
5. **Accept §5 failure-modes table** — including the SC5-derived "producer dies, consumers' mappings survive" semantics as the production guarantee (not a hope).
6. **Accept §6 gate shape** for the eventual Phase 4 close (bit-identity + density + parity + regression + crash-survival).
7. **Accept §7 v1 boundaries** — FP16/BF16 only, TP=1, same-model, no LoRA, no online quant.
8. **Pre-authorize §8 case B** — a ≤10-LOC `cipher_kv_bridge` sugar wrapper without a second adjudication round, with the rotation recorded at close.
9. **Accept §9 build sequence** (P4-2/3/4) as the next four working units, each with the standard 7-item plan → execute → tarball discipline.

On adjudication → **P4-2 (build)** — the plugin package, no vLLM source touched, no GPU until §6 verification.

---

**Evidence cited:**
- vLLM internals spike (this session, read-only, 8 verbatim excerpts from `/home/ubuntu/vllm_env/lib/python3.10/site-packages/vllm/`)
- Track 2 SC3 design memo (`cipher-fusion-evidence/phase_c/TRACK_2_SC3_DESIGN_MEMO.md`)
- Track 2 SC5 closeout (`cipher-fusion-evidence/phase_c/TRACK_2_SC5_CLOSEOUT.md` — line 62 producer-death semantics)
- Track 2 closeout (`cipher-fusion-evidence/phase_c/TRACK_2_CLOSEOUT.md` — anchor table)
- FUTURE_SCOPE/A design memo §2b, Phase 3.5 results
- CP 5.1 plugin precedent (`/home/ubuntu/cipher_vllm_plugin/setup.py`, `cipher_vllm_kv.py`)
