# READ-ONLY PROBE — does vLLM's weight loader allocate through a torch allocator that `use_mem_pool` captures?

**Date:** 2026-05-31. **Type:** READ-ONLY measurement + cited source (NO build, NO commit, anchors unchanged:
deployed `1f305ce6`, staging `7b5f4311`, Marlin + pager tags stand). Decides STEP 2's shape.

## THE ANSWER: **BYPASS** in every realistic deployment — an outer `use_mem_pool(my_pool)` does NOT reliably capture vLLM's weights.
Capture is conditional on two independent things, and the realistic settings fail both:
1. **Process boundary (the dominant one).** Offline `vllm.LLM(model=...)` runs the V1 EngineCore — which loads
   weights (`gpu_worker.py:318 load_model` → `gpu_model_runner.py:4793 model_loader.load_model`) — in a **spawned
   subprocess** by default. A `use_mem_pool` set in the caller process never reaches the worker's address space.
2. **vLLM's own pool (when you'd actually want pageability).** With `enable_sleep_mode=True`, `load_model` runs
   under `_maybe_get_memory_pool_context("weights")` = `allocator.use_memory_pool("weights")` = vLLM's OWN
   `use_mem_pool(its_pool)` (cumem.py:86) → the inner pool shadows any outer one → weights go to vLLM's pool.
Only `enable_sleep_mode=False` AND `VLLM_ENABLE_V1_MULTIPROCESSING=0` (NOT the default) lets an outer pool
capture — and then it indiscriminately captures KV + everything, not just weights, and reinvents a mechanism
vLLM already ships.

## THE DOMINANT FINDING (bigger than the capture binary): **vLLM ALREADY implements CIPHER's single-model pager.**
`vllm/device_allocator/cumem.py` `CuMemAllocator`:
- is a **cuMemCreate/cuMemMap** pluggable allocator — `python_create_and_map`/`python_unmap_and_release` C-ext
  (cumem.py:30-44, 58-63, 71-74) — the SAME VMM primitive CIPHER's pager uses.
- has **tag-based CPU-offload sleep/wake**: `sleep(offload_tags=("weights",))` copies tagged tensors to
  `AllocationData.cpu_backup_tensor` (cumem.py:55 — literally CIPHER's warm copy) then `unmap_and_release` (frees
  HBM); `wake_up()` re-maps + restores. `gpu_worker.sleep()` logs "Sleep mode freed X GiB" (gpu_worker.py:160-179).
  **That is `page_out`/`page_in` with CPU offload.**
- is asserted **single-instance-per-process** — "Sleep mode can only be used for one instance per process"
  (gpu_worker.py:213-215). vLLM's pager canNOT hold multiple distinct models in one process.

## MEASUREMENT MATRIX (pager_vllm_probe.c/.py — a counting torch MemPool allocator wrapping `LLM(...)`; TinyLlama)
| Mode | Config | Outer pool captured | vLLM own-pool (`get_current_usage`) | Verdict |
|------|--------|--------------------:|------------------------------------:|---------|
| **M1** | default mp (realistic) | **0.0 MiB** (load fired in a subprocess; my main-proc counters = 0) | instance=None | **BYPASS (process boundary)** |
| **M2** | mp off, sleep off | 36822 MiB (weights + the 45%-util KV — *everything* in the `with` block) | instance=None (confound control: capture is real, not locality) | captures — but only mp-off + indiscriminate |
| **M3** | sleep on, mp off | 1138 MiB (incidental only) | **34.8 GiB in vLLM's OWN pool** | **BYPASS via vLLM's own pager** (weights in its pool, not mine) |

M3 is the direct proof the advisor asked for: outer pool ≈ 0 of the weights **and** `CuMemAllocator.get_current_usage()`
= 34.8 GiB → the weights bypass to vLLM's own pool, exercising the exact mechanism a vLLM integration must touch.

## INTEGRATION IMPLICATION — STEP 2's shape changes (do NOT "wire like transformers")
The transformers path worked because `from_pretrained().to('cuda')` is in-process and uses the default allocator,
so an outer `use_mem_pool` captures. vLLM is the opposite: subprocess + (under sleep) its own pool. So STEP 2 is
NOT "wrap `LLM()` in `use_mem_pool`." The mechanism already exists inside vLLM (`CuMemAllocator` + sleep/wake).

**The honest delta question STEP 2 hinges on (the fusion-was-vLLM-cargo pattern recurring, [[cipher-fusion-not-vllm-cargo]]):**
**does CIPHER's pager beat simply orchestrating N sleep-mode vLLM instances via `LLM.sleep()`/`wake_up()`?**

The delta is NARROWER than "multi-model," because **N orchestrated vLLM-sleep instances (N processes, sleep the
idle ones) ALSO give multi-model density** — so multi-model alone is not the edge. And the workflow synthesis ties
in the prior density-axis finding ([[cipher-density-axis-cargo]]): **PCIe (~51 GB/s, K misses = K×70ms) is the
SHARED cold-miss bottleneck for BOTH CIPHER and N-vLLM-instances** → raw cold-miss bandwidth CANNOT be the
differentiator (identical for both). So CIPHER's residual lever can only be one of:
- **(i) fewer BYTES moved** — sub-model / partial-LAYER residency (vLLM's tags are all-or-nothing whole
  weights/kv_cache; no per-layer granularity), so CIPHER moves less per swap; OR
- **(ii) co-residence in ONE process** — avoiding N-process + N-CPU-backup overhead that N-sleep-instances pay; OR
- **(iii) eviction-DURING-use / demand paging** — vLLM's sleep/wake requires cooperative quiescence (full-stop),
  CIPHER's proven eviction-during-use coherence does not.

STEP 2 must be SIZED as "does (i)/(ii)/(iii) beat N orchestrated sleep instances on a REAL metric, given PCIe is
shared?" **If none does, this is a fusion/Marlin-style redundant reinvention → scope down or STOP at measurement.**
The integration hook, only if a delta survives, targets the WORKER process (extend/replace `CuMemAllocator` / its
`python_create_and_map` backend), not an outer `use_mem_pool`.

## Adversarial confirmation (workflow vllm-weight-alloc-probe: 4 cited source agents + 2 adversarial refuters)
Both refutation attempts returned **`refuted: false`** — neither claim could be broken:
- **C-ext is cuMemMap-based (binary-confirmed):** `vllm/cumem_allocator.abi3.so` `nm -D` exports
  `cuMemCreate, cuMemMap, cuMemUnmap, cuMemRelease`; `objdump` shows `create_and_map` calls `cuMemCreate` then
  `cuMemMap`. Not just inferred from the python wrapper — confirmed in the compiled extension.
- **Offload is a real CPU warm-copy + free:** `sleep()` (cumem.py ~195-207) creates a CPU backup tensor and copies
  via `libcudart.cudaMemcpy(cpu_ptr, ptr, size)` into `data.cpu_backup_tensor`, then `python_unmap_and_release`.
- **Subprocess is the DEFAULT (the decisive process-boundary citation):** `vllm/envs.py:129
  VLLM_ENABLE_V1_MULTIPROCESSING = True`; `v1/engine/llm_engine.py:165-167` forces `enable_multiprocessing=True`
  when the env is set; `v1/engine/core_client.py:100-101` routes `multiprocess_mode=True` → the EngineCore (and
  its weight load) runs in a separate process by default. This is why M1 (the realistic default) captured 0.
- **Default mode skips the pool + single-instance:** `gpu_worker.py:202-210` `if not enable_sleep_mode: return
  nullcontext()` and the `assert "Sleep mode can only be used for one instance per process"`.

**Verdict (source + adversarially-survived + 3-mode measured): BYPASS in every realistic deployment; vLLM already
has the single-model pager; STEP 2 = size CIPHER's multi-model delta vs N sleep-mode vLLM instances, not a wire.**
