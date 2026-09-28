# SIZING — the memory/density axis toward 100 agents: which lever delivers OVER stock vLLM?

**Date:** 2026-05-31. **Type:** READ-ONLY sizing (no build, no commit; deployed `1f305ce6` / staging `29686678`
/ Marlin gate unchanged). Picks the cargo before any build. **Answer: SAME-model density is vLLM-SUBSUMED
(drop it). DISTINCT-model density is the differentiated cargo — real-and-buildable toward 100 IF INT4 + tiered
residence + the bursty statistical-multiplexing assumption holds; fp16 caps ~5-10 (swap-thrash). Binding
constraint = HBM hot-set vs peak concurrency (mitigated by INT4) + warm-set RAM.**

---

## PART A — SAME-model density: vLLM-SUBSUMED (no cargo)

Stock vLLM, ONE TinyLlama engine, N concurrent same-model requests (continuous batching + APC, both default-on),
`same_model_density.py`:

| N | agg tok/s | meanW | gpu% | tok/s/W |
|---|---|---|---|---|
| 4 | 1,551 | 134 | 64 | 11.6 |
| 16 | 5,604 | 176 | 37 | 31.8 |
| 50 | 8,729 | 193 | 37 | 45.3 |
| 100 | **15,606** | 202 | 31 | **77.4** |

**vLLM delivers 10× aggregate tok/s and 6.7× tok/s/W from N=4→100 on one engine.** That *is* the prior CIPHER
"3-6× tok/W cross-tenant" lever — it was the **batching** lever, and vLLM continuous batching subsumes it:
one engine = **one weight copy** for N same-model requests (CIPHER's cross-tenant weight-sharing can't beat one
copy), and `enable_prefix_caching=True` (`vllm/config/cache.py:91`) already does shared-prefix KV-dedup. So
CIPHER's same-model weight-sharing/KV-dedup adds **~0** over a properly-configured single vLLM engine.
- **Only residual same-model angle:** content-addressed KV-dedup of **non-prefix** shared segments (vLLM's APC
  is prefix-only) — e.g. RAG with a shared document mid-context. Narrow; not a Phase-1 lever. **Same-model: DROP.**

## PART B — DISTINCT-model density: the differentiated axis (vLLM gives NOTHING)

**The gap is real (cited):** vLLM `LLM()` takes a single `model=` (no `models=` list); `max_loras` is same-*base*
adapters (`vllm/config/lora.py:36`), not distinct models. So vLLM is **one base model per engine** → N distinct
models = N engines = N×14.5 GB → **OOM at N>~5**. vLLM cannot serve 100 distinct models on one H100, by design.

**The memory wall:** 7B fp16 = 14.5 GB, INT4 = 3.6 GB; HBM = 80 GB. Resident cap ≈ **5 fp16 / ~20 INT4** (minus
KV/activation). 100 distinct = 1450 GB (fp16) / 360 GB (INT4) of weights — must be **tiered**: HBM hot / RAM warm
/ NVMe cold, paged on demand.

**Weight-swap cost (measured, `torch` pinned H2D, PCIe Gen5 x16 @ 51 GB/s effective):**
- 7B **fp16**: RAM→HBM pull = **283 ms**, evict = 282 ms.
- 7B **INT4**: pull = **70 ms**, evict = 70 ms.

**Feasibility — NOT established; gated on four unmeasured conditions (advisor pushback, two probes run):**
- **Probe (a) aggregate PCIe SERIALIZES (measured):** N concurrent 2 GB H2D pulls on the one Gen5 link = 39 / 78 /
  156 / 311 ms (N=1/2/4/8) → flat **51 GB/s aggregate** (no parallelism — one link). So K correlated cold-misses
  cost **K×70 ms (INT4) serialized**, NOT 70 ms. A peak of ~10 simultaneous misses = **700 ms ≫ the 200 ms
  burst** → thrash. **The binding constraint is aggregate-PCIe-under-correlated-bursts** — the isolated 70 ms
  single-pull number does NOT speak to it.
- **Probe (b) KV starves the resident cap (measured/calc):** Mistral-7B GQA KV = 128 KB/token = 0.5 GiB per 4K
  agent context. Resident weights + KV must share 80 GB: **10 INT4 models → ~80 contexts; 15 → ~44; 20 → only ~8**
  (KV-starved). So the practical hot set is **~10-15 models, not 20** — and agents are context-heavy, tightening
  further.
- **Duty-cycle is the MEAN, thrash is at the PEAK:** 100 × ~5% ≈ 5-15 *mean* concurrent, but multi-agent systems
  **fan out correlated** (one user action → many agents burst together) → transient peak 30-50, which a ~10-15
  hot set can't cover → cold-miss storm → serialized PCIe → thrash. Arrival-correlation is unmeasured and decisive.
- **INT4 quality is load-bearing AND unestablished:** fp16 thrashes (283 ms pulls, ~5 resident — measured), so
  feasibility *requires* INT4 — but that's **100 per-model quality gates** (CIPHER's own FP8 failed PPL +0.567%).
  Not an optimization; a precondition with an open quality gate.
- **Tiering is really two-tier for burst-responsiveness:** NVMe-cold read ~3-7 GB/s → ~700 ms+ for a 3.6 GB INT4
  model ≫ burst → NVMe can't serve within a burst. So the **warm set must hold ~all 100 in RAM (360 GB)**; NVMe
  only for truly-dormant. A real host-spec constraint (≥512 GB RAM).

**The differentiated mechanism (the cargo):** a tiered-residence weight manager (HBM hot / RAM warm, cuMemMap-style
paging) driven by bursty arrival, serving N distinct models on one GPU — which vLLM fundamentally can't do.
CIPHER's Track 2/3 (weight-sharing, DSM migration) + KV-bridge (cuMemMap) are the substrate.

## THE ANSWER (per half)
- **SAME-model density: vLLM-SUBSUMED → DROP.** vLLM continuous batching + APC deliver 6.7× tok/W to N=100 on one
  engine. (The old CIPHER "3-6×" was vs N-separate-instances/N-weight-copies; the honest baseline is vLLM-one-
  engine, which I ran — it already captures it. Trap-guard worked. Only narrow residual: non-prefix KV-dedup.)
- **DISTINCT-model density: the DIFFERENTIATED axis (vLLM can't, confirmed) — but "toward 100" is PLAUSIBLE-ONLY-IF,
  not feasible-as-measured.** It holds only if ALL of: (1) per-model INT4 quality holds, (2) arrivals aren't
  correlated, (3) aggregate PCIe absorbs the peak cold-miss rate, (4) active-set KV fits the ~10-15-model residual
  — **none established.** **Binding constraint = aggregate-PCIe-under-correlated-bursts** (measured to serialize),
  NOT RAM. fp16 caps ~5; INT4 hot set ~10-15; "100" only under benign (uncorrelated, quality-OK) load.
- **Correctness (co-residence):** distinct models share no weights → no cross-model corruption; risk is paging
  coherence (don't serve a half-evicted model) — a residency-state-machine invariant, build-time.

## RECOMMENDATION (process — this is the 3rd sizing turn)
We've sized enough to know the cargo: **distinct-model tiered-residence paging is the differentiated lever; the
binding risk is aggregate-PCIe-under-correlated-bursts + INT4 quality.** Further open-ended sizing has
diminishing returns. The honest next step is **build the distinct-model pager WITH instrumentation, targeting the
binding constraint first** (instrument cold-miss rate under correlated bursts + the PCIe queue), and let the build
surface the real numbers — OR stop. Not a fourth sizing. No code changed; anchors unchanged. Anil picks: build-
with-instrumentation (recommended) or stop.
