# V0 G-O1 ENGINE INCREMENT 3: the SINGLETON-WALL test — M distinct models, one process

**2026-06-03. SINGLETON-WALL DISSOLVED (tagged). Residency-swap: binding term NARROWED, NOT tagged (honest split
per discipline).** NO CIPHER `.so` source change; deployed anchor 1f305ce6 + staging `.so` UNCHANGED; inc-1 module
`cipher_engine.py` BYTE-IDENTICAL (a cache-pool-isolation experiment was tried and REVERTED), inc-1 gate still
PASSES (non-regression). Run with `CIPHER_RT_DISABLE_AUTO_INIT=1`. Artifacts: `pager_g3_coresidence_probe.py`,
`pager_g3_decisive.py`, `pager_g3_alias_diag.py`, `cipher_router.py`/`cipher_router_gate.py` (WIP, see swap caveat).

## VERDICT 1 — SINGLETON-WALL DISSOLVED (SOLID, the moat, TAGGED)

M distinct models, each in its OWN pager region (CIPHER's pluggable allocator, driver VMM boundary) with its OWN
captured static-KV graph (in-process torch.cuda.graph, no vLLM cudagraph-monitor singleton), ALL in ONE process,
each replaying **KL=0** — **with NO workaround**:

| co-residence | per-model KL=0 | HBM |
|---|---|---|
| M=2 (Qwen2-7B + Llama-3.1-8B) | 64/64 each | 45.6GB free |
| M=3 (Qwen2-7B + Llama-3.1-8B + Llama-3.2-1B) | 64/64 each | 42.8GB free |

Distinct first-tokens per model (no gross cross-model interference). This is the make-or-break: M distinct captured
graphs coexist + replay correctly in one process. vLLM's CuMemAllocator + cudagraph-capture singletons would block
this (carried-forward prior finding [[cipher-go1-inprocess-composition]], not re-verified here). **The moat is real:
CIPHER owning the allocator (driver) + per-model capture (in-process) does what vLLM structurally cannot.**

DRIVER-LEVEL split (stated plainly): per-model region reserve/map/unmap = DRIVER-BOUNDARY (cuMemMap); per-model
graph capture+replay = IN-PROCESS. No overclaim.

## Diagnosed binding term: Mistral SWA capture (per-model, orthogonal to co-residence)

Mistral-7B-v0.1 (`sliding_window=4096`) fails static-KV capture even at M=1: graph-vs-eager logit max|Δ|=**6.75**
(not FP noise) from the first replay → a real capture bug, NOT a near-tie, NOT co-residence. Qwen2 (no SWA) and the
Llama family capture KL=0. So the singleton-wall test uses capture-correct distinct models; **Mistral's sliding-window
attention in the captured graph is a separate per-model capture-robustness item** (the inc-1 "per-model capture
robustness = real work" note). Not a scope-down of the wall test — a diagnosed model-architecture binding term.

## VERDICT 2 — RESIDENCY-SWAP at M>capacity: binding term NARROWED, NOT tagged

Routing + LRU residency swap (evict cold model `page_out`, restore requested `page_in`) at M>capacity surfaced a
PROGRESSIVE corruption: serve#1 KL=0 → serve#2 diverges → serve#3 `index_copy_` OOB across repeated/cross-model
page cycles. **Root-caused as far as: it is NOT a pager/weights bug.** The DECISIVE run (`pager_g3_decisive.py`):
- every resident model's `cipher_pager_live_cksum` is **OK (no drift)** from load-time across all cycles, incl. an
  evicted-and-restored model;
- after a diverging swap, an **EAGER decode on a FRESH cache (no captured graph) = 48/48 CORRECT**.
→ weights + VA are restored correctly under churn; the corruption is in the **carried captured-graph / static
prefill-KV**, NOT `page_in`. (`index_copy_` writes the StaticCache on torch's allocator, NOT the pager region.)

Ruled OUT as the sole cause: **cache aliasing.** Two models' StaticCaches were measured 72KB apart (overlap-risk);
isolating each model's cache in its own retained `MemPool` separated them to 16–32GB apart — but the swap corruption
**persisted identically** → cache-overlap is not the (sole) mechanism. The locus is the carried captured-graph +
static-KV reuse across cross-model residency churn; full root cause is OPEN.

**Re-capture-per-serve (re-prefill + fresh graph) MAKES the router green** — `cipher_router_gate.py`: per-request
KL=0 (exact 20/20), across-swap KL=0 9/9, swaps physically real 9/9 (HBM freed >0.5GB each), cross-model misroute
negative-control DETECTED, 4 distinct models in one process (K=2 resident, swap-on-miss). **BUT this is a MASKING fix**
(it recomputes both the graph and the KV, so it can't distinguish the bug from the fix) and folds a ~17–100ms capture
into every serve. Per discipline (do not fold an un-rooted workaround into the engine; do not tag a masked result),
**swap-correctness is NOT tagged.** The honest next step: close the carried-graph/static-KV root cause, then the
likely real fix is **re-prefill-only into a persistent graph** (the advisor's branch-1 prescription, since eager-fresh
is correct), preserving graph persistence at ~prefill cost — and scale-test REQ≥200.

## Density (from co-residence, the proven part)

M=3 distinct 7-8B+1B resident in 32.5GB → ~5 fp16 7-8B co-resident in 80GB (consistent with prior "5 distinct"
ceiling). The pager extends M beyond the resident-K via swap-on-miss; the per-swap cost is currently the
(un-rooted) re-capture tax, so the honest distinct-models-per-GPU at low swap rate ≈ resident capacity (~5 fp16 /
more at int4), with paging beyond pending the swap root-cause.

## Gate status (Mem #11)

1. **Correctness:** singleton-wall per-model KL=0 (M=2, M=3) — PASS, no workaround. Swap per-agent KL=0 only via the
   masking re-capture path — NOT accepted as tagged correctness.
2. **Singleton-wall verdict:** DISSOLVED (stated plainly, M-models-one-process evidence).
3. **Density:** ~5 fp16 7-8B co-resident; swap-beyond pending root-cause.
4. **NON-REGRESSION:** inc-1 module byte-identical + gate PASSES; inc-2 unaffected; OFF byte-identical + anchor
   1f305ce6 UNCHANGED (no `.so` change). Subprocesses os._exit-reaped.

## STOP

Singleton-wall (the moat) DISSOLVED + tagged. Residency-swap root cause is the open item before inc-4. The
carried-graph/static-KV corruption under cross-model churn must be closed (re-prefill-only persistent-graph fix,
then REQ≥200 scale) before swap density is product-claimable. Anil's call: close swap root-cause, or proceed.
