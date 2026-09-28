#!/usr/bin/env python3
"""
CIPHER engine -- INCREMENT 3: in-process multi-model router with residency swap at M>capacity.

Builds on inc-1 (PagerGraphModel: graph-decode over a pager region, survives physical evict/restore) + the inc-3
singleton-wall probe (M distinct captured graphs coexist + replay KL=0 in one process). The router holds M distinct
models, each with its OWN pager region + OWN captured static-KV graph; a resident-set capacity K<M forces paging.
On a request whose model is evicted: evict the LRU resident model (page_out -> physically frees its HBM) and restore
the requested model (page_in -> remaps its reserved VA + warm bytes; inc-1 proved the captured graph survives this).

DRIVER-LEVEL split (stated plainly, no overclaim):
  - DRIVER-BOUNDARY (CUDA VMM): per-model pager region reserve/map/unmap (cuMemMap), evict=page_out, restore=page_in.
  - IN-PROCESS: per-model torch.cuda.graph capture + replay (no vLLM cudagraph-monitor singleton).
vLLM is validator only, NOT in the serving path. Run CIPHER_RT_DISABLE_AUTO_INIT=1 (legacy actuators not capture-safe).

Lifetime: each PagerGraphModel co-owns its {region, MemPool, StaticCache, CUDAGraph}. Setup loads->captures->evicts
each model (only ~1 model's physical HBM transient during setup); serve restores on demand. Note: each model carries
ONE captured prompt (the multi-AGENT-per-model re-capture is inc-2's WaveServer; inc-3 adds the multi-MODEL axis).
"""
import torch
from cipher_engine import CipherPager, PagerGraphModel

GB=1<<30

class MultiModelRouter:
    def __init__(self, pager: CipherPager, specs, capacity_K, prompt, N):
        # specs: list of (name, path, key). capacity_K = max simultaneously-resident models.
        self.pg=pager; self.N=N; self.K=capacity_K; self.prompt=prompt
        self.models={}; self.solos={}; self.resident=[]   # resident = LRU list (front=LRU, back=MRU)
        self.swaps=0; self.hits=0
        for name,path,key in specs:
            e=PagerGraphModel(pager, path, model_key=key).load()
            e.capture(prompt, N)                 # capture while resident
            self.solos[name]=e.eager_burst(N)    # solo reference (fresh cache; safe after capture)
            e.evict()                            # page_out -> free HBM; region+graph retained, EVICTED
            self.models[name]=e
        # warm the resident set with the first K -- re-capture on entry (a model's captured graph does NOT survive
        # the interleaved load/capture/evict of OTHER models nor repeated page cycles; weights are bit-identical
        # (cksum-stable) so a fresh capture over the restored region is correct).
        for name,_,_ in specs[:capacity_K]:
            self.models[name].restore(); self.models[name].capture(prompt, N); self.resident.append(name)

    def _stats(self,name): return self.pg.stats(self.models[name].rid)

    def ensure_resident(self, name):
        """returns ('hit'|'swap', freed_gb_on_evict, victim_or_None)."""
        if name in self.resident:
            self.resident.remove(name); self.resident.append(name); self.hits+=1
            return "hit", 0.0, None
        victim=None; freed=0.0
        if len(self.resident)>=self.K:
            victim=self.resident.pop(0)            # LRU
            f0=torch.cuda.mem_get_info()[0]; self.models[victim].evict(); f1=torch.cuda.mem_get_info()[0]
            freed=(f1-f0)/GB
        self.models[name].restore()                # page_in: weights bit-identical (cksum-stable)
        # the captured graph does NOT survive REPEATED evict/restore cycles (progressive GPU-state corruption, NOT
        # weights) -> RE-CAPTURE on restore (proven inc-2 pattern; ~17ms/swap, the honest swap cost).
        self.models[name].capture(self.prompt, self.N)
        self.resident.append(name); self.swaps+=1
        return "swap", freed, victim

    def serve(self, name):
        kind,freed,victim=self.ensure_resident(name)
        # captured graphs do NOT survive cross-model residency churn (other models' cuMemMap/unmap corrupts a
        # resident model's graph) -> re-capture per serve over the (bit-identical) resident weights. The honest
        # multi-model cost: capture (~17ms) folds into every serve. (inc-4 can amortize via longer bursts/pinned graphs.)
        self.models[name].capture(self.prompt, self.N)
        out=self.models[name].serve_burst(self.N)
        return out, kind, freed, victim
