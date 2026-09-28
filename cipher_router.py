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
    # inc-3b-fix (honest, time-boxed): mechanism NOT characterized. EVIDENCED: a never-evicted model A (own
    # weights/KV/data_ptr all stable) is corrupted by OTHER models' residency churn -> corruption is GLOBAL, so
    # buffer-pinning (Q1) cannot help (REFUTED). NOT "any VMM op invalidates graphs": A stayed KL=0 after B,C were
    # page-OUT'd post-capture; corruption appeared only after a page-IN in the churn loop -- a narrower, un-isolated
    # restore/page-in correlation, NOT a named cause (do not over-claim; 5th attempt at this trigger). An epoch-gated
    # reuse (re-capture only on churn) was attempted and FAULTED; bug-vs-genuinely-unsafe not separated -> conservatively
    # re-capture PER SERVE. Empirically sustains KL=0 at REQ=200, no leak. Mechanism = documented debt for inc-4.
    def __init__(self, pager: CipherPager, specs, capacity_K, prompt, N):
        self.pg=pager; self.N=N; self.K=capacity_K; self.prompt=prompt
        self.models={}; self.solos={}; self.resident=[]
        self.swaps=0; self.hits=0; self.vmm_epoch=0; self.gepoch={}; self.recaptures=0
        for name,path,key in specs:
            e=PagerGraphModel(pager, path, model_key=key).load()
            e.capture(prompt, N); self.solos[name]=e.eager_burst(N)
            e.evict(); self.vmm_epoch+=1          # page_out is a VMM op
            self.models[name]=e; self.gepoch[name]=-1
        for name,_,_ in specs[:capacity_K]:
            self.models[name].restore(); self.vmm_epoch+=1; self.resident.append(name)  # warm (graph re-captured lazily on first serve)

    def _ensure_graph_valid(self, name):
        """Re-capture per serve. The epoch-gated reuse (re-capture only on vmm_epoch change) was TESTED and FAULTS:
        a graph reused between VMM ops still corrupts -> something beyond VMM ops also invalidates live graphs
        (e.g. other models' concurrent graph captures during swaps). So reuse is UNSAFE in practice and the tax is
        PER-SERVE (Q2). re-capture-per-serve sustains KL=0 at REQ=200 with no leak."""
        import time
        torch.cuda.synchronize(); t0=time.time(); self.models[name].capture(self.prompt, self.N); torch.cuda.synchronize()
        self.gepoch[name]=self.vmm_epoch; self.recaptures+=1; return (time.time()-t0)*1000

    def ensure_resident(self, name):
        if name in self.resident:
            self.resident.remove(name); self.resident.append(name); self.hits+=1
            return "hit", 0.0, None
        victim=None; freed=0.0
        if len(self.resident)>=self.K:
            victim=self.resident.pop(0)
            f0=torch.cuda.mem_get_info()[0]; self.models[victim].evict(); self.vmm_epoch+=1; f1=torch.cuda.mem_get_info()[0]
            freed=(f1-f0)/GB
        self.models[name].restore(); self.vmm_epoch+=1     # page_in is a VMM op -> invalidates ALL live graphs
        self.resident.append(name); self.swaps+=1
        return "swap", freed, victim

    def serve(self, name):
        import time
        kind,freed,victim=self.ensure_resident(name)
        cap_ms=self._ensure_graph_valid(name)              # re-capture ONLY if a VMM op happened since capture
        t0=time.time(); out=self.models[name].serve_burst(self.N); torch.cuda.synchronize(); dec_ms=(time.time()-t0)*1000
        return out, kind, freed, victim, cap_ms, dec_ms
