#!/usr/bin/env python3
"""
CIPHER in-process multi-model engine -- INCREMENT 1: CIPHER-owned single-model graph-decode over the pager.

DRIVER-LEVEL, CIPHER-owned: weights live in CIPHER's pager cuMemMap region (CIPHER's pluggable allocator); CIPHER
captures + replays the static-KV decode graph over them. NO vLLM in the serving path (vLLM is validator only). NO
vLLM CuMemAllocator / cudagraph-monitor singletons -> the in-process-composition wall does not apply.

This increment = CORRECTNESS milestone (per-agent KL=0; capture composes with pager residency). It needs NO .so
source change: it composes two proven driver-level bricks -- the pager substrate (cipher_rt_pager.c, default-OFF,
deployed-anchor-unchanged) and today's manual static-KV capture. The legacy CIPHER compute actuators
(Koopman/DET-SVD via cusolver) are NOT torch.cuda.graph-capture-safe; the engine serving path does not use them
(run with CIPHER_RT_DISABLE_AUTO_INIT=1), so they are out of scope here.

LIFETIME CO-OWNERSHIP (the use-after-free lesson): one PagerGraphModel object owns {region id, MemPool, StaticCache,
CUDAGraph, static io tensors}. If the pool/region frees while the captured graph holds pointers into pager VA, the
graph reads freed memory -> the exact dangling-cache OOB this arc keeps hitting. All are instance attributes, freed
together.

INTERFACE (increment 1): load() -> capture() -> serve_burst().  evict()/restore() are the pager residency hooks the
multi-model router (increment 3) will drive. serve_burst is a MONOTONIC burst (the empirically-deterministic replay
vehicle; a re-run single step aliases the graph's private pool and is noisy -- do not build comparators on it).
"""
import ctypes, os, torch
from torch.cuda.memory import CUDAPluggableAllocator
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

DEFAULT_SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"

class _St(ctypes.Structure):
    _fields_=[("cold_miss",ctypes.c_ulong),("pagein_cnt",ctypes.c_ulong),("evict_cnt",ctypes.c_ulong),
              ("state",ctypes.c_int),("ref",ctypes.c_int),("pages_in_flight",ctypes.c_int),
              ("used_bytes",ctypes.c_ulong),("base_va",ctypes.c_ulonglong),
              ("peak_live",ctypes.c_ulong),("cur_live",ctypes.c_ulong)]

class CipherPager:
    """Thin owner of the CIPHER pager .so handle + pluggable allocator (driver-level substrate, Mem #24)."""
    def __init__(self, so_path=DEFAULT_SO):
        self.lib = ctypes.CDLL(so_path)
        L=self.lib
        L.cipher_pager_init.restype=ctypes.c_int
        L.cipher_pager_begin_load.restype=ctypes.c_int; L.cipher_pager_begin_load.argtypes=[ctypes.c_ulonglong,ctypes.c_size_t]
        L.cipher_pager_end_load.restype=ctypes.c_int; L.cipher_pager_end_load.argtypes=[ctypes.c_int]
        L.cipher_pager_page_out.restype=ctypes.c_int; L.cipher_pager_page_out.argtypes=[ctypes.c_int]
        L.cipher_pager_page_in.restype=ctypes.c_int; L.cipher_pager_page_in.argtypes=[ctypes.c_int]
        L.cipher_pager_live_cksum.restype=ctypes.c_ulonglong; L.cipher_pager_live_cksum.argtypes=[ctypes.c_int]
        L.cipher_pager_get_stats.argtypes=[ctypes.c_int,ctypes.POINTER(_St)]
        self.alloc = CUDAPluggableAllocator(so_path,"cipher_pager_malloc","cipher_pager_free")
        assert L.cipher_pager_init()==0, "cipher_pager_init failed"
        torch.cuda.init(); _=torch.zeros(1,device="cuda")
    def stats(self, rid):
        s=_St(); self.lib.cipher_pager_get_stats(rid, ctypes.byref(s)); return s

class PagerGraphModel:
    """One model: weights in a pager region + a captured static-KV decode graph over them. Co-owns all lifetimes."""
    def __init__(self, pager: CipherPager, model_path, model_key=0xC1):
        self.pg=pager; self.path=model_path; self.key=model_key
        self.tok=AutoTokenizer.from_pretrained(model_path); self.V=self.tok.vocab_size
        self.rid=None; self.pool=None; self.m=None
        self.cache=None; self.graph=None; self.sin=None; self.spos=None; self.slog=None
        self.Plen=None; self.Lc=None; self.t1=None; self.dev=None
    # ---- residency (driver-level; the router will drive these in increment 3) ----
    def load(self):
        cpu=AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16)
        pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
        self.rid=self.pg.lib.cipher_pager_begin_load(self.key, int(pb*1.10)+(256<<20))
        self.pool=torch.cuda.MemPool(self.pg.alloc.allocator())
        with torch.cuda.use_mem_pool(self.pool):
            cpu.to("cuda"); torch.cuda.synchronize()
        self.pg.lib.cipher_pager_end_load(self.rid)
        self.m=cpu.eval()
        s=self.pg.stats(self.rid)
        inreg=sum(1 for _,p in self.m.named_parameters() if s.base_va<=p.data_ptr()<s.base_va+s.used_bytes)
        tot=sum(1 for _ in self.m.named_parameters())
        assert inreg==tot, f"routing precondition failed: {inreg}/{tot} weights in pager region"
        return self
    def evict(self):  return self.pg.lib.cipher_pager_page_out(self.rid)   # RESIDENT->EVICTED (copy-free unmap)
    def restore(self):return self.pg.lib.cipher_pager_page_in(self.rid)    # EVICTED->RESIDENT (remap warm bytes, same VA)
    def region_cksum(self): return self.pg.lib.cipher_pager_live_cksum(self.rid)
    def base_va(self): return self.pg.stats(self.rid).base_va
    def _clamp(self,t): return int(max(0,min(self.V-1,int(t))))
    # ---- capture (CIPHER owns the graph; no vLLM monitor) ----
    @torch.no_grad()
    def capture(self, prompt, max_new):
        pids=self.tok(prompt, return_tensors="pt").input_ids.cuda()
        self.dev=pids.device; self.Plen=pids.shape[1]; self.Lc=self.Plen+max_new+64
        def prefill(c): return self.m(pids, cache_position=torch.arange(self.Plen,device=self.dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
        self._prefill=prefill; self._pids=pids
        self.cache=StaticCache(config=self.m.config, max_cache_len=self.Lc); self.t1=prefill(self.cache)
        self.sin=torch.zeros(1,1,dtype=torch.long,device=self.dev); self.sin.fill_(self._clamp(self.t1))
        self.spos=torch.tensor([self.Plen],device=self.dev)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            o=self.m(self.sin, cache_position=self.spos, past_key_values=self.cache, use_cache=True); self.slog=o.logits
        return self
    @torch.no_grad()
    def eager_burst(self, n):  # reference (eager-static), greedy
        c=StaticCache(config=self.m.config, max_cache_len=self.Lc); nt=self._prefill(c); out=[]
        for i in range(n):
            out.append(self._clamp(nt))
            nt=self.m(torch.tensor([[self._clamp(nt)]],device=self.dev), cache_position=torch.tensor([self.Plen+i],device=self.dev),
                      past_key_values=c, use_cache=True).logits[0,-1].argmax()
        return out
    # ---- serve (CIPHER-owned graph-decode; monotonic burst = the deterministic replay vehicle) ----
    @torch.no_grad()
    def serve_burst(self, n):
        toks=[self._clamp(self.t1)]; self.sin.fill_(self._clamp(self.t1)); self.spos.fill_(self.Plen)
        for _ in range(n-1):
            self.graph.replay(); torch.cuda.synchronize()
            nt=self._clamp(self.slog[0,-1].argmax()); toks.append(nt); self.sin.fill_(nt); self.spos.add_(1)
        return toks
