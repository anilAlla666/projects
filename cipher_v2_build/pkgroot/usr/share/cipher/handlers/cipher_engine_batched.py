#!/usr/bin/env python3
"""
CIPHER engine -- INCREMENT 2: batched wave serving over the pager (same-model coalescing).

Extends increment 1 (cipher_engine.PagerGraphModel, single-model graph-decode over the pager). A WaveServer holds
one pager-resident model and serves WAVES: a wave is up to B agents decoding in LOCKSTEP (one batched static-KV
captured graph). Mechanism facts established by probe (pager_g2_*):
  - batched static-KV capture over the pager = per-agent KL=0 modulo near-ties; ZERO cross-row contamination
    (co-tenant invariance dDelta=0.000 on TinyLlama + real 8B).
  - one captured graph CANNOT be reused across waves via eager re-prefill (corrupts the graph's private pool) ->
    each wave RE-CAPTURES after its prefill (the proven inc-1 prefill->capture->burst pattern, batched). Steady-state
    capture ~17ms/wave (TinyLlama); the first capture is dearer (warmup).
LOCKSTEP BOUNDARY (HF StaticCache index_copy_(2, cache_position,...) is batch-shared -> no per-row position): a wave
batches agents AT THE SAME sequence position. No mid-flight join (that needs paged attention = vLLM's core / inc 3+).
Within a wave, agents with shorter generations finish early -> their slots burn compute (work-proportional waste).

Correctness = NO-CONTAMINATION + greedy-match-modulo-near-ties (Anil-ratified): a per-agent divergence-vs-solo is
benign iff its FIRST diff is a sub-delta near-tie (solo top1-top2 margin < the batch-shape logit delta ~0.02).
Run with CIPHER_RT_DISABLE_AUTO_INIT=1 (legacy compute actuators are not capture-safe).
"""
import time, torch
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel

TIE_DELTA = 0.05   # batch-shape logit delta ~0.0156 (TinyLlama)/~0.02 (8B); flip benign iff solo margin below this

class WaveServer:
    """One pager-resident model; serves lockstep batched waves by re-capturing the decode graph per wave."""
    def __init__(self, pager: CipherPager, model_path, model_key=0xC1):
        self.base=PagerGraphModel(pager, model_path, model_key).load()   # weights -> pager region (inc-1 brick)
        self.m=self.base.m; self.tok=self.base.tok; self.V=self.base.V; self.dev="cuda"
        self.cfg=self.m.config
        self._warmed=False
    def _clamp(self,t): return int(max(0,min(self.V-1,int(t))))

    @torch.no_grad()
    def serve_wave(self, prompt_rows, gen_lens):
        """prompt_rows: list of B 1-D LongTensors, all length P (lockstep). gen_lens: per-agent token counts.
        Returns (per_agent_tokens, timing). Decodes max(gen_lens) lockstep steps; early-finished slots ignored."""
        B=len(prompt_rows); P=prompt_rows[0].shape[0]; G=max(gen_lens); Lc=P+G+8
        pids=torch.stack([r.to(self.dev) for r in prompt_rows])              # [B,P]
        cache=StaticCache(config=self.cfg, max_cache_len=Lc)
        first=self.m(pids, cache_position=torch.arange(P,device=self.dev), past_key_values=cache, use_cache=True).logits[:,-1].argmax(dim=-1)
        sin=torch.zeros(B,1,dtype=torch.long,device=self.dev); sin.copy_(first.view(B,1)); spos=torch.tensor([P],device=self.dev)
        g=torch.cuda.CUDAGraph()
        torch.cuda.synchronize(); tc=time.time()
        with torch.cuda.graph(g):
            slog=self.m(sin, cache_position=spos, past_key_values=cache, use_cache=True).logits
        torch.cuda.synchronize(); cap_ms=(time.time()-tc)*1000
        seqs=[[self._clamp(first[b])] for b in range(B)]
        sin.copy_(first.view(B,1)); spos.fill_(P)
        td=time.time()
        for _ in range(G-1):
            g.replay(); torch.cuda.synchronize(); nxt=slog[:,-1].argmax(dim=-1)
            for b in range(B): seqs[b].append(self._clamp(nxt[b]))
            sin.copy_(nxt.view(B,1)); spos.add_(1)
        dec_ms=(time.time()-td)*1000
        del g, cache, slog
        out=[seqs[b][:gen_lens[b]] for b in range(B)]                        # trim each agent to its own gen_len
        waste=sum(G-gl for gl in gen_lens)                                   # slot-steps burned by early-finishers
        return out, {"B":B,"P":P,"G":G,"cap_ms":cap_ms,"dec_ms":dec_ms,"waste_steps":waste,"tot_steps":B*G}

    @torch.no_grad()
    def solo(self, prompt_row, gen_len):
        """batch-1 reference: greedy tokens + per-step top1-top2 margins (for the near-tie classification)."""
        P=prompt_row.shape[0]; Lc=P+gen_len+8
        c=StaticCache(config=self.cfg, max_cache_len=Lc)
        lg=self.m(prompt_row.unsqueeze(0).to(self.dev), cache_position=torch.arange(P,device=self.dev), past_key_values=c, use_cache=True).logits[0,-1]
        out=[]; mar=[]
        for i in range(gen_len):
            top2=lg.topk(2).values; mar.append(float(top2[0]-top2[1]))
            nt=int(lg.argmax()); out.append(self._clamp(nt))
            lg=self.m(torch.tensor([[self._clamp(nt)]],device=self.dev), cache_position=torch.tensor([P+i],device=self.dev), past_key_values=c, use_cache=True).logits[0,-1]
        return out, mar

def classify(agent_tokens, solo_tokens, solo_margins):
    """ratified correctness: 'exact' | 'tie@k' (benign near-tie) | 'FAULT@k' (real divergence)."""
    fd=next((i for i,(x,y) in enumerate(zip(agent_tokens,solo_tokens)) if x!=y), -1)
    if fd<0: return "exact", -1, 0.0
    m=solo_margins[fd]
    return ("tie" if m<TIE_DELTA else "FAULT"), fd, m
