#!/usr/bin/env python3
"""
A2 (i) — DETECTION-COST microbenchmark on the real intercepted GEMM shapes.

For each distinct Mistral-7B intercepted shape (M,K,N) at prefill (M=2048) and
decode (M=1), measure cublas GEMM time vs Huang-Abraham ABFT verify time for:
  - ROW-sum check via precomputed weight col-checksum wcol = W.sum(0):  reads A,C only
    (the task's "A . checksum(B)" direction; cheap for decode by construction)
  - COL-sum check via activation col-checksum acol = A.sum(0):          reads W,C
    (the task's "checksum(A) . B" direction; full weight re-read, bad for decode)
  - BLOCK row-sum check (G column blocks): tighter per-sum noise floor at G x ref cost
Verify cost split into COMPUTE (resid tensor ready; capturable in a CUDA graph) and
FLAG (.item() host sync; capture-illegal -> must be deferred outside the graph).
Timing only -> synthetic same-shape tensors. Clock locked. Events, warmup, median.
"""
import json, time, statistics, torch

OUT="/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a2_micro.json"
HID,INTER,KV,VOCAB=4096,14336,1024,32000
SHAPES=[  # name, K(in), N(out)
 ("q/o_proj",   HID, HID),
 ("k/v_proj",   HID, KV),
 ("gate/up_proj",HID, INTER),
 ("down_proj",  INTER, HID),
 ("lm_head",    HID, VOCAB),
]
REGIMES=[("prefill",2048),("decode",1)]
dev="cuda"; torch.backends.cuda.matmul.allow_tf32=False

def t_event(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)  # ms

def main():
    rows=[]
    for rname,M in REGIMES:
        for sname,K,N in SHAPES:
            A=torch.randn(M,K,device=dev,dtype=torch.float16)
            W=torch.randn(N,K,device=dev,dtype=torch.float16)   # [out,in]
            Wt=W.t().contiguous()
            wcol=W.sum(0).float()              # [K] precomputed weight col-checksum (fp32)
            G=32 if N>=4096 else 8
            # block col-checksum of weight: [K,G]
            pad=(-N)%G; Npad=N+pad
            Wb=torch.cat([W, torch.zeros(pad,K,device=dev,dtype=W.dtype)],0) if pad else W
            wblk=Wb.view(G, Npad//G, K).sum(1).float().t().contiguous()  # [K,G]
            ones_M=torch.ones(M,device=dev,dtype=torch.float16)

            gemm=lambda: torch.matmul(A,Wt)
            C=torch.matmul(A,Wt)
            T=1.0  # placeholder threshold for flag timing

            def v_row_compute():
                r=(A.float()@wcol)            # [M]
                s=C.float().sum(1)            # [M]
                resid=(r-s).abs()
                return resid
            def v_row_full():
                resid=v_row_compute()
                return bool((resid.max()>T).item())   # host sync = capture-illegal flag
            def v_col_compute():
                acol=A.float().sum(0)         # [K]
                ref=acol@Wt.float()           # [N]
                s=C.float().sum(0)            # [N]
                return (ref-s).abs()
            def v_blk_compute():
                Cb=torch.cat([C, torch.zeros(M,pad,device=dev,dtype=C.dtype)],1) if pad else C
                sb=Cb.float().view(M,G,Npad//G).sum(2)   # [M,G]
                rb=A.float()@wblk                        # [M,G]
                return (rb-sb).abs()

            tg=t_event(gemm)
            trc=t_event(v_row_compute)
            trf=t_event(v_row_full)
            tcc=t_event(v_col_compute)
            tbk=t_event(v_blk_compute)
            rows.append(dict(regime=rname,shape=sname,M=M,K=K,N=N,
                gemm_ms=tg,
                row_compute_ms=trc, row_full_ms=trf,
                col_compute_ms=tcc, blk_compute_ms=tbk, blk_G=G,
                row_compute_ovh=trc/tg, row_full_ovh=trf/tg,
                col_compute_ovh=tcc/tg, blk_compute_ovh=tbk/tg))
            print(f"{rname:8s} {sname:13s} M={M:5d} gemm={tg:.4f}ms "
                  f"row_c={trc/tg*100:6.2f}% row_full={trf/tg*100:7.1f}% "
                  f"col_c={tcc/tg*100:7.1f}% blk={tbk/tg*100:6.2f}%",flush=True)
            del A,W,Wt,C; torch.cuda.empty_cache()
    sm=torch.cuda.clock_rate() if hasattr(torch.cuda,'clock_rate') else None
    json.dump(dict(meta=dict(note="GEMM vs ABFT verify, synthetic shapes, locked clock",
                             regimes=REGIMES),rows=rows),open(OUT,"w"))
    print("wrote",OUT,flush=True)

if __name__=="__main__": main()
