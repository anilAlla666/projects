#!/usr/bin/env python3
# STEP 0 -- analytical cost floor for the Freivalds check, written BEFORE any measurement.
# Two floors per shape:
#   (A) FLOP floor (the spec's table): the added MACs as a fraction of the GEMM's M*K*N MACs.
#       u=W*g -> K*N MACs = 1/M ;  v=A*u -> M*K MACs = 1/N ;  Cg=C*g -> M*N MACs = 1/K.
#   (B) BANDWIDTH-aware floor (the REAL prediction): each check is a matrix-vector product
#       with arithmetic intensity ~1 MAC/element, i.e. MEMORY-BOUND. Its time is governed by
#       bytes-moved / HBM-bandwidth, NOT by FLOPs. So time fraction = (bytes/BW) / T_gemm,
#       which is I_eff * (FLOP fraction), where I_eff = GEMM_FLOPS / BW (the roofline ridge).
#       This is the same trap that sank fused-ABFT: the FLOP floor (0.05-0.10%) is two orders
#       optimistic because the operand re-reads are exposed bandwidth, not hidden compute.
import json

# real Mistral-7B prefill shapes, M=2048 (token block)
SHAPES = [
    ("k/v_proj",  2048, 4096, 1024),
    ("q/o_proj",  2048, 4096, 4096),
    ("down_proj", 2048,14336, 4096),
    ("gate/up_proj",2048,4096,14336),
]
# measured bare-GEMM times from step_b_prefill_build/base_result.json (this machine, locked 1200MHz)
BASE = {  # name -> (cutlass_us, cublas_us)
    "v_proj":   (87.84, 47.20),     # == k/v shape N=1024
    "down_proj":(500.29,420.38),
    "gate_proj":(602.11,426.83),
}
# H100 SXM HBM3 peak ~3.35 TB/s; a well-tuned gemv realizes ~2.8-3.2.  Use 3.0 for the floor.
BW = 3.0e12            # bytes/s, realistic gemv bandwidth
PEAK_FP16 = 990e12     # H100 fp16 tensor-core peak FLOP/s

def gemm_us(name, M, K, N):
    # use measured where we have it; else scale by FLOPs from gate (same K/M class)
    if name=="k/v_proj":   return BASE["v_proj"]
    if name=="q/o_proj":
        # q/o has N=4096, K=4096; interpolate from gate(N=14336) & v(N=1024) is unsafe -> mark None, measured later
        return (None, None)
    if name=="down_proj":  return BASE["down_proj"]
    if name=="gate/up_proj":return BASE["gate_proj"]
    return (None,None)

print("="*108)
print("STEP 0  FREIVALDS ANALYTICAL COST FLOOR   (BW=%.1f TB/s, fp16 peak=%.0f TFLOP/s, ridge I*=%.0f)"
      % (BW/1e12, PEAK_FP16/1e12, PEAK_FP16/BW))
print("  check: verify C[M,N]=A[M,K]@W[N,K]^T via random g[N]:  A@(W^T@g) ?= C@g")
print("="*108)
hdr = ("%-13s %5s %6s %6s | %7s %7s %7s %7s | %8s %8s | %8s %8s %8s | %9s")
print(hdr % ("shape","M","K","N","1/M(u)","1/N(v)","1/K(Cg)","FLOOR%","T_cut us","T_cub us",
             "u bw%","v bw%","Cg bw%","BW-FLOOR%"))
print("-"*108)
rows=[]
for name,M,K,N in SHAPES:
    f_u, f_v, f_cg = 1.0/M, 1.0/N, 1.0/K          # FLOP fractions
    flop_floor = 100.0*max(f_u,f_v,f_cg)          # spec's dominant-term floor, %
    t_cut,t_cub = gemm_us(name,M,K,N)
    # bytes moved by each check matvec (fp16 operands)
    by_u  = N*K*2     # read all of W
    by_v  = M*K*2     # read all of A
    by_cg = M*N*2     # read all of C
    # time of each check matvec at BW (us)
    tu, tv, tcg = by_u/BW*1e6, by_v/BW*1e6, by_cg/BW*1e6
    # bandwidth-floor % of GEMM time (use CUTLASS base where known)
    def pct(t_check, t_gemm): return (100.0*t_check/t_gemm) if t_gemm else float('nan')
    u_pct  = pct(tu, t_cut);  v_pct = pct(tv, t_cut);  cg_pct = pct(tcg, t_cut)
    bw_floor_cut = pct(tu+tv+tcg, t_cut)
    bw_floor_cub = pct(tu+tv+tcg, t_cub)
    print(hdr % (name,M,K,N,
                 f"{100*f_u:.3f}",f"{100*f_v:.3f}",f"{100*f_cg:.3f}",f"{flop_floor:.3f}",
                 (f"{t_cut:.1f}" if t_cut else "  ?"),(f"{t_cub:.1f}" if t_cub else "  ?"),
                 (f"{u_pct:.2f}" if t_cut else "?"),(f"{v_pct:.2f}" if t_cut else "?"),
                 (f"{cg_pct:.2f}" if t_cut else "?"),
                 (f"{bw_floor_cut:.2f}" if t_cut else "?")))
    rows.append(dict(name=name,M=M,K=K,N=N,
                     flop_floor_pct=flop_floor, f_u=100*f_u,f_v=100*f_v,f_cg=100*f_cg,
                     cutlass_us=t_cut,cublas_us=t_cub,
                     bytes_u=by_u,bytes_v=by_v,bytes_cg=by_cg,
                     t_u_us=tu,t_v_us=tv,t_cg_us=tcg,
                     u_pct_cut=u_pct,v_pct_cut=v_pct,cg_pct_cut=cg_pct,
                     bw_floor_cut_pct=bw_floor_cut, bw_floor_cub_pct=bw_floor_cub))
print("-"*108)
print("KEY: FLOOR%% = spec's FLOP floor (dominant of 1/M,1/N,1/K). BW-FLOOR%% = layout(i) 3-gemv")
print("     bandwidth prediction vs CUTLASS base.  v bw%% alone = the IRREDUCIBLE term (Cg free in")
print("     epilogue, u free/amortized) = best-case fully-fused Freivalds.")
print()
print("IRREDUCIBLE (best-case fused) = v=A@u term only (reads A once), vs CUTLASS base & cuBLAS base:")
for r in rows:
    if r['cutlass_us']:
        vc = 100*r['t_v_us']/r['cutlass_us']; vb = 100*r['t_v_us']/r['cublas_us']
        print(f"   {r['name']:13s} v=A@u read A={r['bytes_v']/1e6:5.1f}MB -> {r['t_v_us']:5.1f}us  "
              f"= {vc:5.2f}% of CUTLASS / {vb:5.2f}% of cuBLAS   {'PASS' if vb<3 else 'FAIL'}(vs cuBLAS)")
json.dump(rows, open('step0_floor.json','w'), indent=1)
print("\nwrote step0_floor.json")
