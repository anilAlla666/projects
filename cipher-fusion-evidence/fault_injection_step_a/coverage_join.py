import json, math, statistics
A1 = json.load(open('a1_results.json'))
recs = A1['results']
A2 = json.load(open('a2_real.json'))
thr = A2['threshold']; e2e = A2['e2e']; cap = A2['captured']

# map op ptype -> (shape-tag, T_fp32_max, T_fp32_p999)
ptype2tag = {'q_proj':'q/o','o_proj':'q/o','k_proj':'k/v','v_proj':'k/v',
             'gate_proj':'gate/up','up_proj':'gate/up','down_proj':'down','lm_head':'lm_head'}
def T_of(pt, key='T_fp32_max'): return thr[ptype2tag[pt]][key]

# ---- coverage join ----
def caught(r, key='T_fp32_max'):
    if r['naninf']: return True
    return r['abs_delta'] > T_of(r['ptype'], key)

harm = [r for r in recs if r['harmful']]
ben  = [r for r in recs if not r['harmful']]
print(f"total injections {len(recs)}  harmful {len(harm)}  benign {len(ben)}")

# per-op coverage and margins
print("\n%-10s %6s %7s %9s %9s %9s %9s" % ("op","#harm","#caught","cov%","min|d|","Tfp32","margin"))
ops = ['o_proj','down_proj','v_proj','up_proj','gate_proj','lm_head','q_proj','k_proj']
tot_h=tot_c=0
for op in ops:
    h=[r for r in harm if r['ptype']==op]
    c=[r for r in h if caught(r)]
    finite=[r['abs_delta'] for r in h if not r['naninf']]
    mind = min(finite) if finite else float('inf')
    T=T_of(op); tot_h+=len(h); tot_c+=len(c)
    print("%-10s %6d %7d %8.1f%% %9.3g %9.4g %8.1fx" % (op,len(h),len(c),100*len(c)/max(len(h),1),mind,T,(mind/T if T>0 else float('inf'))))
print("%-10s %6d %7d %8.1f%%" % ("TOTAL",tot_h,tot_c,100*tot_c/tot_h))

# false positives on clean (benign): would a benign flip with |d|>T be flagged? (correct behavior, costs a re-exec)
fp=[r for r in ben if caught(r)]
print(f"\nbenign flips that exceed their shape T (correct ABFT 'corruption detected', extra re-exec not a correctness error): {len(fp)}/{len(ben)} = {100*len(fp)/len(ben):.2f}%")
# but the REAL clean-noise false-positive rate is governed by T_fp32_max being the MAX clean residual over 922 rows => 0 clean FP by construction
print("clean-noise false positives: T set = per-shape max clean residual over 922 rows => 0 by construction (margin to min-harmful below)")

# tightest margin overall
allmarg=[(op, min([r['abs_delta'] for r in harm if r['ptype']==op and not r['naninf']]+[float('inf')])/T_of(op)) for op in ops]
allmarg=[(o,m) for o,m in allmarg if math.isfinite(m)]
print("tightest min-harm/T margin:", min(allmarg,key=lambda x:x[1]))

# ---- overhead summary ----
print("\n=== OVERHEAD (A2-B eager end-to-end, real Mistral-7B, 225 linears) ===")
print(f"prefill: off {e2e['prefill_off_ms']:.2f}ms  verify_compute {e2e['prefill_verify_compute_ms']:.2f}ms (+{e2e['prefill_vc_ovh']*100:.1f}%)  verify_full {e2e['prefill_verify_full_ms']:.2f}ms (+{e2e['prefill_vf_ovh']*100:.1f}%)")
print(f"decode : off {e2e['decode_off_ms']:.2f}ms  verify_compute {e2e['decode_verify_compute_ms']:.2f}ms (+{e2e['decode_vc_ovh']*100:.1f}%)  verify_full {e2e['decode_verify_full_ms']:.2f}ms (+{e2e['decode_vf_ovh']*100:.1f}%)")
print(f"\n=== A2-C captured verify (225 row-sum ops, M=1 decode shapes, ONE cuda graph) ===")
print(f"captured replay {cap['captured_verify_ms']:.3f}ms   eager loop {cap['eager_verify_ms']:.3f}ms   capture speedup {cap['eager_verify_ms']/cap['captured_verify_ms']:.2f}x")
# rigorous lower bound: captured verify / eager decode (captured decode <= eager decode, so true frac is HIGHER)
lb = cap['captured_verify_ms']/e2e['decode_off_ms']
print(f"captured-verify overhead LOWER BOUND = captured_verify / EAGER decode = {cap['captured_verify_ms']:.3f}/{e2e['decode_off_ms']:.2f} = {lb*100:.1f}%  (true frac higher: captured decode < eager decode)")
# per-op captured floor
print(f"per-op captured verify floor = {cap['captured_verify_ms']/cap['n_ops']*1000:.2f} us/op  (kernel-count/dispatch floor, not FLOPs)")

# ---- FLOP floor of a FUSED epilogue checksum (analytical, for the re-scope path) ----
print("\n=== FUSED-epilogue checksum FLOP floor (analytical, NOT measured) ===")
HID,INTER,KV,VOCAB=4096,14336,1024,32000
# reference matvec A[M,K]@wcol[K] = M*K MACs vs GEMM M*K*N MACs => factor 1/N
for tag,(K,N) in {'q/o':(HID,HID),'k/v':(HID,KV),'gate/up':(HID,INTER),'down':(INTER,HID),'lm_head':(HID,VOCAB)}.items():
    print(f"  {tag:9s} K={K:5d} N={N:5d}: reference-matvec FLOPs = 1/N of GEMM = {100.0/N:.4f}%  (row-sum of C is a free epilogue byproduct)")
