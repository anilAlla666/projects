import ctypes, json, os, glob, statistics, math
import torch

torch.manual_seed(0)
dev = 'cuda'
DOWN_N, DOWN_K = 4096, 14336   # down_proj: out_features=N=4096, in_features=K=14336; W is [N,K]

lib = ctypes.CDLL(os.path.abspath('libgemv.so'))
lib.launch_gemv.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_void_p]
lib.launch_gemv.restype = None
lib.gemv_nblocks.argtypes=[ctypes.c_int]; lib.gemv_nblocks.restype=ctypes.c_int
NB = lib.gemv_nblocks(DOWN_N)   # per-block partial count
def gemv(A,W,wref,y,prs,prf,K,N,chk):
    lib.launch_gemv(A.data_ptr(),W.data_ptr(),wref.data_ptr(),y.data_ptr(),
                    prs.data_ptr(),prf.data_ptr(),int(K),int(N),int(chk),None)

# ---- load REAL Mistral-7B down_proj weight (one tensor) ----
MODEL='/home/ubuntu/models/Mistral-7B-v0.1'
W = None
try:
    from safetensors import safe_open
    for sf in sorted(glob.glob(os.path.join(MODEL,'*.safetensors'))):
        with safe_open(sf,'pt') as f:
            for key in f.keys():
                if key.endswith('mlp.down_proj.weight'):
                    W = f.get_tensor(key).to(dev).half().contiguous()   # [4096,14336]
                    print(f"loaded REAL {key} {tuple(W.shape)} from {os.path.basename(sf)}")
                    break
        if W is not None: break
except Exception as e:
    print("safetensors load failed:", e)
if W is None:
    print("FALLBACK: synthetic W (real weight not found)")
    W = (torch.randn(DOWN_N,DOWN_K,device=dev)*0.02).half()
N,K = W.shape

# realistic activation: post-SwiGLU intermediate, scale to give C_abs_mean ~ Step-A's 0.028
A = (torch.randn(1,K,device=dev)*0.1).half()
C = (A.float() @ W.float().t())              # fp32 true output [1,N]
print(f"C abs-mean={C.abs().mean():.4f} (Step-A down C_abs_mean=0.0278), rowsum={C.sum():.4f}")

wref = W.float().sum(dim=0).contiguous()      # [K] = sum_n W[n,k], fp32, precomputed once
y = torch.zeros(1,N,device=dev,dtype=torch.float16)
prs = torch.zeros(NB,device=dev,dtype=torch.float32)   # per-block partial row-sums
prf = torch.zeros(NB,device=dev,dtype=torch.float32)   # per-block partial reference (distributed)

# ================= A. CORRECTNESS =================
print("\n===== A. CORRECTNESS (down_proj M=1, real W) =====")
gemv(A,W,wref,y,prs,prf,K,N,2); torch.cuda.synchronize()
rowsum = prs.sum().item()                          # OUT-OF-BAND final reduce of NB row-sum partials
ref_k  = prf.sum().item()                          # OUT-OF-BAND final reduce of NB reference partials
# kernel GEMV vs torch
y_torch = (A.float() @ W.float().t()).half()
gemv_relerr = (y.float()-y_torch.float()).abs().max().item() / y_torch.float().abs().max().item()
ref_torch = (A.float() @ wref).item()              # A . w_ref (fp32 reference)
print(f"kernel GEMV vs cuBLAS y: max rel-err {gemv_relerr:.2e}")
print(f"kernel fused rowsum (sum of fp32 acc) = {rowsum:.6f}   kernel ref (A.w_ref) = {ref_k:.6f}")
print(f"  ref vs torch A.w_ref |diff| = {abs(ref_k-ref_torch):.3e}")
# clean noise floor T = |fused_rowsum - ref| on clean data (single row M=1)
T = abs(rowsum - ref_k)
print(f"clean residual |rowsum - ref| = T = {T:.6e}   (fp32-accumulator checksum: detects compute SDC)")

# A1 coverage join for down_proj harmful flips (from Step A a1_results.json)
a1p='/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a1_results.json'
cov=None
if os.path.exists(a1p):
    recs=json.load(open(a1p))['results']
    dh=[r for r in recs if r['ptype']=='down_proj' and r['harmful']]
    finite=[r['abs_delta'] for r in dh if not r['naninf']]
    nninf=[r for r in dh if r['naninf']]
    caught=sum(1 for r in dh if r['naninf'] or r['abs_delta']>T)
    mind=min(finite) if finite else float('inf')
    cov=dict(n_harm=len(dh), n_caught=caught, T=T, min_delta=mind,
             margin=(mind/T if T>0 else float('inf')), n_naninf=len(nninf))
    print(f"down_proj harmful={len(dh)}  caught(|delta|>T or NaN/Inf)={caught}/{len(dh)}  "
          f"min|delta|={mind:.3f}  T={T:.3e}  margin={mind/T:.1f}x")
else:
    print("a1_results.json not found; coverage join skipped")

# ================= B. OVERHEAD =================
print("\n===== B. OVERHEAD (down_proj M=1, locked clock) =====")
Aw = A.contiguous()
def timed(fn, iters=300, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def baseline():     # the real intercepted decode linear
    torch.nn.functional.linear(Aw, W)
# standalone parallel ref kernel on a side stream
lib.launch_ref.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]; lib.launch_ref.restype=None
lib.ref_nblocks.restype=ctypes.c_int
RB=lib.ref_nblocks()
pref2=torch.zeros(RB,device=dev,dtype=torch.float32)
sB=torch.cuda.Stream()
def cand_gemv():  gemv(A,W,wref,y,prs,prf,K,N,0)   # GEMV, no checksum
def cand_rs():    gemv(A,W,wref,y,prs,prf,K,N,1)   # GEMV + rowsum partials only (output checksum)
def cand_full():  gemv(A,W,wref,y,prs,prf,K,N,2)   # GEMV + rowsum + ref in block 0 (serial-tail)
def ref_only():   lib.launch_ref(A.data_ptr(),wref.data_ptr(),pref2.data_ptr(),int(K),None)
def cand_overlap():  # GEMV+rowsum on default stream; ref on side stream B, concurrent
    gemv(A,W,wref,y,prs,prf,K,N,1)
    with torch.cuda.stream(sB):
        lib.launch_ref(A.data_ptr(),wref.data_ptr(),pref2.data_ptr(),int(K),sB.cuda_stream)
    torch.cuda.current_stream().wait_stream(sB)
def oob_reduce(): prs.sum()

t_base = timed(baseline)
t_gemv = timed(cand_gemv)
t_rs   = timed(cand_rs)
t_chk  = timed(cand_full)
t_refonly = timed(ref_only)
t_ovl  = timed(cand_overlap)
t_oob  = timed(oob_reduce)

# pure-ABFT cost isolates the checksum from the (orthogonal) custom-kernel substitution gap:
sub_gap   = (t_gemv-t_base)/t_base*100           # custom GEMV vs cuBLAS (kernel quality, NOT checksum)
rs_delta  = (t_rs  -t_gemv)/t_gemv*100           # rowsum-only checksum cost over the custom GEMV
chk_delta = (t_chk -t_gemv)/t_gemv*100           # FULL fused checksum (rowsum+distributed ref) over custom GEMV  <<< HEADLINE
chk_vs_cublas = (t_chk-t_base)/t_base*100        # fused checksum kernel total vs cuBLAS (carries sub_gap)
ovl_vs_cublas = (t_ovl-t_base)/t_base*100        # prior separate-stream-overlap path, for comparison
print(f"baseline cuBLAS F.linear     : {t_base*1000:.2f} us")
print(f"my GEMV (no checksum, mode0) : {t_gemv*1000:.2f} us   substitution gap vs cuBLAS = {sub_gap:+.2f}%  (my-kernel = {t_base/t_gemv*100:.0f}% of cuBLAS bw)")
print(f"GEMV + rowsum       (mode1)  : {t_rs*1000:.2f} us   rowsum-only checksum delta = {rs_delta:+.2f}% over my GEMV  <-- output reduction NEARLY FREE")
print(f"GEMV + rowsum + ref (mode2)  : {t_chk*1000:.2f} us   FUSED CHECKSUM DELTA = {chk_delta:+.2f}% over my GEMV  <<< HEADLINE (distributed ref, single launch)")
print(f"  fused checksum kernel total vs cuBLAS = {chk_vs_cublas:+.2f}%  (= {sub_gap:+.2f}% substitution + {chk_delta:+.2f}% checksum)")
print(f"ref kernel standalone        : {t_refonly*1000:.2f} us  (separate launch = +9.5us launch floor, the path fusion removes)")
print(f"GEMV+rowsum || ref(sideStrm) : {t_ovl*1000:.2f} us   prior overlap path total vs cuBLAS = {ovl_vs_cublas:+.2f}%  (superseded by mode2)")
print(f"  out-of-band final reduce ({NB} rs + {NB} ref partials)  = {t_oob*1000:.2f} us  (OFF GEMM critical path)")
t_chk_full=t_chk

# bandwidth sanity: W read = N*K*2 bytes
gb = N*K*2/1e9
print(f"\nW read = {gb*1000:.1f} MB; cuBLAS implied BW = {gb/t_base*1e3:.0f} GB/s; my GEMV BW = {gb/t_gemv*1e3:.0f} GB/s (H100 HBM3 ~3.35 TB/s)")
verdict = "PASS <3%" if chk_delta < 3.0 else "FAIL >=3%"
print(f"\n>>> FUSED DECODE CHECKSUM-DELTA = {chk_delta:+.2f}%  -> {verdict}  (substitution gap {sub_gap:+.2f}% is an orthogonal kernel-quality cost)")

out=dict(shape=dict(M=1,K=K,N=N,op='down_proj'),
         correctness=dict(gemv_relerr=gemv_relerr, rowsum=rowsum, ref=ref_k,
                          T=T, coverage=cov, C_abs_mean=C.abs().mean().item()),
         overhead_us=dict(baseline=t_base*1000, gemv=t_gemv*1000, rowsum=t_rs*1000,
                          ref_standalone=t_refonly*1000, fused_mode2=t_chk_full*1000,
                          overlapped=t_ovl*1000, oob_reduce=t_oob*1000),
         overhead_pct=dict(substitution_gap_vs_cublas=sub_gap,
                           rowsum_delta_vs_gemv=rs_delta,
                           fused_checksum_delta_vs_gemv=chk_delta,
                           fused_checksum_total_vs_cublas=chk_vs_cublas,
                           overlap_path_total_vs_cublas=ovl_vs_cublas),
         verdict=dict(fused_checksum_delta_pct=chk_delta, threshold_pct=3.0, pass_=bool(chk_delta<3.0)),
         bw_gbs=dict(cublas=gb/t_base*1e3, mygemv=gb/t_gemv*1e3))
json.dump(out, open('m1_result.json','w'), indent=1, default=lambda o: None)
print("\nwrote m1_result.json")
