"""W14 Step 3 S3.C LM-head validation harness (Python full-substrate, real TinyLlama).

Per Anil adjudication 2026-05-24 option alpha (two-mode) + option full-substrate.
Pass I (fire mode, EXISTENCE contract): in-distribution X on rank-64 EDMD-live
calibrated shape; handled >= 1, emits == handled, drained advances; top-1
matches D1.3 architectural ceiling (>= 90%).
Pass II (passthrough mode, PRESERVATION contract): off-manifold X; handled = 0
(beta-fired PASSTHROUGH); top-1 >= 99.95%, KL <= 5.5e-5.
"""
import os, sys, ctypes, json, time
os.environ["CIPHER_KOOPMAN"] = "1"
os.environ["CIPHER_REMEMBER"] = "1"
# The default beta threshold 0.05 is designed to keep Koopman from firing on
# real LM head data (per W14 Step 2 G section 5.3 OOD sweep). To exercise the
# fire-path EXISTENCE contract under the integrated stack on REAL LM head data
# (where the D1.3 architectural ceiling at rank-64 is top-1 = 90%, residual ~
# 0.6 by manifold-energy construction), we raise the threshold to 0.7 here.
# This is a harness-only setting; the substrate default 0.05 ships unchanged.
os.environ.setdefault("CIPHER_KOOPMAN_OOD_THRESHOLD", "0.7")

# Step 1: import torch + force cuBLAS load BEFORE libcipher_rt's GOT patch.
import torch
print(f"[S3.C] torch {torch.__version__}, cuda {torch.cuda.is_available()}")
_warmup = torch.empty(8, 8, dtype=torch.float16, device="cuda")
_ = (_warmup @ _warmup).cpu()
torch.cuda.synchronize()
print("[S3.C] cuBLAS warmed up")

# Step 2: load libcipher_rt + run init body. GOT patch walks loaded modules.
LIB = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
lib.InitializeInjection2.restype  = ctypes.c_int
lib.InitializeInjection2.argtypes = []
lib.cipher_koopman_fp16_register_shape.restype  = ctypes.c_int
lib.cipher_koopman_fp16_register_shape.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.cipher_koopman_fp16_ood_max_residual.restype  = ctypes.c_float
lib.cipher_koopman_fp16_ood_max_residual.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
for fname in ("cipher_rt_koopman_calls_total",
              "cipher_rt_koopman_calls_handled",
              "cipher_rt_koopman_calls_skipped",
              "cipher_rt_koopman_remember_emits",
              "cipher_rt_remember_consumer_drained",
              "cipher_rt_remember_consumer_lnn_invocations"):
    f = getattr(lib, fname); f.restype = ctypes.c_ulong; f.argtypes = []
lib.cipher_rt_koopman_is_active.restype = ctypes.c_int
lib.cipher_rt_koopman_is_active.argtypes = []
lib.cipher_rt_remember_consumer_is_active.restype = ctypes.c_int
lib.cipher_rt_remember_consumer_is_active.argtypes = []

lib.InitializeInjection2()
assert lib.cipher_rt_koopman_is_active(), "Koopman engine inactive"
print(f"[S3.C] Koopman active={lib.cipher_rt_koopman_is_active()} "
      f"REMEMBER active={lib.cipher_rt_remember_consumer_is_active()}")

# Step 3: smoke test — confirm torch.matmul reaches the substrate GOT-patched
# cublasGemmEx. (PyTorch may dispatch to cublasLt or other variants; we verify
# explicitly so we know whether the substrate sees torch's calls.)
import torch.nn.functional as F
pre_total = lib.cipher_rt_koopman_calls_total()
A = torch.empty(2048, 64, dtype=torch.float16, device="cuda").normal_()
B = torch.empty(64, 8,    dtype=torch.float16, device="cuda").normal_()
C = (A @ B).contiguous()
torch.cuda.synchronize()
post_smoke = lib.cipher_rt_koopman_calls_total()
print(f"[S3.C SMOKE] torch.matmul fp16 -> substrate calls_total delta = "
      f"{post_smoke - pre_total} (0 = not intercepted; >0 = GOT-patched cublasGemmEx hit)")

# Step 4: real TinyLlama LM head capture.
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
print("[S3.C] Loading TinyLlama-1.1B...")
m = AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B",
    torch_dtype=torch.float16).cuda(); m.eval()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B")
W_real = m.lm_head.weight.detach()   # (32000, 2048) fp16

calib_text = ("Artificial intelligence research focuses on the development "
              "of systems capable of perceiving reasoning learning. Modern "
              "approaches combine neural networks with knowledge.")
inp = tok(calib_text, return_tensors="pt").to("cuda")
cur = inp.input_ids
X_cap, Y_cap = [], []
with torch.no_grad():
    for step in range(2050):
        out = m(cur, output_hidden_states=True)
        X_cap.append(out.hidden_states[-1][0, -1, :].float().clone())
        Y_cap.append(out.logits[0, -1, :].float().clone())
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nt], dim=1)
        if cur.shape[1] > 1024: cur = cur[:, -1024:]
X_calib = torch.stack(X_cap[:2000], dim=0)
Y_calib = torch.stack(Y_cap[:2000], dim=0)
X_in    = torch.stack(X_cap[2000:2010], dim=0).contiguous()
Y_van_in_ref = torch.stack(Y_cap[2000:2010], dim=0).contiguous()   # reference Y captured during vanilla autoreg

K, N, R, M_TEST = 2048, 32000, 64, X_in.shape[0]
torch.manual_seed(0xDEAD)
X_random = torch.randn(M_TEST, K, dtype=torch.float32, device="cuda") * X_in.std().item()

U_x, S_x, Vh = torch.linalg.svd(X_calib, full_matrices=False)
U_x = U_x[:, :R].contiguous()
S_x = S_x[:R].contiguous()
V_x = Vh[:R].T.contiguous()
V_T = V_x.T.contiguous().to("cpu", dtype=torch.float32).contiguous()
K_op = torch.eye(R, dtype=torch.float32)
W_buf = (torch.diag(1.0/S_x) @ U_x.T @ Y_calib).to("cpu", dtype=torch.float32).contiguous()
V_T_np  = V_T.numpy()
K_op_np = K_op.numpy()
W_buf_np = W_buf.numpy()

# Step 5: vanilla baseline BEFORE registration. The fp16 matmul goes through
# substrate (if GOT-patched) -> koopman engine maybe_handle returns PASSTHROUGH
# (shape not registered yet) -> shim falls back to real cublasGemmEx.
pre_total   = lib.cipher_rt_koopman_calls_total()
pre_handled = lib.cipher_rt_koopman_calls_handled()
pre_emits   = lib.cipher_rt_koopman_remember_emits()
X_in_fp16     = X_in.to("cuda", dtype=torch.float16).contiguous()
X_random_fp16 = X_random.to("cuda", dtype=torch.float16).contiguous()
Y_van_in     = X_in_fp16     @ W_real.T
Y_van_random = X_random_fp16 @ W_real.T
torch.cuda.synchronize()
print(f"[S3.C VANILLA BASELINE] total +{lib.cipher_rt_koopman_calls_total()-pre_total} "
      f"handled +{lib.cipher_rt_koopman_calls_handled()-pre_handled} "
      f"emits +{lib.cipher_rt_koopman_remember_emits()-pre_emits}")

rc = lib.cipher_koopman_fp16_register_shape(K, N,
    V_T_np.ctypes.data, K_op_np.ctypes.data, W_buf_np.ctypes.data)
print(f"[S3.C REGISTER] rc={rc} (K={K} N={N} R={R})")
assert rc == 0

resid_in   = lib.cipher_koopman_fp16_ood_max_residual(X_in_fp16.data_ptr(),     M_TEST, K, N)
resid_rand = lib.cipher_koopman_fp16_ood_max_residual(X_random_fp16.data_ptr(), M_TEST, K, N)
print(f"[S3.C OOD PROBE] in-dist residual={resid_in:.6f} off-dist residual={resid_rand:.6f} "
      f"(harness threshold {os.environ['CIPHER_KOOPMAN_OOD_THRESHOLD']})")

# Step 6: Pass I — in-distribution.
pre_p1_handled = lib.cipher_rt_koopman_calls_handled()
pre_p1_emits   = lib.cipher_rt_koopman_remember_emits()
pre_p1_drained = lib.cipher_rt_remember_consumer_drained()
pre_p1_lnn     = lib.cipher_rt_remember_consumer_lnn_invocations()
Y_cip_in = X_in_fp16 @ W_real.T
torch.cuda.synchronize()
time.sleep(0.05)
d_handled = lib.cipher_rt_koopman_calls_handled() - pre_p1_handled
d_emits   = lib.cipher_rt_koopman_remember_emits() - pre_p1_emits
d_drained = lib.cipher_rt_remember_consumer_drained() - pre_p1_drained
d_lnn     = lib.cipher_rt_remember_consumer_lnn_invocations() - pre_p1_lnn
print(f"[S3.C PASS I  TELEMETRY] handled +{d_handled} emits +{d_emits} drained +{d_drained} lnn +{d_lnn}")

# Compare to the vanilla baseline captured pre-registration (Y_van_in).
v_top = Y_van_in.float().argmax(dim=-1)
c_top = Y_cip_in.float().argmax(dim=-1)
top1_in = float((v_top == c_top).float().mean())
log_p = F.log_softmax(Y_van_in.float(),  dim=-1)
log_q = F.log_softmax(Y_cip_in.float(),  dim=-1)
p = log_p.exp()
kl_per = (p * (log_p - log_q)).sum(-1)
kl_mean_in = float(kl_per.mean())
print(f"[S3.C PASS I  QUALITY] top1={top1_in:.4f} kl_mean={kl_mean_in:.6e}")
pass_i_telemetry = (d_handled >= 1) and (d_emits == d_handled) and (d_drained >= 1)
pass_i_quality = (top1_in >= 0.90)
pass_i = pass_i_telemetry and pass_i_quality
print(f"[S3.C PASS I  RESULT] telemetry={'PASS' if pass_i_telemetry else 'FAIL'} "
      f"quality={'PASS' if pass_i_quality else 'FAIL'} overall={'PASS' if pass_i else 'FAIL'}")

# DEBUG: consumer cycle telemetry
for _fn in ("cipher_rt_remember_consumer_active_cycles","cipher_rt_remember_consumer_total_cycles"):
    f=getattr(lib,_fn); f.restype=ctypes.c_ulong; f.argtypes=[]
_active_cycles = lib.cipher_rt_remember_consumer_active_cycles()
_total_cycles  = lib.cipher_rt_remember_consumer_total_cycles()
print(f"[S3.C DEBUG] consumer cycles total={_total_cycles} active={_active_cycles}")
time.sleep(0.5)
_active_cycles2 = lib.cipher_rt_remember_consumer_active_cycles()
_total_cycles2  = lib.cipher_rt_remember_consumer_total_cycles()
print(f"[S3.C DEBUG] after 500ms total={_total_cycles2} (delta {_total_cycles2-_total_cycles}) active={_active_cycles2}")
print(f"[S3.C DEBUG] late drained = {lib.cipher_rt_remember_consumer_drained()}")

# Step 7: Pass II — off-manifold.
pre_p2_handled = lib.cipher_rt_koopman_calls_handled()
pre_p2_emits   = lib.cipher_rt_koopman_remember_emits()
pre_p2_drained = lib.cipher_rt_remember_consumer_drained()
Y_cip_rand = X_random_fp16 @ W_real.T
torch.cuda.synchronize()
time.sleep(0.05)
d2_handled = lib.cipher_rt_koopman_calls_handled() - pre_p2_handled
d2_emits   = lib.cipher_rt_koopman_remember_emits() - pre_p2_emits
d2_drained = lib.cipher_rt_remember_consumer_drained() - pre_p2_drained
print(f"[S3.C PASS II TELEMETRY] handled +{d2_handled} emits +{d2_emits} drained +{d2_drained}")
v_top_r = Y_van_random.float().argmax(dim=-1)
c_top_r = Y_cip_rand.float().argmax(dim=-1)
top1_rand = float((v_top_r == c_top_r).float().mean())
log_p_r = F.log_softmax(Y_van_random.float(), dim=-1)
log_q_r = F.log_softmax(Y_cip_rand.float(),   dim=-1)
pr = log_p_r.exp()
kl_per_r = (pr * (log_p_r - log_q_r)).sum(-1)
kl_mean_rand = float(kl_per_r.mean())
print(f"[S3.C PASS II QUALITY] top1={top1_rand:.4f} kl_mean={kl_mean_rand:.6e}")
pass_ii_telemetry = (d2_handled == 0)
pass_ii_quality   = (top1_rand >= 0.9995) and (kl_mean_rand <= 5.5e-5)
pass_ii = pass_ii_telemetry and pass_ii_quality
print(f"[S3.C PASS II RESULT] telemetry={'PASS' if pass_ii_telemetry else 'FAIL'} "
      f"quality={'PASS' if pass_ii_quality else 'FAIL'} overall={'PASS' if pass_ii else 'FAIL'}")

overall = pass_i and pass_ii
print(f"[S3.C OVERALL] {'PASS' if overall else 'FAIL'}")
result = dict(
    pass_i=pass_i, pass_ii=pass_ii, overall=overall,
    top1_in=top1_in, kl_mean_in=kl_mean_in,
    top1_rand=top1_rand, kl_mean_rand=kl_mean_rand,
    resid_in=resid_in, resid_rand=resid_rand,
    smoke_total=int(post_smoke - pre_total),
    handled_p1=int(d_handled), emits_p1=int(d_emits), drained_p1=int(d_drained),
    handled_p2=int(d2_handled), emits_p2=int(d2_emits), drained_p2=int(d2_drained),
)
with open("/tmp/step13_3_baseline/s3c_result.json","w") as f:
    json.dump(result, f, indent=2)
sys.exit(0 if overall else 1)
