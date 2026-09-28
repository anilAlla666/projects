"""D1.3 KL probe -- add KL measurement to the in-distribution held-out test."""
import ctypes, sys, json, numpy as np, torch
import torch.nn.functional as F
torch.manual_seed(0)

lib = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
lib.cipher_koopman_fp16_register_shape.restype  = ctypes.c_int
lib.cipher_koopman_fp16_register_shape.argtypes = [ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.cipher_koopman_fp16_launch_shape.restype  = ctypes.c_int
lib.cipher_koopman_fp16_launch_shape.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int]

from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B",
    torch_dtype=torch.float16).cuda()
m.eval()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B")

calib_text = ("Artificial intelligence research focuses on the development "
              "of systems capable of perceiving reasoning learning. Modern "
              "approaches combine neural networks with knowledge.")
inp = tok(calib_text, return_tensors="pt").to("cuda")
cur = inp.input_ids
X_cap, Y_cap = [], []
with torch.no_grad():
    for step in range(2050):
        out = m(cur, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].float().clone()
        logits = out.logits[0, -1, :].float().clone()
        X_cap.append(hidden); Y_cap.append(logits)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nt], dim=1)
        if cur.shape[1] > 1024: cur = cur[:, -1024:]
X_calib = torch.stack(X_cap[:2000], dim=0)
Y_calib = torch.stack(Y_cap[:2000], dim=0)
X_test  = torch.stack(X_cap[2000:2010], dim=0).contiguous()
Y_test_true = torch.stack(Y_cap[2000:2010], dim=0).contiguous()

R = 64; K = 2048; N = 32000
U_x, S_x, Vh = torch.linalg.svd(X_calib, full_matrices=False)
U_x = U_x[:, :R].contiguous()
S_x = S_x[:R].contiguous()
V_x = Vh[:R].T.contiguous()
V_T = V_x.T.contiguous()
K_op = torch.eye(R, dtype=torch.float32, device="cuda")
W_buf = torch.diag(1.0/S_x) @ U_x.T @ Y_calib

rc = lib.cipher_koopman_fp16_register_shape(K, N,
    ctypes.c_void_p(V_T.data_ptr()),
    ctypes.c_void_p(K_op.data_ptr()),
    ctypes.c_void_p(W_buf.data_ptr()))

X_test_fp16 = X_test.to(torch.float16).contiguous()
y_cipher_fp16 = torch.full((10, N), -99.0, dtype=torch.float16, device="cuda")
lib.cipher_koopman_fp16_launch_shape(
    ctypes.c_void_p(X_test_fp16.data_ptr()),
    ctypes.c_void_p(y_cipher_fp16.data_ptr()),
    10, K, N)
torch.cuda.synchronize()
y_cipher = y_cipher_fp16.float()

v_top = Y_test_true.argmax(dim=-1)
c_top = y_cipher.argmax(dim=-1)
top1_match = float((v_top==c_top).float().mean())

log_p = F.log_softmax(Y_test_true, dim=-1)
log_q = F.log_softmax(y_cipher, dim=-1)
p = log_p.exp()
kl_per = (p * (log_p - log_q)).sum(-1)

res = {
    "methodology": "D1.3 same-sequence: 2000 calibration + 10 held-out autoregressive continuation",
    "rank": R, "shape": [K, N], "samples": 10, "register_rc": rc,
    "top1_match": top1_match,
    "kl_mean": float(kl_per.mean()),
    "kl_p50": float(kl_per.median()),
    "kl_max": float(kl_per.max()),
    "kl_min": float(kl_per.min()),
    "scope_lock_kl_gate": 5.5e-5,
    "kl_mean_pass_gate": float(kl_per.mean()) <= 5.5e-5,
    "kl_max_pass_gate":  float(kl_per.max())  <= 5.5e-5,
    "ratio_kl_mean_vs_gate": float(kl_per.mean()) / 5.5e-5,
}
print(json.dumps(res, indent=2))
with open("/tmp/step13_3_baseline/d1_kl_probe_result.json","w") as f:
    json.dump(res, f, indent=2)
