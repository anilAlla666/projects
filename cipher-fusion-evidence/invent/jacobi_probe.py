# RESEARCH PROBE: Jacobi/lookahead parallel decoding potential on Mistral-7B.
# Convert serial m=1 GEMV decode into parallel m=K decode (uses idle tensor cores). EXACT (= greedy serial).
# Measure: tokens accepted per forward pass = the single-stream speedup from the idle compute. Draft-free, no retrain.
import torch, glob, time, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"; MIST=glob.glob("/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/")[0]
tok=AutoTokenizer.from_pretrained(MIST)
model=AutoModelForCausalLM.from_pretrained(MIST,dtype=torch.float16,device_map=DEV).eval()
prompts=[
 "The capital of France is Paris, which is famous for",
 "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
 "In machine learning, gradient descent is an optimization algorithm that",
 "Once upon a time, in a small village nestled between two mountains, there lived",
]
GEN=64   # tokens to produce per prompt
@torch.no_grad()
def serial_greedy(ids, n):
    out=[]
    for _ in range(n):
        lg=model(ids).logits[:,-1,:]; nt=lg.argmax(-1,keepdim=True); ids=torch.cat([ids,nt],1); out.append(nt.item())
    return out
@torch.no_grad()
def jacobi(ids0, n, K=16):
    # produce n tokens via Jacobi windows of size K. Count forward passes. Output == serial greedy (exact).
    ids=ids0; produced=[]; fwd=0
    while len(produced)<n:
        k=min(K, n-len(produced))
        # init guess: repeat last token (any init works; converges to fixed point)
        guess=torch.full((1,k), ids[0,-1].item(), device=DEV, dtype=ids.dtype)
        for _ in range(k):  # at most k iterations to converge a length-k window (Jacobi bound)
            cur=torch.cat([ids,guess],1)
            lg=model(cur).logits[0]; fwd+=1
            # position predicting token j (j=0..k-1) is at index len(ids)-1+j
            base=ids.shape[1]-1
            newg=lg[base:base+k].argmax(-1)  # parallel argmax for all k positions
            if torch.equal(newg, guess[0]):  # fixed point
                break
            guess=newg.unsqueeze(0)
        # accept the longest prefix that's stable (greedy-equivalent): accept all k (converged) 
        produced.extend(guess[0].tolist()); ids=torch.cat([ids,guess],1)
    return produced, fwd
print(f"{'prompt':40} {'serial_fwd':>10} {'jacobi_fwd':>10} {'tok/fwd':>8} {'speedup':>8} {'exact?':>7}")
sp=[]
for p in prompts:
    ids=tok(p,return_tensors="pt").input_ids.to(DEV)
    ser=serial_greedy(ids.clone(),GEN)
    jac,fwd=jacobi(ids.clone(),GEN,K=16)
    exact = jac[:GEN]==ser[:GEN]
    tpf=GEN/fwd; speed=GEN/fwd  # serial = GEN forwards; jacobi = fwd forwards; speedup = GEN/fwd
    sp.append(speed)
    print(f"{p[:40]:40} {GEN:>10} {fwd:>10} {tpf:>8.2f} {speed:>7.2f}x {str(exact):>7}")
print(f"\nMEAN single-stream speedup (Jacobi, K=16, draft-free, exact): {st.mean(sp):.2f}x")
print("(serial = GEN forward passes; Jacobi = fwd passes; each Jacobi pass is m=K matmul on IDLE tensor cores)")
