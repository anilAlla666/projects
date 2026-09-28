"""CP 5.4 Step 1.4 — isolation diagnostic (THROWAWAY).

The Step 1.4 sweep found TinyLlama teacher-forced logits blow up (TFGATE KL ~32)
at green-context sizes <=56 SMs (K<=7 groups) for prompts 2 & 4, clean at >=72
SMs. This harness isolates the cause: it uses ONLY torch + torch.cuda.GreenContext
-- NO libcipher_v2, NO cp54_pool, NO /dev/cipher, NO CIPHER substrate at all.
If it reproduces, the fault is PyTorch/cuDNN green contexts, not CIPHER.

For each prompt it forwards the gold [8 x full_len] batch on the full GPU
(reference) and under torch green contexts of 72 / 56 / 40 SMs, and reports the
max |logit difference| vs the full-GPU reference.
"""
import os, json, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
GOLD = json.load(open("/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01/gold.json"))["prompts"]

torch.manual_seed(0)
AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda()
m.train(False)


def fwd(full_ids, gc=None):
    batch = torch.tensor([full_ids] * 8, device="cuda")
    if gc is None:
        with torch.no_grad():
            return m(input_ids=batch).logits.float().cpu()
    strm = gc.Stream()
    gc.set_context()
    with torch.no_grad(), torch.cuda.stream(strm):
        out = m(input_ids=batch).logits.float().cpu()
    torch.cuda.synchronize()
    gc.pop_context()
    return out


print("prompt  K(SMs)   max|logit diff vs full-GPU|   verdict")
for pi in (1, 2, 4):                       # 1 = clean control; 2,4 = blew up
    full = GOLD[pi]["full_ids"]
    ref = fwd(full, None)
    for K in (9, 7, 5):
        gc = torch.cuda.GreenContext.create(K * 8, 0)
        diff = (ref - fwd(full, gc)).abs().max().item()
        verdict = "OK" if diff < 1.0 else "DIVERGED"
        print(f"  {pi}     {K:2d}({K*8:3d})    {diff:14.5f}              {verdict}")
