# CP 5.3 STEP 2B Step 1 — three-arm F1 discriminator, one arm per process.
# v2: the teacher-forced forward runs AFTER the Marlin warm generate, so it
# actually routes through the (compiled+quantized) substrate. v1 ran the tf
# forward first -> measured FP16 in every arm (KL=0.0 artifact).
import os, sys, json, time
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch

ARM    = os.environ["STEP2B_ARM"]
TARGET = "/home/ubuntu/models/Llama-3.1-8B"
DRAFT  = "/home/ubuntu/models/Llama-3.2-1B-Instruct"
PROMPT = ("The Pacific Ocean is the largest ocean on Earth, covering "
          "approximately")
TF_SEQ = (PROMPT + " 165 million square kilometers, more than the combined "
          "land area of all the continents on Earth.")
N = 64
OUT = "/home/ubuntu/cipher-fusion-evidence/cp_5_3/step2b"

try:
    import tenant_register
    tenant_register.register("step2b_arm_" + ARM)
except Exception as e:
    print("[step2b] tenant_register:", repr(e), file=sys.stderr)

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    TARGET, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()

import cipher_spec_decode
cipher_spec_decode.install()
enc = tok(PROMPT, return_tensors="pt").to("cuda")
plen = int(enc.input_ids.shape[-1])

# warm — compiles + quantizes Marlin (substrate arms); also warms the draft
with torch.no_grad():
    m.generate(**enc, max_new_tokens=16, do_sample=False,
               pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

# teacher-forced target logits — NOW routes through warmed Marlin
tf_ids = tok(TF_SEQ, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    tf_logits = m(tf_ids).logits.float().cpu()           # [1,T,V]
torch.cuda.synchronize()
torch.save(tf_logits, f"{OUT}/arm_{ARM}_tf_logits.pt")

# measured speculative-decode acceptance
err = None
res = {"arm": ARM, "target": "Llama-3.1-8B", "draft": "Llama-3.2-1B-Instruct",
       "tf_seq_len": int(tf_ids.shape[-1])}
try:
    t0 = time.time()
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=N, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    wall = time.time() - t0
    st = dict(cipher_spec_decode._last_stats)
    acc, prop, rounds = (st.get("accepted", 0), st.get("proposed", 0),
                         st.get("rounds", 0))
    gen = out[0].tolist()[plen:]
    res.update(accepted=acc, proposed=prop, rounds=rounds,
               accept_rate=acc / max(prop, 1),
               gen_tokens=len(gen), wall_s=round(wall, 2),
               tok_per_round=len(gen) / max(rounds, 1),
               text=tok.decode(gen, skip_special_tokens=True)[:240])
except Exception as e:
    err = repr(e)
    res["error"] = err

json.dump(res, open(f"{OUT}/arm_{ARM}.json", "w"), indent=2)
print("STEP2B_ARM_RESULT", json.dumps(res))
