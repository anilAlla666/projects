"""CP 5.4 Step 1.4 — K<=7 corruption bisection harness (THROWAWAY).

Faithfully replicates the Phase B executor's Phase-2 TFGATE: forward each
prompt's gold [8 x full_len] batch under a green context, KL(gold_logits ||
forward) per row — the gate that blew up (KL=32.56 at K<=7).

BISECT_PHASE env (substring flags, additive bisection):
  ""            Phase 0/1 — direct torch.cuda.GreenContext, no generate rounds
  "gen"         Phase 2   — + warmup & 5 generate() rounds before TFGATE
  "gen pool"    Phase 3   — + green ctx built via cp54_pool.PoolBinding
                            (uses CIPHER_POOL_MAX_GROUPS to set K)
CUDA_INJECTION64_PATH=libcipher_v2.so adds the CUPTI injection (Phase 1+).
"""
import os, json, sys, torch
import torch.nn.functional as F
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
MDIR = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
GOLD = json.load(open(f"{MDIR}/gold.json"))["prompts"]
gold_logits = torch.load(f"{MDIR}/gold_logits.pt", map_location="cuda")
N = 8
DO_GEN = "gen" in os.environ.get("BISECT_PHASE", "")
USE_POOL = "pool" in os.environ.get("BISECT_PHASE", "")

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda()
m.train(False)


def tf_kl(gold_logit, tf_logit):
    lp = F.log_softmax(gold_logit.float(), dim=-1)
    lq = F.log_softmax(tf_logit.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


def gen_rounds(strm):
    def _g(texts):
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            if strm is not None:
                with torch.cuda.stream(strm):
                    m.generate(**enc, max_new_tokens=128, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            else:
                m.generate(**enc, max_new_tokens=128, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
    _g(["warm up the model"] * N)
    for r in range(5):
        _g([GOLD[r]["text"]] * N)


def tfgate_suite(strm):
    kls = []
    for pi in range(5):
        g = GOLD[pi]
        full, plen = g["full_ids"], g["plen"]
        gen = len(full) - plen
        batch = torch.tensor([full] * N, device="cuda")
        with torch.no_grad():
            if strm is not None:
                with torch.cuda.stream(strm):
                    tf = m(input_ids=batch)
            else:
                tf = m(input_ids=batch)
        torch.cuda.synchronize()
        gl = gold_logits[pi]
        kls.append(max(float(tf_kl(gl, tf.logits[row, plen - 1:plen - 1 + gen, :]).max())
                       for row in range(N)))
    return kls


def show(tag, kls):
    bad = [i for i in range(5) if kls[i] > 0.1]
    print(f"  {tag:>14}: " + "  ".join(f"p{i}={kls[i]:.5f}" for i in range(5))
          + (f"   <-- KL>0.1 at prompts {bad}" if bad else "   all clean"))


label = os.environ.get("BISECT_LABEL", "?")
NO_WARMUP = "nowarmup" in os.environ.get("BISECT_PHASE", "")
print(f"=== bisect [{label}]  injection={os.environ.get('CUDA_INJECTION64_PATH','(none)')}"
      f"  DO_GEN={DO_GEN} USE_POOL={USE_POOL} NO_WARMUP={NO_WARMUP} ===")
# The full-GPU pass below also pre-seeds cuDNN/cuBLAS algorithm selection for
# the TFGATE prefill shape at the full 132-SM device. "nowarmup" skips it so
# the green-context forward is the FIRST execution of that shape — exactly the
# executor's situation (it never runs a full-GPU forward of the TFGATE shape).
if not NO_WARMUP:
    show("full-GPU", tfgate_suite(None))

if USE_POOL:
    sys.path.insert(0, "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_b/session1")
    import cp54_pool
    pb = cp54_pool.PoolBinding()
    pb.build_green_ctx()
    if DO_GEN:
        gen_rounds(pb.gc_stream)
    show(f"POOL K={pb._gc_count}", tfgate_suite(pb.gc_stream))
    pb.free()
else:
    for K in (9, 7, 5):
        gc = torch.cuda.GreenContext.create(K * 8, 0)
        strm = gc.Stream()
        gc.set_context()
        if DO_GEN:
            gen_rounds(strm)
        show(f"K={K}({K*8}SM)", tfgate_suite(strm))
        gc.pop_context()
