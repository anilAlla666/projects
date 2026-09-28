"""W.4b.6 Memory-#11 confirmation: teacher-forced KL, batched(B=8) vs solo(B=1).

The N=8 row-3 divergence (20/64) is hypothesised to be a benign near-tie greedy
coin-flip that cascades (first-div gap 0.00781). Decisive test: teacher-force BOTH
the B=8 batch and a B=1 solo with the SAME token sequence (the solo greedy path)
and compare per-step logit KL for each row under IDENTICAL context. If KL stays
~1e-5, the batched substrate is faithful to solo and the token cascade is pure
FP-reduction-order at coin-flips — NOT corruption. A systematic logit error would
show large KL at every step.
"""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("WL_MODEL", "/home/ubuntu/models/Mistral-7B-v0.1")
N = int(os.environ.get("N", "8"))
GEN = int(os.environ.get("GEN_LEN", "64"))
PLEN = int(os.environ.get("PLEN", "12"))
DEV = "cuda"
PROMPTS = [
    "The history of computing spans several distinct technological eras that",
    "Renewable energy adoption worldwide has accelerated rapidly over the recent",
    "Modern distributed systems must carefully balance consistency availability and partition",
    "Advances in materials science have enabled lighter stronger and more durable",
    "The global supply chain experienced unprecedented disruption during the pandemic which",
    "Machine learning models trained on large corpora can exhibit surprising emergent",
    "Urban planners increasingly rely on data driven simulations to forecast traffic",
    "Ocean currents play a critical role in regulating the planet climate by",
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()


def enc(p):
    t = tok(p, return_tensors="pt").input_ids[0]
    if t.shape[0] >= PLEN:
        return t[:PLEN]
    padv = tok.pad_token_id or tok.eos_token_id
    return torch.cat([t, torch.full((PLEN - t.shape[0],), padv, dtype=t.dtype)])


ids = torch.stack([enc(PROMPTS[i % len(PROMPTS)]) for i in range(N)], 0).cuda()  # [N,PLEN]


def solo_greedy(row):
    seq = ids[row:row + 1]
    toks = []
    with torch.no_grad():
        out = m(input_ids=seq, use_cache=True); past = out.past_key_values
        lg = out.logits[:, -1, :]; cur = seq.shape[1]
        for _ in range(GEN):
            nt = lg.argmax(-1); toks.append(int(nt))
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; lg = out.logits[:, -1, :]; cur += 1
    return toks


# solo greedy token paths per row (the teacher-forcing sequences)
solo_toks = [solo_greedy(r) for r in range(N)]

# teacher-force the B=N batch with each row's OWN solo tokens, capture batched logits
forced = torch.stack([torch.tensor(solo_toks[r], device=DEV) for r in range(N)], 0)  # [N,GEN]
with torch.no_grad():
    out = m(input_ids=ids, use_cache=True); past = out.past_key_values
    batched_lg = [out.logits[:, -1, :]]                  # step 0 logits [N,V]
    cur = PLEN
    for s in range(GEN - 1):
        step_tok = forced[:, s].unsqueeze(-1)            # feed solo token (teacher force)
        out = m(input_ids=step_tok, past_key_values=past, use_cache=True,
                cache_position=torch.tensor([cur], device=DEV))
        past = out.past_key_values; batched_lg.append(out.logits[:, -1, :]); cur += 1

# solo teacher-forced logits per row, same forcing sequence
def solo_tf(row):
    seq = ids[row:row + 1]
    with torch.no_grad():
        out = m(input_ids=seq, use_cache=True); past = out.past_key_values
        lgs = [out.logits[:, -1, :]]; cur = seq.shape[1]
        for s in range(GEN - 1):
            st = forced[row, s].view(1, 1)
            out = m(input_ids=st, past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; lgs.append(out.logits[:, -1, :]); cur += 1
    return lgs


print("row | max_step_KL | mean_step_KL | argmax_mismatch_steps/GEN")
worst = 0.0
for r in range(N):
    solo_lgs = solo_tf(r)
    kls, mm = [], 0
    for s in range(GEN):
        bp = F.log_softmax(batched_lg[s][r].float(), -1)
        sp = F.log_softmax(solo_lgs[s][0].float(), -1)
        kl = float((sp.exp() * (sp - bp)).sum())
        kls.append(kl)
        if int(batched_lg[s][r].argmax()) != int(solo_lgs[s][0].argmax()):
            mm += 1
    mx = max(kls); worst = max(worst, mx)
    print("%3d | %.3e | %.3e | %d/%d" % (r, mx, sum(kls) / len(kls), mm, GEN))
print("\nWORST per-step teacher-forced KL across all rows = %.3e" % worst)
print("VERDICT: %s" % ("benign (substrate faithful under identical context; "
      "token cascade is FP-order at near-ties)" if worst < 5e-3 else
      "INVESTIGATE — systematic logit divergence"))
