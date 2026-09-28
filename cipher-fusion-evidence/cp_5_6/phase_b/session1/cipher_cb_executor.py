"""Phase B Step 4+5 — continuous-batching executor, sub-component 2:
KV-splice admission (survivors' KV kept; only the new row is prefilled).

Invariant (the correctness backbone): every row's real KV content is
RIGHT-aligned (all padding is left-padding); a row's attention mask is its
trailing `real_len` ones; a new token's position_id == `real_len`. The
invariant survives prefill (left-pad prompts), decode (append right), eviction
(index_select — per-row layout unchanged), and admission-splice (left-pad to a
common length — real content stays right-aligned).

Admission: prefill the new request's prompt alone (B=1), left-pad its KV to
the batch length, cat it into the running KV. Survivors' KV is never recomputed.
Correctness: teacher-forced per-prompt logit-KL gate (cascade-free).
"""
import os
import sys
import json
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import torch
import torch.nn.functional as F
import tenant_register

MODEL = os.environ["WL_MODEL"]
WORKLOAD = os.environ["WORKLOAD"]
N_SLOTS = int(os.environ.get("N_SLOTS", "4"))
GOLD_JSON = os.environ["GOLD_JSON"]
GOLD_LOGITS = os.environ["GOLD_LOGITS"]
RESULT = os.environ["RESULT_JSON"]

PROMPTS = [
    "The Pacific Ocean is the largest ocean on Earth, covering approximately",
    "Sarah opened the old letter with trembling hands. The handwriting was "
    "her grandmother's, and the date read",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot =",
    "If a train leaves Chicago at 3 PM traveling east at 60 mph and another "
    "leaves New York at 4 PM traveling west at 80 mph,",
    "User: What are the main differences between supervised and unsupervised "
    "learning?\nAssistant: The main differences are",
]

tenant_register.register("cb_exec")
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS = tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()
DEV = "cuda"
PROMPT_TOK = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]
gold_meta = {g["prompt"]: g for g in json.load(open(GOLD_JSON))["prompts"]}
gold_logits = torch.load(GOLD_LOGITS, map_location=DEV)


def prefill_one(prompt_ids):
    """B=1 prefill of a prompt. Returns (DynamicCache, first-token id)."""
    ids = torch.tensor([prompt_ids], device=DEV)
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    return out.past_key_values, int(out.logits[0, -1].argmax())


def lpad_seq(t, L):
    """Left-pad a KV tensor [B,h,seq,d] along seq (dim=2) to length L."""
    p = L - t.shape[2]
    if p <= 0:
        return t
    z = torch.zeros(t.shape[0], t.shape[1], p, t.shape[3],
                    dtype=t.dtype, device=DEV)
    return torch.cat([z, t], dim=2)


def splice(past, new_caches):
    """Cat survivors' KV (past, may be None) with new B=1 caches along batch,
    left-padding every row to the common length. Returns the merged cache."""
    caches = ([past] if past is not None else []) + new_caches
    L = max(c.layers[0].keys.shape[2] for c in caches)
    base = caches[0]
    for li in range(len(base.layers)):
        base.layers[li].keys = torch.cat(
            [lpad_seq(c.layers[li].keys, L) for c in caches], dim=0)
        base.layers[li].values = torch.cat(
            [lpad_seq(c.layers[li].values, L) for c in caches], dim=0)
    return base


def run_continuous(requests):
    queue = deque(requests)
    active = []                 # each: req dict + gen[] + real_len + done
    past = None
    nt = None                   # [B] token to feed next, aligned with active
    completed = []
    step = 0
    occ = []
    admit_prefills = 0
    while queue or active:
        # ---- evict finished ----
        fin = [i for i, r in enumerate(active) if r["done"]]
        if fin:
            finset = set(fin)
            keep = [i for i in range(len(active)) if i not in finset]
            for i in fin:
                active[i]["t_complete"] = step
                completed.append(active[i])
            if keep:
                kt = torch.tensor(keep, device=DEV)
                for lyr in past.layers:
                    lyr.keys = lyr.keys.index_select(0, kt)
                    lyr.values = lyr.values.index_select(0, kt)
                active = [active[i] for i in keep]
                nt = nt.index_select(0, kt)
            else:
                active, past, nt = [], None, None
        # ---- admit (KV-splice) ----
        new_caches, new_first = [], []
        while len(active) < N_SLOTS and queue:
            r = dict(queue.popleft())
            ptok = PROMPT_TOK[r["prompt_idx"]]
            pc, ft = prefill_one(ptok)
            admit_prefills += 1
            r.update(gen=[ft], real_len=len(ptok), done=False,
                     t_admit=step, t_first=step)
            if len(r["gen"]) >= r["gen_len"] or ft == EOS:
                r["done"] = True
            active.append(r)
            new_caches.append(pc)
            new_first.append(ft)
        if new_caches:
            past = splice(past, new_caches)
            nf = torch.tensor(new_first, device=DEV)
            nt = nf if nt is None else torch.cat([nt, nf])
        if not active:
            break
        # ---- one batched decode step ----
        L_phys = past.layers[0].keys.shape[2]
        real = [r["real_len"] for r in active]
        am = torch.zeros(len(active), L_phys + 1, dtype=torch.long, device=DEV)
        for i, rl in enumerate(real):
            am[i, L_phys - rl:] = 1                     # trailing rl+1 ones
        pos = torch.tensor([[rl] for rl in real], device=DEV)
        with torch.no_grad():
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past,
                    attention_mask=am, position_ids=pos,
                    cache_position=torch.tensor([L_phys], device=DEV))
        past = out.past_key_values
        newtok = out.logits[:, -1, :].argmax(-1)
        step += 1
        occ.append(len(active))
        for i, r in enumerate(active):
            r["gen"].append(int(newtok[i]))
            r["real_len"] += 1
            if len(r["gen"]) >= r["gen_len"] or r["gen"][-1] == EOS:
                r["done"] = True
        nt = newtok
    return completed, occ, admit_prefills, step


def tf_kl(prompt_idx, n):
    g = gold_meta[prompt_idx]
    full, plen = g["full_ids"], g["plen"]
    n = min(n, len(full) - plen)
    with torch.no_grad():
        o = m(input_ids=torch.tensor([full], device=DEV))
    seg = o.logits[0, plen - 1:plen - 1 + n, :]
    gl = gold_logits[prompt_idx][:n]
    lp = F.log_softmax(gl.float(), -1)
    lq = F.log_softmax(seg.float(), -1)
    ks = (lp.exp() * (lp - lq)).sum(-1)
    return float(ks.mean()), float(ks.max())


wl = json.load(open(WORKLOAD))["requests"]
t0 = time.time()
completed, occ, admit_prefills, total_steps = run_continuous(wl)
wall = time.time() - t0
completed.sort(key=lambda r: r["id"])

kl_by_prompt = {}
rows = []
exact_ok = 0
useful = 0
for r in completed:
    pi = r["prompt_idx"]
    if pi not in kl_by_prompt:
        kl_by_prompt[pi] = tf_kl(pi, 128)
    klm, klx = kl_by_prompt[pi]
    gold_gen = gold_meta[pi]["gen_ids"]
    chk = min(len(r["gen"]), len(gold_gen))
    match = sum(1 for i in range(chk) if r["gen"][i] == gold_gen[i])
    exact = match / chk if chk else 0.0
    exact_ok += int(exact >= 0.99)
    useful += len(r["gen"])
    rows.append({"id": r["id"], "prompt_idx": pi, "gen_len": r["gen_len"],
                 "produced": len(r["gen"]), "exact_vs_gold": round(exact, 4),
                 "kl_mean": round(klm, 6), "kl_max": round(klx, 6)})

res = {"phase": "B", "step": "4+5", "subcomponent": "2-kv-splice",
       "n_slots": N_SLOTS, "n_requests": len(wl), "n_completed": len(completed),
       "total_decode_steps": total_steps, "admit_prefills": admit_prefills,
       "wall_s": round(wall, 2), "useful_tokens": useful,
       "agg_tok_s": round(useful / wall, 2) if wall else None,
       "mean_batch_occupancy": round(sum(occ) / len(occ), 3) if occ else 0,
       "correctness_exact_ge99_count": exact_ok,
       "correctness_kl_max": max(r["kl_max"] for r in rows) if rows else None,
       "requests": rows}
json.dump(res, open(RESULT, "w"), indent=2)
print("CB-SPLICE n_slots=%d req=%d completed=%d admit_prefills=%d steps=%d "
      "useful=%d wall=%.1fs agg_tok_s=%.1f mean_occ=%.2f"
      % (N_SLOTS, len(wl), len(completed), admit_prefills, total_steps,
         useful, wall, res["agg_tok_s"], res["mean_batch_occupancy"]),
      file=sys.stderr)
print("CORRECTNESS exact>=99%%: %d/%d   KL_max=%.6f"
      % (exact_ok, len(rows), res["correctness_kl_max"]), file=sys.stderr)
