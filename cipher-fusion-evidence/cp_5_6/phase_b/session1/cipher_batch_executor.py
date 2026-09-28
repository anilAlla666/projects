"""Phase B batched-decode executor — continuous-batch loop with per-tenant
generate lengths + finished-sequence eviction (Step 4: heterogeneous batch).

PHASE 1 (throughput): per round, prefill B=N, then a manual decode loop —
each step advances all active rows; a row that reaches its gen_len (or EOS)
is evicted from the batch and KV cache; loop ends when all rows are done.
PHASE 2 (correctness): teacher-forced B=N forward over each prompt's gold,
per-row per-step logit-KL (cascade-free).

Clients send {text, prompt_idx, gen_len}, all 5 up front, then recv 5 responses.
"""
import os
import sys
import json
import time
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import torch
import torch.nn.functional as F
import tenant_register
from batch_ipc import send_msg, recv_msg

SOCK = os.environ.get("BATCH_SOCK", "/tmp/cipher_batch_exec.sock")
MODEL = os.environ["WL_MODEL"]
N_CLIENTS = int(os.environ.get("N_CLIENTS", "4"))
GOLD_JSON = os.environ["GOLD_JSON"]
GOLD_LOGITS = os.environ["GOLD_LOGITS"]
RESULT = os.environ["RESULT_JSON"]
SENTINEL = os.environ.get("SENTINEL", "/tmp/cipher_batch_exec.decode_window")
N_PROMPTS = 5

tenant_register.register("batch_exec")
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
EOS = tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()
gold_meta = {g["prompt"]: g for g in json.load(open(GOLD_JSON))["prompts"]}
gold_logits = torch.load(GOLD_LOGITS, map_location="cuda")
DEV = "cuda"


def cb_decode(texts, gen_lens):
    """Continuous-batch decode with per-row gen_len + eviction.
    Returns (outs[list per row], decode_steps, token_steps, wall_s)."""
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    ids, attn = enc.input_ids, enc.attention_mask
    n = len(texts)
    plen = int(ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    nt = out.logits[:, -1, :].argmax(-1)              # [n]
    active = list(range(n))
    outs = [[] for _ in range(n)]
    cur = plen
    steps = 0
    token_steps = 0
    while active:
        for pos, oi in enumerate(active):
            outs[oi].append(int(nt[pos]))
        token_steps += len(active)
        steps += 1
        still = [pos for pos, oi in enumerate(active)
                 if len(outs[oi]) < gen_lens[oi] and outs[oi][-1] != EOS]
        if not still:
            break
        if len(still) < len(active):
            keep = torch.tensor(still, device=DEV)
            for lyr in past.layers:
                lyr.keys = lyr.keys.index_select(0, keep)
                lyr.values = lyr.values.index_select(0, keep)
            nt = nt.index_select(0, keep)
            attn = attn.index_select(0, keep)
            active = [active[p] for p in still]
        attn = torch.cat(
            [attn, torch.ones((len(active), 1), device=DEV, dtype=attn.dtype)],
            dim=1)
        cp = torch.tensor([cur], device=DEV)
        with torch.no_grad():
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past,
                    attention_mask=attn, use_cache=True, cache_position=cp)
        past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1)
        cur += 1
    torch.cuda.synchronize()
    return outs, steps, token_steps, time.perf_counter() - t0


def tf_kl_row(gold_logit, tf_logit):
    lp = F.log_softmax(gold_logit.float(), dim=-1)
    lq = F.log_softmax(tf_logit.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


if os.path.exists(SOCK):
    os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(64)
print("LISTENING %s" % SOCK, file=sys.stderr, flush=True)
conns = [srv.accept()[0] for _ in range(N_CLIENTS)]
print("all %d clients connected" % N_CLIENTS, file=sys.stderr, flush=True)

_w = tok(["warm up the model"] * N_CLIENTS, return_tensors="pt").to(DEV)
with torch.no_grad():
    m.generate(**_w, max_new_tokens=64, do_sample=False, pad_token_id=EOS)
torch.cuda.synchronize()

all_reqs = [[recv_msg(c) for c in conns] for _ in range(N_PROMPTS)]

rounds = []
with open(SENTINEL, "w") as s:
    s.write("DECODE_START %.3f\n" % time.time())
for r in range(N_PROMPTS):
    texts = [req["text"] for req in all_reqs[r]]
    gen_lens = [int(req["gen_len"]) for req in all_reqs[r]]
    outs, steps, token_steps, wall = cb_decode(texts, gen_lens)
    rounds.append({"outs": outs, "gen_lens": gen_lens, "steps": steps,
                   "token_steps": token_steps, "wall_s": round(wall, 4),
                   "useful_tokens": sum(len(o) for o in outs)})
    print("ROUND %d batch=%d steps=%d token_steps=%d useful=%d wall=%.3fs"
          % (r, len(texts), steps, token_steps,
             sum(len(o) for o in outs), wall), file=sys.stderr, flush=True)
with open(SENTINEL, "a") as s:
    s.write("DECODE_END %.3f\n" % time.time())

for r in range(N_PROMPTS):
    pr = all_reqs[r][0]["prompt_idx"]
    g = gold_meta[pr]
    full, plen_g = g["full_ids"], g["plen"]
    gen_g = len(full) - plen_g
    gold_batch = torch.tensor([full] * N_CLIENTS, device=DEV)
    with torch.no_grad():
        tf = m(input_ids=gold_batch)
    gl = gold_logits[pr]
    kl_pc = []
    for row in range(N_CLIENTS):
        chk = min(rounds[r]["gen_lens"][row], gen_g)
        seg = tf.logits[row, plen_g - 1:plen_g - 1 + chk, :]
        ks = tf_kl_row(gl[:chk], seg)
        kl_pc.append((float(ks.mean()), float(ks.max())))
    rounds[r]["kl_pc"] = kl_pc
    print("TFGATE round=%d prompt=%d kl_max=%.6f"
          % (r, pr, max(x[1] for x in kl_pc)), file=sys.stderr, flush=True)

for r in range(N_PROMPTS):
    for i, c in enumerate(conns):
        klm, klx = rounds[r]["kl_pc"][i]
        send_msg(c, {"output_ids": rounds[r]["outs"][i],
                     "gen": len(rounds[r]["outs"][i]),
                     "kl_mean": klm, "kl_max": klx})

tot_useful = sum(x["useful_tokens"] for x in rounds)
tot_steps = sum(x["token_steps"] for x in rounds)
tot_wall = sum(x["wall_s"] for x in rounds)
json.dump({"role": "executor", "n_clients": N_CLIENTS,
           "rounds": [{"round": r, "steps": rounds[r]["steps"],
                       "token_steps": rounds[r]["token_steps"],
                       "useful_tokens": rounds[r]["useful_tokens"],
                       "gen_lens": rounds[r]["gen_lens"],
                       "wall_s": rounds[r]["wall_s"]} for r in range(N_PROMPTS)],
           "total_useful_tokens": tot_useful, "total_token_steps": tot_steps,
           "total_wall_s": round(tot_wall, 4),
           "agg_tok_s": round(tot_useful / tot_wall, 3) if tot_wall else None,
           "batch_efficiency": round(tot_useful / tot_steps, 4) if tot_steps else None},
          open(RESULT, "w"), indent=2)
print("wrote " + RESULT, file=sys.stderr, flush=True)
for c in conns:
    c.close()
srv.close()
os.unlink(SOCK)
