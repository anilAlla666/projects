"""Phase A — three-pillar driver. PASS 1 (single-tenant) + multi-tenant gate
worker. DECODE regime. Adds logit-KL correctness vs clean-FP16 gold.

MODE=selftest : tiny teacher-forced reproduction check.
MODE=gold     : clean FP16, no substrate. Free-running greedy decode; dump
                gold token IDs (GOLD_JSON) AND per-position gold logits
                (GOLD_JSON sibling 'gold_logits.pt') for KL.
MODE=gate     : one arm / one tenant. Free-run (tok/s, watts-window, FLOPs)
                + teacher-forced gate: per-step argmax agreement AND per-step
                logit-KL(gold || substrate) vs the FP16 gold distribution.

Env: MODE WL_ID WL_MODEL [WL_BATCH=1] [WL_MAX_NEW=128] ARM WL_TENANT_ID
     OUT_JSON GOLD_JSON [CIPHER_SUBSTRATE_MD5]
"""
import os
import sys
import json
import time

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import torch.nn.functional as F
import tenant_register

MODE = os.environ.get("MODE", "gate")
WL_ID = os.environ.get("WL_ID", "WL00")
MODEL = os.environ["WL_MODEL"]
TENANT = os.environ.get("WL_TENANT_ID", "phasea")
ARM = os.environ.get("ARM", "vanilla")
BATCH = int(os.environ.get("WL_BATCH", "1"))
MAX_NEW = int(os.environ.get("WL_MAX_NEW", "128"))
OUT_JSON = os.environ.get("OUT_JSON", "/dev/null")
GOLD_JSON = os.environ.get("GOLD_JSON", "/tmp/phasea_gold.json")
GOLD_LOGITS = os.path.join(os.path.dirname(GOLD_JSON), "gold_logits.pt")
PEAK_FLOPS = 989.4e12

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

tenant = tenant_register.register(TENANT)
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()
N_PARAMS = sum(p.numel() for p in m.parameters())

import cipher_spec_decode
cipher_spec_decode.install()


def kl_gold_vs_subst(gold_logit, subst_logit):
    """KL(gold || substrate) in nats, fp32. Both [vocab] tensors."""
    lp = F.log_softmax(gold_logit.float(), dim=-1)
    lq = F.log_softmax(subst_logit.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum())


def free_run(prompt, capture_logits=False):
    """One timed batched greedy generation. Prompt replicated to WL_BATCH."""
    enc = tok([prompt] * BATCH, return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    torch.cuda.synchronize()
    t0 = time.time()
    p0 = time.perf_counter()
    gkw = dict(max_new_tokens=MAX_NEW, do_sample=False,
               pad_token_id=tok.eos_token_id)
    if capture_logits:
        gkw.update(output_logits=True, return_dict_in_generate=True)
    with torch.no_grad():
        out = m.generate(**enc, **gkw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - p0
    t1 = time.time()
    if capture_logits:
        seq0 = out.sequences[0].tolist()
        # out.logits: tuple(gen_len) of [batch, vocab]; keep row 0
        glog = torch.stack([l[0].half().cpu() for l in out.logits], dim=0)
    else:
        seq0 = out[0].tolist()
        glog = None
    gen_per = len(seq0) - plen
    total_gen = gen_per * BATCH
    st = dict(cipher_spec_decode._last_stats)
    prop = st.get("proposed")
    ar = (st.get("accepted") / prop) if prop else None
    g0 = seq0[plen:]
    distinct = (len(set(g0)) / len(g0)) if g0 else 0.0
    flops = 2.0 * N_PARAMS * BATCH * (plen + gen_per)
    return {"plen": plen, "full_ids": seq0, "gen_ids": g0,
            "gen_per": gen_per, "total_gen": total_gen,
            "wall_s": round(dt, 4), "tok_s": round(total_gen / max(dt, 1e-6), 4),
            "flops": flops, "t0_epoch": round(t0, 3), "t1_epoch": round(t1, 3),
            "accept_rate": ar, "distinct_ratio": round(distinct, 4),
            "text": tok.decode(g0), "gold_logits": glog}


def teacher_forced(gold_full_ids, plen, gold_logits):
    """Incremental KV-cache decode fed the gold prefix. Per step: argmax-vs-
    gold agreement + KL(gold||substrate). gold_logits: [n_gen, vocab] or None."""
    gold = torch.tensor([gold_full_ids], device="cuda")
    n_gen = gold.shape[-1] - plen
    gl = gold_logits.cuda() if gold_logits is not None else None
    past = None
    agree = 0
    mism = []
    kls = []
    with torch.no_grad():
        cur = gold[:, :plen]
        cache_pos = torch.arange(plen, device="cuda")
        attn = torch.ones((1, plen), device="cuda", dtype=torch.long)
        out = m(input_ids=cur, past_key_values=past, use_cache=True,
                cache_position=cache_pos, attention_mask=attn)
        past = out.past_key_values
        for i in range(n_gen):
            slogit = out.logits[0, -1]
            pred = int(slogit.argmax())
            g = int(gold[0, plen + i])
            if pred == g:
                agree += 1
            else:
                mism.append({"pos": i, "pred": pred, "gold": g})
            if gl is not None and i < gl.shape[0]:
                kls.append(kl_gold_vs_subst(gl[i], slogit))
            if i == n_gen - 1:
                break
            cur = gold[:, plen + i:plen + i + 1]
            cache_pos = torch.tensor([plen + i], device="cuda")
            attn = torch.ones((1, plen + i + 1), device="cuda", dtype=torch.long)
            out = m(input_ids=cur, past_key_values=past, use_cache=True,
                    cache_position=cache_pos, attention_mask=attn)
            past = out.past_key_values
    res = {"n_steps": n_gen, "agree": agree,
           "agreement": round(agree / n_gen, 6) if n_gen else 0.0,
           "mismatches": mism[:25]}
    if kls:
        ks = sorted(kls)
        res["kl"] = {"mean": round(sum(kls) / len(kls), 6),
                     "p50": round(ks[len(ks) // 2], 6),
                     "p95": round(ks[min(len(ks) - 1, int(0.95 * len(ks)))], 6),
                     "max": round(ks[-1], 6), "n": len(kls)}
    else:
        res["kl"] = None
    return res


if MODE == "selftest":
    enc = tok([PROMPTS[0]] * BATCH, return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    tf = teacher_forced(out[0].tolist(), plen, None)
    print("PREFLIGHT %s agreement=%.6f" % (WL_ID, tf["agreement"]),
          file=sys.stderr)
    sys.exit(0 if tf["agreement"] >= 0.999999 else 1)

for _p in PROMPTS:
    _we = tok([_p] * BATCH, return_tensors="pt").to("cuda")
    with torch.no_grad():
        m.generate(**_we, max_new_tokens=MAX_NEW, do_sample=False,
                   pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

if MODE == "gold":
    gold = {"wl_id": WL_ID, "model": MODEL, "batch": BATCH, "max_new": MAX_NEW,
            "n_params": N_PARAMS, "n_prompts": len(PROMPTS), "prompts": []}
    glog_store = {}
    for i, p in enumerate(PROMPTS):
        fr = free_run(p, capture_logits=True)
        gold["prompts"].append(
            {"prompt": i, "plen": fr["plen"], "full_ids": fr["full_ids"],
             "gen_ids": fr["gen_ids"], "gen_per": fr["gen_per"],
             "distinct_ratio": fr["distinct_ratio"], "text": fr["text"]})
        glog_store[i] = fr["gold_logits"]
        print("GOLD %s prompt=%d gen=%d distinct=%.3f logits=%s"
              % (WL_ID, i, fr["gen_per"], fr["distinct_ratio"],
                 tuple(fr["gold_logits"].shape)), file=sys.stderr)
    with open(GOLD_JSON, "w") as f:
        json.dump(gold, f, indent=2)
    torch.save(glog_store, GOLD_LOGITS)
    print("wrote %s + %s" % (GOLD_JSON, GOLD_LOGITS), file=sys.stderr)
    sys.exit(0)

# MODE == gate
with open(GOLD_JSON) as f:
    gold = json.load(f)
gold_by_prompt = {g["prompt"]: g for g in gold["prompts"]}
try:
    gold_logits = torch.load(GOLD_LOGITS, map_location="cpu")
except FileNotFoundError:
    gold_logits = {}

SENTINEL = "/tmp/cipher_phasea_%s.decode_window" % TENANT
with open(SENTINEL, "w") as _s:
    _s.write("DECODE_START %.3f\n" % time.time())

free = []
for i, p in enumerate(PROMPTS):
    fr = free_run(p)
    free.append(fr)
    print("RESULT %s prompt=%d tok_s=%.3f total_gen=%d wall=%.3fs"
          % (WL_ID, i, fr["tok_s"], fr["total_gen"], fr["wall_s"]),
          file=sys.stderr)

with open(SENTINEL, "a") as _s:
    _s.write("DECODE_END %.3f\n" % time.time())

results = []
for i, p in enumerate(PROMPTS):
    g = gold_by_prompt[i]
    gl = gold_logits.get(i) if isinstance(gold_logits, dict) else None
    tf = teacher_forced(g["full_ids"], g["plen"], gl)
    fr = free[i]
    results.append({"prompt": i, "plen": fr["plen"], "tok_s": fr["tok_s"],
        "wall_s": fr["wall_s"], "total_gen": fr["total_gen"],
        "gen_per": fr["gen_per"], "flops": fr["flops"],
        "t0_epoch": fr["t0_epoch"], "t1_epoch": fr["t1_epoch"],
        "accept_rate": fr["accept_rate"], "distinct_ratio": fr["distinct_ratio"],
        "tf_agreement": tf["agreement"], "tf_agree": tf["agree"],
        "tf_n_steps": tf["n_steps"], "kl": tf["kl"]})
    klm = tf["kl"]["mean"] if tf["kl"] else None
    print("TFGATE %s prompt=%d tf_agreement=%.4f kl_mean=%s"
          % (WL_ID, i, tf["agreement"],
             ("%.5f" % klm) if klm is not None else "n/a"), file=sys.stderr)

out = {"phase": "A", "wl_id": WL_ID, "arm": ARM, "model": MODEL,
       "batch": BATCH, "max_new": MAX_NEW, "n_params": N_PARAMS,
       "peak_flops": PEAK_FLOPS, "tenant": TENANT,
       "substrate_md5": os.environ.get("CIPHER_SUBSTRATE_MD5", "n/a"),
       "n_prompts": len(PROMPTS), "prompts": results}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("wrote " + OUT_JSON, file=sys.stderr)
