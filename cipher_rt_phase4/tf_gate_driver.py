"""CP 5.6 Priority 2 — teacher-forced correctness gate + free-running TPW
re-measurement driver (finding F1 re-measurement).

MODE=selftest : clean FP16, no substrate. Pre-flight — generate() an 8-token
            sequence, then teacher_forced() it; assert exact agreement. Proves
            the KV-cache teacher-forced loop reproduces generate() before any
            long GPU run is committed.
MODE=gold : clean FP16, no substrate. Free-running greedy decode; dump the
            128-token gold sequence (token IDs + text) per prompt to GOLD_JSON.
MODE=gate : selected arm (env, same toggles as CP 2.4 run_composed.sh):
   (1) free-running greedy decode, timed + power-windowed via DECODE_START/END
       sentinel  -> production TPW measurement (tok/s, ids, text, accept_rate);
   (2) teacher-forced check -> incremental KV-cache decode FED THE GOLD tokens
       (cascade-free; identical forward path to production, so FP-tie-noise-
       free). Per-step argmax-vs-gold agreement over the 128 gold positions.
   Writes per-prompt JSON: free-run metrics + tf agreement -> OUT_JSON.

Arm selection (set by run_cp56_p2.sh): vanilla = no substrate; marlin/allon =
LD_PRELOAD libcipher_rt + CIPHER_* toggles. The driver itself is arm-agnostic.
"""
import os
import sys
import json
import time

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register

MODE = os.environ.get("MODE", "gate")              # selftest | gold | gate
MODEL = os.environ["WL_MODEL"]
TENANT = os.environ.get("WL_TENANT_ID", "cp56p2")
ARM = os.environ.get("ARM", "vanilla")
OUT_JSON = os.environ.get("OUT_JSON", "/dev/null")
GOLD_JSON = os.environ.get("GOLD_JSON", "/tmp/cp56p2_gold.json")
MAX_NEW = int(os.environ.get("WL_MAX_NEW", "128"))

# Verbatim from spec_varied_driver.py — the CP 2.4 varied 5-prompt set, open-
# ended so a fixed 128-token budget does not hit EOS early.
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

# Operator-policy injection — install() no-ops when CIPHER_SPEC=0, so the same
# driver measures the vanilla / marlin baselines and the spec-on all-on arm.
import cipher_spec_decode
cipher_spec_decode.install()


def free_run(prompt):
    """One timed greedy 128-token generation — the production decode path."""
    enc = tok(prompt, return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    torch.cuda.synchronize()
    t0 = time.time()
    p0 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - p0
    t1 = time.time()
    seq = out[0].tolist()
    gen = seq[plen:]
    st = dict(cipher_spec_decode._last_stats)      # {} on non-spec arms
    prop = st.get("proposed")
    ar = (st.get("accepted") / prop) if prop else None
    distinct = (len(set(gen)) / len(gen)) if gen else 0.0
    return {"plen": plen, "full_ids": seq, "gen_ids": gen,
            "gen_tokens": len(gen), "wall_s": round(dt, 4),
            "tok_s": round(len(gen) / max(dt, 1e-6), 4),
            "t0_epoch": round(t0, 3), "t1_epoch": round(t1, 3),
            "accept_rate": ar, "distinct_ratio": round(distinct, 4),
            "text": tok.decode(gen)}


def teacher_forced(gold_full_ids, plen):
    """Incremental KV-cache decode FED THE GOLD prefix at every step.

    Step i consumes gold[plen+i-1] (never the model's own argmax) -> the
    chain cannot cascade. The KV-cache incremental path is numerically the
    SAME forward as free-running greedy decode (explicit cache_position +
    attention_mask match what m.generate() passes), so for the vanilla arm
    this reproduces the gold run exactly (agreement == 100%); substrate arms
    diverge only by genuine substrate numerics. Returns per-step argmax-vs-
    gold agreement over the gold generation positions.
    """
    gold = torch.tensor([gold_full_ids], device="cuda")
    n_gen = gold.shape[-1] - plen
    past = None
    agree = 0
    mism = []
    with torch.no_grad():
        # prompt forward — cache empty, sequence length == plen
        cur = gold[:, :plen]
        cache_pos = torch.arange(plen, device="cuda")
        attn = torch.ones((1, plen), device="cuda", dtype=torch.long)
        out = m(input_ids=cur, past_key_values=past, use_cache=True,
                cache_position=cache_pos, attention_mask=attn)
        past = out.past_key_values
        pred = int(out.logits[0, -1].argmax())
        for i in range(n_gen):
            g = int(gold[0, plen + i])
            if pred == g:
                agree += 1
            else:
                mism.append({"pos": i, "pred": pred, "gold": g})
            if i == n_gen - 1:
                break
            # teacher-force: next input is the GOLD token, not `pred`
            cur = gold[:, plen + i:plen + i + 1]
            cache_pos = torch.tensor([plen + i], device="cuda")
            attn = torch.ones((1, plen + i + 1), device="cuda", dtype=torch.long)
            out = m(input_ids=cur, past_key_values=past, use_cache=True,
                    cache_position=cache_pos, attention_mask=attn)
            past = out.past_key_values
            pred = int(out.logits[0, -1].argmax())
    return {"n_steps": n_gen, "agree": agree,
            "agreement": round(agree / n_gen, 6) if n_gen else 0.0,
            "mismatches": mism[:25]}


if MODE == "selftest":
    # Pre-flight: prove teacher_forced() reproduces m.generate() exactly on a
    # tiny clean-FP16 case, BEFORE committing a 30-min full run.
    enc = tok(PROMPTS[0], return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    full = out[0].tolist()
    tf = teacher_forced(full, plen)
    print("PREFLIGHT plen=%d n_steps=%d agreement=%.6f mismatches=%s"
          % (plen, tf["n_steps"], tf["agreement"], tf["mismatches"]),
          file=sys.stderr)
    if tf["agreement"] >= 0.999999:
        print("PREFLIGHT_OK", file=sys.stderr)
        sys.exit(0)
    print("PREFLIGHT_FAIL -- teacher_forced does not reproduce generate()",
          file=sys.stderr)
    sys.exit(1)

# Warm — run the full prompt set once at the timed budget, untimed. Pays every
# one-time cost up front (Marlin lazy quant + NVRTC compile, cuDNN/Marlin JIT
# across the decode kv_len range, spec-arm draft load) so the timed free-run
# and the teacher-forced loop both observe a fully warm, Marlin-active substrate.
for _p in PROMPTS:
    _we = tok(_p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        m.generate(**_we, max_new_tokens=MAX_NEW, do_sample=False,
                   pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

if MODE == "gold":
    gold = {"model": MODEL, "max_new": MAX_NEW, "n_prompts": len(PROMPTS),
            "prompts": []}
    for i, p in enumerate(PROMPTS):
        fr = free_run(p)
        gold["prompts"].append(
            {"prompt": i, "plen": fr["plen"], "full_ids": fr["full_ids"],
             "gen_ids": fr["gen_ids"], "gen_tokens": fr["gen_tokens"],
             "distinct_ratio": fr["distinct_ratio"], "text": fr["text"]})
        print("GOLD prompt=%d gen_tokens=%d distinct_ratio=%.3f"
              % (i, fr["gen_tokens"], fr["distinct_ratio"]), file=sys.stderr)
    with open(GOLD_JSON, "w") as f:
        json.dump(gold, f, indent=2)
    print("wrote " + GOLD_JSON, file=sys.stderr)
    sys.exit(0)

# MODE == gate
with open(GOLD_JSON) as f:
    gold = json.load(f)
gold_by_prompt = {g["prompt"]: g for g in gold["prompts"]}

# Power-sentinel window — run_cp56_p2.sh keys its nvidia-smi sampler off
# DECODE_START / DECODE_END so watts.csv covers exactly the free-run loop
# (all 5 prompts). The teacher-forced loop runs AFTER DECODE_END and is
# intentionally not power-measured: TPW is a free-running production figure.
SENTINEL = "/tmp/cipher_cp56p2_%s.decode_window" % TENANT
with open(SENTINEL, "w") as _s:
    _s.write("DECODE_START %.3f\n" % time.time())

free = []
for i, p in enumerate(PROMPTS):
    fr = free_run(p)
    free.append(fr)
    print("RESULT prompt=%d tok_s=%.3f gen_tokens=%d wall=%.3fs "
          "accept_rate=%s distinct=%.3f"
          % (i, fr["tok_s"], fr["gen_tokens"], fr["wall_s"],
             ("%.3f" % fr["accept_rate"]) if fr["accept_rate"] is not None
             else "n/a", fr["distinct_ratio"]), file=sys.stderr)

with open(SENTINEL, "a") as _s:
    _s.write("DECODE_END %.3f\n" % time.time())

results = []
for i, p in enumerate(PROMPTS):
    g = gold_by_prompt[i]
    tf = teacher_forced(g["full_ids"], g["plen"])
    fr = free[i]
    row = {"prompt": i, "plen": fr["plen"],
           "tok_s": fr["tok_s"], "wall_s": fr["wall_s"],
           "gen_tokens": fr["gen_tokens"],
           "t0_epoch": fr["t0_epoch"], "t1_epoch": fr["t1_epoch"],
           "accept_rate": fr["accept_rate"],
           "distinct_ratio": fr["distinct_ratio"],
           "free_run_text": fr["text"], "free_run_gen_ids": fr["gen_ids"],
           "tf_agreement": tf["agreement"], "tf_agree": tf["agree"],
           "tf_n_steps": tf["n_steps"], "tf_mismatches": tf["mismatches"]}
    results.append(row)
    print("TFGATE prompt=%d tf_agreement=%.4f (%d/%d)"
          % (i, tf["agreement"], tf["agree"], tf["n_steps"]), file=sys.stderr)

out = {"cp": "5.6", "priority": "P2", "arm": ARM, "model": MODEL,
       "substrate_md5": os.environ.get("CIPHER_SUBSTRATE_MD5", "n/a"),
       "tenant": TENANT, "max_new": MAX_NEW, "n_prompts": len(PROMPTS),
       "prompts": results}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("wrote " + OUT_JSON, file=sys.stderr)
