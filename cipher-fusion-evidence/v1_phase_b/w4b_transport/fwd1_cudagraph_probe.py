"""FWD-1 — launch-bound decode-forward probe (memo §2/§5; fixed-position method).

W.4b.7 found the B=N Mistral-7B decode forward is ~52 ms/step and clock-independent.
QUESTION: launch-overhead bound (eliminable by collapsing the hundreds of per-layer
kernel launches into one graph) vs memory-latency bound (irreducible HBM round-trips)?

METHOD (advisor-directed, correctness-clean): measure ONE decode step at a FIXED cache
position P — full correct attention over slots 0..P — repeated K times WITHOUT advancing
the position. Because nothing varies between repeats, a CUDA-graph replay does the
*identical, full* computation every time (no truncated-attention cheat), so:
  - correctness gate is trivial: replay logits must equal the eager-at-P logits (KL~0);
  - same-work eager-vs-graph timing isolates pure launch overhead.
Verdict: graph replay >= 2x faster than eager-at-P -> the inter-kernel launch gaps were
the cost -> LAUNCH-BOUND (productionize-worth). replay ~= eager -> MEMORY-LATENCY-BOUND
(W.7 next). This sidesteps the growing-KV graph-capture problem (which bakes the
attention span — a real hard problem; the contaminated growing-loop compile/cudagraph
runs are discarded, see FWD_1_FINDINGS).

ONE (B, COND) per process. COND in {eager, compile, cudagraph}. Userspace measurement
only; anchors UNCHANGED.
"""
import os
import sys
import json
import time

MODEL = os.environ["WL_MODEL"]
B = int(os.environ["B"])
COND = os.environ["COND"]                       # eager | compile | cudagraph
PLEN = int(os.environ.get("PLEN", "12"))
P_OFF = int(os.environ.get("P_OFF", "200"))     # fixed measurement position = PLEN+P_OFF
KREP = int(os.environ.get("KREP", "64"))        # timed reps at fixed P
WARM = int(os.environ.get("FWD_WARM", "12"))    # reps skipped before timing window
KL_TOL = float(os.environ.get("FWD_KL_TOL", "7e-5"))
RESULT = os.environ["RESULT_JSON"]
SENT = os.environ["SENTINEL"]

import torch                                     # noqa: E402
import torch.nn.functional as F                  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache  # noqa: E402

DEV = "cuda"
P = PLEN + P_OFF
MAXLEN = P + 4
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
VOCAB = m.config.vocab_size


def enc_ids(p):
    t = tok(p, return_tensors="pt").input_ids[0]
    if t.shape[0] >= PLEN:
        return t[:PLEN]
    padv = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return torch.cat([t, torch.full((PLEN - t.shape[0],), padv, dtype=t.dtype)])


ids = torch.stack([enc_ids(PROMPTS[i % len(PROMPTS)]) for i in range(B)], 0).cuda()


def eager_step(col, cache, cp):
    with torch.no_grad():
        return m(input_ids=col, past_key_values=cache, use_cache=True,
                 cache_position=cp).logits[:, -1, :]


# ---- fill the cache correctly to slots 0..P-1 (eager growing decode = SETUP) --
cache = StaticCache(config=m.config, max_cache_len=MAXLEN)
with torch.no_grad():
    out = m(input_ids=ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(PLEN, device=DEV))
nt = out.logits[:, -1, :].argmax(-1)             # token to write at slot PLEN
growing_ms = []
cur = PLEN
while cur < P:                                   # advance, filling slots PLEN..P-1
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    cp = torch.tensor([cur], device=DEV)
    ev0.record()
    lg = eager_step(nt.unsqueeze(-1), cache, cp)
    ev1.record()
    torch.cuda.synchronize()
    growing_ms.append(ev0.elapsed_time(ev1))
    nt = lg.argmax(-1)
    cur += 1
# cache now has slots 0..P-1 valid; `nt` is the token to feed at the fixed step (cp=P).
feed = nt.unsqueeze(-1).clone()                  # [B,1] constant token for the fixed step


def setlen(v):
    # StaticCache.get_seq_length() auto-increments on EVERY forward (even a re-write of
    # the same slot). Repeating the fixed step would grow it past max_cache_len -> OOB,
    # AND grow the attended span (attending unwritten zero slots) -> wrong logits. Reset
    # the per-layer cumulative_length to P before each rep so every rep is the IDENTICAL,
    # full, correct computation (attends exactly 0..P). (Graph replay doesn't run this
    # python path, so the captured span is fixed at the value set before capture.)
    for L in cache.layers:
        if torch.is_tensor(L.cumulative_length):
            L.cumulative_length.fill_(v)
        else:
            L.cumulative_length = v
# growing-loop guardrail: mean step time near position P (eager, full correct work)
growing_mean = sum(growing_ms[-32:]) / len(growing_ms[-32:]) if growing_ms else 0.0

# ---- eager REFERENCE logits at the fixed step (cache_position = P) ------------
cp_P = torch.tensor([P], device=DEV)
ref_logits = eager_step(feed, cache, cp_P).float().clone()   # writes slot P
torch.cuda.synchronize()


# ---- build the fixed-P step per condition ------------------------------------
note = ""
capture_ok = True
if COND == "eager":
    def pre_rep():
        setlen(P)                                # reset counter; not timed

    def fixed_step():
        return eager_step(feed, cache, cp_P)

elif COND == "compile":
    import torch._inductor.config as _ind         # noqa: E402
    _ind.triton.cudagraph_trees = False
    cmodel = torch.compile(m, mode="reduce-overhead")

    def pre_rep():
        setlen(P)

    def fixed_step():
        with torch.no_grad():
            return cmodel(input_ids=feed, past_key_values=cache, use_cache=True,
                          cache_position=cp_P).logits[:, -1, :]

elif COND == "cudagraph":
    static_out = None
    g = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    setlen(P)                    # keep warmup span at exactly 0..P
                    m(input_ids=feed, past_key_values=cache, use_cache=True,
                      cache_position=cp_P)
        torch.cuda.current_stream().wait_stream(s)
        setlen(P)                                # capture with span fixed at 0..P
        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                static_out = m(input_ids=feed, past_key_values=cache, use_cache=True,
                               cache_position=cp_P).logits[:, -1, :]
    except Exception as e:
        capture_ok = False
        note = "CUDA-graph capture FAILED: %r" % e

    if capture_ok:
        # guarded probe: a single replay + correctness BEFORE the timed loop. Manual
        # capture of HF decode often OOBs on replay (the captured seqlen/mask state is
        # not graph-safe for the growing cache) -> report as a finding, don't force it.
        try:
            g.replay()
            torch.cuda.synchronize()
            _q = static_out.float()
            _kl = float((F.log_softmax(ref_logits, -1).exp()
                         * (F.log_softmax(ref_logits, -1) - F.log_softmax(_q, -1))).sum(-1).max())
            if _kl > KL_TOL:
                capture_ok = False
                note = ("manual capture replay NOT bit-correct (kl=%.3g): HF decode "
                        "graph state not replay-safe; supported path is "
                        "compile(reduce-overhead)." % _kl)
        except Exception as e:
            capture_ok = False
            note = ("manual capture replay RAISED (%r): HF decode growing-KV not "
                    "graph-safe; supported path is compile(reduce-overhead)." % e)

    if capture_ok:
        def pre_rep():
            pass                                 # replay re-runs captured kernels only

        def fixed_step():
            g.replay()
            return static_out
    else:
        # report-only failure: write the finding JSON and exit (avoid a poisoned-context
        # crash in the timed loop). cudagraph signal is carried by the compile condition.
        json.dump({"cond": COND, "B": B, "P": P, "capture_ok": False, "note": note,
                   "correctness_pass": False, "mean_ms_per_step": None}, open(RESULT, "w"),
                  indent=2, default=str)
        print("FWD1fixedP cond=cudagraph B=%d P=%d FAILED: %s" % (B, P, note),
              file=sys.stderr, flush=True)
        sys.exit(0)
else:
    raise SystemExit("unknown COND %s" % COND)

# ---- timed reps at fixed P (no advance) + correctness ------------------------
last = None
evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
       for _ in range(KREP)]
with open(SENT, "w") as f:
    f.write("DECODE_START %.3f\n" % time.time())
for k in range(KREP):
    pre_rep()                                    # counter reset (eager/compile); not timed
    evs[k][0].record()
    last = fixed_step()
    evs[k][1].record()
torch.cuda.synchronize()
with open(SENT, "a") as f:
    f.write("DECODE_END %.3f\n" % time.time())

rep_ms = [evs[k][0].elapsed_time(evs[k][1]) for k in range(KREP)]
steady = rep_ms[WARM:] if len(rep_ms) > WARM else rep_ms
mean_ms = sum(steady) / len(steady) if steady else 0.0
tok_s = (B * 1000.0 / mean_ms) if mean_ms else 0.0

# correctness: fixed-P output must equal eager-at-P reference (identical computation)
q = last.float()
lp = F.log_softmax(ref_logits, dim=-1)
lq = F.log_softmax(q, dim=-1)
kl = (lp.exp() * (lp - lq)).sum(-1)              # [B]
kl_max = float(kl.max())
match = int((ref_logits.argmax(-1) == q.argmax(-1)).sum())
passed = (kl_max <= KL_TOL)
speedup_vs_eager_growing = (growing_mean / mean_ms) if mean_ms else 0.0

res = {"cond": COND, "B": B, "plen": PLEN, "P": P, "krep": KREP, "warm": WARM,
       "capture_ok": capture_ok, "note": note,
       "mean_ms_per_step": round(mean_ms, 4), "tok_s": round(tok_s, 2),
       "growing_eager_mean_ms_nearP": round(growing_mean, 4),
       "kl_max": kl_max, "greedy_match": match, "greedy_total": B,
       "kl_tol": KL_TOL, "correctness_pass": bool(passed)}
json.dump(res, open(RESULT, "w"), indent=2, default=str)
print("FWD1fixedP cond=%s B=%d P=%d ms/step=%.3f tok/s=%.1f kl_max=%.2e match=%d/%d "
      "growing_eager@P=%.3f capture_ok=%s pass=%s %s"
      % (COND, B, P, mean_ms, tok_s, kl_max, match, B, growing_mean, capture_ok,
         passed, note), file=sys.stderr, flush=True)
