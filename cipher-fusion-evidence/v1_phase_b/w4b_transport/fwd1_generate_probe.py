"""FWD-1 — launch-bound decode-forward probe (generate()-based; supported path).

Supersedes the manual fixed-P-repeat probe (fwd1_cudagraph_probe.py), which fought
StaticCache's seqlen/mask machinery (worked on TinyLlama, corrupted Mistral's
sliding-window+GQA state -> wrong, less work, untrustworthy). Per advisor: use HF
generate(), which handles cache/mask/position correctly, so correctness = do the token
sequences match, and timing = wall/max_new_tokens (decode-dominated).

THREE conditions (ONE per process):
  eager_dynamic  : generate(), default DynamicCache (re-alloc each step) = the W.4b.7-era path
  eager_static   : generate(), cache_implementation="static" (pre-alloc)
  compile_static : eager_static + model.forward = torch.compile(reduce-overhead)  [= cudagraphs]

eager_static vs eager_dynamic -> the cache-impl effect (the W.4b.7 52ms gap finding).
compile_static vs eager_static -> the graph/launch-elimination effect (the §3 verdict).
Correctness: each process emits its generated token ids; the runner diffs them vs
eager_dynamic (greedy must match, modulo benign near-ties). Userspace only; anchors
UNCHANGED.
"""
import os
import sys
import json
import time

MODEL = os.environ["WL_MODEL"]
B = int(os.environ["B"])
COND = os.environ["COND"]                        # eager_dynamic | eager_static | compile_static
MAXNEW = int(os.environ.get("MAXNEW", "256"))
NREP = int(os.environ.get("NREP", "3"))          # timed generate() repeats (report min+mean)
NWARM = int(os.environ.get("NWARM", "1"))        # warmup generate()s (trigger compile/capture)
RESULT = os.environ["RESULT_JSON"]
SENT = os.environ["SENTINEL"]

import torch                                      # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

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
tok.padding_side = "left"                         # required for correct batched generation
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

prompts = [PROMPTS[i % len(PROMPTS)] for i in range(B)]
enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
in_len = enc["input_ids"].shape[1]

# greedy, deterministic
gen_kw = dict(max_new_tokens=MAXNEW, do_sample=False, num_beams=1,
              pad_token_id=tok.pad_token_id)

if COND == "eager_dynamic":
    m.generation_config.cache_implementation = None      # DynamicCache (default)
elif COND == "eager_static":
    m.generation_config.cache_implementation = "static"
elif COND == "compile_static":
    m.generation_config.cache_implementation = "static"
    import torch._inductor.config as _ind                # noqa: E402
    _ind.triton.cudagraph_trees = False
    m.forward = torch.compile(m.forward, mode="reduce-overhead")
else:
    raise SystemExit("unknown COND %s" % COND)


def one_generate():
    with torch.no_grad():
        return m.generate(**enc, **gen_kw)


# ---- warmup (trigger compile + cudagraph capture; stabilize clocks) ----------
for _ in range(NWARM):
    _ = one_generate()
torch.cuda.synchronize()

# ---- timed generate() repeats (windowed for the power sampler) ---------------
walls = []
with open(SENT, "w") as f:
    f.write("DECODE_START %.3f\n" % time.time())
out_ids = None
for _ in range(NREP):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ids = one_generate()
    torch.cuda.synchronize()
    walls.append(time.perf_counter() - t0)
with open(SENT, "a") as f:
    f.write("DECODE_END %.3f\n" % time.time())

min_wall = min(walls)
mean_wall = sum(walls) / len(walls)
# per-new-token decode time (prefill is <1% at MAXNEW>=256); tok/s over the batch
ms_per_tok = 1000.0 * min_wall / MAXNEW
tok_s = B * MAXNEW / min_wall
new_tokens = out_ids[:, in_len:].tolist()         # [B, MAXNEW] generated ids for the diff

res = {"cond": COND, "B": B, "model": MODEL, "in_len": in_len, "max_new": MAXNEW,
       "nrep": NREP, "min_wall_s": round(min_wall, 4), "mean_wall_s": round(mean_wall, 4),
       "ms_per_new_token": round(ms_per_tok, 4), "tok_s": round(tok_s, 2),
       "new_tokens": new_tokens}
json.dump(res, open(RESULT, "w"))
print("FWD1gen cond=%s B=%d ms/tok=%.3f tok/s=%.1f min_wall=%.3fs (in_len=%d new=%d)"
      % (COND, B, ms_per_tok, tok_s, min_wall, in_len, MAXNEW), file=sys.stderr, flush=True)
