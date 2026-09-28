"""W.4b.2 plumbing-gap baselines (§5 item 7), vanilla, on THIS pod.

  MODE=inproc : ONE process, B=2 batched HF greedy decode (the in-process
                ceiling the cross-process prototype is compared against).
  MODE=solo   : ONE process, B=1 greedy decode (run two concurrently by the
                harness => the naive-2-concurrent substrate-attributable
                denominator).

Same model / PLEN / GEN as run_formb_proto.sh so the gap is apples-to-apples.
Writes the decode-window sentinel for the external power sampler.
"""
import os
import sys
import json
import time

MODEL = os.environ["WL_MODEL"]
MODE = os.environ["MODE"]
PLEN = int(os.environ["PLEN"])
GEN = int(os.environ["GEN_LEN"])
RESULT = os.environ["RESULT_JSON"]
SENT = os.environ["SENTINEL"]

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

DEV = "cuda"
NB = int(os.environ.get("NB", "2"))            # batch size for the in-process ceiling
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


def decode(ids):
    """Greedy decode GEN steps over a [B, PLEN] batch; return (decode_wall, tokens)."""
    B = ids.shape[0]
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1)
        cur = ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1, GEN):
            cp = torch.tensor([cur], device=DEV)
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past,
                    use_cache=True, cache_position=cp)
            past = out.past_key_values
            nt = out.logits[:, -1, :].argmax(-1)
            cur += 1
        torch.cuda.synchronize()
    return time.perf_counter() - t0, B * (GEN - 1)


tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

n = NB if MODE == "inproc" else 1
off = int(os.environ.get("PROMPT_OFFSET", "0"))   # so concurrent solos differ


def enc_ids(p):
    """Tokenize to EXACTLY PLEN (right-pad if short; tokenizers vary by model)."""
    t = tok(p, return_tensors="pt").input_ids[0]
    if t.shape[0] >= PLEN:
        return t[:PLEN]
    padv = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return torch.cat([t, torch.full((PLEN - t.shape[0],), padv, dtype=t.dtype)])


ids = torch.stack([enc_ids(PROMPTS[(off + i) % len(PROMPTS)])
                   for i in range(n)], 0).cuda()
# warm
with torch.no_grad():
    m(input_ids=ids, use_cache=True)
torch.cuda.synchronize()

with open(SENT, "w") as s:
    s.write("DECODE_START %.3f\n" % time.time())
wall, ntok = decode(ids)
with open(SENT, "a") as s:
    s.write("DECODE_END %.3f\n" % time.time())

json.dump({"mode": MODE, "B": n, "gen_len": GEN, "decode_wall_s": round(wall, 4),
           "decode_tokens": ntok, "decode_tok_s": round(ntok / wall, 3)},
          open(RESULT, "w"), indent=2)
print("BASELINE mode=%s B=%d decode_tok/s=%.1f wall=%.3fs"
      % (MODE, n, ntok / wall, wall), file=sys.stderr, flush=True)
