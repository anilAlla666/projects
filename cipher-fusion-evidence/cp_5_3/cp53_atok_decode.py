#!/usr/bin/env python3
# CP 5.3 STEP 2 gate A-tok — greedy decode token-ID capture.
#
# Runs a fixed greedy decode (deterministic: do_sample=False, num_beams=1,
# KV-cache on so every decode-step GEMM is M=1 -> Marlin-eligible) and dumps
# the generated token IDs as JSON. Run once per (model, libcipher_rt build):
# the partition-aware build vs the full-GPU build. The driver diffs the two
# token streams -> top-1 agreement rate + first-divergence index (§7 A-tok,
# option ii).
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
n_tokens   = int(sys.argv[2])
out_path   = sys.argv[3]

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16).cuda()
model.train(False)   # inference mode (no dropout); avoids the .eval() alias

prompt = ("The history of computing began with mechanical calculators and "
          "evolved through several distinct eras. In the first era,")
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False,
                         num_beams=1, use_cache=True,
                         pad_token_id=(tok.eos_token_id or 0))
gen = out[0, ids.shape[1]:].tolist()

json.dump({"model": model_path, "n_requested": n_tokens,
           "n_generated": len(gen), "prompt": prompt, "tokens": gen},
          open(out_path, "w"))
print(f"[atok] {model_path}: generated {len(gen)} tokens -> {out_path}",
      file=sys.stderr)
