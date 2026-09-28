#!/usr/bin/env python3
# CP 5.3 STEP 2 A-tok investigation — batched greedy decode.
#
# Same fixed greedy decode as cp53_atok_decode.py, but the prompt is
# replicated to a batch of B identical rows so every decode-step GEMM is
# M=B -> exercises Marlin's designed B>=8 regime (cp53_atok_decode.py is
# B=1, outside that regime). All B rows are identical so row 0's token
# stream is the decode under test. Dumps row-0 token IDs as JSON.
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1]
n_tokens   = int(sys.argv[2])
out_path   = sys.argv[3]
batch      = int(sys.argv[4]) if len(sys.argv) > 4 else 8

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16).cuda()
model.train(False)

prompt = ("The history of computing began with mechanical calculators and "
          "evolved through several distinct eras. In the first era,")
ids1 = tok(prompt, return_tensors="pt").input_ids.cuda()
ids = ids1.repeat(batch, 1)   # B identical rows

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False,
                         num_beams=1, use_cache=True,
                         pad_token_id=(tok.eos_token_id or 0))
gen = out[0, ids.shape[1]:].tolist()   # row 0

json.dump({"model": model_path, "n_requested": n_tokens, "batch": batch,
           "n_generated": len(gen), "prompt": prompt, "tokens": gen},
          open(out_path, "w"))
print(f"[atok-b{batch}] {model_path}: generated {len(gen)} tokens -> {out_path}",
      file=sys.stderr)
