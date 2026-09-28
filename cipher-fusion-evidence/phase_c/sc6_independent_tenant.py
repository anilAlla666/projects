#!/usr/bin/env python3
"""Track 2 SC6 — independent-load CONTROL tenant.

The counterfactual: a tenant that loads the model NORMALLY — private weights,
no CIPHER arena, no kmod, no sharing. N of these is what weight-sharing is
measured against. Each tenant loads the model, runs the canonical-prompt
forward, frees its scratch, signals READY, and holds (resident) until the
sentinel sc6_ind_done — so the orchestrator can snapshot the framebuffer with
all surviving tenants co-resident.

argv: [tenant_index] [model_name]
"""
import gc
import os
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
import torch                                                  # noqa: E402
import sc6_models as M                                        # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SENTINEL = HERE + "/sc6_ind_done"


def main():
    idx = int(sys.argv[1])
    model_name = sys.argv[2]
    spec = M.MODELS[model_name]
    path, dtype = spec["path"], spec["dtype"]
    tag = "IND-%d" % idx

    torch.manual_seed(0)
    try:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype).cuda()
        model.train(False)
        ids = tok(M.CANONICAL_PROMPT, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            y = model(ids).logits.float().cpu()
        torch.save(y, "%s/sc6_%s_independent_%d_logits.pt"
                   % (HERE, model_name, idx))
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print("%s OOM — %s" % (tag, str(e)[:120]), flush=True)
        sys.exit(3)        # the orchestrator reads rc==3 as "did not fit"

    print("%s-READY independent load complete" % tag, flush=True)

    t0 = time.time()
    while not os.path.exists(SENTINEL):
        if time.time() - t0 > 600:
            break
        time.sleep(0.25)
    print("%s exiting" % tag, flush=True)


if __name__ == "__main__":
    main()
