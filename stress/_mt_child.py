
import os, sys, time, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mp = sys.argv[1]; n_iter=int(sys.argv[2]); max_new=int(sys.argv[3]); out=sys.argv[4]
tok = AutoTokenizer.from_pretrained(mp)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16, device_map='cuda:0')
total_new=0; total_t=0.0
prompts=['hi there','tell me a joke','what is python','how are you']
for i in range(n_iter):
    for p in prompts:
        ids = tok(p, return_tensors='pt').input_ids.to('cuda:0')
        t0=time.perf_counter()
        o = m.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        total_t += time.perf_counter()-t0
        total_new += int(o.shape[1]-ids.shape[1])
res = {'tenant': os.environ.get('CIPHER_TENANT_ID','?'),
       'pid': os.getpid(), 'tok_per_s': total_new/max(total_t,1e-6),
       'total_new': total_new, 'total_t': total_t}
open(out,'w').write(json.dumps(res))
