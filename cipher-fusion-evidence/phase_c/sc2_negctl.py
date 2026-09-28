# SC2 supplementary: tied-weights check + negative control.
# Negative control: corrupt a rebound (arena-resident) weight in place and
# confirm the forward output changes -> directly proves the forward READS
# the VMM arena memory (not a stale pre-rebind copy).
import sys, json
sys.path.insert(0, '/home/ubuntu/cipher_rt_phase4')
import torch
import cipher_kv_bridge as kvb
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='/home/ubuntu/models/TinyLlama-1.1B'
def dstr(dt): return {torch.float16:'float16',torch.bfloat16:'bfloat16',torch.float32:'float32'}[dt]

kvb.init(64*1024*1024)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
model.eval()
ids = tok("The history of computing spans several distinct eras, each defined by",
          return_tensors='pt').input_ids.cuda()

# tied-weights check (before rebind)
emb = model.model.embed_tokens.weight
lmh = model.lm_head.weight
tied = (emb.data_ptr() == lmh.data_ptr())
print("tie_word_embeddings (embed.data_ptr == lm_head.data_ptr):", tied)

params = list(model.named_parameters())
total = sum(p.numel()*p.element_size() for _,p in params)
arena = kvb.weight_arena_create(int(total)+16*1024*1024, 1)
for name,p in params:
    vt = arena.alloc(list(p.shape), p.element_size(), dstr(p.dtype))
    vt.copy_(p.data); p.data = vt

with torch.no_grad():
    logits_clean = model(ids).logits.float().cpu()

# negative control: perturb an arena-resident weight in place
target = model.model.layers[0].mlp.gate_proj.weight
pi = kvb.page_info(target.data_ptr())
assert pi and pi.get('kind')=='weight', "target not in arena"
with torch.no_grad():
    target.add_(1.0)                         # in-place mutate VMM memory
    logits_corrupt = model(ids).logits.float().cpu()

diff = float((logits_clean - logits_corrupt).abs().max())
print(f"negative control: corrupted layers.0.mlp.gate_proj.weight (arena page)")
print(f"  logits max-abs-diff clean-vs-corrupt: {diff:.4f}")
out = {"tied_weights": bool(tied),
       "n_param_tensors": len(params),
       "neg_control_target": "model.layers.0.mlp.gate_proj.weight",
       "neg_control_logits_diff": diff,
       "neg_control_pass": diff > 1.0}
print("RESULT", json.dumps(out))
json.dump(out, open('/home/ubuntu/cipher-fusion-evidence/phase_c/sc2_negctl_result.json','w'), indent=2)
