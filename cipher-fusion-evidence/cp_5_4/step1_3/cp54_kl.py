"""CP 5.4 Step 1.3 Phase D — teacher-forced KL gate (THROWAWAY).
Usage: cp54_kl.py <baseline.pt> <test.pt> <label>
KL(baseline || test) per token position over the prompt logits; gate max <= 0.1.
Also reports whether the 24-token greedy decode ids match exactly.
"""
import sys, torch
import torch.nn.functional as F

base = torch.load(sys.argv[1]); test = torch.load(sys.argv[2]); label = sys.argv[3]
pb = F.log_softmax(base['logits'][0], dim=-1)
qt = F.log_softmax(test['logits'][0], dim=-1)
kl = (pb.exp() * (pb - qt)).sum(-1)
kl_max, kl_mean = float(kl.max()), float(kl.mean())
ids_match = base['gen_ids'] == test['gen_ids']
gate = kl_max <= 0.1
print(f"[{label}] KL max={kl_max:.3e} mean={kl_mean:.3e}  gen_ids_match={ids_match}  "
      f"KL_GATE(<=0.1)={'PASS' if gate else 'FAIL'}")
sys.exit(0 if (gate and ids_match) else 1)
