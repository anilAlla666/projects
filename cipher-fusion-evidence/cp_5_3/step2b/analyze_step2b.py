# CP 5.3 STEP 2B Step 1 — analyzer. Teacher-forced KL of each substrate arm's
# target logits vs the arm-a (substrate-OFF, FP16) gold, + the decision tree.
import json, torch, torch.nn.functional as F
D = "/home/ubuntu/cipher-fusion-evidence/cp_5_3/step2b"

arms = {a: json.load(open(f"{D}/arm_{a}.json")) for a in "abc"}
tf   = {a: torch.load(f"{D}/arm_{a}_tf_logits.pt")[0] for a in "abc"}   # [T,V]
gold = tf['a']
gp   = F.log_softmax(gold, dim=-1)
gold_argmax = gold.argmax(-1)

res = {"arms": {}, "tf_kl_vs_goldA": {}}
for a in "abc":
    r = arms[a]
    res["arms"][a] = {"accept_rate": round(r.get("accept_rate", -1), 4),
                      "rounds": r.get("rounds"), "tok_per_round": round(r.get("tok_per_round",0),2),
                      "wall_s": r.get("wall_s"), "text_head": r.get("text","")[:90]}
    qp = F.log_softmax(tf[a], dim=-1)
    kl = (gp.exp() * (gp - qp)).sum(-1)                    # KL(gold||arm) per pos
    top1 = (tf[a].argmax(-1) == gold_argmax).float().mean()
    res["tf_kl_vs_goldA"][a] = {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                                "target_top1_agree_vs_gold": float(top1)}

ra = res["arms"]["a"]["accept_rate"]
rb = res["arms"]["b"]["accept_rate"]
rc = res["arms"]["c"]["accept_rate"]
total_collapse = ra - rb
f1_recovered   = rc - rb
residual_gap   = ra - rc
res["decomposition"] = {
    "total_collapse_a_minus_b": round(total_collapse, 4),
    "f1_fix_recovered_c_minus_b": round(f1_recovered, 4),
    "f1_recovered_fraction": round(f1_recovered / max(total_collapse, 1e-9), 3),
    "residual_gap_a_minus_c": round(residual_gap, 4),
}
if rc >= 0.45:
    verdict = "(c)>=0.45 — F1 was the cause; STEP 2B closes by verification"
elif rc < 0.20:
    verdict = "(c)<0.20 — residual is architectural"
else:
    verdict = "(c) in [0.20,0.45) — PARTIAL: F1 dominant + residual to diagnose"
res["verdict"] = verdict
res["gates"] = {"arm_a_reproduces_~0.490": abs(ra-0.490) < 0.05,
                "arm_b_reproduces_collapse": rb < 0.10}

json.dump(res, open(f"{D}/step2b_step1_analysis.json","w"), indent=2)
print(json.dumps(res, indent=2))
