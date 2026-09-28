"""Phase B Session 2 — N-tenant batched-executor rollup. Usage: analyze_session2.py N"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1])
BASE = 0.488                                  # WL01 1-tenant vanilla tok/W
CEIL = {2: 0.921, 4: 1.922, 8: 3.781, 16: 7.289}   # in-process batch-scan tok/W
d = os.path.join(HERE, "n%d" % N)
r = json.load(open(os.path.join(d, "executor_result.json")))
sent = open("/tmp/cipher_batch_exec_n%d.decode_window" % N).read().split()
t0, t1 = float(sent[1]), float(sent[3])
pw = []
for l in open(os.path.join(d, "power.csv")):
    l = l.strip()
    if not l:
        continue
    ts, w = l.split(",")
    if t0 <= float(ts) <= t1:
        pw.append(float(w))
power = sum(pw) / len(pw) if pw else None
tps = r["agg_tok_s"]
tpw = tps / power if power else None
kls = []
for i in range(1, N + 1):
    p = os.path.join(d, "client_t%d.json" % i)
    if not os.path.exists(p):
        continue
    for x in json.load(open(p))["prompts"]:
        if x.get("kl_max") is not None:
            kls.append(x["kl_max"])
klmax = max(kls) if kls else None
ceil = CEIL.get(N)
res = {"N": N, "agg_tok_s": round(tps, 1), "power_w": round(power, 1),
       "agg_tok_w": round(tpw, 4), "vs_1tenant_baseline": round(tpw / BASE, 3),
       "in_process_ceiling_tok_w": ceil,
       "pct_of_ceiling": round(100 * tpw / ceil, 1) if ceil else None,
       "correctness_kl_max": klmax,
       "kl_gate": "PASS" if (klmax is not None and klmax <= 0.1) else "CHECK"}
json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
