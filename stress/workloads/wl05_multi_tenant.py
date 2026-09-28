"""WL05 — Multi-tenant. N child processes share GPU 0."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import write_result, sample_power, avg_power, REPO_ROOT


CHILD = """
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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="baseline")
    ap.add_argument("--model", default="/home/ubuntu/models/Llama-3.2-1B")
    ap.add_argument("--n-tenants", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--n-iter", type=int, default=2)
    args = ap.parse_args()

    cscript = Path("stress/_mt_child.py"); cscript.write_text(CHILD)
    samples = []; stop_evt = threading.Event()
    threading.Thread(target=sample_power, args=(stop_evt, samples, 0), daemon=True).start()
    procs = []; outs = []
    t0 = time.perf_counter()
    for i in range(args.n_tenants):
        env = os.environ.copy()
        env["CIPHER_TENANT_ID"] = f"t{i}"
        env["USE_TF"] = "0"; env["TRANSFORMERS_OFFLINE"]="1"
        opath = f"stress/results/_mt_t{i}.json"
        outs.append((f"t{i}", opath))
        p = subprocess.Popen(
            [sys.executable, str(cscript), args.model, str(args.n_iter),
             str(args.max_new), opath],
            env=env)
        procs.append(p)
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0
    stop_evt.set()
    pw = avg_power(samples)

    rows = []
    for tid, op in outs:
        try:
            r = json.loads(Path(op).read_text())
        except Exception as e:
            r = {"tenant": tid, "error": str(e)}
        rows.append(r)
    tps_vals = [r["tok_per_s"] for r in rows if "tok_per_s" in r]
    spread = (max(tps_vals)/min(tps_vals)) if (tps_vals and min(tps_vals)>0) else 0
    summary = {"wid":"wl05","phase":args.phase,"n_tenants":args.n_tenants,
               "wall_seconds":wall,"avg_watts":pw,
               "tok_per_s_total":sum(tps_vals),
               "tok_per_s_max_min_spread":spread,
               "rows":rows}
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}, indent=2))
    write_result("wl05", args.phase, summary)


if __name__ == "__main__":
    main()
