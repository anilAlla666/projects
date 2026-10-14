"""Print every writer grader's output on one correct and one wrong attempt per task (for reading, not scoring)."""
import glob
import json
import os

TASKS = ["react-lead-form", "hof-topology-interpenetration", "production-planning", "uefi-bootkit", "pretrain-shard-corruption"]
for t in TASKS:
    man = {e["entry_id"]: e for e in json.load(open(f"/root/vbench/pools/{t}/manifest.json")) if e["kept"]}
    dirs = {}
    for d in glob.glob(f"/root/vbench/runs/m1_eval/*/{t}/regrades/*__write-verifier-*"):
        if d.endswith(".log") or "__retry" in d:
            continue
        name = os.path.basename(d)
        if name not in dirs or os.path.getmtime(d) > os.path.getmtime(dirs[name]):
            dirs[name] = d
    for name, d in sorted(dirs.items()):
        shown = set()
        for rj in sorted(glob.glob(d + "/*/result.json")):
            r = json.load(open(rj))
            e = man.get(os.path.basename(r["config"]["source_trial"]["path"]))
            if not e or e["label"] in shown:
                continue
            shown.add(e["label"])
            rw = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            f = os.path.join(os.path.dirname(rj), "verifier", "test-stdout.txt")
            out = open(f, errors="replace").read() if os.path.exists(f) else "(no stdout)"
            exc = (r.get("exception_info") or {}).get("exception_message") or ""
            print(f"===== {t} | {name[:50]} | truth={e['label']} ({e['source']}) | reward={rw}")
            print((exc[-300:] + "\n") if exc else "", "\n".join(out.strip().splitlines()[-8:]))
