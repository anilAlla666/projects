"""Deploy the correct-by-construction writer tasks (blind and constructor) to HUD and write their rows."""
import json, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.hud_publish import LOCAL, SERVER, adapt_one, deploy
from tools.hud_experiment import HUD, adapted_row

TASKS = [3, 4, 5]
rows = []
for cond in ("blind", "constructor"):
    ds = LOCAL / f"cbc-{cond}"
    ds.mkdir(parents=True, exist_ok=True)
    for k in TASKS:
        name = f"write-verifier-cbc-portion-{k}"
        subprocess.run(["rsync", "-a", "--delete", "--partial", "--timeout=120",
                        f"{SERVER}:/root/vbench/cbc_hud/{cond}/{name}", str(ds) + "/"], check=True)
        d, _ = adapt_one(ds / name)
        deploy(HUD, d)
        r = adapted_row(ds, f"cbc-portion-{k}")
        r["slug"] = f"{cond}__cbc-portion-{k}"
        r["verifier"]["slug"] = f"{cond}__cbc-portion-{k}:verify"
        rows.append(r)
        print("deployed", cond, k, d.name, flush=True)
(LOCAL / "exp_rows_cbc.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "rows")
