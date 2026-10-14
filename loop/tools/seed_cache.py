"""Seed the evaluation cache with rewards already computed: each pool's label check (the original tests on the
official image) and any preview regrade. Run on the server from the repo root with PYTHONPATH=."""
import hashlib
import json
import sys
from pathlib import Path

from vb.evaluate import CACHE
from vb.pool import load_manifest
from vb.regrade import read_rewards, tests_hash
from vb.util import TB_ROOT, write_json

for task in json.load(open("configs/m1.json"))["tasks"]:
    kept = [e for e in load_manifest(task) if e["kept"]]
    pool_key = hashlib.sha256(",".join(sorted(e["entry_id"] for e in kept)).encode()).hexdigest()[:12]
    r = read_rewards(Path(f"/root/vbench/pools/{task}/label_check/regrades/original"))
    write_json(CACHE / task / f"original__{tests_hash(TB_ROOT / task / 'tests')}__{pool_key}.json", r)
    print(task, "original seeded:", len(r), "rewards,", len(kept), "kept")
# extra: <task> <writer>/<trial> <regraded job dir>
for arg in sys.argv[1:]:
    task, name, regraded = arg.split(":")
    kept = [e for e in load_manifest(task) if e["kept"]]
    pool_key = hashlib.sha256(",".join(sorted(e["entry_id"] for e in kept)).encode()).hexdigest()[:12]
    writer, trial = name.split("/")
    tests = Path(f"/root/vbench/verifiers/{writer}/{task}/{trial}/tests")
    r = read_rewards(Path(regraded))
    write_json(CACHE / task / f"{writer}__{trial}__{tests_hash(tests)}__{pool_key}.json", r)
    print(task, name, "seeded:", len(r))
