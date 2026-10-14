"""Rerun, one at a time on a quiet server, the correct uefi-bootkit attempts that the official tests failed through
our build route under load. Same route (build_image, offline); only the load differs."""
import glob
import json
import os
import shutil
from pathlib import Path

from vb.pool import _job_config, load_manifest, pool_dir
from vb.regrade import grade, verdict
from vb.util import TB_ROOT

task = "uefi-bootkit"
man = {e["entry_id"]: e for e in load_manifest(task)}
failed = []
for rj in glob.glob(f"/root/vbench/positive_control/{task}/regrades/original_built/*/result.json"):
    r = json.load(open(rj))
    e = man[os.path.basename(r["config"]["source_trial"]["path"])]
    rw = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if e["label"] == "pass" and rw != 1.0:
        failed.append(e["entry_id"])
job = Path("/root/vbench/positive_control/uefi_quiet_job")
shutil.rmtree(job, ignore_errors=True)
job.mkdir(parents=True)
for eid in failed:
    shutil.copytree(pool_dir(task) / "job" / eid, job / eid, symlinks=True)
_job_config(job, task)
rewards = grade(job, TB_ROOT / task, TB_ROOT / task / "tests", Path("/root/vbench/positive_control/uefi_quiet"),
                "original_built_quiet", n_concurrent=1)
for eid in failed:
    print(eid, "under load: fail | quiet:", verdict(rewards.get(eid)))
print("QUIET_DONE")
