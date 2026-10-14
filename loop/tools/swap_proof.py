"""Swap proof: the verifier we name is the verifier that runs, and it runs offline. Run on the server."""
from pathlib import Path
import shutil
from vb.regrade import grade
from vb.taskdir import BASELINES, official_verifier_image
from vb.util import TB_ROOT
task = TB_ROOT / "photonic-waveguide-routing"
job, work = Path("/root/vbench/probe/job1"), Path("/root/vbench/probe/swap3")
shutil.rmtree(work, ignore_errors=True)
plan = [("original", task / "tests", official_verifier_image(task)), ("pass_a", BASELINES["always_pass"], None),
        ("fail", BASELINES["always_fail"], None), ("pass_b", BASELINES["always_pass"], None),
        ("network_probe", BASELINES["network_probe"], None)]
res = {}
for name, tests, image in plan:
    res[name] = list(grade(job, task, tests, work, name, image=image).values())
    print(name, res[name], flush=True)
want = {"original": [1.0], "pass_a": [1.0], "fail": [0.0], "pass_b": [1.0], "network_probe": [0.0]}
bad = {k: (res[k], v) for k, v in want.items() if res[k] != v}
print("SWAP PROOF OK" if not bad else f"SWAP PROOF FAILED {bad}")
