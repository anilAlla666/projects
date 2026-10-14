import json, subprocess, tomllib, hashlib
from pathlib import Path
from vb.util import TB_ROOT
def sha(b): return hashlib.sha256(b).hexdigest()
for t in json.load(open("configs/m1.json"))["tasks"]:
    img = tomllib.loads((TB_ROOT / t / "task.toml").read_text())["environment"]["docker_image"]
    files = [p for d in ("tests", "solution") for p in (TB_ROOT / t / d).rglob("*") if p.is_file() and p.stat().st_size > 64 and p.name not in ("Dockerfile",)]
    names = sorted({p.name for p in files})
    want = {sha(p.read_bytes()): f"{p.parent.name}/{p.name}" for p in files}
    pat = " -o ".join(f"-name {n!r}" for n in names)
    cmd = f"ls -d /tests /solution 2>/dev/null; find / -xdev \\( {pat} \\) -type f 2>/dev/null | head -40 | while read f; do echo \"$f $(sha256sum < \"$f\" | cut -c1-64)\"; done"
    out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "sh", img, "-c", cmd], capture_output=True, text=True).stdout.strip().splitlines()
    hits = [l for l in out if l.split()[-1] in want or l in ("/tests", "/solution")]
    print(t, "| official files checked:", len(files), "| same-name files in image:", len(out), "| IDENTICAL:", hits or "none", flush=True)
