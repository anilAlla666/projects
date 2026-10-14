"""Step 18: runs inside a no-network container. Reads /w/job.json:
    {"solutions": {"<id>": "<python source>", ...}, "inputs": ["<stdin>", ...], "timeout": seconds}
and prints one JSON line: {"<id>": [<stdout or null on error/timeout>, ...], ...}. Each run is a fresh python3
process with a CPU-time and memory limit, so one bad solution cannot stall or starve the others."""
import json
import resource
import subprocess
import sys
import tempfile

job = json.load(open("/w/job.json"))
T = float(job.get("timeout", 4))


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (int(T) + 1, int(T) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


out = {}
with tempfile.TemporaryDirectory() as d:
    for sid, src in job["solutions"].items():
        path = f"{d}/{sid}.py"
        open(path, "w").write(src)
        res = []
        for inp in job["inputs"]:
            try:
                r = subprocess.run([sys.executable, path], input=inp, capture_output=True, text=True, timeout=T,
                                   preexec_fn=limits, errors="replace")
                res.append(r.stdout if r.returncode == 0 else None)
            except subprocess.TimeoutExpired:
                res.append(None)
        out[sid] = res
print(json.dumps(out))
