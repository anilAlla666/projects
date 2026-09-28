# Strict lane isolation. Every gate probe AND every regime handler runs as a fresh,
# fully-reaped subprocess with its own CUDA context. After reaping, a GPU-settle
# barrier waits for the driver to release the (possibly os._exit-orphaned) memory
# before the next unit starts -- so one lane's CUDA context can never poison the next
# (the V.0 cross-phase sequencing problem). A unit crash is caught and reported as
# that-lane-FAILED (nonzero rc + captured stderr); it never propagates.
import subprocess, time

def gpu_used_mb(idx=0):
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(o.stdout.strip().splitlines()[idx])
    except Exception:
        return None

def settle(threshold_mb=2000, tries=60, dt=0.5):
    """Block until the GPU is released back below threshold (orphan-safe) or timeout."""
    for _ in range(tries):
        u = gpu_used_mb()
        if u is None or u < threshold_mb:
            return u
        time.sleep(dt)
    return gpu_used_mb()

def run_unit(argv, env, timeout, settle_after=True):
    """Spawn a fresh isolated subprocess, reap to completion, capture rc/stdout/stderr.
    Never raises on child failure; on timeout the child is killed and reaped."""
    try:
        r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
        res = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "timeout": False}
    except subprocess.TimeoutExpired as e:
        res = {"rc": None, "stdout": e.stdout or "", "stderr": (e.stderr or "") + "\n[TIMEOUT]", "timeout": True}
    if settle_after:
        res["gpu_settled_mb"] = settle()
    return res
