# Regime -> handler dispatch. Each regime runs through the isolation primitive
# (fresh subprocess, own CUDA context, GPU-settled + reaped before the next unit).
# A lane crash is reported as that-lane-FAILED with captured stderr -- never poisons
# the next lane. Promoted from the validated mock router, reframed bench/demo.
import os, re, sys, json
from cipher_platform import isolate

def _parse(regime, job, out):
    m = re.search(r"CIPHER_(?:MOCK_)?RESULT (\{.*\})", out)               # smoke handlers
    if m:
        return json.loads(m.group(1))
    g = re.search(r"\[GATE inc4[^\]]*\]\s+.*->\s+(PASS|FAIL)", out)        # full cipher_inc4.py
    if g:
        return {"regime": regime, "name": job.get("name"), "metrics": {"gate": g.group(1)}, "real_engine": True}
    v = re.search(r"V0C_JSON (\{.*\})", out)                              # full v0_phaseC
    if v:
        return {"regime": regime, "metrics": json.loads(v.group(1)), "real_engine": True}
    return None

def dispatch(regime, cfg, job, quick=False):
    spec = cfg["regimes"][regime]
    handler = spec["quick_handler"] if (quick and "quick_handler" in spec) else spec["handler"]
    env = dict(os.environ)
    inject = {k: (cfg["so_path"] if v == "__SO__" else v) for k, v in spec.get("inject", {}).items()}
    env.update(inject); env["CIPHER_REGIME"] = regime
    inj = [k for k in inject if k in ("CUDA_INJECTION64_PATH", "CIPHER_RT_DISABLE_AUTO_INIT")]
    path = os.path.join(cfg["handlers_dir"], handler)
    # Each attempt is a fresh, GPU-settled isolated process == the inc-3b re-capture mitigation
    # for intrinsic CUDA-graph capture-invalidation flakiness. Reliable lanes succeed on attempt 1
    # (zero extra cost); the flaky agent capture is absorbed. A genuine fault still fails all attempts.
    attempts = spec.get("max_attempts", 3)
    last = None
    for k in range(1, attempts + 1):
        tag = "" if k == 1 else f" (re-capture attempt {k}/{attempts})"
        print(f"  [lane] regime={regime:<8} -> {handler:<24} injection={inj or 'default'} (isolated){tag}", flush=True)
        u = isolate.run_unit([sys.executable, path, regime, json.dumps(job)], env, spec.get("timeout", 600))
        parsed = _parse(regime, job, u["stdout"])
        if parsed is not None:
            parsed["lane_status"] = "OK"; parsed["attempts"] = k
            return parsed
        why = "timeout" if u["timeout"] else (f"crash rc={u['rc']}" if u["rc"] not in (0, None) else "no result marker")
        print(f"  [lane] regime={regime} attempt {k} failed ({why}); gpu_settled={u.get('gpu_settled_mb')}MB", flush=True)
        last = {"regime": regime, "name": job.get("name"), "lane_status": "FAILED",
                "error": why, "rc": u["rc"], "attempts": k, "stderr": (u["stderr"] or "")[-400:]}
    print(f"  [lane] regime={regime} FAILED after {attempts} isolated attempts; next lane unaffected", flush=True)
    return last

def run_manifest(cfg, manifest, quick=False):
    return [dispatch(j["regime"], cfg, j, quick=quick) for j in manifest["jobs"]]
