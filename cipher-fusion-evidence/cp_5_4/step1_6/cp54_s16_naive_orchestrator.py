#!/usr/bin/env python3
"""CP 5.4 Step 1.6B-3 — naive Arm-A orchestrator (THROWAWAY).

Launches N **naive** tenants concurrently (`cp54_s16_naive_tenant.py`), each a
plain PyTorch process — no CIPHER kmod path, no `CUDA_INJECTION64_PATH`, no
`LD_PRELOAD`, no MPS/MIG — sharing the full GPU by ordinary CUDA time-slicing.
A per-round lockstep barrier keeps every tenant's contention window
co-extensive (Step 1.6B-2A precedent — without it, tenants finishing early
escape contention and the variance measurement is invalid).

This is the Arm-A baseline for the Step 1.6 2-arm methodology: the comparison
point that Arm-B (CIPHER mixed deployment, 1.6B-4) measures the substrate
value against.

Three framebuffer measurements (`nvidia-smi memory.used`), per the 1.6B-3/4
design memo §3 decision-3 PUSH: idle (pre-launch) / loaded (all tenants
resident) / exited (post-teardown leak check).

Naive-environment precondition: the CIPHER kmod should be UNLOADED for a true
Arm-A run (`lsmod | grep cipher_kmod` empty). The orchestrator records the
kmod state; it does not itself rmmod (that needs sudo — done by the run
procedure).

argv: --tenants N --rounds R --tag TAG [--outdir DIR]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAIVE_TENANT = os.path.join(HERE, "cp54_s16_naive_tenant.py")
WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
GOLD_JSON = WL01 + "/gold.json"
GOLD_LOGITS = WL01 + "/gold_logits.pt"
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"


def olog(m):
    print("[naive-orch %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def gpu_fb_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def kmod_loaded():
    out = subprocess.run(["lsmod"], capture_output=True, text=True).stdout
    return any(l.startswith("cipher_kmod") for l in out.splitlines())


def barrier_coordinator(tag, members, rounds, stop, per_round_timeout=240.0):
    """Per-round lockstep coordinator — identical protocol to the Arm-B
    orchestrator: wait for every member's `<m>_r<r>.done`, then write
    `all_r<r>.go`."""
    for r in range(rounds):
        t0 = time.time()
        while not stop.is_set():
            if all(os.path.exists("/tmp/cp54_s16_%s_%s_r%d.done"
                                  % (tag, m, r)) for m in members):
                break
            if time.time() - t0 > per_round_timeout:
                olog("BARRIER timeout round %d — missing: %s"
                     % (r, [m for m in members
                            if not os.path.exists(
                                "/tmp/cp54_s16_%s_%s_r%d.done" % (tag, m, r))]))
                stop.set()
                for rr in range(r, rounds):
                    open("/tmp/cp54_s16_%s_all_r%d.go" % (tag, rr), "w").close()
                return
            time.sleep(0.05)
        if stop.is_set():
            return
        open("/tmp/cp54_s16_%s_all_r%d.go" % (tag, r), "w").close()
        olog("BARRIER round %d — all %d members done, released"
             % (r, len(members)))


class NaiveTenant:
    def __init__(self, name, outdir):
        self.name = name
        self.out_json = os.path.join(outdir, "%s.json" % name)
        self.log = open(os.path.join(outdir, "%s.stderr.log" % name), "w")
        self.proc = None
        self.pid = None
        self.ready = threading.Event()
        self.done = threading.Event()

    def launch(self, rounds, tag):
        env = dict(os.environ)
        # explicit naive environment — strip any CIPHER injection/preload
        for k in ("CUDA_INJECTION64_PATH", "LD_PRELOAD",
                  "CIPHER_QOS_CLASS", "CIPHER_SM_COUNT"):
            env.pop(k, None)
        env.update({
            "TENANT": self.name, "OUT_JSON": self.out_json,
            "ROUNDS": str(rounds), "NOSIGNAL": "0", "BARRIER_TAG": tag,
            "WL_MODEL": MODEL, "GOLD_JSON": GOLD_JSON,
            "GOLD_LOGITS": GOLD_LOGITS,
        })
        self.proc = subprocess.Popen(
            [sys.executable, NAIVE_TENANT], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("READY"):
                self.pid = int(line.split()[1])
                self.ready.set()
            elif line == "DONE":
                self.done.set()

    def start(self):
        self.proc.stdin.write("START\n")
        self.proc.stdin.flush()


def tenant_stats(res):
    """Per-tenant decode-latency stats. Warm convention (design memo §3e):
    the timed round 0 is discarded — the tenant already ran an untimed warmup
    round, this drops the first *under-contention* round too."""
    rd = res["rounds_detail"]
    warm = [x for r in rd if r["round"] >= 1 for x in r["decode_ms"]]
    alld = [x for r in rd for x in r["decode_ms"]]

    def stats(xs):
        if len(xs) < 2:
            return {"n": len(xs)}
        xs_s = sorted(xs)
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs)
        return {"n": len(xs), "mean_ms": mean,
                "p50_ms": xs_s[len(xs_s) // 2],
                "p95_ms": xs_s[min(len(xs_s) - 1, int(0.95 * len(xs_s)))],
                "p99_ms": xs_s[min(len(xs_s) - 1, int(0.99 * len(xs_s)))],
                "cv": sd / mean if mean else 0.0,
                "tok_s": 1000.0 / mean if mean else 0.0}
    return {"tenant": res["tenant"], "pid": res.get("pid"),
            "kl_gate_pass": res["kl_gate_pass"], "kl_max": res["kl_max_all"],
            "peak_mem_mib": res.get("max_mem_allocated_mib"),
            "warm": stats(warm), "all_rounds": stats(alld)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--tag", default="naive_sanity")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(HERE, "b3_" + args.tag)
    os.makedirs(outdir, exist_ok=True)
    import glob
    for f in glob.glob("/tmp/cp54_s16_%s_*" % args.tag):
        try:
            os.unlink(f)
        except OSError:
            pass

    kmod = kmod_loaded()
    olog("config: tenants=%d rounds=%d tag=%s  kmod_loaded=%s%s"
         % (args.tenants, args.rounds, args.tag, kmod,
            "  [WARNING: not a pure Arm-A environment]" if kmod else ""))
    fb_idle = gpu_fb_mib()

    members = ["n%d" % i for i in range(args.tenants)]
    tenants = [NaiveTenant(m, outdir) for m in members]
    for t in tenants:
        t.launch(args.rounds, args.tag)
    olog("launched %d naive tenants — waiting READY" % args.tenants)

    t0 = time.time()
    while not all(t.ready.is_set() for t in tenants):
        dead = [t.name for t in tenants if t.proc.poll() is not None
                and not t.ready.is_set()]
        if dead:
            olog("FATAL: tenant(s) exited before READY: %s" % dead)
            for t in tenants:
                if t.proc.poll() is None:
                    t.proc.kill()
            sys.exit(2)
        if time.time() - t0 > 600:
            olog("FATAL: READY timeout")
            sys.exit(2)
        time.sleep(0.5)
    fb_loaded = gpu_fb_mib()
    olog("all %d tenants READY — fb idle=%d loaded=%d MiB"
         % (args.tenants, fb_idle, fb_loaded))

    stop = threading.Event()
    bc = threading.Thread(target=barrier_coordinator,
                          args=(args.tag, members, args.rounds, stop),
                          daemon=True)
    bc.start()
    olog("=== START — release %d naive tenants ===" % args.tenants)
    for t in tenants:
        t.start()

    t0 = time.time()
    while not all(t.done.is_set() for t in tenants):
        if time.time() - t0 > 1200:
            olog("FATAL: DONE timeout")
            stop.set()
            for t in tenants:
                if t.proc.poll() is None:
                    t.proc.kill()
            sys.exit(2)
        time.sleep(0.5)
    for t in tenants:
        t.proc.wait()
    stop.set()
    time.sleep(1)
    fb_exited = gpu_fb_mib()

    # ---- collect + aggregate ----
    per_tenant = []
    for t in tenants:
        if not os.path.exists(t.out_json):
            olog("WARN: %s produced no JSON" % t.name)
            continue
        per_tenant.append(tenant_stats(json.load(open(t.out_json))))

    kl_all = all(p["kl_gate_pass"] for p in per_tenant)
    agg_tok_s = sum(p["warm"].get("tok_s", 0.0) for p in per_tenant)
    cvs = [p["warm"]["cv"] for p in per_tenant if "cv" in p["warm"]]

    result = {
        "step": "1.6B-3", "arm": "A-naive", "tag": args.tag,
        "tenants": args.tenants, "rounds": args.rounds,
        "kmod_loaded_during_run": kmod,
        "fb_mib": {"idle": fb_idle, "loaded": fb_loaded, "exited": fb_exited},
        "all_kl_gates_pass": kl_all,
        "aggregate_tok_s": agg_tok_s,
        "per_tenant_cv_warm": {"min": min(cvs) if cvs else None,
                               "max": max(cvs) if cvs else None,
                               "mean": statistics.mean(cvs) if cvs else None},
        "per_tenant": per_tenant,
    }
    outp = os.path.join(outdir, "naive_result.json")
    json.dump(result, open(outp, "w"), indent=2)
    olog("=== summary ===")
    for p in per_tenant:
        w = p["warm"]
        olog("  %-4s kl_gate=%s  decode mean=%.2fms p99=%.2fms CV=%.4f"
             % (p["tenant"], p["kl_gate_pass"], w.get("mean_ms", 0),
                w.get("p99_ms", 0), w.get("cv", 0)))
    olog("  aggregate_tok_s=%.2f  fb idle/loaded/exited=%d/%d/%d  kl_all=%s"
         % (agg_tok_s, fb_idle, fb_loaded, fb_exited, kl_all))
    olog("VERDICT: %s — wrote %s"
         % ("PASS" if kl_all else "FAIL (a KL gate failed)", outp))
    sys.exit(0 if kl_all else 1)


if __name__ == "__main__":
    main()
