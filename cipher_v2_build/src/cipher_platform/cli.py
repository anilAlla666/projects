#!/usr/bin/env python3
# `cipher` CLI. CIPHER substrate + reproducible Class-D demos/gates + `accelerate`
# (the one real request->tokens path, under the customer's own vLLM). NOT a
# CIPHER-native serving runtime (the handlers run fixed demo workloads).
import os, sys, json, argparse, subprocess
from cipher_platform import config, router, report, gates

def _load_manifest(path):
    with open(path) as f: return json.load(f)

def cmd_demo(args, cfg):
    job = {"regime": "agent", "name": "demo", "agents": args.agents}
    res = [router.dispatch("agent", cfg, job, quick=args.quick)]
    report.scorecard(res); return 0

def cmd_bench(args, cfg):
    manifest = _load_manifest(args.manifest)
    res = router.run_manifest(cfg, manifest, quick=args.quick)
    print("\n--- discipline gate (OFF byte-identical) ---", flush=True)
    g = gates.off_byte_identical(cfg)
    print(f"  OFF byte-identical: baseline={g['baseline']} off={g['off']} -> {'PASS' if g['pass'] else 'FAIL'}", flush=True)
    print(flush=True); report.scorecard(res)
    with open("/tmp/cipher_report.json", "w") as f: json.dump({"results": res, "off_byte_identical": g}, f, indent=2)
    checks = gates.evaluate(res); ok = g["pass"] and all(checks.values())
    print(f"\noverall gate: {'PASS' if ok else 'FAIL'}  {checks}", flush=True)
    return 0 if ok else 1

def cmd_selftest(args, cfg):
    print("=== cipher selftest"+(" --quick" if args.quick else "")+" ===", flush=True)
    # Class-D capture-sensitive lanes FIRST (each isolated), then OFF-byte-identical LAST.
    # Ordering is load-bearing: the OFF probe deliberately runs FULL auto-init (Koopman/EDMD
    # calibration, green-ctx, kmod registration) whose device state survives process reaping
    # and would poison a following CUDA-graph capture lane. So nothing capture-sensitive may
    # run after it. (Same sequencing constraint V.0 hit; resolved by order, not substrate.)
    print("[1/2] Class-D gates (agent FAULT=0 + misroute + density KL=0)...", flush=True)
    manifest = {"jobs": [{"regime": "agent", "name": "selftest", "agents": (12 if args.quick else 100), "bmax": 4},
                         {"regime": "density", "name": "selftest"}]}
    res = router.run_manifest(cfg, manifest, quick=args.quick)
    report.scorecard(res)
    print("\n[2/2] OFF byte-identical (no-regress foundation; runs last -- full-init probe)...", flush=True)
    g = gates.off_byte_identical(cfg)
    print(f"      -> {'PASS' if g['pass'] else 'FAIL'}  baseline={g['baseline']} off={g['off']}", flush=True)
    checks = gates.evaluate(res); ok = g["pass"] and all(checks.values())
    print(f"\nSELFTEST: {'PASS' if ok else 'FAIL'}  off_byte_identical={g['pass']} {checks}", flush=True)
    return 0 if ok else 1

def cmd_report(args, cfg):
    try:
        with open("/tmp/cipher_report.json") as f: d = json.load(f)
        report.scorecard(d["results"])
    except FileNotFoundError:
        print("no prior run; run `cipher bench`/`selftest` first", flush=True); return 1
    return 0

def cmd_accelerate(args, cfg):
    # THE real serving path: exec the customer's command with CIPHER injected under their vLLM.
    env = dict(os.environ); env["CUDA_INJECTION64_PATH"] = cfg["so_path"]
    for k, v in cfg.get("accelerate_env", {"CIPHER_KV_ALLOC": "1", "CIPHER_KVDEDUP": "1"}).items():
        env.setdefault(k, v)
    if not args.cmd:
        print("usage: cipher accelerate -- <command>", flush=True); return 2
    print(f"[accelerate] CUDA_INJECTION64_PATH={cfg['so_path']} -> exec: {' '.join(args.cmd)}", flush=True)
    return subprocess.run(args.cmd, env=env).returncode

def main(argv=None):
    p = argparse.ArgumentParser(prog="cipher", description="CIPHER platform (substrate + Class-D demos/gates + vLLM accelerate)")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--agents", type=int, default=12); d.add_argument("--quick", action="store_true"); d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("bench"); b.add_argument("--manifest", required=True); b.add_argument("--quick", action="store_true"); b.set_defaults(fn=cmd_bench)
    s = sub.add_parser("selftest"); s.add_argument("--quick", action="store_true"); s.set_defaults(fn=cmd_selftest)
    r = sub.add_parser("report"); r.set_defaults(fn=cmd_report)
    a = sub.add_parser("accelerate"); a.add_argument("cmd", nargs=argparse.REMAINDER); a.set_defaults(fn=cmd_accelerate)
    args = p.parse_args(argv)
    cfg = config.load(args.config)
    sys.exit(args.fn(args, cfg))

if __name__ == "__main__":
    main()
