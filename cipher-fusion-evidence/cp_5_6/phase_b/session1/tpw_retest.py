#!/usr/bin/env python3
"""TPW RE-TEST (2026-05-19) — cross-tenant batching tok/W, current substrate.

Re-measures the ONE real, substrate-attributable tok/W lever — cross-tenant
batching — on the CURRENT anchors (kmod 008b3c66 / libcipher_rt 83afd1ca,
post Track 2 + Track 3), reproducing the Phase B headline:

  substrate-attributable tok/W = batched-executor tok/W  /  naive-N-concurrent tok/W

Two arms per scenario, identical model path (float16 + sdpa + greedy
generate, MAX_NEW=128, same gold prompts):
  - BATCHED — `cipher_batch_executor_gen.py` fuses N tenants' decode steps
              into one B=N generate (the substrate).
  - NAIVE   — N independent `tpw_naive_worker.py`, each its own B=1 generate,
              time-slicing the GPU (no substrate).

Power: NVML, sampled 10 Hz, averaged over each arm's decode window.

Scenarios: Mistral-7B-v0.1 N=4 (production-representative) and
TinyLlama-1.1B N=8 (the Phase B headline point).
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXECUTOR = os.path.join(HERE, "cipher_batch_executor_gen.py")
BATCH_CLIENT = os.path.join(HERE, "batch_client.py")
NAIVE_WORKER = os.path.join(HERE, "tpw_naive_worker.py")
PHASE_A = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a"

SCENARIOS = [
    {"label": "Mistral-7B  N=4", "model": "/home/ubuntu/models/Mistral-7B-v0.1",
     "n": 4, "wl": "WL_MISTRAL"},
    {"label": "TinyLlama   N=8", "model": "/home/ubuntu/models/TinyLlama-1.1B",
     "n": 8, "wl": "WL01"},
]


def power_sampler(stop, samples):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:                                    # noqa: BLE001
        print("  power sampler off: %r" % e, flush=True)
        return
    while not stop.is_set():
        try:
            samples.append((time.time(),
                            pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0))
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(0.1)


def mean_power(samples, t0, t1):
    w = [p for ts, p in samples if t0 <= ts <= t1]
    return statistics.mean(w) if w else None


def run_batched(sc, outdir):
    gold = "%s/%s/gold.json" % (PHASE_A, sc["wl"])
    gold_logits = "%s/%s/gold_logits.pt" % (PHASE_A, sc["wl"])
    sock = "/tmp/tpw_batched_%s.sock" % sc["wl"]
    sent = "/tmp/tpw_batched_%s.window" % sc["wl"]
    for p in (sock, sent):
        if os.path.exists(p):
            os.remove(p)
    elog = os.path.join(outdir, "batched_executor.log")
    env = dict(os.environ)
    for k in ("CUDA_INJECTION64_PATH", "LD_PRELOAD"):
        env.pop(k, None)
    env.update({"WL_MODEL": sc["model"], "N_CLIENTS": str(sc["n"]),
                "WL_MAX_NEW": "128", "BATCH_SOCK": sock, "SENTINEL": sent,
                "GOLD_JSON": gold, "GOLD_LOGITS": gold_logits,
                "RESULT_JSON": os.path.join(outdir, "batched_result.json")})
    exe = subprocess.Popen([sys.executable, EXECUTOR], env=env,
                           stdout=open(elog, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while "LISTENING" not in (open(elog).read() if os.path.exists(elog) else ""):
        if exe.poll() is not None:
            print("  batched executor died before LISTENING", flush=True)
            return None
        if time.time() - t0 > 600:
            exe.kill(); return None
        time.sleep(0.3)
    samples, stop = [], threading.Event()
    threading.Thread(target=power_sampler, args=(stop, samples),
                     daemon=True).start()
    for i in range(sc["n"]):
        cenv = dict(env)
        cenv.update({"BATCH_SOCK": sock, "TENANT": "t%d" % i, "GEN_LEN": "128",
                     "OUT_JSON": os.path.join(outdir, "batched_t%d.json" % i)})
        subprocess.Popen([sys.executable, BATCH_CLIENT], env=cenv,
                         stdout=open(os.path.join(outdir, "batched_t%d.log" % i),
                                     "w"), stderr=subprocess.STDOUT)
    exe.wait()
    stop.set()
    time.sleep(0.3)
    res = json.load(open(os.path.join(outdir, "batched_result.json")))
    win = open(sent).read().split()
    wt0, wt1 = float(win[1]), float(win[3])
    pw = mean_power(samples, wt0, wt1)
    return {"agg_tok_s": res["agg_tok_s"], "power_w": pw,
            "tok_w": res["agg_tok_s"] / pw if pw else None,
            "decode_wall_s": res.get("total_wall_s")}


def run_naive(sc, outdir):
    gold = "%s/%s/gold.json" % (PHASE_A, sc["wl"])
    go = "/tmp/tpw_naive_%s.go" % sc["wl"]
    if os.path.exists(go):
        os.remove(go)
    procs = []
    for i in range(sc["n"]):
        env = dict(os.environ)
        for k in ("CUDA_INJECTION64_PATH", "LD_PRELOAD"):
            env.pop(k, None)
        env.update({"WL_MODEL": sc["model"], "GOLD_JSON": gold,
                    "OUT_JSON": os.path.join(outdir, "naive_t%d.json" % i),
                    "GO_FILE": go, "TENANT": "n%d" % i, "MAX_NEW": "128"})
        lg = os.path.join(outdir, "naive_t%d.log" % i)
        procs.append((subprocess.Popen([sys.executable, NAIVE_WORKER], env=env,
                      stdout=open(lg, "w"), stderr=subprocess.STDOUT), lg))
    # wait until every worker has loaded its model
    t0 = time.time()
    while not all("LOADED" in open(lg).read() for _, lg in procs
                  if os.path.exists(lg)):
        if any(p.poll() is not None for p, _ in procs):
            dead = [lg for p, lg in procs if p.poll() is not None
                    and "LOADED" not in (open(lg).read()
                                         if os.path.exists(lg) else "")]
            if dead:
                print("  naive worker died before LOADED: %s" % dead,
                      flush=True)
                return None
        if time.time() - t0 > 600:
            return None
        time.sleep(0.3)
    samples, stop = [], threading.Event()
    threading.Thread(target=power_sampler, args=(stop, samples),
                     daemon=True).start()
    wt0 = time.time()
    open(go, "w").close()                       # release all workers together
    for p, _ in procs:
        p.wait()
    wt1 = time.time()
    stop.set()
    time.sleep(0.3)
    toks, walls = 0, []
    for i in range(sc["n"]):
        r = json.load(open(os.path.join(outdir, "naive_t%d.json" % i)))
        toks += r["tokens"]
        walls.append(r["decode_wall_s"])
    agg = toks / (wt1 - wt0)
    pw = mean_power(samples, wt0, wt1)
    return {"agg_tok_s": agg, "power_w": pw,
            "tok_w": agg / pw if pw else None,
            "decode_wall_s": max(walls)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # --tinyllama-n 12,16 → run extra TinyLlama scaling points (higher N is
    # TinyLlama-only: Mistral-7B at N>4 needs Track 2 weight-sharing for HBM).
    ap.add_argument("--tinyllama-n", default=None,
                    help="comma list of extra TinyLlama N points, e.g. 12,16")
    ap.add_argument("--only-extra", action="store_true",
                    help="run ONLY the --tinyllama-n points, skip the defaults")
    args = ap.parse_args()
    scenarios = [] if args.only_extra else list(SCENARIOS)
    if args.tinyllama_n:
        for n in (int(x) for x in args.tinyllama_n.split(",")):
            scenarios.append({"label": "TinyLlama   N=%d" % n,
                              "model": "/home/ubuntu/models/TinyLlama-1.1B",
                              "n": n, "wl": "WL01"})

    results = []
    for sc in scenarios:
        outdir = os.path.join(HERE, "tpw_retest_%s_n%d" % (sc["wl"], sc["n"]))
        os.makedirs(outdir, exist_ok=True)
        print("\n===== %s =====" % sc["label"], flush=True)
        print("  running BATCHED arm ...", flush=True)
        b = run_batched(sc, outdir)
        print("  batched: %s" % b, flush=True)
        print("  running NAIVE-%d-concurrent arm ..." % sc["n"], flush=True)
        nv = run_naive(sc, outdir)
        print("  naive:   %s" % nv, flush=True)
        ratio = (b["tok_w"] / nv["tok_w"]) if (b and nv and b["tok_w"]
                                               and nv["tok_w"]) else None
        row = {"scenario": sc["label"], "n": sc["n"],
               "batched_tok_w": round(b["tok_w"], 4) if b else None,
               "batched_tok_s": round(b["agg_tok_s"], 1) if b else None,
               "batched_power_w": round(b["power_w"], 1) if b else None,
               "naive_tok_w": round(nv["tok_w"], 4) if nv else None,
               "naive_tok_s": round(nv["agg_tok_s"], 1) if nv else None,
               "naive_power_w": round(nv["power_w"], 1) if nv else None,
               "substrate_attributable_x": round(ratio, 3) if ratio else None}
        results.append(row)
        print("  >>> %s: substrate-attributable tok/W = %.3fx  "
              "(batched %.4f / naive %.4f)"
              % (sc["label"], ratio or 0, b["tok_w"] if b else 0,
                 nv["tok_w"] if nv else 0), flush=True)

    resfile = ("tpw_retest_highn_result.json" if args.only_extra
               else "tpw_retest_result.json")
    json.dump(results, open(os.path.join(HERE, resfile), "w"), indent=2)
    print("\n===== TPW RE-TEST SUMMARY =====", flush=True)
    for r in results:
        print("  %-18s batched %.3f tok/W  /  naive %.3f tok/W  =  %.2fx "
              "substrate-attributable"
              % (r["scenario"], r["batched_tok_w"] or 0,
                 r["naive_tok_w"] or 0, r["substrate_attributable_x"] or 0),
              flush=True)


if __name__ == "__main__":
    main()
