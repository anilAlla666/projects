#!/usr/bin/env python3
"""Track 2 SC4 — model-identity integration driver.

One producer (TinyLlama); two consumer scenarios:
  matching  — consumer model == producer model -> fingerprint MATCH ->
              shared import -> bit-identical forward, ~0 MiB added (SC3
              behaviour, must still hold).
  mismatch  — consumer model = Llama-3.2-1B-Instruct != producer ->
              fingerprint MISMATCH -> consumer logs details, does NOT import,
              falls back to an independent load and runs correctly.
"""
import json
import os
import subprocess
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
TINYLLAMA = "/home/ubuntu/models/TinyLlama-1.1B"
OTHER = "/home/ubuntu/models/Llama-3.2-1B-Instruct"


def gpu_fb_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def run_consumer(mode, consumer_model):
    for f in ("sc4_consumer_logits.pt", "sc4_consumer_result.json"):
        if os.path.exists(HERE + "/" + f):
            os.remove(HERE + "/" + f)
    env = dict(os.environ, SC4_CONSUMER_MODEL=consumer_model)
    clog = open("%s/sc4_consumer_%s.log" % (HERE, mode), "w")
    fb_before = gpu_fb_mib()
    rc = subprocess.run([sys.executable, HERE + "/sc4_consumer.py"], env=env,
                        stdout=clog, stderr=subprocess.STDOUT,
                        timeout=240).returncode
    fb_after = gpu_fb_mib()
    cres = (json.load(open(HERE + "/sc4_consumer_result.json"))
            if os.path.exists(HERE + "/sc4_consumer_result.json") else {})
    # NB: cres carries its own "mode" (consumer path: shared/fallback_*) —
    # keep the scenario name under a distinct key so it is not clobbered.
    v = {"scenario": mode, "rc": rc,
         "consumer_added_mib": fb_after - fb_before, **cres}
    if mode == "matching" and rc == 0:
        import torch
        yp = torch.load(HERE + "/sc4_producer_logits.pt")
        yc = torch.load(HERE + "/sc4_consumer_logits.pt")
        v["bit_identical"] = bool(torch.equal(yp, yc))
        v["logits_max_abs_diff"] = float((yp - yc).abs().max())
    return v


def main():
    for f in ("sc4_producer_logits.pt", "sc4_integration_result.json"):
        if os.path.exists(HERE + "/" + f):
            os.remove(HERE + "/" + f)

    fb_idle = gpu_fb_mib()
    plog = open(HERE + "/sc4_producer.log", "w")
    prod = subprocess.Popen([sys.executable, HERE + "/sc4_producer.py"],
                            stdout=plog, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if (os.path.exists(HERE + "/sc4_producer.log") and
                "PRODUCER ready" in open(HERE + "/sc4_producer.log").read()):
            break
        if prod.poll() is not None:
            print("FAIL — producer exited early"); sys.exit(1)
        time.sleep(0.5)
    fb_producer = gpu_fb_mib()
    print("FB: idle=%d producer-loaded=%d MiB" % (fb_idle, fb_producer),
          flush=True)

    runs = []
    runs.append(run_consumer("matching", TINYLLAMA))
    runs.append(run_consumer("mismatch", OTHER))
    prod.terminate()
    for v in runs:
        extra = (("bit_identical=%s" % v.get("bit_identical"))
                 if v["scenario"] == "matching" else
                 ("mismatch_tier=%s" % (v.get("mismatch") or {}).get(
                     "differing_tier")))
        print("consumer[%s]: rc=%d consumer_mode=%s added=%s MiB %s"
              % (v["scenario"], v.get("rc"), v.get("mode"),
                 v.get("consumer_added_mib"), extra), flush=True)

    m = next(v for v in runs if v["scenario"] == "matching")
    x = next(v for v in runs if v["scenario"] == "mismatch")
    matching_ok = (m.get("rc") == 0 and m.get("mode") == "shared" and
                   m.get("bit_identical") is True and
                   m.get("consumer_added_mib", 1e9) < 500)
    mismatch_ok = (x.get("rc") == 0 and x.get("mode") == "fallback_independent"
                   and bool(x.get("mismatch")) and x.get("forward_ok") is True)

    verdict = {"matching": m, "mismatch": x,
               "matching_scenario_pass": bool(matching_ok),
               "mismatch_scenario_pass": bool(mismatch_ok),
               "PASS": bool(matching_ok and mismatch_ok)}
    json.dump(verdict, open(HERE + "/sc4_integration_result.json", "w"),
              indent=2, default=str)
    print(json.dumps(verdict, indent=2, default=str))
    print("SC4 INTEGRATION:", "PASS" if verdict["PASS"] else "FAIL")
    sys.exit(0 if verdict["PASS"] else 1)


if __name__ == "__main__":
    main()
