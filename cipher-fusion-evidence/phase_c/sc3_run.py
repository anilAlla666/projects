#!/usr/bin/env python3
"""Track 2 SC3 — weight-sharing integration driver.

Launches the producer, then runs the consumer twice — same-VA and forced
offset-relative — verifying for each:
  (1) bit-identical forward pass — Y_producer == Y_consumer  (Item-4 PUSH);
  (2) memory — the consumer adds ~0 weight FB (sharing actually happened);
  (3) the consumer's weight tensors are arena memory (page_info kind=weight).
The two consumer modes also cover the same-VA and offset-relative import paths.
"""
import json
import os
import subprocess
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"


def gpu_fb_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def run_consumer(mode, force_offset):
    """Run one consumer; return (rc, verdict-dict for this mode)."""
    for f in ("sc3_consumer_logits.pt", "sc3_consumer_result.json"):
        if os.path.exists(HERE + "/" + f):
            os.remove(HERE + "/" + f)
    env = dict(os.environ)
    if force_offset:
        env["SC3_FORCE_OFFSET"] = "1"
    clog = open("%s/sc3_consumer_%s.log" % (HERE, mode), "w")
    fb_before = gpu_fb_mib()
    cons = subprocess.run([sys.executable, HERE + "/sc3_consumer.py"],
                          env=env, stdout=clog, stderr=subprocess.STDOUT,
                          timeout=200)
    fb_after = gpu_fb_mib()
    if cons.returncode != 0:
        return cons.returncode, {"mode": mode, "rc": cons.returncode,
                                 "error": "consumer exited nonzero"}
    import torch
    yp = torch.load(HERE + "/sc3_producer_logits.pt")
    yc = torch.load(HERE + "/sc3_consumer_logits.pt")
    cres = json.load(open(HERE + "/sc3_consumer_result.json"))
    pi = cres.get("page_info_big") or {}
    return 0, {
        "mode": mode, "rc": 0,
        "bit_identical": bool(torch.equal(yp, yc)),
        "logits_max_abs_diff": float((yp - yc).abs().max()),
        "same_va": cres.get("same_va"),
        "weight_arena_backed": (isinstance(pi, dict)
                                and pi.get("kind") == "weight"),
        "fb_before_mib": fb_before, "fb_after_mib": fb_after,
        "consumer_added_mib": fb_after - fb_before,
    }


def main():
    for f in ("sc3_producer_logits.pt", "sc3_integration_result.json"):
        if os.path.exists(HERE + "/" + f):
            os.remove(HERE + "/" + f)

    fb_idle = gpu_fb_mib()
    plog = open(HERE + "/sc3_producer.log", "w")
    prod = subprocess.Popen([sys.executable, HERE + "/sc3_producer.py"],
                            stdout=plog, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if (os.path.exists(HERE + "/sc3_producer.log") and
                "PRODUCER ready" in open(HERE + "/sc3_producer.log").read()):
            break
        if prod.poll() is not None:
            print("FAIL — producer exited early"); sys.exit(1)
        time.sleep(0.5)
    fb_producer = gpu_fb_mib()
    producer_w = fb_producer - fb_idle
    print("FB: idle=%d, producer-loaded=%d MiB (weights+act ~%d)"
          % (fb_idle, fb_producer, producer_w), flush=True)

    runs = []
    for mode, fo in (("same_va", False), ("offset_relative", True)):
        rc, v = run_consumer(mode, fo)
        runs.append(v)
        print("consumer[%s]: rc=%d bit_identical=%s added=%s MiB same_va=%s"
              % (mode, rc, v.get("bit_identical"), v.get("consumer_added_mib"),
                 v.get("same_va")), flush=True)

    prod.terminate()

    # a consumer "added ~0" => sharing; allow slack for activations/ctx
    def shared(v):
        return (v.get("consumer_added_mib", 1e9) < producer_w // 2)

    overall = all(v.get("rc") == 0 and v.get("bit_identical") and
                  v.get("weight_arena_backed") and shared(v) for v in runs)
    # the offset-relative run must actually have taken the offset path
    off = next(v for v in runs if v["mode"] == "offset_relative")
    offset_path_exercised = (off.get("same_va") is False)

    verdict = {"producer_weights_mib": producer_w, "runs": runs,
               "offset_relative_path_exercised": offset_path_exercised,
               "PASS": bool(overall)}
    json.dump(verdict, open(HERE + "/sc3_integration_result.json", "w"),
              indent=2, default=str)
    print(json.dumps(verdict, indent=2))
    print("SC3 INTEGRATION:", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
