#!/usr/bin/env python3
"""Track 2 SC5 — weight-sharing integration driver (producer-dies-first).

This is the SC5 crux test. Ordering, enforced here:

  1. producer  — load TinyLlama, pack a VMM weight arena, REGISTER it with
                 the kmod, stay alive.
  2. consumer-A (mode A) — IMPORT the arena, rebind, then BLOCK on a sentinel.
                 A is now a live participant keeping the arena's count > 0.
  3. SIGKILL the producer — no cleanup, no ARENA_LEAVE: the crash path.
  4. ARENA_QUERY — the arena MUST still exist with producer_pid == 0 (the 5 s
                 reaper has cleared the dead producer) and n_consumers == 1.
                 The kmod's own `struct file *` ref + A's import handle keep
                 the physical alive — the producer's death freed nothing.
  5. drop the sentinel — consumer-A wakes and runs its forward pass. It
                 succeeds => a consumer's mapping keeps working after the
                 producer has died.
  6. consumer-B (mode B) — launched only NOW, with the producer long dead.
                 A successful IMPORT => the kmod rendezvous hands a peer the
                 fd with no producer process in existence (SC3's socket
                 transport could not do this).
  7. verify producer / A / B logits are all bit-identical (shared physical
     bytes) and that the consumers added ~0 weight FB (sharing happened).
  8. after both consumers exit, ARENA_QUERY — the arena MUST be reaped
     (participant count 0 => the kmod fput's its held fd).
"""
import json
import os
import signal
import subprocess
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
import sc5_arena_ioctl as aioctl                              # noqa: E402

SENTINEL = HERE + "/sc5_kill_done"
ARTIFACTS = ["sc5_producer_logits.pt", "sc5_consumer_A_logits.pt",
             "sc5_consumer_B_logits.pt", "sc5_consumer_A_result.json",
             "sc5_consumer_B_result.json", "sc5_integration_result.json",
             SENTINEL.split("/")[-1]]


def gpu_fb_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def wait_for(proc, logpath, needle, timeout, what):
    """Block until `needle` appears in logpath, or proc dies, or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(logpath) and needle in open(logpath).read():
            return True
        if proc.poll() is not None:
            print("FAIL — %s exited early (rc=%d)" % (what, proc.returncode))
            return False
        time.sleep(0.5)
    print("FAIL — timeout waiting for '%s' from %s" % (needle, what))
    return False


def find_arena(arena_id):
    for a in aioctl.query():
        if a["arena_id"] == arena_id:
            return a
    return None


def main():
    for f in ARTIFACTS:
        p = HERE + "/" + f
        if os.path.exists(p):
            os.remove(p)

    fb_idle = gpu_fb_mib()
    result = {"stages": {}}

    # ---- 1. producer ----
    plog = HERE + "/sc5_producer.log"
    prod = subprocess.Popen([sys.executable, HERE + "/sc5_producer.py"],
                            stdout=open(plog, "w"), stderr=subprocess.STDOUT)
    if not wait_for(prod, plog, "PRODUCER registered", 240, "producer"):
        prod.kill(); sys.exit(1)
    line = [l for l in open(plog) if "PRODUCER registered" in l][0]
    arena_id = int(line.split("arena_id=")[1].split()[0])
    fb_producer = gpu_fb_mib()
    print("producer up: arena_id=%d  FB idle=%d -> producer=%d MiB"
          % (arena_id, fb_idle, fb_producer), flush=True)

    # ---- 2. consumer-A (mode A: import, rebind, block on sentinel) ----
    alog = HERE + "/sc5_consumer_A.log"
    consA = subprocess.Popen(
        [sys.executable, HERE + "/sc5_consumer.py", str(arena_id), "A"],
        stdout=open(alog, "w"), stderr=subprocess.STDOUT)
    if not wait_for(consA, alog, "A-IMPORTED", 240, "consumer-A"):
        prod.kill(); consA.kill(); sys.exit(1)
    fb_A = gpu_fb_mib()
    q_before = find_arena(arena_id)
    print("consumer-A imported: FB=%d MiB  query=%s" % (fb_A, q_before),
          flush=True)

    # ---- 3. SIGKILL the producer (crash path — no cleanup) ----
    print("SIGKILL producer pid=%d" % prod.pid, flush=True)
    os.kill(prod.pid, signal.SIGKILL)
    prod.wait()

    # ---- 4. wait out a reaper cycle; arena must SURVIVE, producer_pid -> 0 ----
    time.sleep(8)
    q_postkill = find_arena(arena_id)
    survived = (q_postkill is not None)
    producer_cleared = bool(q_postkill and q_postkill["producer_pid"] == 0)
    print("post-SIGKILL: arena survived=%s  query=%s"
          % (survived, q_postkill), flush=True)
    result["stages"]["producer_alive"] = q_before
    result["stages"]["post_sigkill"] = q_postkill

    # ---- 5. release consumer-A's forward pass (producer is dead) ----
    open(SENTINEL, "w").close()
    a_done = wait_for(consA, alog, "A-DONE", 120, "consumer-A")
    consA.wait()

    # ---- 6. consumer-B — joins with the producer long dead ----
    fb_before_B = gpu_fb_mib()
    blog = HERE + "/sc5_consumer_B.log"
    consB = subprocess.Popen(
        [sys.executable, HERE + "/sc5_consumer.py", str(arena_id), "B"],
        stdout=open(blog, "w"), stderr=subprocess.STDOUT)
    b_done = wait_for(consB, blog, "B-DONE", 240, "consumer-B")
    fb_B = gpu_fb_mib()
    consB.wait()
    b_imported = (os.path.exists(blog)
                  and "CONSUMER-B imported" in open(blog).read())

    # ---- 7. bit-identical forward across producer / A / B ----
    import torch
    bit = {}
    try:
        yp = torch.load(HERE + "/sc5_producer_logits.pt")
        ya = torch.load(HERE + "/sc5_consumer_A_logits.pt")
        yb = torch.load(HERE + "/sc5_consumer_B_logits.pt")
        bit = {"A_eq_producer": bool(torch.equal(yp, ya)),
               "B_eq_producer": bool(torch.equal(yp, yb)),
               "A_max_abs_diff": float((yp - ya).abs().max()),
               "B_max_abs_diff": float((yp - yb).abs().max())}
    except FileNotFoundError as e:
        bit = {"error": "missing logits: %s" % e}

    # ---- 8. after both consumers exit, the arena must be reaped ----
    time.sleep(8)
    q_final = find_arena(arena_id)
    reaped = (q_final is None)
    result["stages"]["after_consumers_exit"] = q_final
    print("after consumers exit: arena reaped=%s" % reaped, flush=True)

    producer_w = fb_producer - fb_idle
    a_added = fb_A - fb_producer
    b_added = fb_B - fb_before_B
    aw = json.load(open(HERE + "/sc5_consumer_A_result.json")) \
        if os.path.exists(HERE + "/sc5_consumer_A_result.json") else {}
    bw = json.load(open(HERE + "/sc5_consumer_B_result.json")) \
        if os.path.exists(HERE + "/sc5_consumer_B_result.json") else {}

    def arena_backed(r):
        pi = r.get("page_info_big") or {}
        return isinstance(pi, dict) and pi.get("kind") == "weight"

    checks = {
        "arena_survived_producer_sigkill": survived,
        "producer_pid_cleared_by_reaper": producer_cleared,
        "consumerA_forward_after_producer_death": bool(a_done),
        "consumerB_imported_with_no_producer": bool(b_imported),
        "consumerB_forward_done": bool(b_done),
        "A_bit_identical": bit.get("A_eq_producer", False),
        "B_bit_identical": bit.get("B_eq_producer", False),
        "A_weight_arena_backed": arena_backed(aw),
        "B_weight_arena_backed": arena_backed(bw),
        "A_shared_no_weight_copy": (a_added < max(producer_w // 2, 1)),
        "B_shared_no_weight_copy": (b_added < max(producer_w // 2, 1)),
        "arena_reaped_after_last_participant": reaped,
    }
    overall = all(checks.values())

    result.update({
        "arena_id": arena_id,
        "fb_mib": {"idle": fb_idle, "producer": fb_producer, "after_A": fb_A,
                   "before_B": fb_before_B, "after_B": fb_B},
        "producer_weights_mib": producer_w,
        "consumerA_added_mib": a_added, "consumerB_added_mib": b_added,
        "bit_identical": bit, "checks": checks, "PASS": bool(overall)})
    json.dump(result, open(HERE + "/sc5_integration_result.json", "w"),
              indent=2, default=str)
    print(json.dumps(result, indent=2))
    print("SC5 INTEGRATION:", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
