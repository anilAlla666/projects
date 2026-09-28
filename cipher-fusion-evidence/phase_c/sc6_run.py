#!/usr/bin/env python3
"""Track 2 SC6 — N=4 weight-sharing integration driver.

Runs ONE (model, phase) per invocation — separate process trees keep an
OOM-killed independent child from polluting a shared-phase measurement
(SC6 design memo §6 / advisor note).

  phase = shared       — 1 producer + 4 consumers sharing one kmod arena;
                         producer-dies-first at N=4; bit-identical verify.
  phase = independent  — the control: up to 5 tenants each loading the model
                         privately, no sharing. OOM is recorded as a finding
                         (K/5 fit), not a failure.

Each phase takes THREE framebuffer measurements (design memo §4, decision-3
PUSH): idle (before any tenant), loaded (steady state), exited (clean
teardown — catches a leak that a surface "savings" number would hide).

argv: [model_name] [shared|independent]
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
import sc6_models as M                                        # noqa: E402

KILL_SENTINEL = HERE + "/sc6_kill_done"
IND_SENTINEL = HERE + "/sc6_ind_done"
PY = sys.executable


def clean(model_name):
    import glob
    for p in glob.glob("%s/sc6_%s_*" % (HERE, model_name)):
        os.remove(p)
    for s in (KILL_SENTINEL, IND_SENTINEL):
        if os.path.exists(s):
            os.remove(s)


def wait_line(logpath, needle, proc, timeout, what):
    """Block until `needle` is in logpath; False if proc dies first / timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(logpath) and needle in open(logpath).read():
            return True
        if proc is not None and proc.poll() is not None:
            print("  %s exited early (rc=%s) before '%s'"
                  % (what, proc.returncode, needle))
            return False
        time.sleep(0.5)
    print("  timeout waiting for '%s' from %s" % (needle, what))
    return False


def find_arena(arena_id):
    for a in aioctl.query():
        if a["arena_id"] == arena_id:
            return a
    return None


# ----------------------------------------------------------------------
def run_shared(model_name):
    fb_idle = M.gpu_fb_mib()
    plog = "%s/sc6_%s_producer.log" % (HERE, model_name)
    prod = subprocess.Popen([PY, HERE + "/sc6_producer.py", model_name],
                            stdout=open(plog, "w"), stderr=subprocess.STDOUT)
    if not wait_line(plog, "PRODUCER registered", prod, 600, "producer"):
        prod.kill()
        return {"phase": "shared", "model": model_name, "PASS": False,
                "error": "producer failed to register"}
    line = [l for l in open(plog) if "PRODUCER registered" in l][0]
    arena_id = int(line.split("arena_id=")[1].split()[0])
    print("  producer registered arena_id=%d  fb_idle=%d" % (arena_id, fb_idle))

    # ---- launch N=4 consumers concurrently ----
    cons, clogs = [], []
    for i in range(M.N_CONSUMERS):
        cl = "%s/sc6_%s_consumer_%d.log" % (HERE, model_name, i)
        clogs.append(cl)
        cons.append(subprocess.Popen(
            [PY, HERE + "/sc6_consumer.py", str(arena_id), str(i), model_name],
            stdout=open(cl, "w"), stderr=subprocess.STDOUT))
    ok = all(wait_line(clogs[i], "C%d-FWD1" % i, cons[i], 600, "consumer-%d" % i)
             for i in range(M.N_CONSUMERS))
    if not ok:
        prod.kill()
        for c in cons:
            c.kill()
        return {"phase": "shared", "model": model_name, "PASS": False,
                "error": "a consumer failed forward #1"}

    fb_loaded = M.gpu_fb_mib()
    q_alive = find_arena(arena_id)
    print("  N=4 consumers up  fb_loaded=%d  query=%s" % (fb_loaded, q_alive))

    # ---- producer-dies-first at N=4 ----
    print("  SIGKILL producer pid=%d" % prod.pid)
    os.kill(prod.pid, signal.SIGKILL)
    prod.wait()
    time.sleep(8)                         # one liveness-reaper cycle
    q_postkill = find_arena(arena_id)
    survived = q_postkill is not None
    producer_cleared = bool(q_postkill and q_postkill["producer_pid"] == 0)
    consumers_held = bool(q_postkill and q_postkill["n_consumers"] == 4)
    print("  post-SIGKILL: survived=%s query=%s" % (survived, q_postkill))

    # ---- release forward #2 on all 4 consumers (producer is dead) ----
    open(KILL_SENTINEL, "w").close()
    fwd2 = all(wait_line(clogs[i], "C%d-FWD2-DONE" % i, cons[i], 300,
                         "consumer-%d" % i)
               for i in range(M.N_CONSUMERS))
    for c in cons:
        c.wait()

    time.sleep(8)                         # reaper collects the empty arena
    reaped = find_arena(arena_id) is None
    fb_exited = M.gpu_fb_mib()
    print("  consumers exited: arena reaped=%s  fb_exited=%d"
          % (reaped, fb_exited))

    # ---- bit-identical verification ----
    import torch
    yp = torch.load("%s/sc6_%s_producer_logits.pt" % (HERE, model_name))
    bit = {"pre": [], "post": []}
    backed = []
    for i in range(M.N_CONSUMERS):
        base = "%s/sc6_%s_consumer_%d" % (HERE, model_name, i)
        for when in ("pre", "post"):
            yc = torch.load("%s_%s_logits.pt" % (base, when))
            bit[when].append(bool(torch.equal(yp, yc)))
        r = json.load(open(base + "_result.json"))
        pi = r.get("page_info_big") or {}
        backed.append(isinstance(pi, dict) and pi.get("kind") == "weight")

    checks = {
        "all4_fwd1_bit_identical": all(bit["pre"]),
        "all4_fwd2_bit_identical_after_producer_death": fwd2 and all(bit["post"]),
        "all4_arena_backed": all(backed),
        "arena_survived_producer_sigkill": survived,
        "producer_pid_cleared": producer_cleared,
        "all4_consumers_held_arena": consumers_held,
        "arena_reaped_after_last_participant": reaped,
    }
    return {"phase": "shared", "model": model_name,
            "arena_id": arena_id,
            "fb_mib": {"idle": fb_idle, "loaded": fb_loaded,
                       "exited": fb_exited},
            "bit_identical": bit, "arena_backed": backed,
            "checks": checks, "PASS": all(checks.values())}


# ----------------------------------------------------------------------
def run_independent(model_name):
    fb_idle = M.gpu_fb_mib()
    procs, logs, n_fit = [], [], 0
    for i in range(5):
        lg = "%s/sc6_%s_independent_%d.log" % (HERE, model_name, i)
        logs.append(lg)
        p = subprocess.Popen(
            [PY, HERE + "/sc6_independent_tenant.py", str(i), model_name],
            stdout=open(lg, "w"), stderr=subprocess.STDOUT)
        procs.append(p)
        # staggered launch — wait this tenant fully resident before the next
        if wait_line(lg, "IND-%d-READY" % i, p, 600, "independent-%d" % i):
            n_fit += 1
            print("  independent tenant %d resident (%d/5)" % (i, n_fit))
        else:
            print("  independent tenant %d did NOT fit — OOM at %d/5"
                  % (i, n_fit))
            break

    fb_loaded = M.gpu_fb_mib()
    per_tenant = (fb_loaded - fb_idle) / n_fit if n_fit else 0
    projected_5 = int(fb_idle + 5 * per_tenant)

    open(IND_SENTINEL, "w").close()
    for p in procs:
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    time.sleep(3)
    fb_exited = M.gpu_fb_mib()
    print("  independent: %d/5 fit  fb_loaded=%d  fb_exited=%d"
          % (n_fit, fb_loaded, fb_exited))

    return {"phase": "independent", "model": model_name,
            "n_fit": n_fit, "oom": n_fit < 5,
            "fb_mib": {"idle": fb_idle, "loaded": fb_loaded,
                       "exited": fb_exited},
            "per_tenant_mib": round(per_tenant, 1),
            "fb_loaded_5tenant_projected": projected_5,
            "PASS": n_fit >= 1}


# ----------------------------------------------------------------------
def main():
    model_name, phase = sys.argv[1], sys.argv[2]
    if phase == "shared":
        clean(model_name)            # shared runs first — owns the cleanup
    res = run_shared(model_name) if phase == "shared" \
        else run_independent(model_name)
    outp = "%s/sc6_%s_%s_result.json" % (HERE, model_name, phase)
    json.dump(res, open(outp, "w"), indent=2, default=str)
    print(json.dumps(res, indent=2))
    print("SC6 %s/%s:" % (model_name, phase),
          "PASS" if res.get("PASS") else "FAIL")
    sys.exit(0 if res.get("PASS") else 1)


if __name__ == "__main__":
    main()
