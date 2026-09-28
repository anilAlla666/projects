#!/usr/bin/env python3
"""Track 2 SC5 Phase A — VMM cross-process refcount probe (orchestrator+consumer).

Gates the SC5 fd-custodian design (SC5 design memo §1). Verifies on this
H100 / CUDA 13:
  1. a consumer's imported mapping stays valid after the producer is SIGKILL'd
     (checksum bit-identical before vs after, and a GPU op still works);
  2. the physical memory frees once the last handle drops (producer dead +
     consumer releases -> the 256 MiB arena returns to free).

PASS -> the fd-custodian model holds; proceed to the kmod arena registry.
FAIL -> STOP; SC5 needs the heavier kmod-tracks-memory-state redesign.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
SOCK = "/tmp/sc5_probe.sock"
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402


def fb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def main():
    res = {}
    fb_idle = fb()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    plog = HERE + "/sc5_probe_producer.log"
    prod = subprocess.Popen([sys.executable, HERE + "/sc5_probe_producer.py"],
                            stdout=open(plog, "w"), stderr=subprocess.STDOUT)

    line = None
    t0 = time.time()
    while time.time() - t0 < 150:
        txt = open(plog).read() if os.path.exists(plog) else ""
        for L in txt.splitlines():
            if L.startswith("PRODUCER ready"):
                line = L
        if line:
            break
        if prod.poll() is not None:
            print("FAIL — producer died early"); sys.exit(1)
        time.sleep(0.5)
    kv = dict(tok.split("=") for tok in line.split() if "=" in tok)
    base, size = int(kv["base"], 16), int(kv["size"])
    n, pchecksum = int(kv["N"]), float(kv["checksum"])
    fb_producer = fb()

    # ---- consumer: import the arena ----
    kvb.init(64 * 1024 * 1024)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(SOCK)
    _, fds, _, _ = socket.recv_fds(c, 256, 1)
    c.close()
    fd = fds[0]
    arena = kvb.weight_arena_import(fd, base, size)
    os.close(fd)
    t = arena.view([n], 4, "float32", 0)
    torch.cuda.synchronize()
    checksum_1 = float(t.double().sum().item())

    # ---- KILL the producer (no cleanup) ----
    prod.send_signal(signal.SIGKILL)
    prod.wait()
    time.sleep(2.0)                       # let the producer's teardown finish

    # ---- consumer re-reads + a GPU op AFTER the producer is dead ----
    torch.cuda.synchronize()
    checksum_2 = float(t.double().sum().item())
    gpu_op = float((t * 2.0).double().sum().item())

    survives = (abs(checksum_2 - checksum_1) < 1e-3 and
                abs(checksum_1 - pchecksum) < max(1.0, abs(pchecksum) * 1e-5))
    gpu_ok = abs(gpu_op - 2.0 * checksum_2) < max(1.0, abs(checksum_2) * 1e-3)

    # ---- memory: empty_cache() so torch's caching allocator (grown by the
    #      GPU-op temporaries above) does not mask the arena's release ----
    import gc
    del t                                 # the from_blob view (no-op deleter)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(1.5)
    fb_holding = fb()                     # consumer still holds `arena`

    del arena                             # WeightArena.__del__ -> arena freed
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(1.5)
    fb_after_release = fb()               # last handle dropped

    arena_mib = size // (1024 * 1024)
    freed = (fb_holding - fb_after_release) >= arena_mib * 0.7

    res.update(
        producer_checksum=pchecksum, checksum_1_before_kill=checksum_1,
        checksum_2_after_kill=checksum_2, gpu_op_after_kill=gpu_op,
        survives_producer_death=bool(survives),
        gpu_op_after_kill_ok=bool(gpu_ok),
        fb_idle=fb_idle, fb_producer=fb_producer,
        fb_holding_after_kill=fb_holding,
        fb_after_release=fb_after_release, arena_mib=arena_mib,
        arena_freed_mib=(fb_holding - fb_after_release),
        physical_freed_after_last_handle=bool(freed))
    res["PROBE_PASS"] = bool(survives and gpu_ok and freed)
    json.dump(res, open(HERE + "/sc5_probe_result.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("VMM REFCOUNT PROBE:", "PASS" if res["PROBE_PASS"] else "FAIL")
    sys.exit(0 if res["PROBE_PASS"] else 1)


if __name__ == "__main__":
    main()
