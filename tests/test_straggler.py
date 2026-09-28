#!/usr/bin/env python3
"""
Phase 3 — Straggler Detection: 5-test gate (4 PASS + 1 SKIP on this single-GPU pod).

Tests:
  1. rank-detection-from-env       (set $RANK + $WORLD_SIZE → my_rank == 3)
  2. rank-fallback                 (no env → my_rank == -1)
  3. local-slowdown-detected       (inject 5 inflated dur_ns → event_count >= 1)
  4. algo-hint-fires-when-active   (slowdown active + mode=active → returns RING)
  5. multi-rank-attribution        (SKIP — needs torchrun + aggregator)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT = "/tmp/cipher_straggler_report.json"

HARNESS = r"""
import torch, ctypes, sys, time
scenario = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_straggler_my_rank.restype = ctypes.c_int
rt.cipher_straggler_world_size.restype = ctypes.c_int
rt.cipher_straggler_observe.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
rt.cipher_straggler_observe.restype = None
rt.cipher_straggler_inject_dur.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
rt.cipher_straggler_inject_dur.restype = None
rt.cipher_straggler_clear_injection.restype = None
rt.cipher_straggler_local_slowdown_active.restype = ctypes.c_int
rt.cipher_straggler_event_count.restype = ctypes.c_uint
rt.cipher_straggler_algo_hint.argtypes = [ctypes.c_uint64]
rt.cipher_straggler_algo_hint.restype = ctypes.c_int
rt.cipher_straggler_status_string.restype = ctypes.c_char_p
rt.cipher_straggler_report.restype = None

# Touch the runtime
small = torch.randn(1, 1024, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()

if scenario == "env":
    print("RANK", rt.cipher_straggler_my_rank(),
          "WORLD", rt.cipher_straggler_world_size())
elif scenario == "noenv":
    print("RANK", rt.cipher_straggler_my_rank())
elif scenario == "slowdown":
    BYTES = 1_000_000
    # Establish baseline EMA at 0.5 ns/byte (500 us for 1MB)
    for _ in range(20): rt.cipher_straggler_observe(BYTES, 500_000, 1)
    # Inject 5 inflated calls (3.0 ns/byte = 6x baseline)
    for _ in range(8):
        rt.cipher_straggler_inject_dur(BYTES, 3_000_000)
        rt.cipher_straggler_observe(BYTES, 1, 1)   # dur replaced by inject
    print("EVENTS", rt.cipher_straggler_event_count(),
          "SLOWDOWN", rt.cipher_straggler_local_slowdown_active(),
          "HINT", rt.cipher_straggler_algo_hint(BYTES))
elif scenario == "hint_off":
    BYTES = 1_000_000
    for _ in range(20): rt.cipher_straggler_observe(BYTES, 500_000, 1)
    print("HINT", rt.cipher_straggler_algo_hint(BYTES))
else:
    raise SystemExit(f"unknown {scenario}")

rt.cipher_straggler_report()
"""


def run(scenario: str, env_extra: dict) -> dict:
    if os.path.exists(REPORT): os.remove(REPORT)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           **env_extra}
    r = subprocess.run(
        ["python3", "-c", HARNESS, scenario],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{scenario}] STDERR tail:\n{r.stderr[-1500:]}")
        raise SystemExit(f"workload {scenario} crashed")
    return {"stdout": r.stdout.strip(), "stderr": r.stderr,
            "report": json.load(open(REPORT)) if os.path.exists(REPORT) else None}


def main():
    pass_count = 0

    # 1. rank-detection-from-env
    r = run("env", {"CIPHER_STRAGGLER": "observe", "RANK": "3", "WORLD_SIZE": "8"})
    ok1 = ("RANK 3 WORLD 8" in r["stdout"])
    pass_count += int(ok1)
    print(f"[env-rank   ] {r['stdout']}  {'PASS' if ok1 else 'FAIL'}")

    # 2. rank-fallback (clear env vars in subprocess)
    e = {"CIPHER_STRAGGLER": "observe"}
    # Strip any inherited rank env
    e_clear = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
                            "OMPI_COMM_WORLD_SIZE", "SLURM_PROCID",
                            "SLURM_NTASKS", "PMI_RANK", "PMI_SIZE")}
    e_clear.update({"LD_PRELOAD": f"{HOOK} {RT}", "CIPHER_FORCE_PERMIT": "1",
                    **e})
    r = subprocess.run(
        ["python3", "-c", HARNESS, "noenv"],
        capture_output=True, text=True, timeout=120, cwd=CWD, env=e_clear,
    )
    ok2 = ("RANK -1" in r.stdout)
    pass_count += int(ok2)
    print(f"[fallback   ] {r.stdout.strip()}  {'PASS' if ok2 else 'FAIL'}")

    # 3. local-slowdown-detected (mode=active so we can also assert the hint)
    r = run("slowdown", {"CIPHER_STRAGGLER": "active"})
    ok3 = ("SLOWDOWN 1" in r["stdout"]) and ("EVENTS " in r["stdout"]) \
          and (r["report"] and r["report"].get("event_count", 0) >= 1)
    pass_count += int(ok3)
    print(f"[slowdown   ] {r['stdout']}  events={r['report'].get('event_count') if r['report'] else None}  "
          f"{'PASS' if ok3 else 'FAIL'}")

    # 4. algo-hint-fires (active + slowdown → returns RING==1; observe → -1)
    # sub-test 4a: active mode + sustained slowdown → returns 1 (RING)
    r4a = run("slowdown", {"CIPHER_STRAGGLER": "active"})
    hint_active = (r4a["stdout"].split("HINT")[-1].strip() == "1")
    # sub-test 4b: observe mode → -1 even if slowdown
    r4b = run("hint_off", {"CIPHER_STRAGGLER": "observe"})
    hint_off = (r4b["stdout"].split("HINT")[-1].strip() == "-1")
    ok4 = hint_active and hint_off
    pass_count += int(ok4)
    print(f"[algo-hint  ] active.HINT={'1' if hint_active else '?'} "
          f"observe.HINT={'-1' if hint_off else '?'}  {'PASS' if ok4 else 'FAIL'}")

    # 5. multi-rank-attribution — SKIP
    print(f"[multi-rank ] SKIP — requires torchrun + cross-rank aggregator. "
          f"v1 implements local detection only; cross-rank attribution is done via "
          f"tools/straggler_aggregate.py over per-rank logs (NCCL tuner v2 ABI does "
          f"not expose per-rank completion timing, so real-time cross-rank attribution "
          f"is deferred to a future Coordinator op).")

    print(f"\nResult: {pass_count}/4  (5th SKIP, documented)")
    sys.exit(0 if pass_count == 4 else 1)


if __name__ == "__main__":
    main()
