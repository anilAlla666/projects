#!/usr/bin/env python3
"""
Op 20 THERMOSTAT — three-test gate.

Uses the test-only injection API so we do not need to bake the H100.

1. trigger:           inject hot temp + drift → aggressive must flip on
2. clear-on-cool:     trigger first, then inject cool + no drift → must clear
3. single-signal:     inject only temp (no drift) → must NOT trigger
                      and inject only drift (no temp) → must NOT trigger
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_thermostat_report.json"

# Each test runs the same Python harness with a different scenario flag.
HARNESS = r"""
import torch, ctypes, time, sys
scenario = sys.argv[1]

rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_thermostat_inject_temp.argtypes = [ctypes.c_float]
rt.cipher_thermostat_inject_temp.restype = None
rt.cipher_thermostat_inject_drift_for_shape.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double]
rt.cipher_thermostat_inject_drift_for_shape.restype = None
rt.cipher_thermostat_clear_injection.restype = None
rt.cipher_thermostat_poll.restype = None
rt.cipher_thermostat_force_poll.restype = None
rt.cipher_thermostat_aggressive_active.restype = ctypes.c_int
rt.cipher_thermostat_event_count.restype = ctypes.c_uint
rt.cipher_thermostat_report.restype = None

# Touch a real workload so the runtime warms up + ring is alive.
small = torch.randn(1, 4096, dtype=torch.float16, device="cuda")
W     = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(20):
    torch.mm(small, W)
torch.cuda.synchronize()


if scenario == "trigger":
    rt.cipher_thermostat_inject_temp(70.0)              # establish prev
    rt.cipher_thermostat_force_poll()
    time.sleep(0.6)
    rt.cipher_thermostat_inject_temp(78.0)              # rising → signal 1
    rt.cipher_thermostat_inject_drift_for_shape(4096, 4096, 1, 1.20)  # signal 2
    rt.cipher_thermostat_force_poll()

elif scenario == "cool":
    # First trigger
    rt.cipher_thermostat_inject_temp(70.0); rt.cipher_thermostat_force_poll()
    time.sleep(0.6)
    rt.cipher_thermostat_inject_temp(80.0)
    rt.cipher_thermostat_inject_drift_for_shape(4096, 4096, 1, 1.30)
    rt.cipher_thermostat_force_poll()
    assert rt.cipher_thermostat_aggressive_active() == 1, "trigger pre-condition failed"
    # Now cool down + drop drift → must clear
    time.sleep(0.6)
    rt.cipher_thermostat_inject_temp(68.0)
    rt.cipher_thermostat_inject_drift_for_shape(4096, 4096, 1, 1.0)
    rt.cipher_thermostat_force_poll()

elif scenario == "temp_only":
    rt.cipher_thermostat_inject_temp(70.0); rt.cipher_thermostat_force_poll()
    time.sleep(0.6)
    rt.cipher_thermostat_inject_temp(80.0)              # signal 1 only
    rt.cipher_thermostat_force_poll()

elif scenario == "drift_only":
    rt.cipher_thermostat_inject_temp(60.0); rt.cipher_thermostat_force_poll()
    time.sleep(0.6)
    rt.cipher_thermostat_inject_drift_for_shape(4096, 4096, 1, 1.30)  # signal 2 only
    rt.cipher_thermostat_force_poll()

else:
    raise SystemExit(f"unknown scenario {scenario}")

time.sleep(0.2)
rt.cipher_thermostat_report()
print("AGGRESSIVE=", rt.cipher_thermostat_aggressive_active(),
      "EVENTS=",     rt.cipher_thermostat_event_count())
"""


def run(scenario: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_THERMOSTAT":   "on"}
    r = subprocess.run(
        ["python3", "-c", HARNESS, scenario],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{scenario}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {scenario} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{scenario}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {scenario}: no THERMOSTAT report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    pass_count = 0

    r = run("trigger")
    ok1 = (r.get("aggressive") == 1 and r.get("event_count", 0) >= 1)
    pass_count += int(ok1)
    print(f"[trigger    ] aggressive={r.get('aggressive')} events={r.get('event_count')} "
          f"drift_shapes={r.get('drift_shape_count')}  {'PASS' if ok1 else 'FAIL'}")

    r = run("cool")
    ok2 = (r.get("aggressive") == 0)
    pass_count += int(ok2)
    print(f"[cool       ] aggressive={r.get('aggressive')} (after trigger+cool)  "
          f"{'PASS' if ok2 else 'FAIL'}")

    r1 = run("temp_only")
    r2 = run("drift_only")
    ok3 = (r1.get("aggressive") == 0 and r2.get("aggressive") == 0)
    pass_count += int(ok3)
    print(f"[single-sig ] temp_only.aggr={r1.get('aggressive')} "
          f"drift_only.aggr={r2.get('aggressive')}  {'PASS' if ok3 else 'FAIL'}")

    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
