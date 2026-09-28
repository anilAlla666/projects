#!/usr/bin/env python3
"""Smoke test for cipher_power_cap API."""
import os, ctypes, subprocess, sys, time

os.environ["CIPHER_POWER_CAP"] = "on"

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)  # resolve cuGreenCtx*
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

class Stats(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("original_cap_w", ctypes.c_int),
        ("current_cap_w", ctypes.c_int),
        ("last_applied_batch", ctypes.c_int),
        ("last_applied_cap_w", ctypes.c_int),
        ("apply_calls", ctypes.c_uint64),
        ("restore_calls", ctypes.c_uint64),
        ("nvml_success", ctypes.c_uint64),
        ("subprocess_success", ctypes.c_uint64),
        ("failures", ctypes.c_uint64),
    ]

rt.cipher_power_cap_init.restype = ctypes.c_int
rt.cipher_power_cap_enabled.restype = ctypes.c_int
rt.cipher_power_cap_get_optimal_w.argtypes = [ctypes.c_int]
rt.cipher_power_cap_get_optimal_w.restype  = ctypes.c_int
rt.cipher_power_cap_apply_for_batch.argtypes = [ctypes.c_int]
rt.cipher_power_cap_apply_for_batch.restype  = ctypes.c_int
rt.cipher_power_cap_apply_w.argtypes = [ctypes.c_int]
rt.cipher_power_cap_apply_w.restype  = ctypes.c_int
rt.cipher_power_cap_restore.restype  = ctypes.c_int
rt.cipher_power_cap_stats.argtypes = [ctypes.POINTER(Stats)]
rt.cipher_power_cap_stats.restype  = ctypes.c_int

def cur_cap_via_smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=power.limit",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
    return int(round(float(r.stdout.strip())))

def step(label):
    print(f"\n--- {label} ---")

failures = []
def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        failures.append(msg)

# 1. init
step("init")
rc = rt.cipher_power_cap_init()
check(rc == 1, f"init returned 1 (got {rc})")
check(rt.cipher_power_cap_enabled() == 1, "enabled via env")

s = Stats(); rt.cipher_power_cap_stats(ctypes.byref(s))
print(f"  orig_cap_w={s.original_cap_w}  smi={cur_cap_via_smi()}")
check(s.original_cap_w == 700, f"orig captured as 700W (got {s.original_cap_w})")

# 2. lookup table
step("lookup")
for B, expected in [(1, 300), (2, 300), (7, 300), (8, 400), (16, 400),
                    (32, 700), (33, 700), (63, 700), (64, 500), (128, 500)]:
    got = rt.cipher_power_cap_get_optimal_w(B)
    check(got == expected, f"B={B} → {expected}W (got {got}W)")

# 3. apply for batch=1
step("apply_for_batch(1)")
rc = rt.cipher_power_cap_apply_for_batch(1)
check(rc == 1, "apply returned 1")
time.sleep(0.3)
smi = cur_cap_via_smi()
check(smi == 300, f"hw cap is now 300W (smi={smi})")

# 4. apply for batch=64
step("apply_for_batch(64)")
rc = rt.cipher_power_cap_apply_for_batch(64)
check(rc == 1, "apply returned 1")
time.sleep(0.3)
smi = cur_cap_via_smi()
check(smi == 500, f"hw cap is now 500W (smi={smi})")

# 5. restore
step("restore")
rc = rt.cipher_power_cap_restore()
check(rc == 1, "restore returned 1")
time.sleep(0.3)
smi = cur_cap_via_smi()
check(smi == 700, f"hw cap is back to 700W (smi={smi})")

# 6. apply explicit watts
step("apply_w(250)")
rc = rt.cipher_power_cap_apply_w(250)
check(rc == 1, "apply_w returned 1")
time.sleep(0.3)
smi = cur_cap_via_smi()
check(smi == 250, f"hw cap is now 250W (smi={smi})")
rt.cipher_power_cap_restore()

# 7. final stats
step("stats")
s = Stats(); rt.cipher_power_cap_stats(ctypes.byref(s))
print(f"  apply={s.apply_calls} restore={s.restore_calls} "
      f"nvml_ok={s.nvml_success} subproc_ok={s.subprocess_success} fail={s.failures}")
check(s.apply_calls == 3, f"apply_calls=3 (got {s.apply_calls})")
check(s.failures == 0, f"failures=0 (got {s.failures})")

# Cleanup
rt.cipher_power_cap_restore()
print()
if failures:
    print(f"FAILED: {len(failures)} checks")
    for f in failures: print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
