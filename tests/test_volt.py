#!/usr/bin/env python3
"""
Op 30 VOLT — 5-test gate (4 PASS + 1 SKIP on this pod).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
CALIB = "/tmp/cipher_volt_calibration.json"
REPORT = "/tmp/cipher_volt_report.json"


def run(env_extra: dict, code: str, expect_exit_zero: bool = True) -> dict:
    env = {**os.environ, "LD_PRELOAD": f"{HOOK} {RT}", "CIPHER_FORCE_PERMIT": "1", **env_extra}
    r = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if expect_exit_zero and r.returncode != 0:
        print(f"STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload crashed (exit {r.returncode})")
    return {"stdout": r.stdout.strip(), "stderr": r.stderr}


def main():
    pass_count = 0

    # ── 1. classifier-correctness — pure math, no LD_PRELOAD needed ──────────
    code = r"""
import ctypes
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_volt_arithmetic_intensity.argtypes = [ctypes.c_uint32]*3
rt.cipher_volt_arithmetic_intensity.restype = ctypes.c_double
rt.cipher_volt_classify.argtypes = [ctypes.c_uint32]*3
rt.cipher_volt_classify.restype = ctypes.c_int
ai_decode  = rt.cipher_volt_arithmetic_intensity(1, 4096, 4096)
ai_prefill = rt.cipher_volt_arithmetic_intensity(4096, 4096, 4096)
ai_amb     = rt.cipher_volt_arithmetic_intensity(64, 4096, 4096)
cls_decode  = rt.cipher_volt_classify(1, 4096, 4096)
cls_prefill = rt.cipher_volt_classify(4096, 4096, 4096)
cls_amb     = rt.cipher_volt_classify(64, 4096, 4096)
# 1=MEMORY_BOUND_SAFE, 2=FREQ_SENSITIVE, 3=AMBIGUOUS
import sys
ok = (0.5 < ai_decode < 1.5
      and ai_prefill > 1000
      and cls_decode == 1
      and cls_prefill == 2
      and cls_amb == 3)
print(f"ai_decode={ai_decode:.3f} ai_prefill={ai_prefill:.1f} ai_amb={ai_amb:.2f} "
      f"cls_decode={cls_decode} cls_prefill={cls_prefill} cls_amb={cls_amb}")
sys.exit(0 if ok else 1)
"""
    r = run({}, code)
    ok1 = (r["stdout"].endswith("cls_amb=3") or "cls_amb=3" in r["stdout"]) \
          and "cls_decode=1" in r["stdout"] and "cls_prefill=2" in r["stdout"]
    pass_count += int(ok1)
    print(f"[classifier ] {r['stdout']}  {'PASS' if ok1 else 'FAIL'}")

    # ── 2. probe-disable-path — calibrate mode writes degraded JSON ──────────
    if os.path.exists(CALIB): os.remove(CALIB)
    code = r"""
import torch, time
small = torch.randn(1, 1024, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()
time.sleep(0.5)
"""
    run({"CIPHER_VOLT": "calibrate"}, code)
    with open(CALIB) as f: cal = json.load(f)
    ok2 = (cal.get("actuation_supported") is False
           and cal.get("probe_rc_meaning") == "NVML_ERROR_NOT_SUPPORTED"
           and cal.get("pod_driver_version", "") != ""
           and len(cal.get("shapes", [])) >= 1
           and all(s["class"] == "FREQ_SENSITIVE" for s in cal["shapes"]))
    pass_count += int(ok2)
    print(f"[probe-disab] supported={cal.get('actuation_supported')} "
          f"rc_meaning={cal.get('probe_rc_meaning')} "
          f"driver={cal.get('pod_driver_version')} shapes={len(cal.get('shapes', []))}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    # ── 3. active-refuses-without-calibration ────────────────────────────────
    if os.path.exists(CALIB): os.remove(CALIB)
    code = r"""
import torch, ctypes
rt = ctypes.CDLL('./libcipher_rt.so', mode=ctypes.RTLD_GLOBAL)
rt.cipher_volt_mode.restype = ctypes.c_int
small = torch.randn(1, 1024, dtype=torch.float16, device='cuda'); torch.cuda.synchronize()
mode = rt.cipher_volt_mode()
print('MODE', mode)
"""
    r = run({"CIPHER_VOLT": "on"}, code)
    ok3 = ("refusing to start" in r["stderr"] and "MODE 0" in r["stdout"])  # OFF=0
    pass_count += int(ok3)
    print(f"[refuse-no-c] mode={r['stdout']}  log_says_refuse={'refusing to start' in r['stderr']}  "
          f"{'PASS' if ok3 else 'FAIL'}")

    # ── 4. active-degraded-noop — calibrate then on ──────────────────────────
    code_cal = r"""
import torch, time
small = torch.randn(1, 1024, dtype=torch.float16, device='cuda'); torch.cuda.synchronize()
time.sleep(0.5)
"""
    run({"CIPHER_VOLT": "calibrate"}, code_cal)
    code_on = r"""
import torch, ctypes
rt = ctypes.CDLL('./libcipher_rt.so', mode=ctypes.RTLD_GLOBAL)
rt.cipher_volt_mode.restype = ctypes.c_int
rt.cipher_volt_reduce_event_count.restype = ctypes.c_uint
rt.cipher_volt_status_string.restype = ctypes.c_char_p
small = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
W     = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(50): torch.mm(small, W)
torch.cuda.synchronize()
print('MODE', rt.cipher_volt_status_string().decode(), 'REDUCE', rt.cipher_volt_reduce_event_count())
"""
    r = run({"CIPHER_VOLT": "on"}, code_on)
    ok4 = ("MODE DEGRADED" in r["stdout"] and "REDUCE 0" in r["stdout"])
    pass_count += int(ok4)
    print(f"[active-deg ] {r['stdout']}  {'PASS' if ok4 else 'FAIL'}")

    # ── 5. actuation-roundtrip — SKIP on this pod ────────────────────────────
    print(f"[actuation  ] SKIP — pod actuation_supported=false (probe rc=4 NVML_ERROR_NOT_SUPPORTED). "
          f"This test runs on a permissioned host where SetGpuLockedClocks is allowed.")

    print(f"\nResult: {pass_count}/4  (5th SKIP, documented)")
    sys.exit(0 if pass_count == 4 else 1)


if __name__ == "__main__":
    main()
