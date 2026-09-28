#!/usr/bin/env python3
# MOCK regime handler. Stands in for the REAL validated handlers
# (cipher_inc4.py / v0_phaseC_nf4_cofire.py / v0_phaseA_maxmfu.py) so we can
# test the deployable's ORCHESTRATION end-to-end with NO model load / NO GPU burn.
#
# It does two honest things:
#   1) ECHOES the injection-state env it actually received -> proves the router
#      handed each regime the correct (conflicting) injection state.
#   2) EMITS synthetic Class-D metrics tagged MOCK, shaped like the validated
#      numbers, so the report path can be exercised. These are NOT measurements.
import os, sys, json, time

regime = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CIPHER_REGIME", "?")
job = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

# What injection-state env did we actually inherit from the router?
seen = {k: os.environ[k] for k in (
    "CIPHER_RT_DISABLE_AUTO_INIT", "CUDA_INJECTION64_PATH", "CIPHER_FP8",
    "CIPHER_VOLT", "K") if k in os.environ}

time.sleep(0.3)  # simulate a tiny run

# Synthetic, MOCK Class-D metrics per regime (shaped like the validated results).
if regime == "agent":
    fault = int(os.environ.get("CIPHER_MOCK_FAULT", "0"))  # negative-control hook
    metrics = {"agents_per_gpu": job.get("agents", 100), "models": job.get("models", 4),
               "fault": fault, "fleet_tok_w": 2.0, "p99_short_s": 3.5, "p99_long_s": 10.2}
elif regime == "density":
    metrics = {"models_coresident": job.get("models", 3), "nf4_kl": 0.0, "gb_per_7b": 5.6}
elif regime == "compute":  # secondary, NOT a Class-D headline
    metrics = {"fp8_substituted": 900, "power_delta_pct": -24.0, "note": "secondary/non-headline"}
else:
    metrics = {"error": f"unknown regime {regime}"}

print("CIPHER_MOCK_RESULT " + json.dumps({
    "regime": regime, "name": job.get("name"), "injection_state_seen": seen,
    "metrics": metrics, "mock": True
}), flush=True)
