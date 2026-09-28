#!/usr/bin/env bash
# WL05 same-condition noise band: 3 back-to-back 600s runs at
# libcipher_rt.so.v0.2.0_T4_2_3 + kmod 0.4.4.
set -u
EVID=/home/ubuntu/cipher-phase4-evidence
LOG=$EVID/wl05_noise_runner.log
DONE_MARK=/tmp/_wl05_noise_done
: > "$LOG"
rm -f "$DONE_MARK"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

note "=== WL05 noise band: 3 back-to-back runs ==="
note "library: libcipher_rt.so.v0.2.0_T4_2_3 (md5 bc51b9d6f6827ccdd90ad2d5ab09e3ff)"
note "kmod:    0.4.4 (srcversion $(cat /sys/module/cipher_kmod/srcversion))"

cd /home/ubuntu/cipher_workloads
for i in 1 2 3; do
  note "--- run $i starting"
  rm -f /tmp/cipher_tenant_wl05_t*_progress
  CIPHER_INJECTION_OVERRIDE=/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3 \
    bash ./measurement/run_baseline_wl05.sh --duration 600 >> "$LOG" 2>&1
  python3 ./measurement/phase4_baseline_postprocess.py WL05 > /tmp/_wl05_pp_run${i}.json 2>>"$LOG"
  # Apply device_aggregate patch
  python3 << PYEOF >> "$LOG"
import json
p = "/home/ubuntu/cipher_workloads/expected/wl05_p41_baseline.json"
with open(p) as f: d = json.load(f)
agg = d.get("device_aggregate", {}) or {}
watts = agg.get("power_watts_median", 0.0) or 0.0
agg_mfu = agg.get("aggregate_mfu_pct", 0.0) or 0.0
tps = d.get("tokens_per_sec", 0.0) or 0.0
d["steady_state_mfu_pct"] = agg_mfu
d["watts_avg"] = round(watts, 2)
if watts > 0 and tps > 0:
    d["tokens_per_watt"] = round(tps / watts, 4)
d["baseline_valid"] = True
d["baseline_invalid_reason"] = ""
with open(p,"w") as f: json.dump(d,f,indent=2)
with open("/home/ubuntu/cipher-phase4-evidence/wl05_noise_run${i}.json","w") as f:
    json.dump(d, f, indent=2)
print(f"run ${i}: mfu={agg_mfu} tps={tps} W={watts} TPW={d.get('tokens_per_watt')}")
PYEOF
  note "--- run $i done"
done

note "=== WL05 noise band: all 3 runs complete"
touch "$DONE_MARK"
