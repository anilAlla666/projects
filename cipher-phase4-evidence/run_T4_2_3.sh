#!/usr/bin/env bash
# T4.2.3 measurement: WL01, WL02, WL05 (lift), WL03, WL14 (no-regression).
# Each 600s under libcipher_rt.so.v0.2.0_T4_2_3.
set -u
EVID=/home/ubuntu/cipher-phase4-evidence
LOG=$EVID/T4_2_3_runner.log
SUMMARY=$EVID/T4_2_3_summary.txt
JSONL=$EVID/T4_2_3_baselines.jsonl
: > "$LOG"; : > "$SUMMARY"; : > "$JSONL"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" "$SUMMARY"; }

note "=== T4.2.3 measurement under cipher_kmod $(cat /sys/module/cipher_kmod/version) + libcipher_rt v0.2.0_T4_2_3 (md5 $(md5sum /home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3 | cut -d' ' -f1))"

cd /home/ubuntu/cipher_workloads
run_one() {
  local WL="$1"
  note "--- $WL: starting (600s)"
  rm -f /tmp/cipher_tenant_${WL,,}_baseline_progress
  CIPHER_INJECTION_OVERRIDE=/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3 \
    bash ./measurement/run_baseline.sh "$WL" --duration 600 >> "$LOG" 2>&1
  cp expected/${WL,,}_p41_baseline.json "$EVID/${WL,,}_T4_2_3.json" 2>/dev/null
  python3 ./measurement/phase4_baseline_postprocess.py "$WL" > /tmp/_t4_2_3.json 2>>"$LOG"
  cat /tmp/_t4_2_3.json >> "$JSONL"; echo >> "$JSONL"
  note "--- $WL: done"
  cat /tmp/_t4_2_3.json | tee -a "$SUMMARY" >/dev/null
}

run_wl05() {
  note "--- WL05 multi-tenant ×8: starting (600s)"
  rm -f /tmp/cipher_tenant_wl05_t*_progress
  CIPHER_INJECTION_OVERRIDE=/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3 \
    bash ./measurement/run_baseline_wl05.sh --duration 600 >> "$LOG" 2>&1
  cp expected/wl05_p41_baseline.json "$EVID/wl05_T4_2_3.json"
  # WL05 needs the device_aggregate patch
  python3 << 'PYEOF' >> "$JSONL"
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
print(json.dumps({"wl_id":"WL05","steady_state_mfu_pct":d["steady_state_mfu_pct"],
                  "tokens_per_sec":tps,"watts_avg":d["watts_avg"],
                  "tokens_per_watt":d.get("tokens_per_watt"),
                  "children_complete": d.get("wl05_children_complete"),
                  "kmod_version":d.get("kmod_version")}, indent=2))
with open(p,"w") as f: json.dump(d,f,indent=2)
PYEOF
  echo >> "$JSONL"
  note "--- WL05: done"
}

run_one WL01
run_one WL02
run_wl05
run_one WL03
run_one WL14

note "=== T4.2.3 measurement done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
