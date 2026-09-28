#!/usr/bin/env bash
set -u
EVID=/home/ubuntu/cipher-phase4-evidence
LOG=$EVID/diagnostics.log
JSONL=$EVID/diagnostics.jsonl
: > "$LOG"
: > "$JSONL"
note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

note "=== diagnostic 1: WL01 baseline repeat (noise band) — libcipher_v2 on kmod 0.4.4"
cd /home/ubuntu/cipher_workloads
rm -f /tmp/cipher_tenant_wl01_baseline_progress
# default override is libcipher_v2 — leave CIPHER_INJECTION_OVERRIDE unset
bash ./measurement/run_baseline.sh WL01 --duration 600 >> "$LOG" 2>&1
python3 ./measurement/phase4_baseline_postprocess.py WL01 > /tmp/_d1.json 2>>"$LOG"
cp expected/wl01_p41_baseline.json $EVID/wl01_baseline_repeat.json
{ echo '{"label":"WL01 baseline repeat (libcipher_v2 on kmod 0.4.4)",'; tail -n +2 /tmp/_d1.json; } >> "$JSONL"
echo >> "$JSONL"
note "diagnostic 1 done: $(cat /tmp/_d1.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"MFU={d[\"steady_state_mfu_pct\"]} tps={d[\"tokens_per_sec\"]} W={d[\"watts_avg\"]} TPW={d[\"tokens_per_watt\"]}")')"

note "=== diagnostic 2: WL14 bisect — libcipher_v2 on kmod 0.4.4"
rm -f /tmp/cipher_tenant_wl14_baseline_progress
bash ./measurement/run_baseline.sh WL14 --duration 600 >> "$LOG" 2>&1
python3 ./measurement/phase4_baseline_postprocess.py WL14 > /tmp/_d2.json 2>>"$LOG"
cp expected/wl14_p41_baseline.json $EVID/wl14_libcipher_v2_on_0.4.4.json
{ echo '{"label":"WL14 libcipher_v2 on kmod 0.4.4 (bisect)",'; tail -n +2 /tmp/_d2.json; } >> "$JSONL"
echo >> "$JSONL"
note "diagnostic 2 done: $(cat /tmp/_d2.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"MFU={d[\"steady_state_mfu_pct\"]} tps={d[\"tokens_per_sec\"]} W={d[\"watts_avg\"]} TPW={d[\"tokens_per_watt\"]}")')"

note "=== diagnostics done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
