#!/usr/bin/env bash
# CP 5.5 pre-work plugin-routed Koopman feasibility probe driver.
# Runs three variants in separate processes:
#   baseline: no hook (CIPHER_PROBE_HOOK unset)
#   python:   Python-only wrapper around UnquantizedLinearMethod.apply
#   ctypes:   Python wrapper + ctypes call to cipher_rt_koopman_skip_dtype
# Each variant ships a JSON result file. All three are aggregated post-run.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=/home/ubuntu/vllm_env/bin/python
HARNESS=$(pwd)/probe_harness.py

# Common env across all three runs. CIPHER_KV_ALLOC=0 keeps the KV-bridge
# allocator out of the picture (probe is measuring hook overhead, not KV path).
# LD_PRELOAD points at the option-2-complete substrate so worker GOT patches
# install via the existing cipher_vllm_plugin Step 0 path.
export CIPHER_KV_ALLOC=${CIPHER_KV_ALLOC:-0}
export CIPHER_REGISTER_MODEL=${CIPHER_REGISTER_MODEL:-0}
export CIPHER_REGISTER_STREAMS=${CIPHER_REGISTER_STREAMS:-0}
export LD_PRELOAD=${LD_PRELOAD:-/home/ubuntu/cipher_rt_phase4/libcipher_rt.so}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

# Per-variant output paths.
mkdir -p results
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "results/${TS}"

run_one() {
    local variant=$1
    shift
    local out="results/${TS}/${variant}.json"
    rm -f /tmp/cipher_probe_*.json
    if [ "$variant" = "baseline" ]; then
        unset CIPHER_PROBE_HOOK
    else
        export CIPHER_PROBE_HOOK="$variant"
    fi
    echo "=== variant=${variant} out=${out} CIPHER_PROBE_HOOK=${CIPHER_PROBE_HOOK:-<unset>} ==="
    "$PYTHON" "$HARNESS" --variant "$variant" --out "$out" "$@"
}

# Skip-variant via env override (e.g. CIPHER_PROBE_VARIANTS="ctypes" to redo one)
VARIANTS=${CIPHER_PROBE_VARIANTS:-"baseline python ctypes"}
SCRIPT_ARGS=("$@")
for v in $VARIANTS; do
    run_one "$v" "${SCRIPT_ARGS[@]}"
done

echo "=== aggregating ==="
"$PYTHON" -c "
import glob, json, os, statistics, sys
ts = '${TS}'
files = sorted(glob.glob(f'results/{ts}/*.json'))
results = {}
for fp in files:
    d = json.load(open(fp))
    results[d['variant']] = d
agg = {'ts': ts, 'variants': {}}
for v, d in results.items():
    agg['variants'][v] = {
        'mean_tok_s': d['mean_tok_s'],
        'stdev_tok_s': d['stdev_tok_s'],
        'mean_elapsed_s': d['mean_elapsed_s'],
        'reps_measured': d['reps_measured'],
        'crossings_total': d.get('crossings_total'),
        'gpu_pre': d['pre'].get('gpu'),
    }
if 'baseline' in agg['variants']:
    base = agg['variants']['baseline']['mean_tok_s']
    base_elapsed = agg['variants']['baseline']['mean_elapsed_s']
    for v in ('python', 'ctypes'):
        if v in agg['variants']:
            h = agg['variants'][v]
            h_tps = h['mean_tok_s']
            h_elapsed = h['mean_elapsed_s']
            h['tok_s_delta_pct'] = 100.0 * (h_tps - base) / base
            h['elapsed_delta_pct'] = 100.0 * (h_elapsed - base_elapsed) / base_elapsed
            v_tokens = results[v]['tokens']
            cr = h.get('crossings_total') or 0
            if cr:
                # Per-call overhead = delta_elapsed_per_token / calls_per_token
                total_tokens = h['reps_measured'] * v_tokens
                calls_per_token = cr / total_tokens if total_tokens else 0
                overhead_per_token_s = (h_elapsed - base_elapsed) / v_tokens
                if calls_per_token:
                    h['calls_per_token'] = calls_per_token
                    h['overhead_per_call_us'] = 1e6 * overhead_per_token_s / calls_per_token
                else:
                    h['calls_per_token'] = None
                    h['overhead_per_call_us'] = None
print(json.dumps(agg, indent=2))
with open(f'results/{ts}/aggregate.json', 'w') as f:
    json.dump(agg, f, indent=2)
print(f'wrote results/{ts}/aggregate.json', file=sys.stderr)
"
echo "=== done; results in results/${TS}/ ==="
