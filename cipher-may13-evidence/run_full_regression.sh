#!/bin/bash
# Run every existing test under the full-ops-on environment.
# Each test runs in its own process with a 90s timeout. Pass = exit 0.
#
# Tests that depend on the LD_PRELOAD hook get launched with the hook+
# libcuda preloaded; tests that load rt themselves via ctypes still
# work because the hook just sits idle when its symbols aren't called.

set -u
cd /home/ubuntu/op31-prod-fix

export CIPHER_SENSE=on
export CIPHER_SHIELD=on
export CIPHER_SUSTAIN=on
export CIPHER_THERMOSTAT=on
export CIPHER_PULSE=on
export CIPHER_VOLT=on
export CIPHER_HIBERNATE=on
export CIPHER_LOOP=on
export CIPHER_CONTINUITY=on
export CIPHER_SUBSTITUTE_V2=on
export CIPHER_WEIGHT_COMPRESS=on
export CIPHER_FUSION_KERNELS=on
export CIPHER_FP8_COMPUTE=on
# Other commonly-gated ops referenced by the regression list.
export CIPHER_PREDICT=on
export CIPHER_GUARD=on
export CIPHER_DETERMINISM=on
export CIPHER_TOPOLOGY=on
export CIPHER_TRACE=on
export CIPHER_FAIRNESS=on
export CIPHER_CARBON=on
export CIPHER_RECEIPT=on
export CIPHER_COMPLY=on
export CIPHER_PERSIST_ENGINE=on
export CIPHER_GRAPH=on
export CIPHER_KV_COMPRESS=on
export CIPHER_NCCL_V4=on
export CIPHER_PARTITION_ROUTER=on
export CIPHER_THERMAL_FEEDBACK=on
export CIPHER_VMM=on

export LD_PRELOAD="/home/ubuntu/op31-prod-fix/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"

PASS=0
FAIL=0
SKIP=0
FAIL_LIST=()
SKIP_LIST=()

for t in tests/test_*.py; do
    name=$(basename "$t")
    # Skip benchmark / heavy-workload tests that aren't pass/fail.
    case "$name" in
        test_int4_mistral.py|test_megakernel.py|test_attention_koopman.py|test_edmd_live_calibration.py|test_int4_gemv_perf.py|test_kv_v3_rope.py|test_pattern6_paraminfo.py|test_int4_wiring.py)
            echo "SKIP  $name  (heavy / model-dependent / benchmark)"
            SKIP=$((SKIP+1)); SKIP_LIST+=("$name")
            continue;;
    esac
    out=$(timeout 90 python3 "$t" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS  $name"
        PASS=$((PASS+1))
    else
        echo "FAIL  $name  rc=$rc"
        echo "$out" | tail -10 | sed 's/^/      /'
        FAIL=$((FAIL+1))
        FAIL_LIST+=("$name")
    fi
done

echo
echo "============================================================"
echo " REGRESSION SUMMARY: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    echo " FAIL: ${FAIL_LIST[*]}"
fi
if [ ${#SKIP_LIST[@]} -gt 0 ]; then
    echo " SKIP: ${SKIP_LIST[*]}"
fi
exit $FAIL
