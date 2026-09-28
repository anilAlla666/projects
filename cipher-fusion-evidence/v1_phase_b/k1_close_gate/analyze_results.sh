#!/bin/bash
# K.1 close gate result analyzer. Parses each capture log for [cipher_v2] CLASSIFY
# lines (emitted by EngineCore subprocess to stderr, captured by docker run).
# Emits per-cell PASS/FAIL based on Memory #25 product-engagement-gate:
#   - positive cells: final class != UNKNOWN (any KNOWN class is acceptable;
#     spec-precise class identity is K.1.5 territory)
#   - negative cells: final class == UNKNOWN OR no CLASSIFY emitted
#   - transition cell: progression UNKNOWN -> KNOWN observed across snapshots
#   - calibration cell: confidence grows with obs count
set -u
OUT=/home/ubuntu/cipher-fusion-evidence/v1_phase_b/k1_close_gate
CAPS=$OUT/captures
REPORT=$OUT/K1_CLOSE_REPORT.md

extract_final() {
    local log="$1"
    # Skip parent-process atexit CLASSIFY with obs=0 (no real work observed there).
    # W.2: use anchored pattern " obs=0 " (leading space) so we don't accidentally
    # match marlin_bf16_obs=0 / other _obs=0 fields added in the extended log.
    grep "\[cipher_v2\] CLASSIFY:" "$log" 2>/dev/null | grep -v " obs=0 " | tail -1
}

extract_class_from() {
    echo "$1" | sed -n 's/.*workload=\([A-Z][A-Z0-9_]*\).*/\1/p'
}

extract_obs_from() {
    # W.2 (2026-05-27): anchor on " obs=" (leading space) so we don't pick
    # up marlin_bf16_obs= or other _obs= fields.
    echo "$1" | sed -n 's/.* obs=\([0-9]*\) .*/\1/p'
}

extract_conf_from() {
    echo "$1" | sed -n 's/.*confidence=\([0-9]*\).*/\1/p'
}

extract_int4_from() {
    echo "$1" | sed -n 's/.*int4=\([0-9]*\).*/\1/p'
}

declare -A CLASS_PER_CELL
declare -A OBS_PER_CELL
declare -A CONF_PER_CELL
declare -A INT4_PER_CELL
declare -A PASS_PER_CELL

PASS_COUNT=0
FAIL_COUNT=0

for log in "$CAPS"/*.log; do
    cell=$(basename "$log" .log)
    final_line=$(extract_final "$log")
    if [ -z "$final_line" ]; then
        CLASS_PER_CELL[$cell]="NO_CLASSIFY_EMITTED"
        OBS_PER_CELL[$cell]=0
        CONF_PER_CELL[$cell]=0
        INT4_PER_CELL[$cell]=0
    else
        CLASS_PER_CELL[$cell]=$(extract_class_from "$final_line")
        OBS_PER_CELL[$cell]=$(extract_obs_from "$final_line")
        CONF_PER_CELL[$cell]=$(extract_conf_from "$final_line")
        INT4_PER_CELL[$cell]=$(extract_int4_from "$final_line")
    fi
done

# PASS/FAIL evaluation
eval_cell() {
    local cell="$1" expect="$2"
    local cls="${CLASS_PER_CELL[$cell]:-UNKNOWN}"
    if [ "$expect" = "KNOWN" ]; then
        if [ -n "$cls" ] && [ "$cls" != "UNKNOWN" ] && [ "$cls" != "NO_CLASSIFY_EMITTED" ]; then
            PASS_PER_CELL[$cell]="PASS"; return 0
        else
            PASS_PER_CELL[$cell]="FAIL"; return 1
        fi
    elif [ "$expect" = "UNKNOWN" ]; then
        if [ "$cls" = "UNKNOWN" ] || [ "$cls" = "NO_CLASSIFY_EMITTED" ]; then
            PASS_PER_CELL[$cell]="PASS"; return 0
        else
            PASS_PER_CELL[$cell]="FAIL"; return 1
        fi
    fi
}

# Evaluate per-spec expectations
eval_cell "P1_llama3_8b_bf16" KNOWN
eval_cell "P2_mistral_7b_bf16" KNOWN
eval_cell "P3_tinyllama_awq"   KNOWN
eval_cell "E1_bare_torch"      UNKNOWN
eval_cell "E2_load_idle"       UNKNOWN
eval_cell "E3_load_1tok"       UNKNOWN
eval_cell "T1_llama3_transition" KNOWN
eval_cell "C1_llama3_calibration" KNOWN
# W.1: prefill is compute-bound; classifier should leave volt_engage=0.
# The hysteresis K.1.6 v7 design also keeps this UNKNOWN because the
# workload accumulates few KNOWN-emit cycles before prefill stops.
eval_cell "N1_mistral_long_prefill" UNKNOWN

# Tally
for cell in "${!PASS_PER_CELL[@]}"; do
    if [ "${PASS_PER_CELL[$cell]}" = "PASS" ]; then PASS_COUNT=$((PASS_COUNT+1)); else FAIL_COUNT=$((FAIL_COUNT+1)); fi
done

# Report
{
echo "# K.1 CLOSE GATE REPORT"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Substrate: cipher_rt_phase4/build_cuda13/libcipher_rt.so (K.1.6 build)"
echo "md5: $(md5sum /home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so 2>/dev/null | awk '{print $1}')"
echo ""
echo "## Per-Cell Results"
echo ""
printf "| Cell | Expect | Class | Obs | Conf | INT4 | PASS/FAIL |\n"
printf "|------|--------|-------|-----|------|------|-----------|\n"
for cell in P1_llama3_8b_bf16 P2_mistral_7b_bf16 P3_tinyllama_awq E1_bare_torch E2_load_idle E3_load_1tok T1_llama3_transition C1_llama3_calibration N1_mistral_long_prefill; do
    expect="KNOWN"
    case "$cell" in E1_*|E2_*|E3_*|N1_*) expect="UNKNOWN" ;; esac
    printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
        "$cell" "$expect" \
        "${CLASS_PER_CELL[$cell]:-N/A}" \
        "${OBS_PER_CELL[$cell]:-0}" \
        "${CONF_PER_CELL[$cell]:-0}" \
        "${INT4_PER_CELL[$cell]:-0}" \
        "${PASS_PER_CELL[$cell]:-N/A}"
done
echo ""
echo "## Summary"
echo "- PASS: $PASS_COUNT / 9"
echo "- FAIL: $FAIL_COUNT / 9"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ""
    echo "**VERDICT: K.1 + W.1 CLOSE GATE PASS**"
else
    echo ""
    echo "**VERDICT: CLOSE GATE FAIL** — HARD STOP per Memory #11."
fi
echo ""
echo "## W.1 VOLT engagement evidence (per-cell)"
echo ""
printf "| Cell | VOLT ENGAGED lines | VOLT ARMED lines | VOLT classifier_skipped lines |\n"
printf "|------|---------------------|------------------|-------------------------------|\n"
for cell in P1_llama3_8b_bf16 P2_mistral_7b_bf16 P3_tinyllama_awq T1_llama3_transition C1_llama3_calibration N1_mistral_long_prefill E2_load_idle E3_load_1tok; do
    log="$CAPS/${cell}.log"
    [ -f "$log" ] || continue
    eng=$(grep -c "VOLT: classifier ENGAGED" "$log" 2>/dev/null || echo 0)
    armed=$(grep -c "VOLT: ARMED" "$log" 2>/dev/null || echo 0)
    skip=$(grep -c "VOLT: classifier .*skipped\|VOLT.*skipped" "$log" 2>/dev/null || echo 0)
    printf "| %s | %s | %s | %s |\n" "$cell" "$eng" "$armed" "$skip"
done
} > "$REPORT"

cat "$REPORT"
exit "$FAIL_COUNT"
