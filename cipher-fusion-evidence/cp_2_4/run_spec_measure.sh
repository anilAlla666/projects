#!/usr/bin/env bash
# CP 2.4 sub-task (iii) — Mistral-arm spec-decode lift measurement.
# n=5 matched pairs: arm OFF (CIPHER_SPEC=0, Marlin-only baseline) vs arm ON
# (CIPHER_SPEC=on, Marlin + n-gram spec). Both CIPHER_MARLIN=on. tok/s from the
# run_for_duration progress 'end' line; matched-pair lift = mean(on/off).
# Secondary criterion (user adjudication) — document the lift, no pass target.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/spec_measure_driver.py
N="${N:-5}"
DUR="${DUR:-60}"
OUT="${HERE}/spec_measure"
mkdir -p "$OUT"
rm -f "$OUT"/*.progress "$OUT"/*.log

run() {  # <tid> <spec-mode>
  local tid="$1" spec="$2"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  rm -f "$prog"
  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" WL_DURATION="$DUR" \
    CIPHER_MARLIN=on CIPHER_SPEC="$spec" CIPHER_SPEC_DRAFT=ngram CIPHER_VOLT=off \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 "$DRIVER" > "${OUT}/${tid}.log" 2>&1
  [[ -f "$prog" ]] && cp "$prog" "${OUT}/${tid}.progress"
  tail -1 "$prog" 2>/dev/null || echo "  (no progress — see ${tid}.log)"
}

echo "=== CP 2.4 spec-decode Mistral-arm measurement ==="
echo "bin md5=$(md5sum "$BIN" | cut -c1-8)  n=$N pairs  dur=${DUR}s"
echo "workload: Mistral-7B B=1 decode, CIPHER_MARLIN=on; arm OFF/ON = CIPHER_SPEC 0/on"
echo
for ((i=1; i<=N; i++)); do
  echo "--- pair $i  spec=OFF (Marlin-only baseline) ---"
  run "spm_p${i}_off" 0
  echo "--- pair $i  spec=ON  (Marlin + n-gram spec) ---"
  run "spm_p${i}_on" on
done
echo
echo "=== measurement complete; analyze: python3 ${HERE}/analyze_spec_measure.py ==="
