#!/usr/bin/env bash
# CP 2.4 sub-task (iii) — varied-prompt spec-decode lift measurement.
# n=5 matched pairs; each pair runs both arms over the 5-prompt set
# (factual/narrative/code/reasoning/conversational, 128 tokens each):
#   arm OFF = CIPHER_SPEC=0  (Marlin-only baseline)
#   arm ON  = CIPHER_SPEC=on (Marlin + speculative decode)
# Both CIPHER_MARLIN=on, CIPHER_VOLT=off. 5 prompts x 5 pairs = 25 generations
# per arm; analyze_spec_varied.py computes the 25-point matched-pair lift.
#
# Parameterised by env (set per arm — Mistral / Llama):
#   TAG     short label (mistral|llama) -> output dir spec_varied_<TAG>
#   TARGET  target model path
#   DRAFT   draft policy: "ngram" (Mistral arm) or a 1B model path (Llama arm)
#   N       pairs (default 5)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/spec_varied_driver.py

TAG="${TAG:?set TAG (mistral|llama)}"
TARGET="${TARGET:?set TARGET (target model path)}"
DRAFT="${DRAFT:?set DRAFT (ngram | draft model path)}"
N="${N:-5}"
OUT="${HERE}/spec_varied_${TAG}"
mkdir -p "$OUT"
rm -f "$OUT"/*.json "$OUT"/*.log

run() {  # <pair> <off|on>
  local pair="$1" arm="$2"
  local spec=0; [[ "$arm" == on ]] && spec=on
  local tid="sv_${TAG}_p${pair}_${arm}"
  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
    OUT_JSON="${OUT}/p${pair}_${arm}.json" \
    CIPHER_MARLIN=on CIPHER_SPEC="$spec" CIPHER_SPEC_DRAFT="$DRAFT" \
    CIPHER_VOLT=off \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 "$DRIVER" > "${OUT}/p${pair}_${arm}.log" 2>&1
  local rc=$?
  [[ $rc -ne 0 ]] && echo "  pair $pair $arm: EXIT $rc — see p${pair}_${arm}.log"
  grep -h "^RESULT" "${OUT}/p${pair}_${arm}.log" 2>/dev/null | sed 's/^/  /'
}

echo "=== CP 2.4 spec-decode varied-prompt measurement [$TAG] ==="
echo "bin md5=$(md5sum "$BIN" | cut -c1-8)  target=$TARGET"
echo "draft=$DRAFT  n=$N pairs x 5 prompts x 128 tok"
echo
for ((i=1; i<=N; i++)); do
  echo "--- pair $i  arm=OFF (Marlin-only baseline) ---"
  run "$i" off
  echo "--- pair $i  arm=ON  (Marlin + spec) ---"
  run "$i" on
done
echo
echo "=== measurement complete; analyze: TAG=$TAG python3 ${HERE}/analyze_spec_varied.py ==="
