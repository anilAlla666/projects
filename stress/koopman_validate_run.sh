#!/bin/bash
# Koopman correctness + perf validation:
#   Phase A (no CIPHER): generate VALIDATE_N_TOKENS tokens, save token grid
#   Phase B (CIPHER + Koopman): same, save token grid
#   Compare token-by-token, report tok/s, tok/W, substitution stats
set -u
cd "$(dirname "$0")"/..
ROOT=$(pwd)
source /home/ubuntu/cipher-test-venv/bin/activate

RUN_DIR="${RUN_DIR:-$ROOT/stress2/koopman_validate_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "[ORCH] Run dir: $RUN_DIR"

VALIDATE_N_TOKENS="${VALIDATE_N_TOKENS:-1000}"
VALIDATE_BATCH="${VALIDATE_BATCH:-8}"
VALIDATE_PREFILL="${VALIDATE_PREFILL:-64}"
VALIDATE_MODEL_PATH="${VALIDATE_MODEL_PATH:-/home/ubuntu/models/Llama-3.1-8B}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export VALIDATE_N_TOKENS VALIDATE_BATCH VALIDATE_PREFILL VALIDATE_MODEL_PATH

# ----- Phase A — baseline (no CIPHER) -------------------------------------
echo
echo "============================================================"
echo "Phase A — baseline (no CIPHER)"
echo "============================================================"
A_DIR="$RUN_DIR/phase_a_baseline"
mkdir -p "$A_DIR"
unset LD_PRELOAD
PHASE=A VALIDATE_OUT_PATH="$A_DIR/result.json" \
    python3 stress/koopman_validation.py 2>&1 | tee "$A_DIR/log.txt"

# ----- Phase B — CIPHER full stack with Koopman ---------------------------
echo
echo "============================================================"
echo "Phase B — CIPHER (Koopman + EDMD-live + content-hash cache)"
echo "============================================================"
B_DIR="$RUN_DIR/phase_b_cipher"
mkdir -p "$B_DIR/counters"
export LD_PRELOAD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
export CIPHER_FP8_COMPUTE=on
export CIPHER_FUSION_KERNELS=on
export CIPHER_FUSION=on
export CIPHER_FLOW_RECORD=on
export CIPHER_FLOW_MATCH=on
export CIPHER_FLOW_SUBSTITUTE=on
export CIPHER_THERMOSTAT=on
export CIPHER_PREDICT=on
export CIPHER_GUARD=on
export CIPHER_TRACE=on
export CIPHER_RECEIPT=on
export CIPHER_CARBON=on
export CIPHER_DETERMINISM=off
export CIPHER_TOPOLOGY=on
export CIPHER_PIPELINE=on
export CIPHER_CONTINUITY=on
export CIPHER_LOOP=on
export CIPHER_SPECULATE_CHECK=on
export CIPHER_PERSIST=on
export CIPHER_PERSIST_ENGINE=on
export CIPHER_FAIRNESS=off                 # single tenant
export CIPHER_ARBITRATE=off
export CIPHER_EDMD_LIVE=on
export CIPHER_EDMD_HOOK=on                 # NEW: enables substitution-feed
export CIPHER_USE_CACHE=1                  # NEW: cache is now safe (content-hash only)
export CIPHER_ATTN_KOOPMAN=on
export CIPHER_WORKLOAD_DETECT=on
export CIPHER_WORKLOAD_REPORT=on
export CIPHER_COUNTERS_DUMP_DIR="$B_DIR/counters"
export CIPHER_COMPLY_AUDIT_PATH="$B_DIR/audit.jsonl"
PHASE=B VALIDATE_OUT_PATH="$B_DIR/result.json" \
    python3 stress/koopman_validation.py 2>&1 | tee "$B_DIR/log.txt"

# ----- Compare ------------------------------------------------------------
echo
echo "============================================================"
echo "Comparison report"
echo "============================================================"
python3 - <<EOF | tee "$RUN_DIR/COMPARISON.md"
import json, glob
A = json.load(open("$A_DIR/result.json"))
B = json.load(open("$B_DIR/result.json"))

# Token-by-token equality (greedy decode → must match if substitution is exact)
def cmp_tokens(a, b):
    eq_per_seq = []
    for sa, sb in zip(a, b):
        n = min(len(sa), len(sb))
        eq = sum(1 for i in range(n) if sa[i] == sb[i])
        eq_per_seq.append((eq, n))
    return eq_per_seq

eq = cmp_tokens(A["token_seqs"], B["token_seqs"])

# Aggregate Koopman counter sum across phase B
import os
counters_dir = "$B_DIR/counters"
op_sum = {}
n_files = 0
for f in glob.glob(os.path.join(counters_dir, "*.json")):
    try:
        o = json.load(open(f))
        for k, v in o.get("ops", {}).items():
            op_sum[k] = op_sum.get(k, 0) + int(v)
        n_files += 1
    except Exception:
        pass

print("# Koopman validation report")
print()
print("## Performance")
print()
print("| metric | A baseline | B CIPHER | ratio |")
print("|---|---|---|---|")
print(f"| tok/s          | {A['tokens_per_s']:8.1f} | {B['tokens_per_s']:8.1f} | {B['tokens_per_s']/A['tokens_per_s']:.3f}× |")
print(f"| mean watts     | {A['mean_watts']:8.1f} | {B['mean_watts']:8.1f} | {B['mean_watts']/A['mean_watts']:.3f}× |")
print(f"| tok/W          | {A['tok_per_w']:8.3f} | {B['tok_per_w']:8.3f} | {B['tok_per_w']/A['tok_per_w']:.3f}× |")
print(f"| total tokens   | {A['total_tokens']} | {B['total_tokens']} |  |")
print(f"| elapsed (s)    | {A['elapsed_s']:.1f} | {B['elapsed_s']:.1f} |  |")

print()
print("## Coherence (greedy decode token-by-token equality)")
print()
print("| seq | matched / total | first divergence |")
print("|---|---|---|")
for i, (e, n) in enumerate(eq):
    div = "—" if e == n else f"token {e}"
    print(f"| {i} | {e}/{n} ({100*e/n:.1f}%) | {div} |")
total_eq = sum(e for e, _ in eq)
total_n  = sum(n for _, n in eq)
print()
print(f"**Aggregate: {total_eq}/{total_n} = {100*total_eq/total_n:.2f}% identical tokens**")

print()
print("## CIPHER op activity (Phase B, summed across counter dump files)")
print(f"({n_files} counter dump files)")
print()
print("| op | calls |")
print("|---|---|")
key_ops = ["WORKLOAD_OBSERVE","SUBSTITUTE_FP8","SUBSTITUTE_KOOPMAN",
           "FLOW_SUBSTITUTE","FUSE","ADAPT","REMEMBER","VALIDATE",
           "SPECULATE_CHECK","SPECULATE_WRITE","FAIRNESS","CARBON","RECEIPT",
           "TRACE","CLASSIFY","RING_WRITE","FLOW_RECORD","FLOW_MATCH",
           "PERSIST","THERMOSTAT","GUARD","PREDICT","TOPOLOGY"]
for k in key_ops:
    if k in op_sum:
        print(f"| {k} | {op_sum[k]:,} |")

print()
print("## Sample decoded text (first 200 chars)")
print()
for i in range(min(4, len(A.get("sample_decoded", [])))):
    print(f"### Seq {i}")
    print(f"- A: \`{A['sample_decoded'][i]}\`")
    print(f"- B: \`{B['sample_decoded'][i]}\`")
EOF
echo
echo "[ORCH] DONE → $RUN_DIR/COMPARISON.md"
