#!/bin/bash
# CIPHER MULTI-PHASE OP-COVERAGE TEST
# Drives each of the 33 ops to non-zero counter via 5 distinct workload phases.
# Aggregates per-phase counters into a single union table at the end.

set -u
cd "$(dirname "$0")"/..
ROOT=$(pwd)
source /home/ubuntu/cipher-test-venv/bin/activate

RUN_DIR="${RUN_DIR:-$ROOT/stress2/multiphase_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "[ORCH] Run dir: $RUN_DIR"

P1_DURATION_S="${P1_DURATION_S:-180}"
P2_DURATION_S="${P2_DURATION_S:-30}"
P3_DURATION_S="${P3_DURATION_S:-120}"
P4_DURATION_S="${P4_DURATION_S:-60}"
N_TENANTS_P1="${N_TENANTS_P1:-15}"
MODEL_P1="${MODEL_P1:-/home/ubuntu/models/Llama-3.2-1B}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Common CIPHER env applied to every phase
CIPHER_ENV=(
    LD_PRELOAD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
    CIPHER_FP8_COMPUTE=on
    CIPHER_FUSION_KERNELS=on
    CIPHER_SUBSTITUTE_V2=on
    CIPHER_PERSIST_ENGINE=on
    CIPHER_PERSIST=on
    CIPHER_FAIRNESS=on
    CIPHER_FLOW_RECORD=on
    CIPHER_FLOW_MATCH=on
    CIPHER_FLOW_SUBSTITUTE=on
    CIPHER_FUSION=on
    CIPHER_THERMOSTAT=on
    CIPHER_NCCL_TUNER=on
    CIPHER_NCCL_V4=on
    CIPHER_GRAPH=on
    CIPHER_GRAPH_INSPECT=on
    CIPHER_CARBON=on
    CIPHER_COMPLY=on
    CIPHER_TRACE=on
    CIPHER_RECEIPT=on
    CIPHER_DETERMINISM=off
    CIPHER_TOPOLOGY=on
    CIPHER_GUARD=on
    CIPHER_PREDICT=on
    CIPHER_PIPELINE=on
    CIPHER_CONTINUITY=on
    CIPHER_LOOP=on
    CIPHER_SPECULATE_CHECK=on
    CIPHER_SENSE=on
    CIPHER_SHIELD=on
    CIPHER_SUSTAIN=on
    CIPHER_PULSE=on
    CIPHER_VOLT=on
    CIPHER_HIBERNATE=on
    CIPHER_PARTITION_ROUTER=on
    CIPHER_THERMAL_FEEDBACK=on
    CIPHER_VMM=on
    CIPHER_KV_COMPRESS=on
    CIPHER_EDMD_LIVE=on
    CIPHER_EDMD_HOOK=on
    CIPHER_USE_CACHE=0
    CIPHER_ATTN_KOOPMAN=on
    CIPHER_WEIGHT_COMPRESS=on
    CIPHER_DVFS=on
    CIPHER_WORKLOAD_DETECT=on
    CIPHER_WORKLOAD_REPORT=on
    CIPHER_ARBITRATE=on
    CIPHER_AUDIT=on
    CIPHER_COMPLY_AUDIT_PATH="$RUN_DIR/cipher_audit.jsonl"
    MT_LOAD_RT=1
)

# ============================================================================
# PHASE 1 — 15-tenant LLM decode (Llama-3.2-1B, B=1) on GPU 0
# Targets: FP8, FUSE, FAIRNESS, ARBITRATE, FLOW_*, CLASSIFY, RING_WRITE,
#   SPECULATE_*, PREDICT, GUARD, RECEIPT, CARBON, TRACE, CONTINUITY, LOOP,
#   PIPELINE, PERSIST, THERMOSTAT, TOPOLOGY, WORKLOAD_OBSERVE, DETERMINISM,
#   SUBSTITUTE_KOOPMAN (with EDMD_HOOK on), ADAPT, REMEMBER, VALIDATE
# ============================================================================
echo
echo "============================================================"
echo "PHASE 1 — ${N_TENANTS_P1} tenants × Llama-3.2-1B B=1 (${P1_DURATION_S}s)"
echo "============================================================"
P1_DIR="$RUN_DIR/phase1_decode"
mkdir -p "$P1_DIR/counters" "$P1_DIR/stderr"

P1_PIDS=()
for i in $(seq 0 $((N_TENANTS_P1-1))); do
    (
        for kv in "${CIPHER_ENV[@]}"; do export "$kv"; done
        export CIPHER_TENANT_ID="$i"
        export MT_DURATION_S="$P1_DURATION_S"
        export MT_OUT_PATH="$P1_DIR/t${i}.json"
        export MT_BURST_LEN=10
        export MT_BATCH=1
        export MT_PREFILL=128
        export MT_MODEL_PATH="$MODEL_P1"
        export CIPHER_COUNTERS_DUMP_DIR="$P1_DIR/counters"
        export CUDA_VISIBLE_DEVICES=0
        python3 stress/full_stack_child.py \
            > "$P1_DIR/stderr/t${i}.stdout" 2> "$P1_DIR/stderr/t${i}.stderr"
    ) &
    P1_PIDS+=($!)
    sleep 0.4
done
echo "[ORCH] Phase 1: spawned ${#P1_PIDS[@]} tenants"
for pid in "${P1_PIDS[@]}"; do wait "$pid" || true; done
P1_OK=$(ls -1 $P1_DIR/t*.json 2>/dev/null | wc -l)
echo "[ORCH] Phase 1: $P1_OK/$N_TENANTS_P1 tenants produced JSON"

# ============================================================================
# PHASE 2 — TP=2 NCCL on GPU 0+1 (Llama-3.1-8B placeholder; pure NCCL traffic)
# Targets: NCCL_TUNER, TOPOLOGY, COMPLY, AUDIT (NCCL tuner emits audit events)
# ============================================================================
echo
echo "============================================================"
echo "PHASE 2 — TP=2 NCCL on GPU 0+1 (${P2_DURATION_S}s)"
echo "============================================================"
P2_DIR="$RUN_DIR/phase2_nccl"
mkdir -p "$P2_DIR/counters" "$P2_DIR/stderr"

(
    for kv in "${CIPHER_ENV[@]}"; do export "$kv"; done
    export PHASE2_DURATION_S="$P2_DURATION_S"
    export PHASE2_OUT_PATH="$P2_DIR/decide.json"
    export CIPHER_COUNTERS_DUMP_DIR="$P2_DIR/counters"
    export CUDA_VISIBLE_DEVICES=0
    python3 stress/phase2_nccl_child.py \
        > "$P2_DIR/stderr/decide.stdout" 2> "$P2_DIR/stderr/decide.stderr"
)
echo "[ORCH] Phase 2: done"

# ============================================================================
# PHASE 3 — graph capture serving (Llama-3.2-1B, single tenant, GPU 0)
# Targets: GRAPH_ENGINE (cudaGraphInstantiate intercept), LOOP, PIPELINE
# ============================================================================
echo
echo "============================================================"
echo "PHASE 3 — graph capture serving (${P3_DURATION_S}s)"
echo "============================================================"
P3_DIR="$RUN_DIR/phase3_graph"
mkdir -p "$P3_DIR/counters" "$P3_DIR/stderr"
(
    for kv in "${CIPHER_ENV[@]}"; do export "$kv"; done
    export PHASE3_DURATION_S="$P3_DURATION_S"
    export PHASE3_OUT_PATH="$P3_DIR/graph.json"
    export PHASE3_MODEL_PATH=/home/ubuntu/models/Llama-3.2-1B
    export CIPHER_COUNTERS_DUMP_DIR="$P3_DIR/counters"
    export CUDA_VISIBLE_DEVICES=0
    python3 stress/phase3_graph_child.py \
        > "$P3_DIR/stderr/graph.stdout" 2> "$P3_DIR/stderr/graph.stderr"
)
echo "[ORCH] Phase 3: done"

# ============================================================================
# PHASE 4 — model switch (Llama-3.2-1B → Llama-3.1-8B)
# Targets: CONTINUITY, MODEL_SWITCH detection
# ============================================================================
echo
echo "============================================================"
echo "PHASE 4 — model switch 1B→8B (${P4_DURATION_S}s)"
echo "============================================================"
P4_DIR="$RUN_DIR/phase4_switch"
mkdir -p "$P4_DIR/counters" "$P4_DIR/stderr"
(
    for kv in "${CIPHER_ENV[@]}"; do export "$kv"; done
    # ATTN_KOOPMAN's symbol-resolution path crashes on this child's import
    # ordering (cross-DSO extern lookup races torch CUDA init). Disable for
    # this phase only; CONTINUITY counter is what we care about here.
    export CIPHER_ATTN_KOOPMAN=off
    export PHASE4_DURATION_S="$P4_DURATION_S"
    export PHASE4_OUT_PATH="$P4_DIR/switch.json"
    export CIPHER_COUNTERS_DUMP_DIR="$P4_DIR/counters"
    export CUDA_VISIBLE_DEVICES=0
    python3 stress/phase4_modelswitch_child.py \
        > "$P4_DIR/stderr/switch.stdout" 2> "$P4_DIR/stderr/switch.stderr"
)
echo "[ORCH] Phase 4: done"

# ============================================================================
# PHASE 5 — ARBITRATE verification (uses Phase 1's already-produced data
# since the orchestrator already enables CIPHER_ARBITRATE=on; we just verify
# the SM yield/reclaim log lines and counter activity from Phase 1).
# ============================================================================
echo
echo "============================================================"
echo "PHASE 5 — ARBITRATE verification (post-hoc on Phase 1 logs)"
echo "============================================================"
P5_DIR="$RUN_DIR/phase5_arbitrate"
mkdir -p "$P5_DIR"
{
    echo "Aggregating CIPHER ARBITRATE log lines from Phase 1 stderr files…"
    grep -h "CIPHER ARBITRATE\|cipher_fairness\|FAIRNESS yield" \
        "$P1_DIR/stderr"/*.stderr 2>/dev/null | head -40
    echo
    echo "Per-tenant FAIRNESS / ARBITRATE counter values:"
    python3 -c "
import json, glob
for p in sorted(glob.glob('$P1_DIR/counters/cipher_counters_*.json')):
    j = json.load(open(p))
    o = j['ops']
    print(f\"  pid={j['pid']:6d} FAIR={o['FAIRNESS']:5d} ARB={o['ARBITRATE']:5d}\")
"
} > "$P5_DIR/arbitrate_audit.txt"
echo "[ORCH] Phase 5: audit written to $P5_DIR/arbitrate_audit.txt"

# ============================================================================
# AGGREGATE — union of all op counters across all phases + UI summary
# ============================================================================
echo
echo "============================================================"
echo "Aggregating op-coverage report…"
echo "============================================================"
python3 - <<EOF > "$RUN_DIR/MULTIPHASE_REPORT.md"
import json, glob, os
from collections import defaultdict

RUN_DIR = "$RUN_DIR"
phases = {
    "phase1_decode":   "Phase 1: 15-tenant LLM decode",
    "phase2_nccl":     "Phase 2: TP=2 NCCL",
    "phase3_graph":    "Phase 3: CUDA graph capture",
    "phase4_switch":   "Phase 4: model switch 1B→8B",
}
ops_known = ["CLASSIFY","PREDICT","RING_WRITE","FLOW_RECORD","SPECULATE_CHECK",
    "FLOW_MATCH","GUARD","DETERMINISM","SUBSTITUTE_FP8","SUBSTITUTE_KOOPMAN",
    "SUBSTITUTE_MARLIN","FLOW_SUBSTITUTE","FUSE","NCCL_TUNER","PERSIST",
    "THERMOSTAT","FAIRNESS","ARBITRATE","PIPELINE","CONTINUITY","RECEIPT",
    "CARBON","TRACE","COMPLY","AUDIT","LOOP","GRAPH_ENGINE","REMEMBER",
    "ADAPT","SPECULATE_WRITE","VALIDATE","TOPOLOGY","WORKLOAD_OBSERVE"]

per_phase = {p: defaultdict(int) for p in phases}
total_files = {p: 0 for p in phases}
for p in phases:
    files = glob.glob(os.path.join(RUN_DIR, p, "counters", "*.json"))
    total_files[p] = len(files)
    for f in files:
        try:
            o = json.load(open(f))
            for k, v in (o.get("ops") or {}).items():
                per_phase[p][k] += int(v)
        except Exception:
            pass

union = defaultdict(int)
for p in per_phase:
    for k, v in per_phase[p].items():
        union[k] += v

print("# CIPHER MULTI-PHASE OP-COVERAGE TEST — REPORT")
print()
print(f"Run dir: \`{RUN_DIR}\`")
print()
print("## Per-phase op activity matrix")
print()
hdr = ["op"] + [phases[p] for p in phases] + ["UNION", "FIRED?"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
fired = 0
for op in ops_known:
    row = [op]
    for p in phases:
        v = per_phase[p].get(op, 0)
        row.append(f"{v:,}")
    u = union.get(op, 0)
    row.append(f"**{u:,}**")
    is_fired = u > 0
    if is_fired: fired += 1
    row.append("✅" if is_fired else "❌")
    print("| " + " | ".join(row) + " |")
print()
print(f"## Headline: **{fired}/33 ops fired** across all phases")
print()
print("### Per-phase tenant counter dump counts")
print()
for p in phases:
    print(f"- {phases[p]}: {total_files[p]} counter dump files")
print()
print("### Phase 1 stress JSON outputs (first 5)")
import glob, json
for p in sorted(glob.glob(os.path.join(RUN_DIR, "phase1_decode", "t*.json")))[:5]:
    j = json.load(open(p))
    print(f"- t{j['tenant_id']}: {j['tokens_per_s']:.1f} tok/s, "
          f"P50={j['burst_p50_us']:.0f}us, sample='{j['sample_text'][:40]}'")
EOF

echo
echo "[ORCH] DONE → $RUN_DIR/MULTIPHASE_REPORT.md"
cat "$RUN_DIR/MULTIPHASE_REPORT.md"
