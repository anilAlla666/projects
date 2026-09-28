#!/bin/bash
# Phase 1.5.2 orchestrator. For each workload in argv (default A B C D):
#   run "without kmod" arm (kmod off)
#   insmod cipher_kmod.ko
#   run "with kmod" arm
#   rmmod cipher_kmod
# Appends one JSON line per workload-condition tuple to RESULTS.

set -uo pipefail

KMOD=/workspace/cipher_kmod/cipher_kmod.ko
HARNESS=/workspace/cipher_kmod/phase_1_5_2_overhead.py
RESULTS=/workspace/cipher_kmod/phase_1_5_2_results.jsonl

declare -A NSAMP=([A]=10 [B]=5 [C]=3 [D]=5)

WORKLOADS=("$@")
if [ ${#WORKLOADS[@]} -eq 0 ]; then
    WORKLOADS=(A B C D)
fi

if lsmod | grep -q ^cipher_kmod; then
    echo "ERROR: cipher_kmod already loaded; rmmod first" >&2
    exit 1
fi

if [ ! -f "$KMOD" ]; then
    echo "ERROR: $KMOD not found" >&2
    exit 1
fi

for W in "${WORKLOADS[@]}"; do
    N=${NSAMP[$W]:-5}
    echo ""
    echo "================================================================"
    echo "  WORKLOAD $W (N=$N each arm)"
    echo "================================================================"

    # Without kmod first (clean slate)
    if lsmod | grep -q ^cipher_kmod; then
        sudo rmmod cipher_kmod
    fi
    echo "--- without kmod (N=$N) ---"
    OUT=$(python3 "$HARNESS" --workload "$W" --condition without --n "$N" --warmup 1)
    echo "$OUT" >> "$RESULTS"
    echo "$OUT"

    # Insmod and run with-kmod arm. Trust insmod's own exit code rather
    # than polling /proc/modules (which we observed racing once).
    echo "--- insmod ---"
    if ! sudo insmod "$KMOD"; then
        echo "ERROR: sudo insmod failed for $W" >&2
        exit 1
    fi
    echo "--- with kmod (N=$N) ---"
    OUT=$(python3 "$HARNESS" --workload "$W" --condition with --n "$N" --warmup 1)
    echo "$OUT" >> "$RESULTS"
    echo "$OUT"

    echo "--- rmmod ---"
    if ! sudo rmmod cipher_kmod; then
        echo "ERROR: sudo rmmod failed for $W" >&2
        exit 1
    fi
done

echo ""
echo "All workloads complete. Results in $RESULTS"
