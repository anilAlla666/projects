#!/bin/bash
# CP 5.3 STEP 2 gate C — two concurrent 8-SM green-context partitions.
#
# Launches two cp53_step2_harness processes in "multi" role. A file barrier
# inside the harness aligns their PARTITION-phase dispatches so the two
# green-ctx GEMMs run simultaneously. Precondition (scope memo §6 item 5):
# the two processes must bind DIFFERENT green groups — if they hash-collide
# onto the same group_id the pair is respawned with fresh tenant handles.
#
# Gate C verdict (§7 C): (i) both complete, no deadlock; (ii) both meet
# A-num/B; (iii) the two partitions' %smid sets are DISJOINT.
set -u
HARNESS=./cp53_step2_harness
WD=300            # per-process watchdog (s) — a KU1 deadlock is killed here
MAX_TRIES=6

for try in $(seq 1 $MAX_TRIES); do
    BAR="$(pwd)/cp53_gatec_barrier.$$"
    rm -rf "$BAR"; mkdir -p "$BAR"
    rm -f cp53_step2_multi_A.result cp53_step2_multi_B.result
    # distinct tenant handles per attempt; the green_ctx hash is pid^handle
    THA=$(( (RANDOM<<8) ^ try ^ 0x1111 ))
    THB=$(( (RANDOM<<8) ^ try ^ 0x7777 ))
    echo "=== gate C attempt $try : tenant_handles A=$THA B=$THB barrier=$BAR ==="

    timeout $WD $HARNESS multi $THA cp53_step2_multi_A.result "$BAR" \
        > cp53_step2_multi_A.runlog 2>&1 &
    PA=$!
    timeout $WD $HARNESS multi $THB cp53_step2_multi_B.result "$BAR" \
        > cp53_step2_multi_B.runlog 2>&1 &
    PB=$!
    wait $PA; RA=$?
    wait $PB; RB=$?
    echo "proc A exit=$RA  proc B exit=$RB"

    if [ $RA -eq 124 ] || [ $RB -eq 124 ]; then
        echo "RESULT gate_c=DEADLOCK detail=watchdog_timeout (KU1) A=$RA B=$RB"
        echo "  -> a partition-phase dispatch hung; see runlogs for last line"
        rm -rf "$BAR"; exit 1
    fi
    if [ ! -s cp53_step2_multi_A.result ] || [ ! -s cp53_step2_multi_B.result ]; then
        echo "RESULT gate_c=FAIL detail=missing_result_file A=$RA B=$RB"
        rm -rf "$BAR"; exit 1
    fi

    GA=$(awk '/^GROUP_ID/{print $2}' cp53_step2_multi_A.result)
    GB=$(awk '/^GROUP_ID/{print $2}' cp53_step2_multi_B.result)
    if [ "$GA" = "$GB" ]; then
        echo "  green-group COLLISION (both group $GA) — respawning"
        rm -rf "$BAR"
        continue
    fi

    echo "  groups distinct: A=$GA B=$GB  (precondition satisfied)"
    rm -rf "$BAR"

    # ── verdict ───────────────────────────────────────────────────────────
    AA=$(awk '/^ANUM_ALL/{print $2}' cp53_step2_multi_A.result)
    AB=$(awk '/^ANUM_ALL/{print $2}' cp53_step2_multi_B.result)
    BA=$(awk '/^B_ALL/{print $2}'    cp53_step2_multi_A.result)
    BB=$(awk '/^B_ALL/{print $2}'    cp53_step2_multi_B.result)
    UA=$(awk '/^UNION_SMIDS/{print $2}' cp53_step2_multi_A.result)
    UB=$(awk '/^UNION_SMIDS/{print $2}' cp53_step2_multi_B.result)
    echo "  A: group=$GA anum=$AA b=$BA smids=$UA"
    echo "  B: group=$GB anum=$AB b=$BB smids=$UB"

    OVERLAP=$(python3 -c "
a=set('$UA'.split(',')) if '$UA' else set()
b=set('$UB'.split(',')) if '$UB' else set()
ov=sorted(a&b, key=lambda x:int(x))
print(','.join(ov) if ov else 'NONE')
")
    echo "  smid overlap: $OVERLAP"

    DISJOINT=FAIL; [ "$OVERLAP" = "NONE" ] && DISJOINT=PASS
    NODEAD=PASS;   { [ $RA -ne 0 ] && [ $RA -ne 1 ]; } && NODEAD=FAIL
                   { [ $RB -ne 0 ] && [ $RB -ne 1 ]; } && NODEAD=FAIL
    VERDICT=PASS
    for v in "$AA" "$AB" "$BA" "$BB" "$DISJOINT" "$NODEAD"; do
        [ "$v" = "PASS" ] || VERDICT=FAIL
    done
    echo "RESULT gate_c=$VERDICT anum_A=$AA anum_B=$AB b_A=$BA b_B=$BB \
disjoint=$DISJOINT no_deadlock=$NODEAD overlap=$OVERLAP"
    [ "$VERDICT" = "PASS" ] && exit 0 || exit 1
done

echo "RESULT gate_c=INCONCLUSIVE detail=green_group_collision_x$MAX_TRIES"
exit 2
