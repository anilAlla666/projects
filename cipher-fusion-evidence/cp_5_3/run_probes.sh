#!/bin/bash
# CP 5.3 STEP 1 falsification probes — push below the committed sweep floor.
# grid in {1,2,4}: grid=1 = no split-K (one CTA does all); grid 2,4 force
# multi-slice handoff per block (S2 large-K, §3.1) and grid <= work-tile
# count so NO idle blocks (S4, the other end of §3.3).
RUNLOG=cp53_splitk_probes.runlog
: > $RUNLOG
echo "CP 5.3 STEP 1 falsification probes — $(date -u +%FT%TZ)" | tee -a $RUNLOG
echo "harness md5: $(md5sum cp53_splitk_diag | cut -d' ' -f1)" | tee -a $RUNLOG
echo "=========================================================" | tee -a $RUNLOG
for shp in 1 3; do
  for grid in 4 2 1; do
    echo "" | tee -a $RUNLOG
    echo ">>> PROBE shape=$shp grid=$grid  $(date -u +%T)" | tee -a $RUNLOG
    timeout 90 ./cp53_splitk_diag $shp $grid >> $RUNLOG 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then
      echo "VERDICT-CELL shape=$shp grid=$grid DEADLOCK (timeout 90s, killed)" | tee -a $RUNLOG
    elif [ $rc -ne 0 ]; then
      echo "VERDICT-CELL shape=$shp grid=$grid ERROR rc=$rc" | tee -a $RUNLOG
    else
      echo "VERDICT-CELL shape=$shp grid=$grid OK" | tee -a $RUNLOG
    fi
  done
done
echo "" | tee -a $RUNLOG
echo "=== probes complete $(date -u +%FT%TZ) ===" | tee -a $RUNLOG
