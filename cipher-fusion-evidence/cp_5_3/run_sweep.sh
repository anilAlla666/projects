#!/bin/bash
# CP 5.3 STEP 1 grid sweep — 4 shapes x grids {132,66,33,16,8}, G=128.
RUNLOG=cp53_splitk_sweep.runlog
: > $RUNLOG
echo "CP 5.3 STEP 1 split-K generalisation sweep — $(date -u +%FT%TZ)" | tee -a $RUNLOG
echo "anchors: kmod e2f50452 libcipher_rt c2c5d313 libcipher_v2 86618c30" | tee -a $RUNLOG
echo "harness md5: $(md5sum cp53_splitk_diag | cut -d' ' -f1)" | tee -a $RUNLOG
echo "=========================================================" | tee -a $RUNLOG
for shp in 0 1 2 3; do
  for grid in 132 66 33 16 8; do
    echo "" | tee -a $RUNLOG
    echo ">>> shape=$shp grid=$grid  $(date -u +%T)" | tee -a $RUNLOG
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
echo "=== sweep complete $(date -u +%FT%TZ) ===" | tee -a $RUNLOG
