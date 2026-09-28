#!/bin/bash
# Marlin GC-on-free — full close-gate sequence on the FINAL binary (b16cfe59).
# Runs GPU-exclusive in order: synthetic → concurrent → churn → regression →
# fp-cost(ON vs OFF) → 30-min soak. Each step logs PASS/FAIL; soak runs last.
set +e
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
export LD_LIBRARY_PATH="$CU13L:$TORCHLIB:$LD_LIBRARY_PATH"
MD5=$(md5sum "$SO" | cut -d' ' -f1)
echo "########## CLOSE GATES on .so md5=$MD5 ##########"

echo "===== [1/6] SYNTHETIC (fp ON) ====="
MARLIN_SO="$SO" CIPHER_MARLIN_GC_FP=1 MARLIN_GC_SYN_OUT=/home/ubuntu/gate_synthetic.json \
  timeout 300 python3 /home/ubuntu/marlin_gc_synthetic.py >/home/ubuntu/gate_synthetic.log 2>&1
echo "synthetic: $(python3 -c "import json;print('PASS' if json.load(open('/home/ubuntu/gate_synthetic.json'))['OVERALL_PASS'] else 'FAIL')" 2>/dev/null || echo ERR)"

echo "===== [2/6] CONCURRENT safe-reclaim ====="
CUDA_INJECTION64_PATH="$SO" CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO="$SO" \
  CONC_SEC=30 CONC_OUT=/home/ubuntu/gate_concurrent.json \
  timeout 220 python3 /home/ubuntu/marlin_gc_concurrent.py >/home/ubuntu/gate_concurrent.log 2>&1
echo "concurrent: $(python3 -c "import json;print('PASS' if json.load(open('/home/ubuntu/gate_concurrent.json'))['PASS'] else 'FAIL')" 2>/dev/null || echo ERR)"

echo "===== [3/6] CHURN (BOUND + UNBOUND) ====="
CUDA_INJECTION64_PATH="$SO" CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO="$SO" \
  MARLIN_GC_CHURN_OUT=/home/ubuntu/gate_churn.json CHURN_CYCLES=12 \
  timeout 1500 python3 /home/ubuntu/marlin_gc_churn.py >/home/ubuntu/gate_churn.log 2>&1
echo "churn: $(grep -aE '^CHURN ' /home/ubuntu/gate_churn.log)"

echo "===== [4/6] REGRESSION (byte-identical OFF + additivity + test_step3) ====="
bash /home/ubuntu/marlin_gc_regression.sh >/home/ubuntu/gate_regression.log 2>&1
grep -aE 'byte_identical_all|n_of_n|PASS|FAIL|OVERALL' /home/ubuntu/gate_regression.log | tail -12

echo "===== [5/6] FP-COST (fingerprint ON vs OFF, single-tenant) ====="
CUDA_INJECTION64_PATH="$SO" CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 FPCOST_OUT=/home/ubuntu/gate_fpcost_on.json \
  timeout 300 python3 /home/ubuntu/marlin_gc_fpcost.py >/home/ubuntu/gate_fpcost_on.log 2>&1
echo "  $(grep -a FPCOST /home/ubuntu/gate_fpcost_on.log)"
CUDA_INJECTION64_PATH="$SO" CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=0 FPCOST_OUT=/home/ubuntu/gate_fpcost_off.json \
  timeout 300 python3 /home/ubuntu/marlin_gc_fpcost.py >/home/ubuntu/gate_fpcost_off.log 2>&1
echo "  $(grep -a FPCOST /home/ubuntu/gate_fpcost_off.log)"
python3 -c "
import json
on=json.load(open('/home/ubuntu/gate_fpcost_on.json')); off=json.load(open('/home/ubuntu/gate_fpcost_off.json'))
ov=(on['median_ms']-off['median_ms'])/off['median_ms']*100
print('  fp overhead (median): %.1f%% (ON %.2fms vs OFF %.2fms)'%(ov,on['median_ms'],off['median_ms']))
" 2>/dev/null

echo "===== [6/6] 30-MIN SOAK ====="
CUDA_INJECTION64_PATH="$SO" CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO="$SO" \
  D9_SOAK_SEC=1800 MARLIN_GC_SOAK_OUT=/home/ubuntu/gate_soak.json \
  timeout 2100 python3 /home/ubuntu/marlin_gc_soak.py >/home/ubuntu/gate_soak.log 2>&1
echo "soak: $(grep -aE '^SOAK ' /home/ubuntu/gate_soak.log)"
echo "########## CLOSE GATES DONE ##########"
