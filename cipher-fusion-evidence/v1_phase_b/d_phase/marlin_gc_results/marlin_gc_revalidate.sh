#!/bin/bash
# Targeted re-validation on the leak-proof-borrow binary (944f5706).
# Output/mem-neutral gates (churn-KL 12/12, 30-min soak, synthetic, additivity)
# transfer from b16cfe59 — the only behavioral change is the bf16 error-path
# borrow release. Re-run: HARD STOP + changed/new paths.
set +e
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
export LD_LIBRARY_PATH="$CU13L:$TORCHLIB:$LD_LIBRARY_PATH"
echo "########## RE-VALIDATION on .so md5=$(md5sum $SO|cut -d' ' -f1) ##########"

echo "===== [A] borrow balance — NORMAL (no fault) ====="
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO=$SO \
  BORROW_OUT=/home/ubuntu/reval_borrow_normal.json timeout 300 python3 /home/ubuntu/marlin_gc_borrow_test.py >/home/ubuntu/reval_borrow_normal.log 2>&1
echo "  $(grep -a BORROW_TEST /home/ubuntu/reval_borrow_normal.log)"

echo "===== [B] borrow balance — FAULT-INJECTED bf16 error path ====="
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 CIPHER_MARLIN_GC_FAULT_BF16=1 MARLIN_SO=$SO \
  BORROW_OUT=/home/ubuntu/reval_borrow_fault.json timeout 300 python3 /home/ubuntu/marlin_gc_borrow_test.py >/home/ubuntu/reval_borrow_fault.log 2>&1
echo "  $(grep -a BORROW_TEST /home/ubuntu/reval_borrow_fault.log)"

echo "===== [C] concurrent safe-reclaim (re-confirm UAF closed) ====="
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO=$SO \
  CONC_SEC=30 CONC_OUT=/home/ubuntu/reval_concurrent.json timeout 220 python3 /home/ubuntu/marlin_gc_concurrent.py >/home/ubuntu/reval_concurrent.log 2>&1
echo "  concurrent: $(python3 -c "import json;d=json.load(open('/home/ubuntu/reval_concurrent.json'));print('PASS' if d['PASS'] else 'FAIL','errors=%d garbage=%d B_iters=%d'%(len(d['errors']),d['garbage_outputs'],d['B_evict_reclaim_iters']))" 2>/dev/null)"

echo "===== [D] byte-identical OFF (HARD STOP) on shipped binary ====="
TAG=vanilla timeout 1200 python3 /home/ubuntu/d9_fp8_close_offreg.py 2>/dev/null | grep -E 'DONE'
CUDA_INJECTION64_PATH=$SO TAG=gcoff2 timeout 1200 python3 /home/ubuntu/d9_fp8_close_offreg.py 2>/dev/null | grep -E 'DONE'
python3 /home/ubuntu/marlin_gc_compare.py vanilla gcoff2 off_byte_identical_944f

echo "===== [E] short BOUND churn (6 cycles) + borrow==0 assert ====="
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on CIPHER_MARLIN_GC_FP=1 MARLIN_SO=$SO \
  MARLIN_GC_CHURN_OUT=/home/ubuntu/reval_churn.json CHURN_CYCLES=6 timeout 900 python3 /home/ubuntu/marlin_gc_churn.py >/home/ubuntu/reval_churn.log 2>&1
echo "  churn: $(grep -aE '^CHURN ' /home/ubuntu/reval_churn.log)"
python3 -c "import ctypes; l=ctypes.CDLL('$SO'); l.cipher_rt_marlin_engine_dispatch_borrow.restype=ctypes.c_int; print('  post-churn dispatch_borrow =', l.cipher_rt_marlin_engine_dispatch_borrow())"
echo "########## RE-VALIDATION DONE ##########"
