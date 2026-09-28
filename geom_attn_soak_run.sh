#!/bin/bash
set +e
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
export LD_LIBRARY_PATH="$CU13L:$LD_LIBRARY_PATH"
echo "===== [1] OVERHEAD: GEOM_ATTN ON ====="
CUDA_INJECTION64_PATH=$SO CIPHER_GEOM_ATTN=1 CIPHER_MARLIN=on MARLIN_SO=$SO OVH_OUT=/home/ubuntu/geom_ovh_on.json \
  timeout 300 python3 /home/ubuntu/geom_attn_overhead.py 2>/dev/null
echo "===== [2] OVERHEAD: GEOM_ATTN OFF ====="
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on MARLIN_SO=$SO OVH_OUT=/home/ubuntu/geom_ovh_off.json \
  timeout 300 python3 /home/ubuntu/geom_attn_overhead.py 2>/dev/null
python3 -c "
import json
on=json.load(open('/home/ubuntu/geom_ovh_on.json')); off=json.load(open('/home/ubuntu/geom_ovh_off.json'))
ov=(on['median_ms']-off['median_ms'])/off['median_ms']*100
print('OVERHEAD geom_attn ON vs OFF: %.2f%% (ON %.2fms / OFF %.2fms)'%(ov,on['median_ms'],off['median_ms']))
"
echo "===== [3] 30-MIN SOAK (GEOM_ATTN ON) ====="
CUDA_INJECTION64_PATH=$SO CIPHER_GEOM_ATTN=1 CIPHER_MARLIN=on MARLIN_SO=$SO D9_SOAK_SEC=1800 \
  timeout 2100 python3 /home/ubuntu/geom_attn_soak.py 2>/home/ubuntu/geom_attn_soak.err
echo "soak: $(grep -aE '^GEOM-SOAK ' /home/ubuntu/geom_attn_soak.err 2>/dev/null; grep -aE '^GEOM-SOAK ' /tmp/geom_soak_stdout 2>/dev/null)"
echo "===== DONE ====="
