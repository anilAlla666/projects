#!/bin/bash
# §4 re-confirm MFU+quality on 145a8942 (md5-stamped) then §3 30-min FP8 soak. Run after §1 PASS.
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L
MD5=$(md5sum $SO | cut -d' ' -f1)
echo "### §4 MFU on $MD5 (sdpa 700W) ###"
sudo nvidia-smi -pl 700 >/dev/null 2>&1
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 D9_ATTN=sdpa timeout 600 python3 /home/ubuntu/d9_fp8_mfu.py 2>/dev/null | grep '\[off\]'; mv -f /home/ubuntu/d9_fp8_mfu_off.json /home/ubuntu/d9_clz_mfu_off.json 2>/dev/null
CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_MARLIN=0 D9_ATTN=sdpa timeout 600 python3 /home/ubuntu/d9_fp8_mfu.py 2>/dev/null | grep '\[fp8\]'; mv -f /home/ubuntu/d9_fp8_mfu_fp8.json /home/ubuntu/d9_clz_mfu_fp8.json 2>/dev/null
python3 -c "import json; o=json.load(open('/home/ubuntu/d9_clz_mfu_off.json')); f=json.load(open('/home/ubuntu/d9_clz_mfu_fp8.json')); json.dump({'so_md5':'$MD5','bf16':o,'fp8':f,'fp8_vs_bf16_wallclock':round(o['fwd_ms']/f['fwd_ms'],3)}, open('/home/ubuntu/d9_clz_mfu_summary.json','w'), indent=2); print('MFU bf16',o['MFU_vs989_pct'],'fp8',f['MFU_vs989_pct'],'speedup',round(o['fwd_ms']/f['fwd_ms'],3))"
echo "### §4 quality on $MD5 (PPL+MMLU+KL all-layers incl lm_head) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 timeout 1200 python3 /home/ubuntu/d9_fp8_quality.py 2>/dev/null | grep -E '\[ref\]|WROTE'
CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_MARLIN=0 timeout 1200 python3 /home/ubuntu/d9_fp8_quality.py 2>/dev/null | grep -E '\[fp8\]|WROTE'
python3 /home/ubuntu/d9_fp8_quality_compare.py 2>&1 | tail -6
python3 -c "import json; d=json.load(open('/home/ubuntu/d9_fp8_quality_result.json')); d['so_md5']='$MD5'; json.dump(d,open('/home/ubuntu/d9_clz_quality_result.json','w'),indent=2); print('quality stamped',d.get('ppl_delta_pct'),d.get('mmlu_delta_abs'))"
echo "### §3 30-min FP8 soak on $MD5 ###"
CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_MARLIN=0 D9_SOAK_SEC=1800 timeout 2200 python3 /home/ubuntu/d9_fp8_close_soak.py 2>/dev/null | grep -E 'soak t=|SOAK DONE'
echo "### CLOSE 4+3 DONE ###"
