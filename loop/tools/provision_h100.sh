#!/bin/bash
# Step 14: make a fresh GPU host ready to run verify loops with our judge, copying everything from the primary H100.
# Run from the laptop with the key loaded in the agent (ssh-add), so the new host can pull from the primary:
#     ssh -A ubuntu@<new-ip> 'bash -s' < tools/provision_h100.sh
set -u
SRC=ubuntu@209.20.157.193
log() { echo "[$(date +%H:%M:%S)] $*"; }
sudo -n usermod -aG docker ubuntu
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/s13 ~/models ~/vbdata ~/vbrepo2 ~/twdata
ssh -n -o StrictHostKeyChecking=accept-new $SRC 'cat ~/freeze_vl.txt' > ~/freeze_vl.txt
ssh -n $SRC 'cat ~/freeze_orch.txt' > ~/freeze_orch.txt
ssh -n $SRC 'cat ~/freeze_hfv.txt' > ~/freeze_hfv.txt
log "envs"
(uv venv -q -p 3.12 ~/vl && VIRTUAL_ENV=~/vl uv pip install -q -r ~/freeze_vl.txt > ~/s13/vl.log 2>&1; echo VL_DONE $? >> ~/s13/vl.log) &
(uv venv -q -p 3.12 ~/orch && VIRTUAL_ENV=~/orch uv pip install -q -r ~/freeze_orch.txt > ~/s13/orch.log 2>&1) &
(uv venv -q -p 3.12 ~/hfv && VIRTUAL_ENV=~/hfv uv pip install -q -r ~/freeze_hfv.txt >/dev/null 2>&1 \
  && ~/hfv/bin/hf download Qwen/Qwen3.8-27B-FP8 --local-dir ~/models/qwen38-27b-fp8 > ~/s13/dl_fp8.log 2>&1; echo FP8_DONE $? >> ~/s13/dl_fp8.log) &
log "files from the primary"
ssh -n $SRC 'tar -C /home/ubuntu -cf - s13/adapter1_hf s13/serve_lora.sh s13/tw_items_s1.txt vbrepo2/vb vbrepo2/tools vbrepo2/templates vbrepo2/runs/step13/graders vbrepo2/runs/step13/tw_graders vbdata/hud_tasks vbdata/candidates vbdata/tw_graders twdata' \
  | tar -C /home/ubuntu -xf -
ln -sfn ~/vbdata ~/vbrepo2/data
sed -i "s/--max-num-seqs 8/--max-num-seqs 16/" ~/s13/serve_lora.sh  # H100 80GB: room for 16 conversations
log "grading images from the primary"
IMGS=$(ssh -n $SRC "docker images --format '{{.Repository}}' | grep '^vbhud-' | sort -u | tr '\n' ' '")
ssh -n $SRC "docker save $IMGS" | sudo -n docker load > /dev/null
log "images: $(sudo -n docker images --format '{{.Repository}}' | grep -c '^vbhud-')"
wait
grep -q "FP8_DONE 0" ~/s13/dl_fp8.log || { log "MODEL DOWNLOAD FAILED"; exit 1; }
log "serving"
sed -i "s/--port 8000/--port \${PORT:-8000}/" ~/s13/serve_lora.sh
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for g in $(seq 0 $((NG - 1))); do  # one server per GPU, ports 8000.., so loops spread across GPUs
  CUDA_VISIBLE_DEVICES=$g PORT=$((8000 + g)) LORA="--lora-modules trained=/home/ubuntu/s13/adapter1_hf" \
    nohup bash ~/s13/serve_lora.sh > ~/s13/serve_$g.log 2>&1 < /dev/null &
done
for g in $(seq 0 $((NG - 1))); do
  until grep -qE "Application startup complete|initialization failed" ~/s13/serve_$g.log; do sleep 10; done
  echo "gpu $g: $(grep -oE "GPU KV cache size: [0-9,]+ tokens|startup complete|initialization failed" ~/s13/serve_$g.log | tail -2 | tr "\n" " ")"
done
log "PROVISION_DONE"
