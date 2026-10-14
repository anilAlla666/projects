set -u
cd ~/vbrepo2
log(){ echo "[$(date -u +%H:%M) UTC] $*" >> ~/s13/rl.log; }
log "training adapter4"
~/tr/bin/torchrun --standalone --nproc_per_node 4 tools/train_judge.py --data runs/step13/rl1.jsonl --out /home/ubuntu/s13/adapter4 \
  --view outputs --max-len 98304 --seg 8 --no-offload --epochs 1 --lr 2e-5 > ~/s13/train4.log 2>&1
grep -q '"DONE"' ~/s13/train4.log || { log "TRAIN_FAILED"; exit 1; }
log "trained: $(grep '"DONE"' ~/s13/train4.log)"
~/tr/bin/python - >> ~/s13/rl.log 2>&1 <<"PY"
import shutil, pathlib
from safetensors.torch import load_file, save_file
s, d = pathlib.Path("/home/ubuntu/s13/adapter4"), pathlib.Path("/home/ubuntu/s13/adapter4_hf")
d.mkdir(exist_ok=True)
w = load_file(s / "adapter_model.safetensors")
w2 = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v for k, v in w.items()}
assert len(w2) == len(w) and all(".language_model.layers." in k for k in w2)
save_file(w2, d / "adapter_model.safetensors", metadata={"format": "pt"})
for f in ("adapter_config.json", "README.md"):
    if (s / f).exists(): shutil.copy(s / f, d / f)
print("RENAMED", len(w2))
PY
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]; do sleep 5; done
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) LORA="--lora-modules trained4=/home/ubuntu/s13/adapter4_hf" setsid nohup bash ~/s13/serve_lora.sh > ~/s13/serve5_$g.log 2>&1 < /dev/null & done
for g in 0 1; do until grep -qE "Application startup complete|initialization failed" ~/s13/serve5_$g.log; do sleep 10; done; done
grep -q "Application startup complete" ~/s13/serve5_0.log && grep -q "Application startup complete" ~/s13/serve5_1.log || { log SERVE_FAILED; exit 1; }
~/tr/bin/python /home/ubuntu/s13/rl_check.py >> ~/s13/rl.log 2>&1
grep -q "RL_CHECK PASS" ~/s13/rl.log || { log "RL_CHECK_FAILED"; exit 1; }
VB_TRIM=outputs VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:8000/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py trained4:r4:production-planning > runs/step17_ck_production.log 2>&1 < /dev/null &
VB_TRIM=outputs VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:8001/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py trained4:r4:photonic-waveguide-routing > runs/step17_ck_photonic.log 2>&1 < /dev/null &
log "CHECKPOINT_TEST_STARTED"
