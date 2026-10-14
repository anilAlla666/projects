set -u
# Step 19 student on the 4xH100: train on the teacher rounds, check, then the held-out gate (mp, production, photonic).
cd ~/vbrepo2
log(){ echo "[$(date -u +%H:%M) UTC] $*" >> ~/s13/ck19.log; }
~/orch/bin/python tools/teacher_build.py runs/step19_oa2.log runs/step19_ob.log runs/step19_smoke.log > runs/step13/teacher.jsonl 2> runs/step13/teacher.count
log "data: $(cat runs/step13/teacher.count); held-out names: $(grep -c -E 'layout-config-recreation2|photonic-waveguide-routing|mp-checkpoint-consolidation|production-planning' runs/step13/teacher.jsonl)"
~/tr/bin/python tools/train_judge.py --data runs/step13/teacher.jsonl --out /tmp/dry --view outputs --max-len 98304 --seg 8 --dry 2>/dev/null | grep '^{' >> ~/s13/ck19.log
~/tr/bin/torchrun --standalone --nproc_per_node 4 tools/train_judge.py --data runs/step13/teacher.jsonl --out /home/ubuntu/s13/adapter6 \
  --view outputs --max-len 98304 --seg 8 --no-offload --epochs 3 --lr 1e-4 > ~/s13/train6.log 2>&1
grep -q '"DONE"' ~/s13/train6.log || { log "TRAIN_FAILED"; exit 1; }
log "trained: $(grep '"DONE"' ~/s13/train6.log)"
~/tr/bin/python - >> ~/s13/ck19.log 2>&1 <<"PY"
import shutil, pathlib
from safetensors.torch import load_file, save_file
s, d = pathlib.Path("/home/ubuntu/s13/adapter6"), pathlib.Path("/home/ubuntu/s13/adapter6_hf")
d.mkdir(exist_ok=True)
w = load_file(s / "adapter_model.safetensors")
w2 = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v for k, v in w.items()}
assert len(w2) == len(w) and all(".language_model.layers." in k for k in w2)
save_file(w2, d / "adapter_model.safetensors", metadata={"format": "pt"})
shutil.copy(s / "adapter_config.json", d / "adapter_config.json")
print("RENAMED", len(w2))
PY
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]; do sleep 5; done
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) LORA="--lora-modules trained6=/home/ubuntu/s13/adapter6_hf" setsid nohup bash ~/s13/serve_lora.sh > ~/s13/serve7_$g.log 2>&1 < /dev/null & done
for g in 0 1 2; do until grep -qE "Application startup complete|initialization failed" ~/s13/serve7_$g.log; do sleep 10; done; grep -q "Application startup complete" ~/s13/serve7_$g.log || { log "SERVE_FAILED $g"; exit 1; }; done
~/orch/bin/python - >> ~/s13/ck19.log 2>&1 <<"PY"
import httpx2 as httpx
def ask(m):
    r = httpx.post("http://127.0.0.1:8000/v1/chat/completions", timeout=600, json={"model": m, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "A grader rejects every candidate with: FAIL: /app/out.json missing. What do you check first? Answer in two sentences."}],
        "chat_template_kwargs": {"reasoning_effort": "medium"}})
    c = r.json()["choices"][0]["message"]; return (c.get("reasoning") or "") + "|" + (c.get("content") or "")
a, a2, b = ask("qwen38-27b"), ask("qwen38-27b"), ask("trained6")
print("BASE_REPEATABLE", a == a2); print("ADAPTER_CHECK", "DIFFERENT" if b not in (a, a2) else "IDENTICAL")
PY
grep -q "BASE_REPEATABLE True" ~/s13/ck19.log && grep -q "ADAPTER_CHECK DIFFERENT" ~/s13/ck19.log || { log ADAPTER_CHECK_FAILED; exit 1; }
g=0
for t in mp-checkpoint-consolidation production-planning photonic-waveguide-routing; do
  VB_TRIM=outputs VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:$((8000+g))/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py trained6:s19:$t > runs/step19_gate_$t.log 2>&1 < /dev/null &
  g=$((g+1))
done
log "GATE_STARTED"
