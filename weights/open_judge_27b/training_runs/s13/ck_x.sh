set -u
# Step 16 checkpoint on the 4xH100: cutoff, score, data, length fit, train adapter3, serve, check, test.
cd ~/vbrepo2
log(){ echo "[$(date -u +%H:%M) UTC] $*" >> ~/s13/ck.log; }
until [ "$(date -u +%H%M)" -ge 1700 ]; do sleep 30; done          # 18:00 BST cutoff
pkill -f "[x]args -a runs/collect"; pkill -f "[r]un_step13_tw.py"; sleep 5; docker ps -q | xargs -r docker kill > /dev/null
log "collection stopped at the cutoff"
for f in runs/collect/o8_gpu*.log runs/collect/o9_gpu*.log runs/collect/o1[012]_gpu*.log; do
  n=$(basename $f .log); ~/orch/bin/python -u tools/tw_score.py $f > runs/collect/score_$n.txt 2>&1 &
done; wait
log "own rounds kept: $(cat runs/collect/score_o*.txt | grep '^{' | grep -vc '"kept": \[\]') of $(cat runs/collect/score_o*.txt | grep -c '^{')"
until [ -f ~/s13/NU_READY ]; do sleep 30; done
~/orch/bin/python tools/sft_build.py own o8,o9,o10,o11,o12,nu > runs/step13/sft_ck.jsonl 2> runs/step13/sft_ck.count
log "data: $(cat runs/step13/sft_ck.count); held-out names found: $(grep -c -E 'layout-config-recreation2|photonic-waveguide-routing|mp-checkpoint-consolidation|production-planning' runs/step13/sft_ck.jsonl)"
python3 -c "
import json,collections; r=[json.loads(l) for l in open('runs/step13/sft_ck.jsonl')]
print('by tag', dict(collections.Counter(x['file'].split('/')[3].rsplit('__',1)[-1] for x in r)))" >> ~/s13/ck.log
pkill -f "[v]llm serve"; sleep 40; pkill -f "[E]ngineCore"
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]; do sleep 5; done
CUDA_VISIBLE_DEVICES=0 ~/tr/bin/python -u ~/s13/fit.py 65536,98304,131072 1 4096 2>&1 | grep -E "^L=" >> ~/s13/ck.log
L=$(grep -E "^L=[0-9]+ offload=True ok" ~/s13/ck.log | sed -E 's/^L=([0-9]+).*/\1/' | sort -n | tail -1)
[ -n "$L" ] || L=32768
for S in 4 8 16; do
  ~/tr/bin/python tools/train_judge.py --data runs/step13/sft_ck.jsonl --out /tmp/dry --view outputs --max-len $L --seg $S --dry 2>/dev/null | grep '^{' | sed "s/^/seg $S: /" >> ~/s13/ck.log
done
# the smallest seg whose 2 epochs fit in about 2 hours at the measured 4,000 tokens/s
S=$(python3 - <<PY
import json,re
best=None
for line in open("/home/ubuntu/s13/ck.log"):
    m=re.match(r".*seg (\d+): (\{.*\})", line)
    if m:
        s,d=int(m.group(1)),json.loads(m.group(2))
        if best is None and d["tokens"]*2/4000/60 <= 130: best=s
print(best or 16)
PY
)
log "training: max_len $L, seg $S"
~/tr/bin/torchrun --standalone --nproc_per_node 4 tools/train_judge.py --data runs/step13/sft_ck.jsonl --out /home/ubuntu/s13/adapter3 \
  --view outputs --max-len $L --seg $S --no-offload > ~/s13/train3.log 2>&1
grep -q '"DONE"' ~/s13/train3.log || { log "TRAIN_FAILED"; exit 1; }
log "trained: $(grep '"DONE"' ~/s13/train3.log)"
~/tr/bin/python - >> ~/s13/ck.log 2>&1 <<"PY"
import shutil, pathlib
from safetensors.torch import load_file, save_file
s, d = pathlib.Path("/home/ubuntu/s13/adapter3"), pathlib.Path("/home/ubuntu/s13/adapter3_hf")
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
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) LORA="--lora-modules trained3=/home/ubuntu/s13/adapter3_hf" setsid nohup bash ~/s13/serve_lora.sh > ~/s13/serve4_$g.log 2>&1 < /dev/null & done
for g in 0 1; do until grep -qE "Application startup complete|initialization failed" ~/s13/serve4_$g.log; do sleep 10; done; done
grep -q "Application startup complete" ~/s13/serve4_0.log && grep -q "Application startup complete" ~/s13/serve4_1.log || { log SERVE_FAILED; exit 1; }
~/orch/bin/python - >> ~/s13/ck.log 2>&1 <<"PY"
import httpx2 as httpx
def ask(m):
    r = httpx.post("http://127.0.0.1:8000/v1/chat/completions", timeout=600, json={"model": m, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "A grader rejects every candidate with: FAIL: /app/out.json missing. What do you check first? Answer in two sentences."}],
        "chat_template_kwargs": {"reasoning_effort": "medium"}})
    c = r.json()["choices"][0]["message"]; return (c.get("reasoning") or "") + "|" + (c.get("content") or "")
a, a2, b = ask("qwen38-27b"), ask("qwen38-27b"), ask("trained3")
print("BASE_REPEATABLE", a == a2); print("ADAPTER_CHECK", "DIFFERENT" if b not in (a, a2) else "IDENTICAL")
PY
grep -q "BASE_REPEATABLE True" ~/s13/ck.log && grep -q "ADAPTER_CHECK DIFFERENT" ~/s13/ck.log || { log ADAPTER_CHECK_FAILED; exit 1; }
VB_TRIM=outputs VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:8000/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py trained3:c3:production-planning > runs/step16_ck_production.log 2>&1 < /dev/null &
VB_TRIM=outputs VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:8001/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py trained3:c3:photonic-waveguide-routing > runs/step16_ck_photonic.log 2>&1 < /dev/null &
log "CHECKPOINT_TEST_STARTED"
