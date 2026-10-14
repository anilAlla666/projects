set -u
# Step 14 on the 4xH100: after data-parallel training, rename adapter2's keys, serve it on all four GPUs, check it is
# applied, and start three retrained runs per held-out task (one task per GPU, its three runs in one process).
cd ~/s13
until grep -q "TR2_DONE" train2.log 2>/dev/null; do sleep 30; done
grep -q '"DONE"' train2.log || { echo "TRAIN_FAILED $(date +%H:%M)" >> eval_x2.log; exit 1; }
echo "TRAIN_DONE $(date +%H:%M) $(grep '"DONE"' train2.log)" >> eval_x2.log
~/tr/bin/python - >> eval_x2.log 2>&1 <<"PY"
import shutil, pathlib
from safetensors.torch import load_file, save_file
s, d = pathlib.Path("/home/ubuntu/s13/adapter2"), pathlib.Path("/home/ubuntu/s13/adapter2_hf")
d.mkdir(exist_ok=True)
w = load_file(s / "adapter_model.safetensors")
w2 = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v for k, v in w.items()}
assert len(w2) == len(w) and all(".language_model.layers." in k for k in w2)
save_file(w2, d / "adapter_model.safetensors", metadata={"format": "pt"})
for f in ("adapter_config.json", "README.md"):
    if (s / f).exists(): shutil.copy(s / f, d / f)
nz = sum(1 for k, v in w2.items() if "lora_B" in k and v.abs().sum().item() > 0)
print("RENAMED", len(w2), "tensors; lora_B non-zero", nz, "of", sum("lora_B" in k for k in w2))
PY
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -lt 1000 ]; do sleep 5; done
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) LORA="--lora-modules trained2=/home/ubuntu/s13/adapter2_hf" setsid nohup bash ~/s13/serve_lora.sh > ~/s13/serve2_$g.log 2>&1 < /dev/null & done
for g in 0 1 2 3; do until grep -qE "Application startup complete|initialization failed" ~/s13/serve2_$g.log; do sleep 10; done; done
for g in 0 1 2 3; do grep -q "Application startup complete" ~/s13/serve2_$g.log || { echo "SERVE_FAILED gpu $g" >> eval_x2.log; exit 1; }; done
~/orch/bin/python - >> eval_x2.log 2>&1 <<"PY"
import httpx2 as httpx
def ask(m):
    r = httpx.post("http://127.0.0.1:8000/v1/chat/completions", timeout=600, json={"model": m, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "user", "content": "A grader rejects every candidate with: FAIL: /app/out.json missing. What do you check first? Answer in two sentences."}],
        "chat_template_kwargs": {"reasoning_effort": "medium"}})
    j = r.json(); c = j["choices"][0]["message"]; return (c.get("reasoning") or "") + "|" + (c.get("content") or "")
a, a2, b = ask("qwen38-27b"), ask("qwen38-27b"), ask("trained2")
print("BASE_REPEATABLE", a == a2)
print("ADAPTER_CHECK", "DIFFERENT" if b not in (a, a2) else "IDENTICAL", len(a), len(b))
print("BASE:", a[:300].replace(chr(10), " ")); print("TRAINED2:", b[:300].replace(chr(10), " "))
PY
grep -q "BASE_REPEATABLE True" eval_x2.log && grep -q "ADAPTER_CHECK DIFFERENT" eval_x2.log || { echo ADAPTER_CHECK_FAILED >> eval_x2.log; exit 1; }
cd ~/vbrepo2; g=0
for t in layout-config-recreation2 photonic-waveguide-routing mp-checkpoint-consolidation production-planning; do
  VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:$((8000+g))/v1 setsid nohup ~/orch/bin/python -u tools/run_step13_pairs.py \
    trained2:t2a:$t trained2:t2b:$t trained2:t2c:$t > runs/step13_t2_$t.log 2>&1 < /dev/null &
  g=$((g+1))
done
echo "EVAL_STARTED $(date +%H:%M)" >> ~/s13/eval_x2.log
