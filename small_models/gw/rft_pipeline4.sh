#!/usr/bin/env bash
# Reward-filtered round on the standalone-file writer (run4): 4 drafts per training task, run first on the unfixed
# code only; the drafts that apply and fail there are run on every labelled fix of their task; keep, per task, a draft
# that passes every correct fix and catches a wrong one; continue training run4 on those; same exam as gen_all4.sh.
set -u
L=~/gw/rft4.log; M=Qwen/Qwen2.5-Coder-7B-Instruct; R=~/gw/rft4
until grep -q generated ~/gw/gen_all4.log 2>/dev/null; do sleep 15; done  # the GPUs are free once the exam is written
export PATH=$HOME/vllmenv/bin:$PATH
cd ~/verifier_model
serve() {
  setsid nohup ~/vllmenv/bin/vllm serve $M --enable-lora --lora-modules gw=$1 --max-lora-rank 64 --tensor-parallel-size 4 \
    --max-model-len 24576 --port 8000 --gpu-memory-utilization 0.85 > ~/gw/vllm_rft4.log 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
}
mkdir -p $R
cp ~/gw/rft/instances.json $R/
mkdir -p $R/base_only
~/vmenv/bin/python - > $L <<'EOF'
import json, sys
sys.path.insert(0, "gw")
from standalone import INSTRUCTION, INSTRUCTION_OLD
R = "/home/ubuntu/gw/rft4"
rows = [json.loads(l) for l in open("/home/ubuntu/gw/rft/prompts.jsonl")]
rows = [{**r, "prompt": r["prompt"].replace(INSTRUCTION_OLD, INSTRUCTION)} for r in rows]
open(f"{R}/prompts.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
ins = json.load(open(f"{R}/instances.json"))
json.dump([dict(i, attempts=[]) for i in ins], open(f"{R}/base_only/instances.json", "w"))
print(sum(INSTRUCTION in r["prompt"] for r in rows), "of", len(rows), "prompts carry the new instruction")
EOF
echo "$(date -u +%T) drafting" >> $L
serve $HOME/gw/run4/adapter
~/vmenv/bin/python gw/generate.py $R/prompts.jsonl gw $R/drafts --n 4 >> $L 2>&1
pkill -f "vllm serv[e]"; sleep 20
until grep -q GEN_ALL4_DONE ~/gw/gen_all4.log 2>/dev/null; do sleep 15; done  # the containers: after the exam's
echo "$(date -u +%T) drafts on the unfixed code" >> $L
~/vmenv/bin/python gw/run_graders.py $R/base_only $R/drafts $R/runs --workers 40 --all >> $L 2>&1
mkdir -p $R/live && ~/vmenv/bin/python - >> $L <<'EOF'
import json, pathlib, shutil
R = pathlib.Path("/home/ubuntu/gw/rft4")
n = 0
for d in (R / "runs").iterdir():
    r = d / "base" / "out" / "result.json"
    if r.is_file() and any(v != "passed" for v in json.loads(r.read_text()).get("outcomes", {}).values()):
        shutil.copy(R / "drafts" / f"{d.name}.patch", R / "live" / f"{d.name}.patch"); n += 1
print(json.dumps({"drafts_failing_on_unfixed": n}))
EOF
echo "$(date -u +%T) live drafts on every fix" >> $L
~/vmenv/bin/python gw/run_graders.py $R $R/live $R/runs --workers 40 --all >> $L 2>&1
~/vmenv/bin/python gw/rft_score.py $R $R/runs $R/drafts $R/selected.jsonl >> $L 2>&1
mkdir -p $R/data && cp $R/selected.jsonl $R/data/train.jsonl
echo "$(date -u +%T) training" >> $L
~/vllmenv/bin/python -m torch.distributed.run --nproc_per_node 4 gw/train.py $R/data ~/gw/run5 --max-len 12288 --epochs 2 --lr 1e-4 \
  --init-adapter ~/gw/run4/adapter >> ~/gw/train5.log 2>&1
[ -d ~/gw/run5/adapter ] || { echo "no adapter" >> $L; exit 1; }
echo "$(date -u +%T) exam" >> $L
serve $HOME/gw/run5/adapter
~/vmenv/bin/python gw/generate.py ~/gw/data4/eval_prompts_labelled.jsonl gw ~/gw/gen5/gw >> $L 2>&1
pkill -f "vllm serv[e]"
~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen5/gw ~/gw/runs5/gw --workers 32 >> $L 2>&1
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs4/base ~/gw/runs4/gw ~/gw/runs5/gw ~/gw/runs4/sonnet > ~/gw/scores5.json 2>> $L
echo RFT4_DONE >> $L
