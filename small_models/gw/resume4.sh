#!/usr/bin/env bash
# After the runner learned to add missing `diff --git` headers: re-run every exam grader run that failed to apply
# (all three writers), score the exam, then continue the reward-filtered round from its first container stage.
set -u
L=~/gw/rft4.log; M=Qwen/Qwen2.5-Coder-7B-Instruct; R=~/gw/rft4
export PATH=$HOME/vllmenv/bin:$PATH
cd ~/verifier_model
serve() {
  setsid nohup ~/vllmenv/bin/vllm serve $M --enable-lora --lora-modules gw=$1 --max-lora-rank 64 --tensor-parallel-size 4 \
    --max-model-len 24576 --port 8000 --gpu-memory-utilization 0.85 > ~/gw/vllm_rft4.log 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
}
redo() {  # drop the runs whose grader did not apply, so run_graders runs them again
  for f in $(grep -l grader_patch_failed $1/*/*/out/result.json 2>/dev/null); do sudo -n rm -rf "$(dirname "$(dirname "$f")")"; done
}
echo "$(date -u +%T) exam re-run" >> $L
for m in base gw sonnet; do
  redo ~/gw/runs4/$m
  ~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen4/$m ~/gw/runs4/$m --workers 32 >> $L 2>&1
done
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs4/base ~/gw/runs4/gw ~/gw/runs4/sonnet > ~/gw/scores4.json 2>> $L
echo EXAM4_SCORED >> $L
redo $R/runs
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
