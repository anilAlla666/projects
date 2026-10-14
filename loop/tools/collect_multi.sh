#!/bin/bash
# Step 14: collect the model's own rounds on training cases across every GPU of a host. One runner per GPU server
# (ports 8000..), each on a DISJOINT set of tasks (two runners preparing one task collide), each running the given
# sampling passes in turn (a pass = one round per case under its own tag).
#     bash tools/collect_multi.sh ITEMS_FILE WORKERS_PER_GPU TAG [TAG ...]       (on the GPU host, in ~/vbrepo2)
set -u
items=$1; workers=$2; shift 2; tags="$*"
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
mkdir -p runs/collect
python3 - "$items" "$NG" <<'EOF'
import sys, collections
items, ng = open(sys.argv[1]).read().split(), int(sys.argv[2])
by_task = collections.defaultdict(list)
for i in items:
    by_task[i.split(":")[0]].append(i)
groups = [[] for _ in range(ng)]
for k, (t, its) in enumerate(sorted(by_task.items(), key=lambda kv: -len(kv[1]))):
    min(groups, key=len).extend(its)
for g, its in enumerate(groups):
    open(f"runs/collect/items_{g}.txt", "w").write("\n".join(its) + "\n")
    print(f"gpu {g}: {len(its)} cases")
EOF
for g in $(seq 0 $((NG - 1))); do
  (for t in $tags; do
     VB_ROUNDS=1 VB_DATA=/home/ubuntu/vbdata VB_LOCAL_URL=http://127.0.0.1:$((8000 + g))/v1 \
       xargs -a runs/collect/items_$g.txt ~/orch/bin/python -u tools/run_step13_tw.py ${SERVED:-qwen38-27b} $t $workers \
       > runs/collect/${t}_gpu$g.log 2>&1
   done) < /dev/null > /dev/null 2>&1 &
done
echo "launched $NG runners, tags: $tags"
