#!/bin/bash
# Make tasks for newly qualified libraries every minute while qualification runs, then once more at the end.
cd ~/verifier_model
while ps -eo args | grep -q "^/home/ubuntu/vmenv/bin/python -c"; do
  VM_WORKERS=32 ~/vmenv/bin/python scripts/build_judge_data.py tasks ~/judge/libs >> ~/judge/tasks.log 2>&1
  sleep 60
done
VM_WORKERS=80 ~/vmenv/bin/python scripts/build_judge_data.py tasks ~/judge/libs >> ~/judge/tasks.log 2>&1
echo "TASKS_FINAL_DONE" >> ~/judge/tasks.log
