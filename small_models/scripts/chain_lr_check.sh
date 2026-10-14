#!/bin/bash
# On the instance: wait for the code corpus, build the tokenizer and token streams, then run the four
# learning-rate checks in parallel, one GPU each. Progress lines go to ~/data/chain.log.
set -e
cd ~/verifier_model
log() { echo "$(date -u +%H:%M:%S) $*" >> ~/data/chain.log; }
log start
while [ ! -f ~/data/code/manifest.json ]; do
  pgrep -f "^/home/ubuntu/vmenv/bin/python -m vm.corpus.code" > /dev/null || { log "code build died"; exit 1; }
  sleep 30
done
log "code corpus done: $(cat ~/data/code/manifest.json | head -c 300 | tr '\n' ' ')"
~/vmenv/bin/python scripts/build_tokenizer.py ~/data > ~/data/tok_build.log 2>&1 || { log "tokenizer build failed"; exit 1; }
log "tokens done: $(tail -1 ~/data/tok_build.log)"
for i in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((i-1)) ~/vmenv/bin/python -m torch.distributed.run --nproc_per_node 1 --master_port $((29500+i)) \
    vmt/pretrain.py configs/pretrain_dev_lr$i.toml > ~/data/lr$i.log 2>&1 &
done
wait
log "lr check done"
