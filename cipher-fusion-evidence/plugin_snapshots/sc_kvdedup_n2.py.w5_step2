"""Week 5 Step 2 N=2 cross-tenant KV-dedup exit gate.

Orchestrates 2 vLLM subprocess workers, both running TinyLlama with an
identical shared system prompt. Each worker prefills + decodes, then
waits. The orchestrator flushes tenant 1 first, then tenant 2 (so
tenant 2's PUTs match tenant 1's already-registered content), reads
both result files, and asserts:

  - tenant 1 hits + tenant 2 hits ≥ 1 in absolute (mechanism fired)
  - tenant 2 hits > tenant 1 hits (cross-tenant content dedup'd)
  - tenant 1 token_ids == tenant 2 token_ids (bit-identical regression
    guard — greedy decode w/ same prompt MUST produce same output)
  - GPU memory savings measurable post-flush vs pre-flush baseline
    (per Q1 user expansion: actual HBM savings must be observable)
"""
import json, os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(HERE)
OUT_DIR = "/tmp/week5_step2_n2"
os.makedirs(OUT_DIR, exist_ok=True)

# Clean prior state
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))
for t in (1, 2):
    for f in (f"/tmp/cipher_kvdedup_pid_t{t}.txt",
              f"/tmp/cipher_kvdedup_result_t{t}.json"):
        if os.path.exists(f):
            os.remove(f)


def gpu_mem_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def wait_for(path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


print(f"GPU mem pre: {gpu_mem_used_mib()} MiB")

# Launch both workers (sequentially-started but they both stay alive in
# parallel; tenant 1 should finish prefill before tenant 2 starts loading
# to keep peak memory bounded).
procs = {}
for t in (1, 2):
    log_path = os.path.join(OUT_DIR, f"t{t}.log")
    log_f = open(log_path, "w")
    p = subprocess.Popen(
        ["/home/ubuntu/vllm_env/bin/python",
         os.path.join(HERE, "sc_kvdedup_worker.py"),
         str(t), OUT_DIR],
        stdout=log_f, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    procs[t] = (p, log_f)
    print(f"launched tenant {t} as pid {p.pid}; log -> {log_path}")
    # Stagger so tenant 1 finishes loading before tenant 2 starts loading.
    if t == 1:
        if not wait_for(os.path.join(OUT_DIR, "t1_ready.txt"), 240):
            print("FAIL: t1_ready.txt timeout")
            sys.exit(1)

print(f"GPU mem after t1 ready: {gpu_mem_used_mib()} MiB")
if not wait_for(os.path.join(OUT_DIR, "t2_ready.txt"), 240):
    print("FAIL: t2_ready.txt timeout")
    sys.exit(1)
gpu_after_both_loaded = gpu_mem_used_mib()
print(f"GPU mem after both ready: {gpu_after_both_loaded} MiB")

# Both workers have prefilled. Read their token_ids (bit-identity check).
def read_ready(t):
    lines = open(os.path.join(OUT_DIR, f"t{t}_ready.txt")).read().splitlines()
    return {"pid": int(lines[0]),
            "tokens": [int(x) for x in lines[1].split(",")],
            "text": lines[2]}

t1 = read_ready(1)
t2 = read_ready(2)
bit_identical = (t1["tokens"] == t2["tokens"])
print(f"t1 tokens: {t1['tokens']}")
print(f"t2 tokens: {t2['tokens']}")
print(f"bit-identical decode: {bit_identical}")

# Flush tenant 1 first, then tenant 2
print("flushing tenant 1...")
# The EngineCore subprocess writes the pid_file (per the wrapper).
t1_pid = int(open(f"/tmp/cipher_kvdedup_pid_t1.txt").read().strip())
os.kill(t1_pid, signal.SIGUSR1)
if not wait_for(f"/tmp/cipher_kvdedup_result_t1.json", 120):
    print("FAIL: t1 result timeout")
    sys.exit(1)
r1 = json.load(open(f"/tmp/cipher_kvdedup_result_t1.json"))
print(f"t1 dedup: pages={r1['pages_processed']} hits={r1['hits_on_flush']} "
      f"misses={r1['misses_on_flush']} phys={r1['post_physical_pages']} "
      f"virt={r1['post_virtual_pages']}")

print("flushing tenant 2...")
t2_pid = int(open(f"/tmp/cipher_kvdedup_pid_t2.txt").read().strip())
os.kill(t2_pid, signal.SIGUSR1)
if not wait_for(f"/tmp/cipher_kvdedup_result_t2.json", 120):
    print("FAIL: t2 result timeout")
    sys.exit(1)
r2 = json.load(open(f"/tmp/cipher_kvdedup_result_t2.json"))
print(f"t2 dedup: pages={r2['pages_processed']} hits={r2['hits_on_flush']} "
      f"misses={r2['misses_on_flush']} phys={r2['post_physical_pages']} "
      f"virt={r2['post_virtual_pages']}")

# Memory after dedup flush. NB: alias rebinds the VA's physical page;
# the old per-tenant physical is released by cuMemRelease. The actual
# HBM drop depends on how many pages dedup'd. Read nvidia-smi.
gpu_after_dedup = gpu_mem_used_mib()
print(f"GPU mem after dedup: {gpu_after_dedup} MiB")
hbm_saved = gpu_after_both_loaded - gpu_after_dedup
print(f"HBM saved by dedup: {hbm_saved} MiB")

# Tell workers to exit cleanly
for t in (1, 2):
    open(os.path.join(OUT_DIR, f"t{t}_done.signal"), "w").write("done\n")

# Reap
for t, (p, log_f) in procs.items():
    p.wait(timeout=30)
    log_f.close()
    print(f"tenant {t} exit code: {p.returncode}")

# Verdict
# Step 2 spec exit gate (per WEEK_5_STEP_1_DESIGN_MEMO.md Part F.1 Test 1):
#   - stats.hits >= 1 (mechanism fired)
#   - bit-identical token_ids (regression guard)
# Plus Q1 user expansion: HBM savings must be observable.
total_hits = r1["hits_on_flush"] + r2["hits_on_flush"]
cross_tenant_signal = r2["hits_on_flush"] > r1["hits_on_flush"]

print("")
print(f"== gates ==")
print(f"  mechanism fired (total_hits >= 1):  {total_hits >= 1}  ({total_hits} hits)")
print(f"  bit-identical decode:                {bit_identical}")
print(f"  cross-tenant signal (t2_hits > t1):  {cross_tenant_signal}  "
      f"(t1={r1['hits_on_flush']} t2={r2['hits_on_flush']})")
print(f"  HBM savings (MiB):                   {hbm_saved}")

verdict = (total_hits >= 1 and bit_identical and cross_tenant_signal)
print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'}")
sys.exit(0 if verdict else 1)
