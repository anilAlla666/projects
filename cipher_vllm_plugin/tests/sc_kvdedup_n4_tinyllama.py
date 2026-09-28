"""Week 5 Step 3.B — TinyLlama N=4 cross-tenant dedup smoke.

Extends sc_kvdedup_n2.py to 4 tenants (1..4). Same shared system prompt.
Flush in sequence t1 → t2 → t3 → t4 so each subsequent tenant's PUTs
hit tenant 1's already-registered content.

Pre-MISTRAL gate: must PASS before Step 3.C (Mistral-7B + KL gate)."""
import json, os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/tmp/week5_step3_4_n4_tinyllama"
os.makedirs(OUT_DIR, exist_ok=True)

N = 4
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))
for t in range(1, N + 1):
    for f in (f"/tmp/cipher_kvdedup_pid_t{t}.txt",
              f"/tmp/cipher_kvdedup_result_t{t}.json"):
        if os.path.exists(f):
            os.remove(f)


def gpu_mem_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())


def wait_for(path, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


print(f"GPU mem pre: {gpu_mem_used_mib()} MiB")

procs = {}
for t in range(1, N + 1):
    log_path = os.path.join(OUT_DIR, f"t{t}.log")
    log_f = open(log_path, "w")
    p = subprocess.Popen(
        ["/home/ubuntu/vllm_env/bin/python",
         os.path.join(HERE, "sc_kvdedup_worker.py"),
         str(t), OUT_DIR],
        stdout=log_f, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    procs[t] = (p, log_f)
    print(f"launched tenant {t} as pid {p.pid}")
    # Stagger so tenant 1 finishes loading before next loads.
    if not wait_for(os.path.join(OUT_DIR, f"t{t}_ready.txt"), 300):
        print(f"FAIL: t{t}_ready.txt timeout — killing all workers")
        for tt, (pp, lf) in procs.items():
            try: pp.kill()
            except: pass
            lf.close()
        # Also kill EngineCore subprocesses (vLLM v1 spawns these)
        subprocess.run(["pkill", "-9", "-f", "EngineCore"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "sc_kvdedup_worker"], stderr=subprocess.DEVNULL)
        time.sleep(3)
        sys.exit(1)
    print(f"  tenant {t} ready (GPU mem: {gpu_mem_used_mib()} MiB)")

gpu_after_all_loaded = gpu_mem_used_mib()
print(f"\nGPU mem after all {N} ready: {gpu_after_all_loaded} MiB")

# Read tokens for bit-identity check
tokens_per_t = {}
text_per_t = {}
for t in range(1, N + 1):
    lines = open(os.path.join(OUT_DIR, f"t{t}_ready.txt")).read().splitlines()
    tokens_per_t[t] = [int(x) for x in lines[1].split(",")]
    text_per_t[t] = lines[2]

print(f"\ntokens per tenant:")
for t in range(1, N + 1):
    print(f"  t{t}: {tokens_per_t[t]}")

bit_identical = all(tokens_per_t[t] == tokens_per_t[1] for t in range(2, N + 1))
print(f"bit-identical decode across all N={N}: {bit_identical}")

# Flush sequentially
results = {}
for t in range(1, N + 1):
    print(f"\nflushing tenant {t}...")
    t_pid = int(open(f"/tmp/cipher_kvdedup_pid_t{t}.txt").read().strip())
    os.kill(t_pid, signal.SIGUSR1)
    if not wait_for(f"/tmp/cipher_kvdedup_result_t{t}.json", 120):
        print(f"FAIL: t{t} result timeout")
        sys.exit(1)
    r = json.load(open(f"/tmp/cipher_kvdedup_result_t{t}.json"))
    results[t] = r
    print(f"  t{t}: pages={r['pages_processed']} hits={r['hits_on_flush']} "
          f"misses={r['misses_on_flush']} phys={r['post_physical_pages']} "
          f"virt={r['post_virtual_pages']}")

gpu_after_dedup = gpu_mem_used_mib()
hbm_saved = gpu_after_all_loaded - gpu_after_dedup
print(f"\nGPU mem after dedup: {gpu_after_dedup} MiB")
print(f"HBM saved by dedup at N={N}: {hbm_saved} MiB")

# Tell workers to exit cleanly
for t in range(1, N + 1):
    open(os.path.join(OUT_DIR, f"t{t}_done.signal"), "w").write("done\n")
for t, (p, lf) in procs.items():
    p.wait(timeout=30)
    lf.close()

# Gates
total_hits = sum(r["hits_on_flush"] for r in results.values())
cross_tenant_signal_count = 0
for t in range(2, N + 1):
    if results[t]["hits_on_flush"] > results[1]["hits_on_flush"]:
        cross_tenant_signal_count += 1

print(f"\n== Step 3.B gates ==")
print(f"  all 4 workers loaded + decoded:         True")
print(f"  all 4 SIGUSR1 flushes completed:        True")
print(f"  bit-identical decode across all 4:      {bit_identical}")
print(f"  mechanism fired (total_hits >= 3):      {total_hits >= 3}  ({total_hits} hits)")
print(f"  cross-tenant signal (t2/t3/t4 > t1):    "
      f"{cross_tenant_signal_count}/3  "
      f"(t1={results[1]['hits_on_flush']}, t2={results[2]['hits_on_flush']}, "
      f"t3={results[3]['hits_on_flush']}, t4={results[4]['hits_on_flush']})")
print(f"  HBM saved (>= 30 GiB target):           {hbm_saved} MiB")

verdict = (bit_identical and total_hits >= 3 and cross_tenant_signal_count >= 2
           and hbm_saved >= 30 * 1024)
print(f"\nSTEP 3.B VERDICT: {'PASS' if verdict else 'FAIL'}")
sys.exit(0 if verdict else 1)
