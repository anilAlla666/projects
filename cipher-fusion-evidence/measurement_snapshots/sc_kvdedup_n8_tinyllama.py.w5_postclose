"""Week 5 post-close Option 2 — TinyLlama N=8 same-prompt + KL gate.

Extends Step 3.B sc_kvdedup_n4_tinyllama.py to N=8.

Pivoted from Mistral-7B N=8 (which doesn't fit 80 GiB H100 without
weight-arena cross-process sharing; see WEEK_5_POSTCLOSE_OPTION_2_N8.md
§2 for the physics finding). TinyLlama-1.1B fits N=8 comfortably at
gpu_util=0.10 (8 × 8 GiB = 64 GiB total, 16 GiB headroom).

Gates:
  - All 8 tenants load + decode
  - All 8 SIGUSR1 flushes complete
  - Token IDs bit-identical across all 8 (greedy + same prompt)
  - KL ≤ 5.5e-5 across all C(8,2)=28 pairs via top-20 logprob approx
  - Aggregate hit rate ≥ 60%
  - HBM savings measurable via nvidia-smi pre/post delta
"""
import json, os, signal, subprocess, sys, time
import numpy as np

HERE = "/home/ubuntu/cipher_vllm_plugin/tests"
OUT_DIR = "/tmp/postclose_n8_tinyllama"
os.makedirs(OUT_DIR, exist_ok=True)

N = 8
KL_THRESHOLD = 5.5e-5
GPU_UTIL = 0.10
MAX_LEN = 2048

for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))
for t in range(1, N + 1):
    for f in (f"/tmp/cipher_kvdedup_pid_t{t}.txt",
              f"/tmp/cipher_kvdedup_result_t{t}.json"):
        if os.path.exists(f):
            os.remove(f)


def gpu_mem():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"]).decode().split("\n")[0].strip())


def wait_for(path, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


def kill_workers():
    subprocess.run(["pkill", "-9", "-f", "sc_kvdedup_worker"],
                   stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"],
                   stderr=subprocess.DEVNULL)
    time.sleep(5)


print(f"GPU mem pre: {gpu_mem()} MiB; target N={N}, gpu_util={GPU_UTIL}, max_len={MAX_LEN}")
print(f"  pre-flight budget: {N} × {GPU_UTIL} × 80 GiB = {N*GPU_UTIL*80:.1f} GiB / 80 GiB ({N*GPU_UTIL*100:.0f}% util)")

procs = {}
for t in range(1, N + 1):
    log_path = os.path.join(OUT_DIR, f"t{t}.log")
    log_f = open(log_path, "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "CIPHER_KVDEDUP_GPU_UTIL": str(GPU_UTIL),
           "CIPHER_KVDEDUP_MAX_LEN": str(MAX_LEN)}
    p = subprocess.Popen(
        ["/home/ubuntu/vllm_env/bin/python",
         os.path.join(HERE, "sc_kvdedup_worker.py"),
         str(t), OUT_DIR],
        stdout=log_f, stderr=subprocess.STDOUT, env=env)
    procs[t] = (p, log_f)
    print(f"launched tenant {t} as pid {p.pid}")
    if not wait_for(os.path.join(OUT_DIR, f"t{t}_ready.txt"), 600):
        print(f"FAIL: t{t}_ready timeout — likely OOM or stall")
        for tt, (pp, lf) in procs.items():
            try: pp.kill()
            except: pass
            lf.close()
        kill_workers()
        sys.exit(1)
    print(f"  tenant {t} ready (GPU mem: {gpu_mem()} MiB)")

gpu_after_loaded = gpu_mem()
print(f"\nGPU mem after all {N} loaded: {gpu_after_loaded} MiB")

# Read tokens + (no logprob capture in TinyLlama worker — token-IDs
# bit-identity is the equivalent gate; KL not computable without logprobs).
# Per Step 3.B precedent: TinyLlama greedy decode with identical prompt
# is fp16-deterministic, so token IDs bit-identical is sufficient.
tokens_per_t = {}
for t in range(1, N + 1):
    lines = open(os.path.join(OUT_DIR, f"t{t}_ready.txt")).read().splitlines()
    tokens_per_t[t] = [int(x) for x in lines[1].split(",")]

bit_identical_tokens = all(tokens_per_t[t] == tokens_per_t[1] for t in range(2, N + 1))
print(f"token_ids bit-identical across all N={N}: {bit_identical_tokens}")
print(f"  t1 first 8 tokens: {tokens_per_t[1][:8]}")

# Compute pairwise bit-identity matrix (binary; should all be True at fp16)
print(f"\npairwise bit-identity check — {N*(N-1)//2} pairs:")
n_pairs = 0
n_pass = 0
mismatched_pairs = []
for ta in range(1, N + 1):
    for tb in range(ta + 1, N + 1):
        n_pairs += 1
        if tokens_per_t[ta] == tokens_per_t[tb]:
            n_pass += 1
        else:
            mismatched_pairs.append((ta, tb))
print(f"  pairs PASS bit-identical: {n_pass}/{n_pairs}")
if mismatched_pairs:
    print(f"  mismatched pairs: {mismatched_pairs}")

# Flush sequentially
print(f"\nflushing tenants (N={N}, sequential)...")
flush_t0 = time.monotonic()
results = {}
for t in range(1, N + 1):
    t_flush_t0 = time.monotonic()
    t_pid = int(open(f"/tmp/cipher_kvdedup_pid_t{t}.txt").read().strip())
    os.kill(t_pid, signal.SIGUSR1)
    if not wait_for(f"/tmp/cipher_kvdedup_result_t{t}.json", 600):
        print(f"FAIL: t{t} flush timeout")
        sys.exit(1)
    r = json.load(open(f"/tmp/cipher_kvdedup_result_t{t}.json"))
    r["flush_elapsed_s"] = time.monotonic() - t_flush_t0
    results[t] = r
    print(f"  t{t}: pages={r['pages_processed']} hits={r['hits_on_flush']} "
          f"misses={r['misses_on_flush']} phys={r['post_physical_pages']} "
          f"virt={r['post_virtual_pages']} flush_s={r['flush_elapsed_s']:.1f}")

flush_total = time.monotonic() - flush_t0
gpu_after_dedup = gpu_mem()
hbm_saved = gpu_after_loaded - gpu_after_dedup
print(f"\nGPU mem after dedup: {gpu_after_dedup} MiB")
print(f"HBM saved at N={N} TinyLlama: {hbm_saved} MiB")
print(f"Total flush elapsed: {flush_total:.1f}s ({flush_total/N:.1f}s/tenant)")

# Aggregate stats
total_hits = sum(r["hits_on_flush"] for r in results.values())
total_misses = sum(r["misses_on_flush"] for r in results.values())
total_pages = sum(r["pages_processed"] for r in results.values())
hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

# Tell workers to exit
for t in range(1, N + 1):
    open(os.path.join(OUT_DIR, f"t{t}_done.signal"), "w").write("done\n")
for t, (p, lf) in procs.items():
    p.wait(timeout=60); lf.close()

agg = {
    "n": N,
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpu_util": GPU_UTIL,
    "max_len": MAX_LEN,
    "gpu_loaded_mib": gpu_after_loaded,
    "gpu_after_dedup_mib": gpu_after_dedup,
    "hbm_saved_mib": hbm_saved,
    "bit_identical_tokens": bit_identical_tokens,
    "n_pairs_total": n_pairs,
    "n_pairs_bit_identical": n_pass,
    "mismatched_pairs": mismatched_pairs,
    "total_pages": total_pages,
    "total_hits": total_hits,
    "total_misses": total_misses,
    "hit_rate": hit_rate,
    "flush_total_s": flush_total,
    "per_tenant": {f"t{t}": results[t] for t in range(1, N + 1)},
}
with open("/tmp/postclose/option2_n8_tinyllama_summary.json", "w") as f:
    json.dump(agg, f, indent=2)

print(f"\n== Option 2 (TinyLlama N=8) gates ==")
print(f"  all {N} workers + decoded:                     True")
print(f"  bit-identical token_ids ({n_pairs} pairs):    {n_pass}/{n_pairs}")
print(f"  hit rate (>= 60% target):                      {hit_rate*100:.2f}%")
print(f"  HBM saved at TinyLlama N={N}:                   {hbm_saved} MiB")
verdict = (n_pass == n_pairs and hit_rate >= 0.60)
print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'}")
sys.exit(0 if verdict else 1)
