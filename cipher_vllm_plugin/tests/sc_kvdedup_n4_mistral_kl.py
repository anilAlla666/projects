"""Week 5 Step 3.C — Mistral-7B N=4 same-prompt + KL gate (LOAD-BEARING).

4 vLLM workers, each running Mistral-7B-v0.1 with identical 4K-token
shared system prompt. Each worker captures top-20 logprobs per decode
step. Orchestrator computes pairwise KL across all 6 pairs (C(4,2)).

Gates:
  - All 4 decoded successfully
  - All 4 SIGUSR1 flushes completed
  - Token IDs identical across all 4 tenants (greedy + same prompt)
  - KL <= 5.5e-5 across all 6 pairs (load-bearing — per scope-lock Part F.1 Test 2)
  - All 4 tenant dedup STATS read"""
import json, os, signal, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/tmp/week5_step3_4_n4_mistral"
os.makedirs(OUT_DIR, exist_ok=True)

N = 4
KL_THRESHOLD = 5.5e-5
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


def wait_for(path, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


def kl_top_k(p_lps, p_ids, q_lps, q_ids):
    """Approximate KL(p||q) using top-K logprob entries from each.

    p_lps, q_lps: numpy arrays of shape (n_steps, K) — log-probabilities
    p_ids, q_ids: numpy arrays of shape (n_steps, K) — token IDs

    For each step, considers the UNION of token IDs in p and q. For
    tokens missing from q's top-K, uses min(q_lps[step]) - 5 as a
    pessimistic lower bound (back-off). Returns scalar mean-KL across
    steps. Per-step KL = sum over union(tok) p(tok) * log(p(tok)/q(tok))."""
    n_steps = p_lps.shape[0]
    total = 0.0
    valid = 0
    for s in range(n_steps):
        p_tok = {int(p_ids[s, k]): float(p_lps[s, k])
                 for k in range(p_lps.shape[1])
                 if p_ids[s, k] >= 0}
        q_tok = {int(q_ids[s, k]): float(q_lps[s, k])
                 for k in range(q_lps.shape[1])
                 if q_ids[s, k] >= 0}
        if not p_tok or not q_tok:
            continue
        q_min = min(q_tok.values()) - 5.0
        # KL(p||q) over union of supports
        kl = 0.0
        for tok, p_lp in p_tok.items():
            q_lp = q_tok.get(tok, q_min)
            p_prob = float(np.exp(p_lp))
            kl += p_prob * (p_lp - q_lp)
        total += kl
        valid += 1
    return total / max(valid, 1)


print(f"GPU mem pre: {gpu_mem_used_mib()} MiB")

procs = {}
for t in range(1, N + 1):
    log_path = os.path.join(OUT_DIR, f"t{t}.log")
    log_f = open(log_path, "w")
    p = subprocess.Popen(
        ["/home/ubuntu/vllm_env/bin/python",
         os.path.join(HERE, "sc_kvdedup_mistral_worker.py"),
         str(t), OUT_DIR],
        stdout=log_f, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    procs[t] = (p, log_f)
    print(f"launched tenant {t} as pid {p.pid}")
    if not wait_for(os.path.join(OUT_DIR, f"t{t}_ready.txt"), 600):
        print(f"FAIL: t{t}_ready.txt timeout (Mistral-7B load + 32-token decode)")
        for tt, (pp, lf) in procs.items():
            try: pp.kill()
            except: pass
            lf.close()
        subprocess.run(["pkill", "-9", "-f", "EngineCore"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "sc_kvdedup_mistral_worker"], stderr=subprocess.DEVNULL)
        time.sleep(3)
        sys.exit(1)
    print(f"  tenant {t} ready (GPU mem: {gpu_mem_used_mib()} MiB)")

gpu_after_all_loaded = gpu_mem_used_mib()
print(f"\nGPU mem after all {N} loaded + decoded: {gpu_after_all_loaded} MiB")

# Load token sequences + logprobs
tokens_per_t = {}
text_per_t = {}
lps_per_t = {}
ids_per_t = {}
for t in range(1, N + 1):
    lines = open(os.path.join(OUT_DIR, f"t{t}_ready.txt")).read().splitlines()
    tokens_per_t[t] = [int(x) for x in lines[1].split(",")]
    text_per_t[t] = lines[2]
    data = np.load(os.path.join(OUT_DIR, f"t{t}_logprobs.npz"))
    lps_per_t[t] = data["topk_logprobs"]
    ids_per_t[t] = data["topk_ids"]

bit_identical_tokens = all(tokens_per_t[t] == tokens_per_t[1] for t in range(2, N + 1))
print(f"\ntoken_ids bit-identical across all N={N}: {bit_identical_tokens}")
print(f"t1 first 8 tokens: {tokens_per_t[1][:8]}")

# Pairwise KL
print("\npairwise KL (top-20 logprob approximation):")
kl_matrix = {}
kl_max = 0.0
kl_all_pass = True
for ta in range(1, N + 1):
    for tb in range(ta + 1, N + 1):
        kl_ab = kl_top_k(lps_per_t[ta], ids_per_t[ta],
                         lps_per_t[tb], ids_per_t[tb])
        kl_ba = kl_top_k(lps_per_t[tb], ids_per_t[tb],
                         lps_per_t[ta], ids_per_t[ta])
        kl = max(abs(kl_ab), abs(kl_ba))  # symmetric upper bound
        kl_matrix[(ta, tb)] = kl
        kl_max = max(kl_max, kl)
        status = "PASS" if kl <= KL_THRESHOLD else "FAIL"
        print(f"  ({ta},{tb}): max-KL = {kl:.6e}  threshold {KL_THRESHOLD:.1e}  {status}")
        if kl > KL_THRESHOLD:
            kl_all_pass = False

# Flush sequentially
print("\nflushing tenants...")
results = {}
for t in range(1, N + 1):
    t_pid = int(open(f"/tmp/cipher_kvdedup_pid_t{t}.txt").read().strip())
    os.kill(t_pid, signal.SIGUSR1)
    # Mistral-7B at gpu_util=0.20 ~ 8192 pages per tenant; HIT path
    # ~50-80ms per page → up to 10 min per tenant flush. Budget 900s.
    if not wait_for(f"/tmp/cipher_kvdedup_result_t{t}.json", 900):
        print(f"FAIL: t{t} flush result timeout (900s budget)")
        sys.exit(1)
    r = json.load(open(f"/tmp/cipher_kvdedup_result_t{t}.json"))
    results[t] = r
    print(f"  t{t}: pages={r['pages_processed']} hits={r['hits_on_flush']} "
          f"misses={r['misses_on_flush']} phys={r['post_physical_pages']} "
          f"virt={r['post_virtual_pages']}")

gpu_after_dedup = gpu_mem_used_mib()
hbm_saved = gpu_after_all_loaded - gpu_after_dedup
print(f"\nGPU mem after dedup: {gpu_after_dedup} MiB")
print(f"HBM saved at N={N} Mistral-7B: {hbm_saved} MiB")

# Aggregate stats for Step 3.D hit-rate analysis
total_hits = sum(r["hits_on_flush"] for r in results.values())
total_misses = sum(r["misses_on_flush"] for r in results.values())
total_pages = sum(r["pages_processed"] for r in results.values())
hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
print(f"\naggregate: pages={total_pages} hits={total_hits} misses={total_misses} "
      f"hit_rate={hit_rate*100:.2f}%")

# Tell workers to exit
for t in range(1, N + 1):
    open(os.path.join(OUT_DIR, f"t{t}_done.signal"), "w").write("done\n")
for t, (p, lf) in procs.items():
    p.wait(timeout=60)
    lf.close()

# Save aggregate result
agg = {
    "n": N,
    "bit_identical_tokens": bit_identical_tokens,
    "kl_max": kl_max,
    "kl_threshold": KL_THRESHOLD,
    "kl_all_pass": kl_all_pass,
    "kl_matrix": {f"({a},{b})": v for (a, b), v in kl_matrix.items()},
    "hbm_saved_mib": hbm_saved,
    "total_hits": total_hits,
    "total_misses": total_misses,
    "total_pages": total_pages,
    "hit_rate": hit_rate,
    "per_tenant": {f"t{t}": results[t] for t in range(1, N + 1)},
}
with open("/tmp/week5_step3_4/n4_mistral_kl_summary.json", "w") as f:
    json.dump(agg, f, indent=2)

print("\n== Step 3.C gates ==")
print(f"  all 4 workers + decoded:             True")
print(f"  bit-identical token_ids:             {bit_identical_tokens}")
print(f"  KL <= {KL_THRESHOLD:.1e} all 6 pairs: {kl_all_pass} (max={kl_max:.6e})")
print(f"  HBM saved at Mistral N=4:            {hbm_saved} MiB")
print(f"  aggregate hit rate:                  {hit_rate*100:.2f}%")

verdict = (bit_identical_tokens and kl_all_pass)
print(f"\nSTEP 3.C VERDICT: {'PASS' if verdict else 'FAIL'}")
sys.exit(0 if verdict else 1)
