"""E6 — FAISS GPU recall test.

100K vectors d=128, 10K queries, k=10. Compare GPU vs exact CPU recall.
"""
import os, sys, json, time, ctypes, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"


def main():
    if USE_CIPHER:
        rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()
    import faiss

    n_db, n_q, d, k = 100_000, 10_000, 128, 10
    rng = np.random.default_rng(42)
    db = rng.standard_normal((n_db, d)).astype(np.float32)
    qs = rng.standard_normal((n_q, d)).astype(np.float32)

    # Reference: CPU exact (Flat) — small enough to be feasible.
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(db)
    cpu_D, cpu_I = cpu_index.search(qs, k)

    # GPU (Flat).
    res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatL2(res, d)
    t0 = time.perf_counter()
    gpu_index.add(db)
    add_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    gpu_D, gpu_I = gpu_index.search(qs, k)
    search_s = time.perf_counter() - t0

    # Recall@k = fraction of CPU top-k contained in GPU top-k.
    # For exact L2 vs exact L2 GPU should be 100%. Slight tie-breaks possible.
    matches = 0
    for i in range(n_q):
        s = set(int(x) for x in cpu_I[i])
        t = set(int(x) for x in gpu_I[i])
        matches += len(s & t)
    recall = matches / (n_q * k)

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, n_db=n_db, n_q=n_q, d=d, k=k,
                   add_s=add_s, search_s=search_s,
                   queries_per_s=n_q/search_s, recall=recall)
    out_path = os.path.join(os.path.dirname(__file__), f"e6_faiss{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[E6] add={add_s:.2f}s search={search_s:.2f}s ({n_q/search_s:.0f} q/s) "
          f"recall@{k}={recall*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
