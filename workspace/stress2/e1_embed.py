"""E1 — sentence-transformers all-MiniLM-L6-v2, 10K sentences.

Compares CIPHER vs baseline: sentences/sec, watts, sentences/W, and
cosine similarity between CIPHER and baseline embeddings (must be >0.99).
"""
import os, sys, json, time, threading, ctypes, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
N_SENTENCES = int(os.environ.get("N_SENTENCES", "10000"))


def main():
    if USE_CIPHER:
        rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"),
                         mode=ctypes.RTLD_GLOBAL)
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"[E1] cipher={USE_CIPHER} sentences={N_SENTENCES}", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")

    # Diverse sentences with mild variation
    base = [
        "The future of GPU computing is to make every joule count.",
        "Energy efficiency means doing more useful work per watt.",
        "Modern AI workloads stress memory bandwidth and tensor throughput.",
        "Distributed training requires careful overlap of compute and communication.",
        "Speculative decoding can amortize the cost of large model inference.",
        "FP8 quantization preserves most accuracy while halving weight bandwidth.",
        "Persistent L2 windows benefit memory-bound kernels with repeated reuse.",
        "NCCL all-reduce dominates the cost of synchronous data parallel training.",
        "The H100 SXM has 132 streaming multiprocessors and 80 GB of HBM3 memory.",
        "Inference servers must balance latency tail against throughput at peak load.",
    ]
    sentences = [base[i % len(base)] + f" Doc#{i}." for i in range(N_SENTENCES)]

    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()
    t0 = time.perf_counter()
    embs = model.encode(sentences, batch_size=128, convert_to_numpy=True,
                        show_progress_bar=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    sps = N_SENTENCES / elapsed
    s_w = sps / mean_w if mean_w > 0 else 0

    # Save embedding hash (first 50 sentences) so we can compare across runs.
    head = embs[:50].tobytes()
    embs_head_hash = hashlib.sha256(head).hexdigest()[:16]

    suffix = "_baseline" if not USE_CIPHER else ""
    payload = dict(cipher=USE_CIPHER, n=N_SENTENCES, elapsed_s=elapsed,
                   sps=sps, mean_w=mean_w, s_w=s_w,
                   embs_head_hash=embs_head_hash,
                   embs_dim=int(embs.shape[1]), embs_shape=list(embs.shape))
    out_path = os.path.join(os.path.dirname(__file__), f"e1_embed{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    # Save the embedding head itself for cosine similarity later.
    import numpy as np
    np.save(out_path.replace(".json", "_head.npy"), embs[:50])
    print(f"[E1] sps={sps:.1f} W={mean_w:.0f} s/W={s_w:.3f} "
          f"head_hash={embs_head_hash}", flush=True)


if __name__ == "__main__":
    main()
