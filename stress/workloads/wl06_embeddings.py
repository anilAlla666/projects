"""WL06 — Embeddings (sentence-transformers MiniLM)"""
from __future__ import annotations
import argparse, hashlib, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import write_result, sample_power, avg_power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="baseline")
    ap.add_argument("--n-sents", type=int, default=200)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    sents = [f"This is sentence number {i} about topic {i%17}." for i in range(args.n_sents)]
    samples = []
    stop_evt = threading.Event()
    threading.Thread(target=sample_power, args=(stop_evt, samples, 0), daemon=True).start()
    t0 = time.perf_counter()
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    embs = m.encode(sents, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    dt = time.perf_counter() - t0
    stop_evt.set()
    pw = avg_power(samples)
    rate = args.n_sents / dt
    h = hashlib.sha256(embs.tobytes()).hexdigest()[:16]
    summary = {"wid": "wl06", "phase": args.phase, "n_sents": args.n_sents,
               "encode_seconds": dt, "sents_per_s": rate,
               "avg_watts": pw, "sents_per_w": rate / pw if pw > 0 else 0,
               "load_seconds": t_load, "emb_hash": h, "emb_shape": list(embs.shape)}
    print(json.dumps(summary, indent=2))
    write_result("wl06", args.phase, summary)


if __name__ == "__main__":
    main()
