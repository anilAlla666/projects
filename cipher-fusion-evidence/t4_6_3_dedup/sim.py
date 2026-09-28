"""T4.6.3 Phase 1 — cross-tenant KV page-level dedup simulation.

Pure data analysis on the Mooncake FAST'25 traces. No GPU, no S2b.
Answers the binary: does cross-tenant 2 MiB-page KV dedup deliver a
multi-tenant density win?

Model (per the T4.6.3 prompt):
  - Mooncake block = 512 tokens; each hash_id is a privacy-preserving
    equality-preserving block id.
  - A CIPHER 2 MiB page (S2b allocator) holds one (layer, role) KV slab
    chunk. page_blocks = floor(2 MiB / KV_bytes_per_block).
  - A request's hash_ids are segmented into pages of N consecutive
    blocks; page n = blocks [n*N, (n+1)*N). Partial trailing page kept.
  - Global dedup table keyed on the page's tuple of block hashes; a page
    is a dedup HIT if an identical tuple was seen earlier (cross-request
    == cross-tenant).
  - dedup_ratio = hits / total_pages.

This is an UPPER BOUND: block-id equality is assumed to imply
byte-identical page content. Page-internal byte alignment is
unobservable from the trace. Phase 2 validates the achieved fraction.
"""
import json, random, sys, statistics

BLOCK_TOKENS = 512
PAGE_BYTES = 2 * 1024 * 1024
DTYPE_BYTES = 2  # FP16
TRACE_DIR = "/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup/traces"

# Both named models: 8 KV heads x 128 head_dim (GQA). Identical KV
# geometry -> identical page_blocks. Reported per the prompt's step 7.
MODELS = {
    "Mistral-7B": {"kv_heads": 8, "head_dim": 128},
    "Llama-3-8B": {"kv_heads": 8, "head_dim": 128},
}
FLAVORS = {
    "conversation": f"{TRACE_DIR}/conversation_trace.jsonl",
    "toolagent":    f"{TRACE_DIR}/toolagent_trace.jsonl",
    "synthetic":    f"{TRACE_DIR}/synthetic_trace.jsonl",
}
SUBSET_FRAC = 0.8
SUBSET_TRIALS = 5
random.seed(20260515)


def page_blocks(kv_heads, head_dim):
    """N consecutive 512-token blocks that fit in one 2 MiB page,
    for one layer / one role (S2b allocates K and V as separate slabs)."""
    kv_bytes_per_token = kv_heads * head_dim * DTYPE_BYTES
    kv_bytes_per_block = kv_bytes_per_token * BLOCK_TOKENS
    return PAGE_BYTES // kv_bytes_per_block, kv_bytes_per_block


def load(path):
    reqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            reqs.append(list(r.get("hash_ids", [])))
    return reqs


def dedup(requests, N):
    """Return (total_pages, hits). Cross-request global dedup table."""
    table = set()
    total = hits = 0
    for hash_ids in requests:
        for i in range(0, len(hash_ids), N):
            page = tuple(hash_ids[i:i + N])
            if not page:
                continue
            total += 1
            if page in table:
                hits += 1
            else:
                table.add(page)
    return total, hits


def ratio_of(requests, N):
    t, h = dedup(requests, N)
    return (h / t if t else 0.0), t, h


def subset_bounds(requests, N):
    """5 random SUBSET_FRAC subsamples -> ratio spread."""
    k = max(1, int(len(requests) * SUBSET_FRAC))
    rs = []
    for _ in range(SUBSET_TRIALS):
        sub = random.sample(requests, k)
        r, _, _ = ratio_of(sub, N)
        rs.append(r)
    return rs


def main():
    print("=== T4.6.3 Phase 1 — cross-tenant KV page dedup (Mooncake) ===\n")

    # page_blocks per model
    pb = {}
    for m, p in MODELS.items():
        n, kvb = page_blocks(p["kv_heads"], p["head_dim"])
        pb[m] = n
        print(f"  {m}: kv_heads={p['kv_heads']} head_dim={p['head_dim']} "
              f"-> KV/block={kvb/1048576:.2f} MiB -> page_blocks N={n}")
    distinct_N = sorted(set(pb.values()))
    print(f"  -> distinct page_blocks values across models: {distinct_N}")
    if len(distinct_N) == 1:
        print(f"  -> both models coincide at N={distinct_N[0]} "
              f"(identical 8x128 KV geometry)\n")

    # primary N(s) + sensitivity
    sens_N = sorted(set(distinct_N) | {1, 4})
    results = {}

    for flavor, path in FLAVORS.items():
        reqs = load(path)
        n_tok_blocks = sum(len(r) for r in reqs)
        print(f"--- {flavor}: {len(reqs)} requests, "
              f"{n_tok_blocks} prefill blocks total ---")
        results[flavor] = {}
        for N in sens_N:
            full_r, total, hits = ratio_of(reqs, N)
            rs = subset_bounds(reqs, N)
            tag = "  (PRIMARY)" if N in distinct_N else "  (sensitivity)"
            print(f"  N={N}{tag}: total_pages={total} hits={hits} "
                  f"dedup_ratio={full_r*100:.2f}%  "
                  f"| 5x{int(SUBSET_FRAC*100)}% subsets: "
                  f"mean={statistics.mean(rs)*100:.2f}% "
                  f"sd={statistics.pstdev(rs)*100:.2f}% "
                  f"min={min(rs)*100:.2f}% max={max(rs)*100:.2f}%")
            results[flavor][N] = {
                "total_pages": total, "hits": hits,
                "dedup_ratio": full_r,
                "subset_mean": statistics.mean(rs),
                "subset_sd": statistics.pstdev(rs),
                "subset_min": min(rs), "subset_max": max(rs),
            }
        print()

    # headline
    primary_N = distinct_N[0]
    print("=== HEADLINE (primary N, the gate number is toolagent) ===")
    for flavor in FLAVORS:
        r = results[flavor][primary_N]
        print(f"  {flavor:13s} N={primary_N}: "
              f"dedup_ratio = {r['dedup_ratio']*100:.2f}%  "
              f"(subset {r['subset_min']*100:.1f}-{r['subset_max']*100:.1f}%)")

    with open("/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup/sim_results.json", "w") as f:
        json.dump({"page_blocks": pb, "primary_N": primary_N,
                   "results": {fl: {str(n): v for n, v in d.items()}
                               for fl, d in results.items()}}, f, indent=2)
    print("\nwrote sim_results.json")


if __name__ == "__main__":
    main()
