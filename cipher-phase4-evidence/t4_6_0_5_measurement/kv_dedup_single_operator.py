"""Sanity check: in a SINGLE-OPERATOR scenario (one sys prompt, many end-users),
how does cross-tenant dedup at block=16 change?

This is the canonical CIPHER deployment: a SaaS company deploys CIPHER for
their own service; their N concurrent users share infrastructure on a few H100s.
The 50-100 tenants/H100 marvel pitch maps to this scenario.

vs the multi-operator scenario in kv_dedup_measurement.py (4 sys prompts).
"""
import random, json, hashlib
import numpy as np
random.seed(43)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Single operator with ONE shared sys prompt
SHARED_SYS = """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries.
Guidelines:
- Be concise but thorough
- Cite sources where appropriate
- Acknowledge uncertainty when present
- Avoid speculation about future events
- Follow safety guidelines: refuse harmful requests, decline to generate misleading content, redirect medical/legal/financial questions to professionals.
Format responses with clear structure when answering complex questions. Use markdown for code, bullet points for lists, and bold for emphasis."""

user_templates = [
    "What is the capital of {x}?", "Explain {y} in simple terms.",
    "Write a short essay about {z}.", "How do I {a} in Python?",
    "Compare {b1} and {b2}.", "Help me with {c}.",
    "What are best practices for {d}?", "Summarize {e}.",
]
fills = {
    "x": ["France", "Japan", "Brazil", "Egypt", "Canada"],
    "y": ["recursion", "blockchain", "neural nets", "the Renaissance"],
    "z": ["climate change", "ML", "the moon landing", "Shakespeare"],
    "a": ["sort", "parse JSON", "make HTTP requests", "write tests"],
    "b1": ["PostgreSQL", "React", "Docker"], "b2": ["MongoDB", "Vue", "Podman"],
    "c": ["git", "Docker", "make", "cargo"],
    "d": ["code reviews", "API design", "incident response"],
    "e": ["a paper", "a meeting transcript", "a contract"],
}


def make_trace(tid):
    n_turns = random.randint(5, 10)
    turns = []
    for _ in range(n_turns):
        tmpl = random.choice(user_templates)
        kwargs = {k: random.choice(v) for k, v in fills.items()}
        user_msg = tmpl.format(**kwargs)
        # Random short assistant response (per-tenant variation)
        asst_words = random.randint(20, 60)
        asst_msg = " ".join(random.choice([
            "Sure", "Of course", "Great question", "Here is",
            "consider edge cases first", "the documentation has examples",
            "you can test this incrementally", "the library handles most cases",
            "Specifically", "Let me explain", "—", "."
        ]) for _ in range(asst_words))
        turns.append((user_msg, asst_msg))
    history = "\n".join(f"[INST] {u} [/INST] {a}" for (u, a) in turns[:-1])
    prompt = SHARED_SYS + "\n\n" + history + f"\n[INST] {turns[-1][0]} [/INST]"
    return {"tenant_id": tid, "prompt": prompt}


# Generate 32 single-operator end-user tenants
traces = [make_trace(f"single_op_user_{i:03d}") for i in range(32)]
for t in traces:
    t["tokens"] = tok(t["prompt"], add_special_tokens=False)["input_ids"]

lens = [len(t["tokens"]) for t in traces]
print(f"single-operator chat: n={len(traces)} tokens mean={np.mean(lens):.0f} "
      f"median={np.median(lens):.0f} min={min(lens)} max={max(lens)}")

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME  = 0x100000001b3
def fnv1a(toks):
    h = FNV_OFFSET
    for t in toks:
        for byte_idx in range(4):
            h ^= (t >> (8 * byte_idx)) & 0xff
            h = (h * FNV_PRIME) & 0xffffffffffffffff
    return h


def cross_tenant_dedup(traces_subset, block_size):
    """count block-instances vs unique-hashes across the subset."""
    seen = set()
    n_total = 0
    n_unique = 0
    for t in traces_subset:
        toks = t["tokens"]
        for i in range(0, len(toks), block_size):
            blk = toks[i:i+block_size]
            if not blk: continue
            h = fnv1a(blk)
            n_total += 1
            if h not in seen:
                seen.add(h)
                n_unique += 1
    return n_total / n_unique if n_unique else 0.0


print("\n=== single-operator chat: cross-tenant dedup ===")
for N in [2, 4, 8, 16, 32]:
    if N > len(traces): continue
    ratios = []
    for _ in range(20):
        subset = random.sample(traces, N)
        for bs in [1, 4, 16, 64]:
            ratios.append((bs, cross_tenant_dedup(subset, bs)))
    by_bs = {}
    for bs, r in ratios:
        by_bs.setdefault(bs, []).append(r)
    for bs in [1, 4, 16, 64]:
        rs = sorted(by_bs[bs])
        med = rs[len(rs)//2]
        print(f"  N={N} block={bs}: median dedup ratio = {med:.2f}  "
              f"(min={rs[0]:.2f} max={rs[-1]:.2f})")


# Position-stratified analysis at block=16, N=8
print("\n=== single-operator position-stratified (block=16, N=8) ===")
subset = random.sample(traces, 8)
BINS = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 100000)]
blocks_in_bin = {b: [] for b in BINS}
for t in subset:
    toks = t["tokens"]
    for i in range(0, len(toks), 16):
        blk = toks[i:i+16]
        if not blk: continue
        h = fnv1a(blk)
        for lo, hi in BINS:
            if lo <= i < hi:
                blocks_in_bin[(lo, hi)].append(h); break
for b, hs in blocks_in_bin.items():
    if hs:
        nt = len(hs); nu = len(set(hs))
        print(f"  pos {b}: n_total={nt} n_unique={nu} ratio={nt/nu:.2f}")
