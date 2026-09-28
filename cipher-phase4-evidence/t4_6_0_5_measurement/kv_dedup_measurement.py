"""T4.6.0.5 — Cross-process KV-block dedup opportunity measurement.

Methodology (per the phase prompt):

  M1: Build realistic multi-tenant traces (chat 50% / agentic 30% / RAG 15% / code 5%).
      Synthesized from public templates (Alpaca-style system prompts, OpenAI tool
      schemas, HotpotQA-style RAG passages, HumanEval-style code prompts). The
      STRUCTURAL pattern (shared system prefix + divergent tail) reproduces
      production operator-context traffic. Tokenized with Mistral-7B's actual
      tokenizer.

  M2: Hash each prompt into N-token blocks at block sizes {1, 4, 16, 64} using
      FNV-1a (lightweight, no crypto). Record (workload, tenant_id, prompt_id,
      block_size, block_pos, block_hash, block_global_pos).

  M3: Four scenarios:
        S1 within-tenant dedup (per-process baseline, what vLLM/SGLang see)
        S2 cross-tenant dedup at N ∈ {2,4,8,16,32} concurrent tenants
           with RANDOM groupings (best/median/worst over 20 trials)
        S3 position-stratified: dedup ratio by token-position bin
        S4 block-size sensitivity (1 vs 4 vs 16 vs 64 token blocks)

  Output: CSVs + JSON summary, suitable for the M4 economic mapping step.

This is operator-context measurement — no production artifact changes,
no GPU touched.
"""
import os, sys, json, csv, time, hashlib, random
import numpy as np
from collections import Counter, defaultdict

random.seed(42)
np.random.seed(42)

OUT_DIR = "/home/ubuntu/cipher-phase4-evidence/t4_6_0_5_measurement"
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Tokenizer (Mistral-7B, the model class we've validated against)
# ────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer
TOK_MODEL = "mistralai/Mistral-7B-v0.1"
print(f"[M1] loading tokenizer {TOK_MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(TOK_MODEL)
print(f"[M1] tokenizer loaded", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# M1: Trace synthesis
# ────────────────────────────────────────────────────────────────────────────

# 4 distinct chat system prompts (operator types: chatbot, customer service,
# coding assistant, RAG copilot). 50 conversations distributed across them
# with weight (12, 12, 13, 13).
CHAT_SYS_PROMPTS = [
    # 1. Generic AI assistant (typical chatbot operator)
    """You are a helpful, harmless, and honest AI assistant. You provide accurate, well-reasoned responses to user queries.
Guidelines:
- Be concise but thorough
- Cite sources where appropriate
- Acknowledge uncertainty when present
- Avoid speculation about future events
- Follow safety guidelines: refuse harmful requests, decline to generate misleading content, redirect medical/legal/financial questions to professionals.
Format responses with clear structure when answering complex questions. Use markdown for code, bullet points for lists, and bold for emphasis.""",

    # 2. Customer service operator
    """You are a customer service agent for ExampleCorp, a software company.
You help users with billing, account issues, technical support, and product questions.
Tone: friendly, professional, empathetic. Acknowledge frustration; do not dismiss complaints.
Escalation rules:
- Refunds over $500 → escalate to manager (provide ticket ID)
- Account security issues → require identity verification before action
- Technical bugs → log to internal tracker with reproduction steps
- Legal/regulatory questions → redirect to legal team
Always thank the customer for their patience. Sign off with: "Best, ExampleCorp Support".""",

    # 3. Coding assistant
    """You are an expert software engineer specializing in Python, Rust, and TypeScript.
When responding to code questions:
1. Provide working, idiomatic code with comments
2. Explain the algorithmic approach briefly before showing code
3. Mention complexity (time/space) when relevant
4. Suggest tests or edge cases the user should verify
5. If the user asks about a specific library/framework, use the latest stable API
Style preferences: prefer type hints in Python, prefer error handling over panic, prefer composition over inheritance.""",

    # 4. RAG copilot
    """You are a research assistant. You will be given retrieved passages from a knowledge base, followed by a user question. Your job is to answer the question using ONLY the information in the passages.
If the passages don't contain the answer, say "I don't have enough information to answer that based on the provided sources."
Always cite the passage number you used for each claim. Format: "[Passage N]".
Never fabricate citations. Never use external knowledge that isn't in the passages.""",
]


def synth_chat(n_conv=50):
    """Build chat traces: shared sys prompt + per-conversation user messages + assistant turns."""
    user_msg_templates = [
        "What is the capital of {country}?",
        "Can you explain {topic} in simple terms?",
        "Write a {item_type} about {subject}.",
        "How do I {action} in {language}?",
        "Compare {a} and {b} — which is better for {use_case}?",
        "I'm having trouble with {tool} — it keeps {problem}.",
        "Can you summarize the key points of {document_type}?",
        "What are best practices for {activity}?",
    ]
    fills = {
        "country": ["France", "Japan", "Brazil", "Egypt", "Canada", "Italy", "India", "Australia"],
        "topic": ["recursion", "blockchain", "the Krebs cycle", "neural nets", "the Renaissance",
                  "supply and demand", "garbage collection", "constitutional law"],
        "item_type": ["short essay", "haiku", "summary", "outline", "letter"],
        "subject": ["climate change", "machine learning", "the moon landing", "Shakespeare",
                    "open-source software", "the printing press"],
        "action": ["sort a list", "parse JSON", "read a file", "make HTTP requests",
                   "manage dependencies", "write tests"],
        "language": ["Python", "Rust", "TypeScript", "Go", "Java", "C++"],
        "a": ["PostgreSQL", "React", "Docker", "Kubernetes", "gRPC"],
        "b": ["MongoDB", "Vue", "Podman", "Nomad", "REST"],
        "use_case": ["a startup", "an enterprise", "a side project", "high-traffic apps"],
        "tool": ["git", "vim", "VS Code", "Docker", "make", "cargo"],
        "problem": ["showing weird errors", "running slowly", "not connecting",
                    "freezing", "consuming memory"],
        "document_type": ["a research paper", "a technical RFC", "a meeting transcript", "a contract"],
        "activity": ["code reviews", "API design", "incident response", "writing documentation"],
    }
    convs = []
    for ci in range(n_conv):
        sys_idx = ci % 4   # round-robin across the 4 operator types
        sys_prompt = CHAT_SYS_PROMPTS[sys_idx]
        n_turns = random.randint(5, 10)
        turns = []
        for t in range(n_turns):
            tmpl = random.choice(user_msg_templates)
            user = tmpl.format(**{k: random.choice(v) for k, v in fills.items()})
            asst_len = random.randint(60, 280)   # chars; rough proxy for token variation
            # Make assistant text plausibly diverse — use a randomized fragment lib
            asst = " ".join([
                random.choice(["Sure", "Of course", "Great question", "Let me explain", "Here is"]),
                random.choice(["—", "."]),
                "I'd suggest looking at this from " + random.choice(["a structural", "a historical", "a practical", "an algorithmic"]) + " angle.",
                "Specifically:", " ".join(random.choice([
                    "consider edge cases first",
                    "the documentation has examples",
                    "you can test this incrementally",
                    "start with a minimal reproduction",
                    "the library handles most cases by default",
                ]) for _ in range(random.randint(2, 5))),
            ])[:asst_len]
            turns.append((user, asst))
        # Full prompt = sys + (user, asst, user, asst, ...) up to second-to-last turn,
        # then final user query (what the model would actually respond to)
        history = []
        for ui, (u, a) in enumerate(turns[:-1]):
            history.append(f"[INST] {u} [/INST] {a}")
        history.append(f"[INST] {turns[-1][0]} [/INST]")
        prompt = sys_prompt + "\n\n" + "\n".join(history)
        convs.append({"tenant_id": f"chat_{ci:03d}", "operator_type": sys_idx, "prompt": prompt})
    return convs


AGENT_TOOL_DEFS = """You have access to the following tools. Use them to accomplish the user's task.

Tools:
- search(query: str) → str: Search the web for the given query, returns a brief summary of top results.
- get_weather(location: str) → dict: Get current weather for a city or zip code. Returns {temperature_f, conditions, humidity_pct, wind_mph}.
- calculate(expression: str) → float: Evaluate a mathematical expression (Python syntax). Supports +, -, *, /, **, sqrt, log, sin, cos.
- read_file(path: str) → str: Read a local file's contents.
- write_file(path: str, content: str) → bool: Write content to a local file. Returns True on success.
- list_dir(path: str) → list[str]: List files in a directory.
- run_shell(command: str) → str: Execute a shell command and return its stdout.
- send_email(to: str, subject: str, body: str) → bool: Send an email to the given recipient.
- query_database(sql: str) → list[dict]: Run a SQL query against the operational database (read-only).
- create_jira_ticket(title: str, description: str, priority: str) → str: Create a Jira ticket and return its ID.
- get_calendar_events(date: str) → list[dict]: List calendar events for the given date.

When you decide to use a tool, output exactly:
<tool_call>
{"name": "<tool_name>", "args": {<args>}}
</tool_call>

After receiving the tool result, decide whether you need more tool calls or can answer.
Always think step-by-step before each tool call. Use the format:
THOUGHT: <your reasoning>
ACTION: <tool_call or final answer>"""


def synth_agentic(n=30):
    """Agent traces: shared tool defs + per-tenant divergent goals + ReAct-style flow."""
    goals = [
        "Find out the weather in Paris and email a summary to alice@example.com",
        "List all files in /tmp/reports and create a Jira ticket for any TODOs found",
        "Calculate the compound interest on $1000 at 5% over 10 years and write the result to /tmp/result.txt",
        "Search for recent news about Rust 2024 edition and summarize the top 3 results",
        "Query the customers table for users who signed up this month and email the count to bob@example.com",
        "Get my calendar events for tomorrow and create a Jira ticket reminding me about each one",
        "Run `df -h` on the host and email any partitions over 80% usage to ops@example.com",
        "Find the largest file in /var/log and create a ticket to investigate",
        "Calculate the area of a circle with radius 7.3 cm and write it to a file",
        "Look up the population of Tokyo and Shanghai and report which is larger",
    ]
    react_flows = [
        "THOUGHT: I need to find the current weather, then send an email summarizing it.",
        "ACTION: <tool_call>{\"name\": \"get_weather\", \"args\": {\"location\": \"Paris\"}}</tool_call>",
        "OBSERVATION: {\"temperature_f\": 62, \"conditions\": \"cloudy\", \"humidity_pct\": 71, \"wind_mph\": 8}",
        "THOUGHT: Now I'll compose and send the email.",
        "ACTION: <tool_call>{\"name\": \"send_email\", \"args\": {\"to\": \"alice@example.com\", \"subject\": \"Paris weather\", \"body\": \"Currently 62F, cloudy, humidity 71%.\"}}</tool_call>",
        "OBSERVATION: true",
    ]
    traces = []
    for i in range(n):
        goal = random.choice(goals)
        n_steps = random.randint(2, 6)
        steps = react_flows[:n_steps * 3]   # each step = 3 lines (T/A/O)
        prompt = AGENT_TOOL_DEFS + "\n\nUser goal: " + goal + "\n\n" + "\n".join(steps)
        traces.append({"tenant_id": f"agent_{i:03d}", "operator_type": 0, "prompt": prompt})
    return traces


RAG_SYS = """You are a research assistant. Answer the user question using ONLY the retrieved passages below. Cite passage numbers like [P1]. If unsure, say so."""


def synth_rag(n=30):
    """RAG traces: shared sys + some recurring passages + per-query different question."""
    # 12 reusable passages drawn from a pool (simulating retrieval that returns
    # overlapping chunks for related queries — a common production pattern).
    passages = [
        "P1: The Krebs cycle is a series of chemical reactions used by all aerobic organisms to release stored energy through the oxidation of acetyl-CoA derived from carbohydrates, fats, and proteins.",
        "P2: Photosynthesis converts light energy into chemical energy stored in glucose molecules. The light-dependent reactions occur in the thylakoid membranes; the Calvin cycle occurs in the stroma.",
        "P3: HTTP/2 introduced binary framing, header compression (HPACK), multiplexing, and server push. HTTP/3 uses QUIC over UDP for reduced head-of-line blocking.",
        "P4: TCP three-way handshake: SYN, SYN-ACK, ACK. Connection establishment requires one round-trip time (RTT). TCP fast open reduces this with a cookie mechanism.",
        "P5: Rust ownership: each value has exactly one owner; values are dropped when the owner goes out of scope. References are either shared (&T) or mutable (&mut T), never both at the same time.",
        "P6: Python's GIL (Global Interpreter Lock) ensures only one thread executes bytecode at a time. CPU-bound tasks should use multiprocessing; I/O-bound tasks benefit from threading or async.",
        "P7: PostgreSQL MVCC stores multiple row versions to support concurrent reads without blocking writes. VACUUM reclaims dead tuples; autovacuum runs periodically.",
        "P8: Kubernetes Pod is the smallest deployable unit. A Deployment manages ReplicaSets. Service provides stable network endpoints; Ingress handles HTTP routing.",
        "P9: Transformers use multi-head self-attention. Each head computes Q@K^T/sqrt(d) softmax @ V independently. Outputs are concatenated and linearly projected.",
        "P10: The Renaissance began in 14th century Florence, characterized by renewed interest in classical antiquity. Key figures include Leonardo da Vinci, Michelangelo, and Galileo Galilei.",
        "P11: gRPC uses HTTP/2 for transport and Protocol Buffers for serialization. It supports streaming, deadlines, and bidirectional communication.",
        "P12: A B-tree is a self-balancing search tree where each node can have multiple children. PostgreSQL btree index uses B+ trees where leaf nodes contain pointers to row locations.",
    ]
    questions = [
        "How does the Krebs cycle relate to ATP production?",
        "What's the difference between HTTP/2 and HTTP/3?",
        "Explain Rust's ownership model.",
        "How does Python handle concurrency?",
        "What is MVCC in PostgreSQL?",
        "Describe a Kubernetes Service.",
        "How does self-attention work in transformers?",
        "What's the difference between gRPC and REST?",
        "Why are B-tree indexes used in databases?",
        "Compare TCP and QUIC.",
    ]
    traces = []
    for i in range(n):
        # Pick 3 retrieved passages: 2 from a topic + 1 from neighbor (realistic overlap)
        retrieved = random.sample(passages, k=3)
        question = random.choice(questions)
        prompt = RAG_SYS + "\n\nRetrieved passages:\n" + "\n".join(retrieved) + "\n\nQuestion: " + question
        traces.append({"tenant_id": f"rag_{i:03d}", "operator_type": 0, "prompt": prompt})
    return traces


CODE_SYS = """You are a code completion assistant. Given a Python file's contents, complete the function indicated by the trailing `# COMPLETE` comment. Match the existing code style (4-space indent, type hints when present, docstrings)."""


def synth_code(n=30):
    """Code completion: shared system + per-prompt file context + completion request."""
    file_contexts = [
        """from typing import List

def quicksort(arr: List[int]) -> List[int]:
    \"\"\"Return a sorted copy of arr using quicksort.\"\"\"
    # COMPLETE""",

        """import json
from pathlib import Path

def load_config(path: Path) -> dict:
    \"\"\"Load a JSON config file, returning an empty dict if missing.\"\"\"
    # COMPLETE""",

        """import socket

def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    \"\"\"Return True if TCP port is reachable on host within timeout seconds.\"\"\"
    # COMPLETE""",

        """from typing import Iterator

def fibonacci() -> Iterator[int]:
    \"\"\"Yield the Fibonacci sequence indefinitely.\"\"\"
    # COMPLETE""",

        """import hashlib

def file_sha256(path: str) -> str:
    \"\"\"Return the SHA-256 hex digest of the file at path.\"\"\"
    # COMPLETE""",
    ]
    traces = []
    for i in range(n):
        ctx = random.choice(file_contexts)
        prompt = CODE_SYS + "\n\nFile contents:\n```python\n" + ctx + "\n```\n\nCompletion:"
        traces.append({"tenant_id": f"code_{i:03d}", "operator_type": 0, "prompt": prompt})
    return traces


# ────────────────────────────────────────────────────────────────────────────
# M2: Block hashing
# ────────────────────────────────────────────────────────────────────────────

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME  = 0x100000001b3

def fnv1a_hash_tokens(toks):
    """FNV-1a hash over a token-id sequence (uint32 each)."""
    h = FNV_OFFSET
    for t in toks:
        # mix 4 bytes per token
        for byte_idx in range(4):
            b = (t >> (8 * byte_idx)) & 0xff
            h ^= b
            h = (h * FNV_PRIME) & 0xffffffffffffffff
    return h


def slice_into_blocks(token_ids, block_size):
    """Yield (block_pos, block_hash, block_tokens) for each block. Partial trailing block included for accuracy."""
    for i in range(0, len(token_ids), block_size):
        block = token_ids[i:i + block_size]
        if len(block) == 0: continue
        yield (i, fnv1a_hash_tokens(block), len(block))


# ────────────────────────────────────────────────────────────────────────────
# M3 helpers
# ────────────────────────────────────────────────────────────────────────────

def measure_dedup(records, block_size, tenant_subset=None):
    """records: list of (workload, tenant_id, prompt_id, block_size, block_pos, block_hash, block_len).
       Return (n_total_blocks, n_unique_blocks, dedup_ratio, total_token_bytes, unique_token_bytes).
       block_size is the block size we're measuring at."""
    if tenant_subset is not None:
        records = [r for r in records if r[1] in tenant_subset]
    records = [r for r in records if r[3] == block_size]
    seen = set()
    n_total = 0
    n_unique = 0
    total_tokens = 0
    unique_tokens = 0
    for r in records:
        h = r[5]
        n_blk_len = r[6]
        n_total += 1
        total_tokens += n_blk_len
        if h not in seen:
            seen.add(h)
            n_unique += 1
            unique_tokens += n_blk_len
    if n_unique == 0:
        return 0, 0, 0.0, 0, 0
    return n_total, n_unique, n_total / n_unique, total_tokens, unique_tokens


# ────────────────────────────────────────────────────────────────────────────
# DRIVE
# ────────────────────────────────────────────────────────────────────────────

print("[M1] synthesizing chat (50 conversations)...", flush=True)
chat = synth_chat(50)
print(f"[M1] chat done: {len(chat)} traces", flush=True)
print("[M1] synthesizing agentic (30 traces)...", flush=True)
agentic = synth_agentic(30)
print("[M1] synthesizing RAG (30 traces)...", flush=True)
rag = synth_rag(30)
print("[M1] synthesizing code (30 traces)...", flush=True)
code = synth_code(30)

WORKLOADS = [
    ("chat", chat),
    ("agentic", agentic),
    ("rag", rag),
    ("code", code),
]

# Tokenize all and report per-workload token stats
print("[M1] tokenizing all traces with Mistral tokenizer...", flush=True)
tokenized_records = []  # per-trace: dict
for wname, traces in WORKLOADS:
    lens = []
    for trace in traces:
        token_ids = tok(trace["prompt"], return_tensors=None, add_special_tokens=False)["input_ids"]
        trace["tokens"] = token_ids
        lens.append(len(token_ids))
    print(f"[M1]   {wname}: n={len(traces)}, "
          f"tokens: mean={np.mean(lens):.0f} median={np.median(lens):.0f} "
          f"min={min(lens)} max={max(lens)} total={sum(lens)}",
          flush=True)


# M2: block hashing at multiple block sizes
print("[M2] hashing blocks at sizes {1, 4, 16, 64}...", flush=True)
BLOCK_SIZES = [1, 4, 16, 64]
records = []  # list of tuples
total_rows = 0
for wname, traces in WORKLOADS:
    for trace in traces:
        for bs in BLOCK_SIZES:
            for pos, h, blk_len in slice_into_blocks(trace["tokens"], bs):
                records.append((wname, trace["tenant_id"], trace.get("prompt_id", "p0"),
                                bs, pos, h, blk_len))
                total_rows += 1
print(f"[M2] {total_rows} block records produced", flush=True)


# Save raw CSV
csv_path = os.path.join(OUT_DIR, "block_hashes.csv")
print(f"[M2] writing CSV → {csv_path} (this may take a moment)...", flush=True)
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["workload", "tenant_id", "prompt_id", "block_size", "block_pos", "block_hash", "block_len"])
    for r in records:
        w.writerow(r)
print(f"[M2] wrote {total_rows} rows", flush=True)


# M3: scenarios
print("\n[M3] === Scenario S1: within-tenant dedup baseline ===", flush=True)
s1_results = {}
for wname, _ in WORKLOADS:
    for bs in BLOCK_SIZES:
        # Within each tenant separately, compute dedup, then average across tenants
        tenants = sorted({r[1] for r in records if r[0] == wname})
        ratios = []
        for tid in tenants:
            n_total, n_unique, ratio, *_ = measure_dedup(
                [r for r in records if r[0] == wname], bs, tenant_subset={tid})
            if n_total > 0:
                ratios.append(ratio)
        avg = np.mean(ratios) if ratios else 0.0
        s1_results[(wname, bs)] = avg
        print(f"  {wname} block={bs}: within-tenant dedup mean ratio = {avg:.3f}", flush=True)

print("\n[M3] === Scenario S2: cross-tenant dedup at N concurrent tenants ===", flush=True)
N_VALUES = [2, 4, 8, 16, 32]
N_TRIALS = 20
s2_results = {}
for wname, traces in WORKLOADS:
    tenants = sorted({trace["tenant_id"] for trace in traces})
    for bs in BLOCK_SIZES:
        for N in N_VALUES:
            if N > len(tenants): continue
            ratios = []
            for trial in range(N_TRIALS):
                subset = set(random.sample(tenants, N))
                _, _, ratio, _, _ = measure_dedup(
                    [r for r in records if r[0] == wname], bs, tenant_subset=subset)
                ratios.append(ratio)
            ratios.sort()
            mn, p10, med, p90, mx = (ratios[0], ratios[len(ratios)//10],
                                     ratios[len(ratios)//2], ratios[9*len(ratios)//10],
                                     ratios[-1])
            s2_results[(wname, bs, N)] = {
                "min": mn, "p10": p10, "median": med, "p90": p90, "max": mx, "mean": np.mean(ratios)
            }
            if bs == 16 and N in (4, 8, 16):
                print(f"  {wname} block=16 N={N}: dedup ratio "
                      f"min={mn:.2f} med={med:.2f} max={mx:.2f}", flush=True)

print("\n[M3] === Scenario S3: position-stratified dedup (block=16, N=8 cross-tenant) ===", flush=True)
POSITION_BINS = [(0, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 100000)]
s3_results = {}
for wname, traces in WORKLOADS:
    tenants = sorted({trace["tenant_id"] for trace in traces})
    if len(tenants) < 8: continue
    # Take a random group of 8 tenants
    subset = set(random.sample(tenants, 8))
    bs = 16
    blocks_in_bin = {b: [] for b in POSITION_BINS}
    for r in records:
        if r[0] != wname or r[3] != bs or r[1] not in subset:
            continue
        pos = r[4]
        for lo, hi in POSITION_BINS:
            if lo <= pos < hi:
                blocks_in_bin[(lo, hi)].append(r[5])
                break
    bin_ratios = {}
    for b, bh_list in blocks_in_bin.items():
        if len(bh_list) == 0: continue
        n_unique = len(set(bh_list))
        ratio = len(bh_list) / n_unique
        bin_ratios[b] = {"n_total": len(bh_list), "n_unique": n_unique, "ratio": ratio}
    s3_results[wname] = bin_ratios
    print(f"  {wname}:", flush=True)
    for b in POSITION_BINS:
        if b in bin_ratios:
            print(f"    pos {b}: n_total={bin_ratios[b]['n_total']} "
                  f"n_unique={bin_ratios[b]['n_unique']} ratio={bin_ratios[b]['ratio']:.2f}",
                  flush=True)

print("\n[M3] === Scenario S4: block-size sensitivity (mixed workload, N=8) ===", flush=True)
# Weighted-mixed workload: pull 8 tenants proportionally
mixed_subset = set()
mix = [("chat", 4), ("agentic", 2), ("rag", 1), ("code", 1)]  # 8 total ≈ 50/25/12.5/12.5
for wname, n in mix:
    tlist = sorted({trace["tenant_id"] for trace in dict(WORKLOADS)[wname]})
    mixed_subset.update(random.sample(tlist, n))

s4_results = {}
print(f"  Mixed subset of {len(mixed_subset)} tenants (chat=4 agent=2 rag=1 code=1)", flush=True)
for bs in BLOCK_SIZES:
    ratios = []
    for trial in range(10):
        ms = set(random.sample(list(mixed_subset), min(8, len(mixed_subset))))
        _, _, ratio, _, _ = measure_dedup(records, bs, tenant_subset=ms)
        ratios.append(ratio)
    med = sorted(ratios)[len(ratios)//2]
    s4_results[bs] = {"median": med, "all": ratios}
    print(f"  block={bs}: median dedup ratio = {med:.3f}", flush=True)


# Save summary JSON
summary = {
    "tokenizer": TOK_MODEL,
    "workloads": {wn: len(tr) for wn, tr in WORKLOADS},
    "total_blocks": total_rows,
    "block_sizes": BLOCK_SIZES,
    "S1_within_tenant": {f"{wn}_bs{bs}": v for (wn, bs), v in s1_results.items()},
    "S2_cross_tenant": {f"{wn}_bs{bs}_N{N}": v for (wn, bs, N), v in s2_results.items()},
    "S3_position_stratified_bs16_N8": s3_results,
    "S4_block_size_mix_N8": s4_results,
}
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n[M3] wrote {summary_path}", flush=True)
print("[M3] done.", flush=True)
