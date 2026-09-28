#!/usr/bin/env python3
"""Mistral-7B decode with CUDA Graph capture/replay.

Strategy:
  1. StaticCache pre-allocates KV at max length (no resize during decode).
  2. Run a few warmup decode steps normally (graph-incompatible ops settle).
  3. Capture exactly one decode step into a CUDA graph using
     torch.cuda.graph(g).
  4. For subsequent tokens: copy the new input_ids into the captured input
     tensor in-place, advance cache_position, replay the graph.

Compare:
  - Eager (HF generate): baseline tok/s
  - Captured + replayed:  decode tok/s

Env:
    CIPHER_GRAPH_N_TOKENS — tokens to generate (default 128)
"""
import os
import sys
import time
import torch

import warnings
warnings.filterwarnings("ignore")

cipher_active = "libcipher_hook" in os.environ.get("LD_PRELOAD", "")
print(f"[{'CIPHER' if cipher_active else 'BASELINE'}] mistral graph-capture decode")

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import StaticCache
    HAS_STATIC_CACHE = True
except ImportError:
    HAS_STATIC_CACHE = False
    print("StaticCache not available; falling back to manual implementation")

MODEL    = "mistralai/Mistral-7B-v0.1"
N_TOKENS = int(os.environ.get("CIPHER_GRAPH_N_TOKENS", "128"))
MAX_LEN  = 384
PROMPT   = ("The future of artificial intelligence in GPU computing is to make "
            "every joule of energy count.")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda")

prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids.shape[1]
print(f"prompt_len={prompt_len}, generating {N_TOKENS} tokens")


# ── Eager baseline (HF generate) ────────────────────────────────────────────
def run_eager():
    with torch.no_grad():
        _ = model.generate(prompt_ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=N_TOKENS,
                             do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return N_TOKENS / elapsed, elapsed, out


# ── Graph-captured decode ───────────────────────────────────────────────────
def run_graph():
    if not HAS_STATIC_CACHE:
        return None, None, None

    # 1) Prefill phase — fill the StaticCache with prompt tokens. This is
    #    NOT captured in the graph; only the per-token decode loop is.
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=MAX_LEN,
        device="cuda", dtype=torch.float16,
    )

    with torch.no_grad():
        cache_position = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(
            input_ids=prompt_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)         # (1, 1)

    # 2) Static decode buffers — bound to the graph capture.
    input_ids   = next_token.detach().clone()           # (1, 1) int64
    cache_pos   = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits  = torch.empty(1, 1, model.config.vocab_size,
                              device="cuda", dtype=torch.float16)

    # 3) Warmup: run a few decode steps in the side stream so cuDNN/cuBLAS
    #    pick their kernels and any one-off allocations settle.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            with torch.no_grad():
                o = model(
                    input_ids=input_ids,
                    cache_position=cache_pos,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    # 4) Capture one decode step.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(
                input_ids=input_ids,
                cache_position=cache_pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        out_logits.copy_(o.logits)
        new_tok = out_logits.argmax(-1)
        input_ids.copy_(new_tok)
    torch.cuda.synchronize()

    # 5) Timed loop — replay the graph.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generated = []
    for step in range(N_TOKENS):
        cache_pos += 1
        g.replay()
        # input_ids has been updated in place by the graph copy; capture it
        # before next replay.
        generated.append(input_ids.clone())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    full = torch.cat([prompt_ids, *generated], dim=1)
    return N_TOKENS / elapsed, elapsed, full


print("\n=== Eager baseline (HF generate) ===")
tps_eager, t_eager, out_eager = run_eager()
print(f"  {N_TOKENS} tokens in {t_eager:.3f}s = {tps_eager:.2f} tok/s")
print(f"  output[:120]: {tokenizer.decode(out_eager[0])[:120]!r}")

print("\n=== CUDA Graph capture + replay ===")
tps_graph, t_graph, out_graph = run_graph()
if tps_graph is None:
    print("  [skipped] StaticCache not available")
else:
    print(f"  {N_TOKENS} tokens in {t_graph:.3f}s = {tps_graph:.2f} tok/s")
    print(f"  speedup over eager: {tps_graph/tps_eager:.2f}x")
    print(f"  output[:120]: {tokenizer.decode(out_graph[0])[:120]!r}")
