#!/usr/bin/env python3
"""Measure graph-capture decode tok/s + power + MFU."""
import os, sys, time, subprocess, threading, json
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

cipher_active = "libcipher_hook" in os.environ.get("LD_PRELOAD", "")
MODEL = "mistralai/Mistral-7B-v0.1"
MAX_LEN = 384
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")
N_TOKENS = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
nparams = sum(p.numel() for p in model.parameters())
prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids.shape[1]


def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception:
            pass
        time.sleep(0.2)


def measure(label, tps_fn):
    samples = []; stop = threading.Event()
    t = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True)
    t.start()
    elapsed, generated = tps_fn()
    stop.set(); t.join(timeout=2)
    pw  = [s[0] for s in samples[1:]]
    clk = [s[1] for s in samples[1:]]
    mean_pw  = sum(pw) / max(len(pw), 1)
    mean_clk = sum(clk) / max(len(clk), 1)
    tps      = generated / elapsed
    flops    = 2.0 * nparams * generated
    tflops   = flops / elapsed / 1e12
    mfu      = tflops / 989.0 * 100.0
    tok_w    = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:30s} tps={tps:7.2f}  power={mean_pw:5.1f}W  clk={mean_clk:.0f}MHz  "
          f"TFLOPS={tflops:5.2f}  MFU={mfu:5.2f}%  tok/W={tok_w:6.3f}")


def eager_run():
    with torch.no_grad():
        _ = model.generate(prompt_ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, N_TOKENS


def graph_run():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cache_position = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cache_position,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(1, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)

    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids.copy_(out_logits.argmax(-1))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        cache_pos += 1
        g.replay()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, N_TOKENS


print(f"=== Mistral-7B fp16, 256 tokens, batch=1, CIPHER={'on' if cipher_active else 'off'} ===")
measure("eager (HF generate)",   eager_run)
measure("graph capture + replay", graph_run)
