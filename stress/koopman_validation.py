#!/usr/bin/env python3
"""Side-by-side Koopman validation:
  - Phase A: baseline (no CIPHER ctypes load), generate N tokens
  - Phase B: CIPHER with EDMD_HOOK, generate same N tokens
  - Compare: token-equality, decoded text, tok/s, watts (sampled)

Spawned by an outer orchestrator that controls LD_PRELOAD + env. The
script picks ONE phase via PHASE env var (A or B) and writes JSON to
VALIDATE_OUT_PATH."""
import os, sys, json, time, ctypes, subprocess, threading

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.normpath(os.path.join(ROOT, "..", "libcipher_rt.so"))


def sample_power_watts(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            o = subprocess.check_output(
                ["nvidia-smi", "-i", "0",
                 "--query-gpu=power.draw,clocks.gr",
                 "--format=csv,noheader,nounits"], timeout=2).decode().strip()
            pw, clk = o.split(",")
            samples.append((time.perf_counter(), float(pw), float(clk)))
        except Exception:
            pass
        time.sleep(0.5)


def main():
    phase     = os.environ.get("PHASE", "A").upper()
    n_tokens  = int(os.environ.get("VALIDATE_N_TOKENS", "1000"))
    batch     = int(os.environ.get("VALIDATE_BATCH", "8"))
    prefill   = int(os.environ.get("VALIDATE_PREFILL", "64"))
    model_path = os.environ.get("VALIDATE_MODEL_PATH",
                                  "/home/ubuntu/models/Llama-3.1-8B")
    out_path  = os.environ["VALIDATE_OUT_PATH"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    rt = None
    if phase == "B":
        rt = ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"[{phase}] loading {model_path}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    for p in model.parameters(): p.requires_grad_(False)

    # Apply CIPHER fusion patches in CIPHER phase if RT loaded successfully
    if rt is not None and os.environ.get("CIPHER_FUSION_KERNELS", "off") in ("on","1","true"):
        try:
            sys.path.insert(0, ROOT)
            import stress_common as sc
            sc.patch_fusion(rt, model)
            print(f"[{phase}] CIPHER fusion patches applied", file=sys.stderr)
        except Exception as ex:
            print(f"[{phase}] fusion patch failed: {ex}", file=sys.stderr)

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile = (prefill + len(ids_one) - 1) // len(ids_one)
    prompt = ids_one.repeat(n_tile)[:prefill].unsqueeze(0).repeat(batch, 1).to("cuda:0")

    MAX_LEN = prefill + n_tokens + 4
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN, max_batch_size=batch)

    # Power sampler
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_power_watts, args=(stop, samples), daemon=True)
    sampler.start()

    # Prefill
    torch.cuda.synchronize()
    t_total = time.perf_counter()
    with torch.no_grad():
        cp = torch.arange(prefill, device="cuda:0", dtype=torch.long)
        o = model(input_ids=prompt, cache_position=cp,
                  past_key_values=cache, use_cache=True, return_dict=True)
        input_ids = o.logits[:, -1:].argmax(-1).clone()
        # Track ALL generated tokens for comparison
        all_tokens = [input_ids.clone()]
        cache_pos = torch.tensor([prefill], device="cuda:0", dtype=torch.long)

        # Decode
        for _ in range(n_tokens - 1):
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
            input_ids = o.logits.argmax(-1)
            all_tokens.append(input_ids.clone())
            cache_pos += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_total
    stop.set(); sampler.join(timeout=2)

    # Stack tokens — shape [n_tokens, batch, 1]
    token_tensor = torch.cat(all_tokens, dim=1).cpu()  # [batch, n_tokens]
    token_list   = token_tensor.tolist()

    decoded = [tok.decode(token_list[i], skip_special_tokens=True)[:200]
               for i in range(min(batch, 4))]

    if samples:
        watts = [s[1] for s in samples]
        mean_w = sum(watts) / len(watts)
    else:
        mean_w = 0.0

    total_tokens_generated = n_tokens * batch
    json.dump(dict(
        phase = phase,
        n_tokens_per_seq = n_tokens,
        batch = batch,
        total_tokens = total_tokens_generated,
        elapsed_s = elapsed,
        tokens_per_s = total_tokens_generated / elapsed,
        mean_watts = mean_w,
        tok_per_w = (total_tokens_generated / elapsed) / mean_w if mean_w > 0 else 0.0,
        token_seqs = token_list,           # full token grid for comparison
        sample_decoded = decoded,
    ), open(out_path, "w"), indent=2)
    print(f"[{phase}] {total_tokens_generated} toks in {elapsed:.1f}s "
          f"= {total_tokens_generated/elapsed:.1f} tok/s, mean_w={mean_w:.1f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
