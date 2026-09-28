#!/usr/bin/env python3
"""Phase 3 — graph capture child. Single tenant, Llama-3.2-1B fp16,
torch.cuda.graph for the decode loop. Exercises GRAPH_ENGINE counter via
the cipher_graph_inspect intercept of cudaGraphInstantiate, plus LOOP
(period detection) and PIPELINE (graph capture coordination)."""
import os, sys, time, json, ctypes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.normpath(os.path.join(ROOT, "..", "libcipher_rt.so"))


def main():
    duration  = float(os.environ.get("PHASE3_DURATION_S", "120"))
    out_path  = os.environ["PHASE3_OUT_PATH"]
    model_path = os.environ.get("PHASE3_MODEL_PATH",
                                 "/home/ubuntu/models/Llama-3.2-1B")
    load_rt   = os.environ.get("MT_LOAD_RT", "1") != "0"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    if load_rt and os.path.exists(RT_PATH):
        try:
            ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)
        except Exception as ex:
            print(f"CIPHER rt load failed: {ex}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    for p in model.parameters(): p.requires_grad_(False)

    PROMPT_LEN  = 64
    DECODE_LEN  = 16
    MAX_LEN     = PROMPT_LEN + DECODE_LEN + 4
    ids = tok("Energy efficiency means doing more useful work per watt. ",
              return_tensors="pt").input_ids[0]
    ids = ids.repeat(PROMPT_LEN // len(ids) + 1)[:PROMPT_LEN].unsqueeze(0).to("cuda:0")

    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
    cp = torch.arange(PROMPT_LEN, device="cuda:0", dtype=torch.long)
    with torch.no_grad():
        o = model(input_ids=ids, cache_position=cp,
                  past_key_values=cache, use_cache=True, return_dict=True)
        input_ids = o.logits[:, -1:].argmax(-1).clone()
        cache_pos = torch.tensor([PROMPT_LEN], device="cuda:0", dtype=torch.long)

        # Warmup the decode step on a side stream before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
                input_ids.copy_(o.logits.argmax(-1))
                cache_pos += 1
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture the decode step into a CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
            input_ids.copy_(o.logits.argmax(-1))

        # Drive replay
        n_replay = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            g.replay()
            n_replay += 1
            if n_replay % 256 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    json.dump(dict(replays=n_replay, elapsed_s=elapsed,
                   replays_per_s=n_replay/elapsed if elapsed>0 else 0),
              open(out_path, "w"), indent=2)
    print(f"graph replays: {n_replay} in {elapsed:.1f}s = {n_replay/elapsed:.1f}/s")


if __name__ == "__main__":
    main()
