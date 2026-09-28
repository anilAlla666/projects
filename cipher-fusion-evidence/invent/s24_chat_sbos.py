# SINGLE-BOS chat re-measurement (panel fix, 2026-06-12). All prior "chat" numbers (06-11 chat
# harness 8355/6611, 06-12 probe chat point) were DOUBLE-BOS: apply_chat_template(tokenize=False)
# emits bos, then vLLM's text path re-adds one (renderers add_special_tokens=True). Here we build
# token ids with apply_chat_template(tokenize=True) — single BOS — and pass TokensPrompt, so the
# drafter sees its true training encoding. S_SPEC=1 also reports v1 spec counters.
import os, time, json
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def main():
    M = os.environ["S_MODEL"]; NREQ = int(os.environ.get("S_NREQ", "32")); OUT = os.environ["S_OUT"]
    SPEC = os.environ.get("S_SPEC", "0") == "1"
    kw = dict(model=M, quantization="fp8", enforce_eager=False, gpu_memory_utilization=0.82,
              max_model_len=2048, disable_log_stats=not SPEC)
    if SPEC:
        kw["speculative_config"] = {"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":5}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    seeds = ["The history of the Roman Empire begins with",
             "Photosynthesis is the process by which plants",
             "In 1969, the Apollo program achieved",
             "The fundamental theorem of calculus states that",
             "Deep in the Amazon rainforest, researchers discovered"]
    msgs = [[{"role":"user","content":"Tell me about: "+seeds[i % len(seeds)]}] for i in range(NREQ)]
    def to_ids(m):
        out = tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)
        if not isinstance(out, list): out = out["input_ids"]
        if out and isinstance(out[0], list): out = out[0]
        return out
    idlists = [to_ids(m) for m in msgs]
    assert all(ids.count(tok.bos_token_id) == 1 for ids in idlists), "expected exactly one BOS"
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in idlists]
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, min_tokens=256)

    def spec_stats():
        try:
            ms = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")}
            return {k: v for k, v in ms.items() if "spec" in k}
        except Exception as e:
            return {}

    llm.generate(prompts, sp, use_tqdm=False)  # warm
    best = None; acc = None
    for _ in range(3):
        s0 = spec_stats()
        t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
        s1 = spec_stats()
        g = sum(len(x.outputs[0].token_ids) for x in o); tps = g / dt
        if best is None or tps > best:
            best = tps
            if SPEC and s1:
                dr = s1.get("vllm:spec_decode_num_drafts", 0) - s0.get("vllm:spec_decode_num_drafts", 0)
                at = s1.get("vllm:spec_decode_num_accepted_tokens", 0) - s0.get("vllm:spec_decode_num_accepted_tokens", 0)
                acc = round(at / dr, 4) if dr else None
    res = {"model": M, "spec": SPEC, "nreq": NREQ, "bos": "single",
           "plugins_env": os.environ.get("VLLM_PLUGINS", "UNSET"),
           "tok_s": round(best, 1), "accepted_per_step": acc,
           "out": o[0].outputs[0].text[:80]}
    json.dump(res, open(OUT, "w")); print("SBOS", json.dumps(res))

if __name__ == "__main__":
    main()
