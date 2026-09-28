# Acceptance-rate probe (2026-06-12): WHY does EAGLE flip negative on raw-continuation prompts?
# One engine instance per config; run the SAME engine on (a) raw varied prompts, (b) chat-templated
# varied prompts; report tok/s + vLLM SpecDecoding acceptance stats per prompt set.
# Acceptance source: vllm.v1.metrics SpecDecoding stats via the engine's stat loggers is awkward to
# reach in-process, so we use the documented logging path (disable_log_stats=False) AND compute a
# direct proxy: spec-on vs spec-off wall-clock needs two engines, so instead we read
# the aggregated spec metrics from the engine core after each batch via llm.llm_engine.
import os, time, json
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

QUANT = os.environ.get("S_QUANT", "")
if QUANT == "torchao":
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    _orig_from_config = TorchAOConfig.from_config.__func__
    def _patched_from_config(cls, config):
        obj = _orig_from_config(cls, config)
        obj.is_checkpoint_torchao_serialized = False
        return obj
    TorchAOConfig.from_config = classmethod(_patched_from_config)

from vllm import LLM, SamplingParams

def main():
    M = os.environ["S_MODEL"]; NREQ = int(os.environ.get("S_NREQ", "32")); OUT = os.environ["S_OUT"]
    DO_CHAT = os.environ.get("S_CHAT", "1") == "1"
    kw = dict(model=M, enforce_eager=False, gpu_memory_utilization=0.82, max_model_len=2048,
              disable_log_stats=False,
              speculative_config={"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":5})
    if QUANT: kw["quantization"] = QUANT
    llm = LLM(**kw)
    seeds = ["The history of the Roman Empire begins with",
             "Photosynthesis is the process by which plants",
             "In 1969, the Apollo program achieved",
             "The fundamental theorem of calculus states that",
             "Deep in the Amazon rainforest, researchers discovered"]
    raw = [seeds[i % len(seeds)] for i in range(NREQ)]
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, min_tokens=256)

    def spec_stats():
        # v1 engine: aggregated scheduler stats live on the output processor's stat loggers;
        # fall back to None if the attribute path moved.
        try:
            from vllm.v1.metrics.reader import Counter
            ms = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")}
            return {k: v for k, v in ms.items() if "spec" in k or "draft" in k or "accept" in k}
        except Exception as e:
            return {"err": str(e)[:120]}

    res = {"model": M, "quant_arg": QUANT or None, "nreq": NREQ,
           "plugins_env": os.environ.get("VLLM_PLUGINS", "UNSET"), "sets": {}}

    def run_set(name, prompts):
        llm.generate(prompts, sp, use_tqdm=False)        # warm
        s0 = spec_stats()
        t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
        s1 = spec_stats()
        g = sum(len(x.outputs[0].token_ids) for x in o)
        delta = {k: (s1.get(k, 0) - s0.get(k, 0)) for k in s1 if not isinstance(s1[k], str)}
        # acceptance rate = accepted / drafted over THIS set only
        acc = None
        dk = [k for k in delta if "draft" in k and "accept" not in k]
        ak = [k for k in delta if "accept" in k]
        if dk and ak and delta[dk[0]]:
            acc = round(delta[ak[0]] / delta[dk[0]], 4)
        res["sets"][name] = {"tok_s": round(g / dt, 1), "accept_rate": acc, "delta": delta,
                             "out": o[0].outputs[0].text[:60]}

    run_set("raw", raw)
    if DO_CHAT:
        tok = llm.get_tokenizer()
        chat = [tok.apply_chat_template([{"role":"user","content":"Tell me about: "+s}],
                                        tokenize=False, add_generation_prompt=True) for s in raw]
        run_set("chat", chat)
    json.dump(res, open(OUT, "w")); print("ACCEPT", json.dumps(res))

if __name__ == "__main__":
    main()
