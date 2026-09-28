# WikiText-2 PPL gate for the torchao FP8+2:4 route (prereg P7, 2026-06-12).
# Teacher-forced via prompt_logprobs on IDENTICAL disjoint token chunks across all arms.
# Chunks are built with the meta-llama/Llama-3.1-8B tokenizer for EVERY arm (all four arms share
# the Llama-3.1 tokenizer) and fed as prompt_token_ids, so the scored token stream is bit-identical.
# torchao arm: same __main__ guard + online-quantize monkeypatch as the throughput harness.
import os, json, math
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

QUANT = os.environ.get("Q_QUANT", "")  # "", "fp8", "torchao"
if QUANT == "torchao":
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    _orig_from_config = TorchAOConfig.from_config.__func__
    def _patched_from_config(cls, config):
        obj = _orig_from_config(cls, config)
        obj.is_checkpoint_torchao_serialized = False
        return obj
    TorchAOConfig.from_config = classmethod(_patched_from_config)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CHUNK = 1024
NCHUNK = 32  # 32 x 1023 scored tokens ~ 32.7k

def main():
    M = os.environ["Q_MODEL"]; OUT = os.environ["Q_OUT"]
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tok(text, add_special_tokens=False).input_ids
    chunks = [ids[i*CHUNK:(i+1)*CHUNK] for i in range(NCHUNK)]
    assert all(len(c) == CHUNK for c in chunks), "ran out of text"

    kw = dict(model=M, enforce_eager=os.environ.get("Q_EAGER","0")=="1",
              gpu_memory_utilization=0.82, max_model_len=2048, disable_log_stats=True)
    if QUANT: kw["quantization"] = QUANT
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=c) for c in chunks], sp, use_tqdm=False)
    nll = 0.0; n = 0
    for o in outs:
        for d in o.prompt_logprobs[1:]:  # first position has no context
            if d:
                nll += -list(d.values())[0].logprob; n += 1
    ppl = math.exp(nll / n)
    res = {"model": M, "quant_arg": QUANT or None, "eager": kw["enforce_eager"],
           "plugins_env": os.environ.get("VLLM_PLUGINS", "UNSET"),
           "ppl": round(ppl, 4), "n_tokens": n, "chunk": CHUNK, "nchunk": NCHUNK,
           "quant": str(getattr(llm.llm_engine.vllm_config.model_config, 'quantization', None))}
    json.dump(res, open(OUT, "w")); print("PPL", json.dumps(res))

if __name__ == "__main__":
    main()
