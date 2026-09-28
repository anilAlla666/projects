# MMLU 5-shot adjudication arm (2026-06-12, ADDITIONAL post-prereg evidence, labeled as such).
# The locked PPL gate FAILED on Qb/Qc=1.26 (pruning cost), but NM's recovery claims are
# downstream-task claims. This measures task accuracy: dense base vs sparse bf16 vs sparse-G2.
# 500 deterministic test questions (seed 0) across all subjects, 5-shot from dev split,
# prediction = argmax over logprobs of " A"/" B"/" C"/" D" on one greedy token.
import os, json, random
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

QUANT = os.environ.get("Q_QUANT", "")
if QUANT == "torchao":
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    _orig_from_config = TorchAOConfig.from_config.__func__
    def _patched_from_config(cls, config):
        obj = _orig_from_config(cls, config)
        obj.is_checkpoint_torchao_serialized = False
        return obj
    TorchAOConfig.from_config = classmethod(_patched_from_config)

from vllm import LLM, SamplingParams

LETTERS = ["A", "B", "C", "D"]

def fmt_q(ex, with_answer):
    s = ex["question"].strip() + "\n"
    for i, c in enumerate(ex["choices"]):
        s += f"{LETTERS[i]}. {c}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[ex['answer']]}\n\n"
    return s

def main():
    M = os.environ["Q_MODEL"]; OUT = os.environ["Q_OUT"]; N = int(os.environ.get("Q_N", "500"))
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    rng = random.Random(0)
    idx = rng.sample(range(len(test)), N)
    dev_by_subj = {}
    for ex in dev:
        dev_by_subj.setdefault(ex["subject"], []).append(ex)
    prompts, golds = [], []
    for i in idx:
        ex = test[i]
        shots = dev_by_subj.get(ex["subject"], [])[:5]
        head = f"The following are multiple choice questions (with answers) about {ex['subject'].replace('_',' ')}.\n\n"
        prompts.append(head + "".join(fmt_q(s, True) for s in shots) + fmt_q(ex, False))
        golds.append(ex["answer"])
    kw = dict(model=M, enforce_eager=False, gpu_memory_utilization=0.82, max_model_len=4096,
              disable_log_stats=True)
    if QUANT: kw["quantization"] = QUANT
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    # drop prompts that don't fit (deterministic — identical set across arms); report the count
    keep = [i for i, p in enumerate(prompts)
            if len(tok.encode(p)) <= kw["max_model_len"] - 8]
    dropped = len(prompts) - len(keep)
    prompts = [prompts[i] for i in keep]; golds = [golds[i] for i in keep]
    letter_ids = {L: tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS}
    sp = SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    correct = 0; scored = 0; no_letter = 0; bits = []
    for o, g in zip(outs, golds):
        lp = o.outputs[0].logprobs[0]
        cand = {L: lp[tid].logprob for L, tid in letter_ids.items() if tid in lp}
        if not cand: no_letter += 1; bits.append(-1); continue
        pred = max(cand, key=cand.get)
        ok = int(pred == LETTERS[g])
        scored += 1; correct += ok; bits.append(ok)
    acc = correct / scored if scored else None
    res = {"model": M, "quant_arg": QUANT or None, "n": N, "dropped_overlen": dropped,
           "scored": scored, "no_letter": no_letter, "acc": round(acc, 4) if acc else None,
           "bits": bits,
           "plugins_env": os.environ.get("VLLM_PLUGINS", "UNSET"),
           "quant": str(getattr(llm.llm_engine.vllm_config.model_config, 'quantization', None))}
    json.dump(res, open(OUT, "w")); print("MMLU", json.dumps(res))

if __name__ == "__main__":
    main()
