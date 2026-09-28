"""Workload driver registry — M1.T9 24-workload validation matrix.

Each driver module exposes:
    NAME       — short canonical id (e.g. "LLM_DECODE_SINGLE")
    PROMPT_KEY — prompts.jsonl category to feed it
    DEFAULT_MODEL — which local model the driver targets by default
    EXPECTED_OPS — set of CIPHER op-ids that should fire when CIPHER is on
    run(model_path, prompts, phase_label, out_dir, **kwargs) -> dict
        Runs the workload, returns the summary dict (matches score.aggregate).

The registry below is also the pod-local coverage map: status ∈
    "runnable"  — local model is present, framework installed
    "blocked"   — needs framework/model not present on this pod
    "gap"       — workload class needs research before driver exists
"""
from dataclasses import dataclass


@dataclass
class WorkloadEntry:
    id:           str
    name:         str
    driver_mod:   str | None     # e.g. "wl01_decode_single"; None = no driver yet
    default_model: str
    framework:    str            # transformers, vllm, sglang, trt-llm, diffusers, peft, autoawq
    status:       str            # runnable / blocked / gap
    note:         str = ""


# Pod-local snapshot. `status` reflects what /home/ubuntu/op31-prod-fix
# can run today — see make_inventory.py to refresh.
WORKLOADS: list[WorkloadEntry] = [
    WorkloadEntry("wl01", "LLM_DECODE_SINGLE",     "wl01_decode_single",
                  "unsloth/Llama-3.2-1B",          "transformers", "runnable"),
    WorkloadEntry("wl02", "LLM_DECODE_BATCHED",    None,
                  "mistralai/Mistral-7B-v0.1",     "vllm",        "blocked",
                  "vLLM not installed in pod"),
    WorkloadEntry("wl03", "LLM_PREFILL",           None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "driver not yet written"),
    WorkloadEntry("wl04", "LLM_SERVING_VLLM",      None,
                  "mistralai/Mistral-7B-v0.1",     "vllm",        "blocked",
                  "vLLM serving requires vLLM install"),
    WorkloadEntry("wl05", "MULTI_TENANT",          "wl05_multi_tenant",
                  "unsloth/Llama-3.2-1B",          "transformers", "runnable"),
    WorkloadEntry("wl06", "EMBEDDINGS",            None,
                  "sentence-transformers/all-MiniLM-L6-v2",
                  "sentence-transformers", "runnable",
                  "driver not yet written"),
    WorkloadEntry("wl07", "LORA_FINETUNING",       None,
                  "unsloth/Llama-3.2-1B",          "peft",        "blocked",
                  "PEFT not installed"),
    WorkloadEntry("wl08", "DIFFUSION",             None,
                  "stabilityai/sdxl-turbo",        "diffusers",   "runnable",
                  "driver not yet written"),
    WorkloadEntry("wl09", "SPEECH",                None,
                  "Systran/faster-whisper-small",  "faster-whisper", "runnable",
                  "driver not yet written"),
    WorkloadEntry("wl10", "SPECULATIVE_DECODING",  None,
                  "mistralai/Mistral-7B-v0.1",     "vllm",        "blocked"),
    WorkloadEntry("wl11", "AGENTIC",               None,
                  "unsloth/Llama-3.2-1B",          "transformers", "gap",
                  "needs tool-calling harness"),
    WorkloadEntry("wl12", "BATCH_PROCESSING",      None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "driver not yet written"),
    WorkloadEntry("wl13", "LONG_CONTEXT",          None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "32K context test driver not yet written"),
    WorkloadEntry("wl14", "TORCH_COMPILE",         None,
                  "unsloth/Llama-3.2-1B",          "transformers", "runnable",
                  "torch.compile + Inductor driver pending"),
    WorkloadEntry("wl15", "MOE_MODELS",            None,
                  "mistralai/Mixtral-8x22B-v0.1",  "vllm",        "blocked",
                  "Mixtral-8x22B is 280GB; needs quantized variant"),
    WorkloadEntry("wl16", "PREFIX_CACHING",        None,
                  "mistralai/Mistral-7B-v0.1",     "vllm",        "blocked"),
    WorkloadEntry("wl17", "TRAINING_FULL",         None,
                  "unsloth/Llama-3.2-1B",          "deepspeed",   "blocked",
                  "DeepSpeed not installed"),
    WorkloadEntry("wl18", "MULTI_GPU_TP",          None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "TP=2 only on this pod (2× H100); driver pending"),
    WorkloadEntry("wl19", "VISION",                None,
                  "openai/clip-vit-base-patch32",  "transformers", "blocked",
                  "no vision model in HF cache"),
    WorkloadEntry("wl20", "MULTIMODAL",            None,
                  "llava-hf/llava-1.5-7b-hf",      "transformers", "blocked",
                  "no multimodal model cached"),
    WorkloadEntry("wl21", "CODE_GENERATION",       None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "driver pending"),
    WorkloadEntry("wl22", "RAG_PIPELINE",          None,
                  "mistralai/Mistral-7B-v0.1",     "transformers", "runnable",
                  "driver pending"),
    WorkloadEntry("wl23", "MODEL_SWITCH",          None,
                  "unsloth/Llama-3.2-1B",          "transformers", "runnable",
                  "driver pending"),
    WorkloadEntry("wl24", "QUANTIZED_NATIVE",      None,
                  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                  "autoawq", "blocked", "AutoAWQ not installed"),
]


def by_id(wid: str) -> WorkloadEntry:
    for w in WORKLOADS:
        if w.id == wid or w.name == wid:
            return w
    raise KeyError(f"unknown workload {wid}")


def runnable_ids() -> list[str]:
    return [w.id for w in WORKLOADS if w.status == "runnable" and w.driver_mod]
