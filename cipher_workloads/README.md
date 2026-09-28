# cipher_workloads — Phase 4.0.9 harness

Binding workload launcher set for WL01–WL24. Built as the measurement substrate for Phase 4 cluster checkpoints (P4.2–P4.7) and P4.8 ship gate.

## Quick start

```bash
# One-time install (frameworks + models — ~30-40 min total)
./setup.sh

# Run a single workload for 10 minutes with a specific tenant ID
./run_wl.sh WL01 --duration 600 --tenant-id alpha

# Multi-tenant ×8 (WL05 is special — spawns 8 subprocess tenants)
WL_DURATION=600 ./drivers/wl05_multitenant_x8.sh

# Sample per-tenant MFU in parallel
./measurement/collect_mfu.sh alpha 600 > /tmp/alpha_mfu.csv

# Verify the run was clean
python3 measurement/verify_run.py WL01 alpha
```

## Directory layout

```
cipher_workloads/
├── README.md
├── setup.sh                       framework installs + model downloads
├── run_wl.sh                      top-level launcher
├── drivers/                       24 workload drivers
│   ├── wl01_decode_b1.py
│   ├── ...
│   └── wl24_awq.py
├── launch_lib/                    shared helpers
│   ├── tenant_register.py         sets CIPHER_TENANT_ID + CUDA_INJECTION64_PATH
│   ├── run_for_duration.py        timed loop + progress logging
│   └── mfu_compute.py             /metrics parsing + MFU calc
├── measurement/
│   ├── collect_mfu.sh             per-tenant MFU sampling
│   └── verify_run.py              post-run sanity check
├── expected/                      baseline MFU JSONs (populated during T4.0.9.6)
└── models/                        (symlinks/refs into ~/.cache/huggingface)
```

## Model substitution table (binding)

Gated models require an HF token. On Lambda pods without one, drivers fall back to open models. The substitution is documented per-driver and does not invalidate CIPHER lift measurement (we measure CIPHER's relative MFU lift, not absolute model performance).

| WL | Spec model | Open substitute (driver default) |
|---|---|---|
| WL01, WL02 | Llama-3.2-1B | **TinyLlama-1.1B-Chat-v1.0** |
| WL03 | Llama-3.1-8B | **Mistral-7B-v0.1** |
| WL04 | vLLM serving (any) | TinyLlama via vLLM |
| WL05 | Multi-tenant ×8 of WL01 | 8× TinyLlama (one driver per tenant) |
| WL06 | MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 (open, no substitution) |
| WL07 | LoRA fine-tune | TinyLlama + LoRA |
| WL08 | SDXL | stabilityai/stable-diffusion-xl-base-1.0 (no substitution) |
| WL09 | Whisper | openai/whisper-large-v3 (or env override) |
| WL10 | Speculative decode | vLLM (Mistral target + TinyLlama draft); falls back to vanilla if vLLM doesn't support it |
| WL11 | Agentic multi-turn | TinyLlama with synthetic turn loop |
| WL12 | Batch | TinyLlama B=64 |
| WL13 | Long context 32K | Mistral-7B (native 32K) |
| WL14 | torch.compile | TinyLlama under torch.compile |
| WL15 | MoE (Mixtral-class) | **TinyLlama proxy** — flagged in driver; not a true MoE measurement |
| WL16 | Prefix caching | vLLM with `enable_prefix_caching=True` + TinyLlama |
| WL17 | Training full | TinyLlama full-param (no LoRA) |
| WL18 | Multi-GPU TP | **DEFERRED on single-GPU pods**; driver detects and exits with `deferred-multi-gpu` |
| WL19 | CLIP | openai/clip-vit-base-patch32 |
| WL20 | LLaVA | llava-hf/llava-1.5-7b-hf |
| WL21 | Code gen | TinyLlama with code prompts |
| WL22 | RAG | MiniLM + TinyLlama 3-doc corpus |
| WL23 | Model switch | alternating TinyLlama / Mistral |
| WL24 | AWQ | TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ |

## Per-driver environment overrides

Each driver reads:
- `WL_DURATION` — seconds to run (default 600 = 10 min)
- `WL_MODEL` — Hugging Face repo override (driver-specific defaults)
- `CIPHER_TENANT_ID` — tenant identity (set by run_wl.sh)
- `CUDA_INJECTION64_PATH` — libcipher_v2 (set by run_wl.sh)

WL09 additionally honors `WL_WHISPER_SIZE` (default `base`).
WL13 honors `WL_CTX` (default 32768).
WL23 honors `WL_MODEL_A` / `WL_MODEL_B`.

## Adding a new workload

1. Create `drivers/wlNN_<name>.py`
2. Import `tenant_register`, `run_for_duration`
3. Define a `step()` returning `(ops_completed, tokens_or_units)`
4. Call `run_for_duration.run(step, DURATION, "WLNN", tenant, "unit_label")`
5. Add a case in `run_wl.sh`

## Baseline data

`expected/wlNN_p41_baseline.json` is populated during T4.0.9.6 (P4.1 baseline measurement pass). Each file contains:

```json
{
  "wl_id":      "WL01",
  "model":      "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "substitute": true,
  "duration_s": 600,
  "tenant_id":  "wl01_baseline",
  "mfu_pct":    {"min": ..., "p50": ..., "max": ...},
  "sm_util":    {...},
  "iters":      N,
  "tokens":     N,
  "kmod_version":    "0.4.0",
  "kmod_srcversion": "..."
}
```

P4.2–P4.7 measurements re-run each WL under the active cluster and compute lift relative to the P4.1 baseline.

## Known gaps in P4.0.9 deliverable

- **`setup.sh` not run yet.** All Python frameworks heavy (vllm, diffusers, whisper, peft, sentence-transformers, autoawq) need install; some models need download. Estimated 30–40 minutes wall-clock.
- **WL01–WL05, WL07, WL11, WL12, WL14, WL15, WL17, WL21, WL23 may work without setup.sh** if torch + transformers + TinyLlama/Mistral already cached. The driver imports will fail loudly if anything's missing.
- **WL18 deferred** on single-GPU pod (documented above).
- **T4.0.9.6 baseline measurement pass deferred** — not in P4.0.9 driver-write window. Will run as separate execution before P4.2 starts.
