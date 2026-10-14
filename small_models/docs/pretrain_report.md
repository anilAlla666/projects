# Pretraining report (8 Oct 2026)

**MAIN (116.5M parameters) was pretrained on 2.33B tokens in about 75 minutes on 4 H100s, at about 530k tokens/s.
Final held-out loss: 0.813 on code and 3.143 on English.**

| Setting | Value |
|---|---|
| Corpus | Python from 8,252 train-split, permissively licensed PyPI packages (926M tokens) and FineWeb-Edu (1.00B tokens); 0.5% of documents held out per source |
| Mix | 60% code, 40% English; code seen about 1.5 times |
| Tokenizer | byte-level BPE, 32,768 tokens, trained on 400 MB of each source |
| Batch | 4 GPUs x 32 x 2,048 = 262,144 tokens per step, 8,888 steps; the last 5% at 8,192-token sequences |
| Optimiser | Muon 0.02 for hidden matrices, AdamW 0.003 for the rest, warm-up 2%, cosine to 10% |
| Learning rates | chosen by a 150M-token check on DEV (table below) |

Learning-rate check (DEV, 150M tokens, final held-out loss):

| Muon / AdamW | Code | English |
|---|---:|---:|
| 0.01 / 0.003 | 1.387 | 4.130 |
| **0.02 / 0.003** | **1.379** | **4.122** |
| 0.04 / 0.003 | 1.414 | 4.169 |
| 0.02 / 0.001 | 1.502 | 4.344 |

MAIN held-out loss during pretraining:

| Step | Tokens (B) | Code | English |
|---:|---:|---:|---:|
| 500 | 0.13 | 1.335 | 4.082 |
| 1,000 | 0.26 | 1.144 | 3.772 |
| 1,500 | 0.39 | 1.075 | 3.616 |
| 2,000 | 0.52 | 1.028 | 3.541 |
| 2,500 | 0.66 | 1.011 | 3.450 |
| 3,000 | 0.79 | 1.003 | 3.424 |
| 3,500 | 0.92 | 0.887 | 3.394 |
| 4,000 | 1.05 | 0.962 | 3.335 |
| 4,500 | 1.18 | 0.912 | 3.344 |
| 5,000 | 1.31 | 0.892 | 3.284 |
| 5,500 | 1.44 | 0.877 | 3.259 |
| 6,000 | 1.57 | 0.851 | 3.240 |
| 6,500 | 1.70 | 0.822 | 3.211 |
| 7,000 | 1.84 | 0.866 | 3.172 |
| 7,500 | 1.97 | 0.801 | 3.153 |
| 8,000 | 2.10 | 0.899 | 3.160 |
| 8,500 | 2.23 | 0.811 | 3.195 |
| 8,888 | 2.33 | 0.812 | 3.143 |

Files: `experiments/pretrain/main_20261008T065216Z/` holds `log.jsonl`, `val.jsonl`, `config.toml` and `weights.pt`
(model only). The full checkpoints with optimiser state stay on the instance.
