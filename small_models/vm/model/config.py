from __future__ import annotations

from dataclasses import dataclass

SEGMENTS = {"task": 0, "diff": 1, "evidence": 2, "untrusted": 3}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden: int = 2048
    attn_period: int = 4  # every attn_period-th layer is full attention: three delta layers to one
    chunk: int = 32
    gate_lower: float = -1.0
    n_segments: int = 4

    def is_attention(self, i: int) -> bool:
        return (i + 1) % self.attn_period == 0


MAIN = ModelConfig()
DEV = ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, ffn_hidden=1024)
MID = ModelConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, ffn_hidden=1408)
TINY = ModelConfig(vocab_size=512, d_model=64, n_layers=4, n_heads=1, n_kv_heads=1, head_dim=64, ffn_hidden=128)
