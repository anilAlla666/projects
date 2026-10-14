"""The judge: a hybrid stack of Kimi Delta Attention and full attention with segment embeddings,
a tied language-model head (pretraining), a verdict head and an evidence-pointer head."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vm.model.config import ModelConfig
from vm.model.layers import KDA, Attention, SwiGLU, ZRMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attention: bool):
        super().__init__()
        self.norm1 = ZRMSNorm(cfg.d_model)
        self.mixer = (Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) if attention
                      else KDA(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.chunk, cfg.gate_lower))
        self.norm2 = ZRMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def __call__(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Judge(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.segment = nn.Embedding(cfg.n_segments, cfg.d_model)
        self.layers = [Block(cfg, cfg.is_attention(i)) for i in range(cfg.n_layers)]
        self.norm = ZRMSNorm(cfg.d_model)
        self.verdict_head = nn.Linear(cfg.d_model, 1)
        self.pointer_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, tokens, segments):
        x = self.embed(tokens) + self.segment(segments)
        for block in self.layers:
            x = block(x)
        return self.norm(x)

    def lm_logits(self, hidden):
        return self.embed.as_linear(hidden)

    def _at(self, hidden, pos):
        return hidden[mx.arange(hidden.shape[0]), pos]

    def verdict_logit(self, hidden, pos):
        return self.verdict_head(self._at(hidden, pos))[:, 0]

    def pointer_logits(self, hidden, pos, line_mask):
        query = self.pointer_query(self._at(hidden, pos))  # [B, d]
        scores = (hidden @ query[:, :, None])[..., 0] / math.sqrt(self.cfg.d_model)
        return mx.where(line_mask, scores, -1e9)


def n_params(module: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(module.parameters()))
