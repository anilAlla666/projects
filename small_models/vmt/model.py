"""PyTorch twin of vm/model/model.py. Embeddings are initialised with std d_model**-0.5, as MLX does."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from vm.model.config import ModelConfig
from vmt.layers import KDA, Attention, SwiGLU, ZRMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attention: bool):
        super().__init__()
        self.norm1 = ZRMSNorm(cfg.d_model)
        self.mixer = (Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) if attention
                      else KDA(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.chunk, cfg.gate_lower))
        self.norm2 = ZRMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Judge(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.segment = nn.Embedding(cfg.n_segments, cfg.d_model)
        for e in (self.embed, self.segment):
            nn.init.normal_(e.weight, std=cfg.d_model**-0.5)
        self.layers = nn.ModuleList(Block(cfg, cfg.is_attention(i)) for i in range(cfg.n_layers))
        self.norm = ZRMSNorm(cfg.d_model)
        self.verdict_head = nn.Linear(cfg.d_model, 1)
        self.pointer_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, tokens, segments):
        x = self.embed(tokens) + self.segment(segments)
        for block in self.layers:
            x = block(x)
        return self.norm(x)

    def lm_weight(self):
        return self.embed.weight

    def lm_logits(self, hidden):
        return hidden @ self.embed.weight.to(hidden.dtype).T

    def _at(self, hidden, pos):
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]

    def verdict_logit(self, hidden, pos):
        return self.verdict_head(self._at(hidden, pos))[:, 0]

    def pointer_logits(self, hidden, pos, line_mask):
        query = self.pointer_query(self._at(hidden, pos))
        scores = (hidden @ query[:, :, None])[..., 0] / math.sqrt(self.cfg.d_model)
        return scores.masked_fill(~line_mask, -1e9)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
