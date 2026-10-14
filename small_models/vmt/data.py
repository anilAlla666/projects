"""Token streams (memory-mapped uint16) and a seeded sampler that mixes sources by weight. Each sequence is a
random window of one source; documents inside it are separated by <|eos|> (id 0)."""
from __future__ import annotations

import numpy as np
import torch


class TokenStream:
    def __init__(self, path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.tokens)


class MixedSampler:
    def __init__(self, streams: dict, weights: dict, seed: int, rank: int):
        self.names = sorted(streams)
        self.streams = streams
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng([seed, rank])

    def batch(self, batch_size: int, seq: int) -> torch.Tensor:
        src = self.rng.choice(len(self.names), size=batch_size, p=self.p)
        out = np.empty((batch_size, seq + 1), dtype=np.int64)
        for i, s in enumerate(src):
            toks = self.streams[self.names[s]].tokens
            start = self.rng.integers(0, len(toks) - seq - 1)
            out[i] = toks[start : start + seq + 1]
        return torch.from_numpy(out)
