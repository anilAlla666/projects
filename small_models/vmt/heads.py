import math

import torch
import torch.nn as nn


class AuxHeads(nn.Module):
    """Two auxiliary targets read from the verdict position: how many hidden tests fail (4 buckets), and which
    line of the task's function holds the planted bug (a pointer over the function's line starts)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bucket = nn.Linear(d_model, 4)
        self.bug_query = nn.Linear(d_model, d_model, bias=False)
        self.d = d_model

    def _at(self, hidden, pos):
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]

    def bucket_logits(self, hidden, pos):
        return self.bucket(self._at(hidden, pos))

    def bug_logits(self, hidden, pos, code_mask):
        q = self.bug_query(self._at(hidden, pos))
        return ((hidden @ q[:, :, None])[..., 0] / math.sqrt(self.d)).masked_fill(~code_mask, -1e9)
