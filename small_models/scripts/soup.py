"""Average the weights of checkpoints from one fine-tuning trajectory into one checkpoint (same architecture, same
start), so the result costs one forward pass per verdict.

    python scripts/soup.py OUT.pt CKPT [CKPT ...]
"""
import sys
from pathlib import Path

import torch


def main():
    out, paths = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    avg = {}
    for part in ("judge", "heads"):
        keys = states[0][part].keys()
        avg[part] = {k: sum(s[part][k].float() for s in states) / len(states) for k in keys}
        avg[part] = {k: v.to(states[0][part][k].dtype) for k, v in avg[part].items()}
    torch.save({**avg, "config": states[0]["config"], "soup_of": [str(p) for p in paths]}, out)
    print("wrote", out, "from", len(paths))


if __name__ == "__main__":
    main()
