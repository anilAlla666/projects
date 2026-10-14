"""Export a judge checkpoint for delivery: weights in bfloat16 (half the size), plus the calibration it was chosen
with, so the file alone reproduces the reported verdicts.

    python scripts/export_bf16.py CKPT CALIB.json THRESHOLD OUT.pt
"""
import json
import sys
from pathlib import Path

import torch

ckpt, calib, th, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
s = torch.load(ckpt, map_location="cpu", weights_only=True)
bf = {part: {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in s[part].items()} for part in ("judge", "heads")}
torch.save({**bf, "config": s["config"], "calibration": json.loads(calib.read_text()), "threshold": th, "source": str(ckpt)}, out)
print(out, round(out.stat().st_size / 1e6, 1), "MB")
