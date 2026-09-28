#!/usr/bin/env python3
# F1 audit — dump Mistral-7B layer-0 q_proj weight to raw FP16 (K x N row-major).
# The bytes are fed verbatim to BOTH Marlin and the cuBLAS reference, so the
# PyTorch (out,in) labelling is irrelevant — it is just a 4096x4096 FP16 matrix.
import sys, json, torch
from safetensors import safe_open

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
KEY   = "model.layers.0.self_attn.q_proj.weight"
out   = sys.argv[1]

idx   = json.load(open(MODEL + "/model.safetensors.index.json"))
shard = idx["weight_map"][KEY]
with safe_open(MODEL + "/" + shard, framework="pt") as f:
    t = f.get_tensor(KEY)
print("key:", KEY, "| dtype on disk:", t.dtype, "| shape:", tuple(t.shape))
a = t.to(torch.float16).contiguous().cpu().numpy()
a.tofile(out)
print("wrote", out, "|", a.shape, a.dtype, "|", a.nbytes, "bytes")
