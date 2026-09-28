# Archived 2:4 verification (panel fix): checks the checkpoint is genuinely 2:4 along the
# reduction (input) dim, with a wrong-axis control. CPU-only.
import glob, torch, json
from safetensors import safe_open
snap = glob.glob("/home/ubuntu/.cache/huggingface/hub/models--neuralmagic--Sparse-Llama-3.1-8B-2of4/snapshots/*/")[0]
out = {}
for fp in sorted(glob.glob(snap + "*.safetensors")):
    with safe_open(fp, framework="pt") as f:
        for n in f.keys():
            if not any(k in n for k in ("down_proj","gate_proj","q_proj","o_proj")): continue
            if not any(l in n for l in ("layers.0.","layers.15.","layers.31.")): continue
            w = f.get_tensor(n).float()
            indim = (w.reshape(-1,4) != 0).sum(1).le(2).float().mean().item()      # K-dim groups
            outdim = (w.t().contiguous().reshape(-1,4) != 0).sum(1).le(2).float().mean().item()  # control
            out[n] = {"shape": list(w.shape), "zero_frac": round((w==0).float().mean().item(),4),
                      "indim_2of4": round(indim,6), "outdim_control": round(outdim,4)}
print(json.dumps(out, indent=1))
