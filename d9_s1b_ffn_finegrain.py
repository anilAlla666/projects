"""
D.9 §1b confound-closer: FFN-only FP8 with PRODUCTION granularity
(per-CHANNEL weight scales + per-TOKEN activation scales) vs the crude
per-TENSOR absmax used in the M3 measurement.

Gate: full-forward token-identity over diverse prompts, greedy + sampled.
This tests the advisor's claim that FFN (81% of GEMM time, the class we
NEED safe to reach 85%) will NOT cross 100% token-identity even at
production granularity, because down-proj activation outliers are
per-channel and per-token scaling does not remove them.

If FFN clears 100% -> flips toward PARTIAL/REACHABLE.
If FFN still declines -> confound closed, CEILING hardens to high confidence.
"""
import torch, torch.nn as nn, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
DEV = "cuda"
FP8 = torch.float8_e4m3fn
FMAX = 448.0  # E4M3 max representable

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEV)
model.eval()

# 50 diverse prompts (code, prose, math, multilingual, factual)
PROMPTS = [
 "The capital of France is", "def fibonacci(n):", "In 1492, Columbus",
 "The mitochondria is the", "To be or not to be,", "import numpy as np",
 "The derivative of x^2 is", "Once upon a time", "The speed of light is",
 "SELECT * FROM users WHERE", "El gato esta en", "The boiling point of water",
 "class Animal:", "Newton's second law states", "The square root of 144 is",
 "def quicksort(arr):", "The French Revolution began in", "for i in range(10):",
 "Photosynthesis converts", "The largest planet is", "git commit -m",
 "The chemical symbol for gold is", "while True:", "The Pythagorean theorem",
 "Machine learning is", "public static void main", "The human body has",
 "import torch.nn as", "The Great Wall of China", "lambda x: x +",
 "The theory of relativity", "async def fetch(", "Water is composed of",
 "The first president of", "try:\n    result =", "DNA stands for",
 "The Fibonacci sequence", "if __name__ ==", "The currency of Japan is",
 "Quantum mechanics describes", "return sorted(", "The tallest mountain is",
 "A binary tree is", "The periodic table has", "console.log(",
 "Gravity causes objects to", "The internet was invented", "x = [i**2 for",
 "The human genome contains", "Neural networks consist of",
]
GEN_TOK = 24

def quantize_per_token_act(x):
    # x: [tokens, K] fp16. per-TOKEN (row) scale.
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / FMAX                      # [tokens,1]
    xq = (x / scale).clamp(-FMAX, FMAX).to(FP8)
    return xq, scale

def quantize_per_channel_w(w):
    # w stored as nn.Linear.weight: [out, in]. _scaled_mm wants B=[K,N] col-major.
    # We quantize per OUTPUT channel (per-row of weight) = per-N column of B.
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)  # [out,1]
    scale = amax / FMAX
    wq = (w / scale).clamp(-FMAX, FMAX).to(FP8)
    return wq, scale  # wq [out,in], scale [out,1]

class FP8LinearFineGrain(nn.Module):
    """Replaces an nn.Linear; per-channel weight + per-token activation FP8."""
    def __init__(self, lin):
        super().__init__()
        self.out_f, self.in_f = lin.out_features, lin.in_features
        wq, wscale = quantize_per_channel_w(lin.weight.data.float())  # wq [out,in], wscale [out,1]
        # _scaled_mm RowWise wants B = [K, N] = [in, out] in fp8, column-major.
        # wq is [out,in] row-major; wq.t() is [in,out] column-major == exactly that layout.
        self.register_buffer("B", wq.t())  # logical [in,out], fp8, column-major (B-native)
        # scale_b must be (1, N) float32 contiguous
        self.register_buffer("scale_b", wscale.reshape(1, -1).to(torch.float32).contiguous())  # [1,out]
        self.bias = lin.bias

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_f)
        xq, xscale = quantize_per_token_act(x2.float())  # xq [T,K], xscale [T,1] fp8/float
        out = torch._scaled_mm(
            xq.contiguous(), self.B,
            scale_a=xscale.to(torch.float32).contiguous(),  # [T,1]
            scale_b=self.scale_b,                            # [1,out]
            out_dtype=torch.float16,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_f)

def get_ffn_linears(m):
    names = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and any(k in name for k in ("gate_proj","up_proj","down_proj")):
            names.append(name)
    return names

def set_module(root, dotted, new):
    parts = dotted.split("."); obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    setattr(obj, parts[-1], new)

@torch.no_grad()
def gen(model, prompt, sample):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=GEN_TOK, do_sample=sample,
                         temperature=1.0 if sample else None,
                         top_p=1.0 if sample else None,
                         top_k=0 if sample else None,
                         pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()

# reference (fp16)
torch.manual_seed(123)
ref_greedy = [gen(model, p, False) for p in PROMPTS]
torch.manual_seed(123)
ref_sampled = [gen(model, p, True) for p in PROMPTS]
print(f"[ref] captured {len(PROMPTS)} prompts x{GEN_TOK} tok", flush=True)

# swap FFN linears -> fine-grain FP8
ffn = get_ffn_linears(model)
print(f"[swap] {len(ffn)} FFN linears -> per-channel-w + per-token-act FP8", flush=True)
import copy
orig = {}
for n in ffn:
    parts = n.split("."); obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    lin = getattr(obj, parts[-1]); orig[n] = lin
    # NOTE: do NOT call .half() here — it would upcast the fp8 B buffer and the
    # float32 scale buffers, breaking _scaled_mm. Buffers are already on DEV.
    set_module(model, n, FP8LinearFineGrain(lin).to(DEV))

torch.manual_seed(123)
g_match = sum(gen(model, p, False) == ref_greedy[i] for i,p in enumerate(PROMPTS))
torch.manual_seed(123)
s_match = sum(gen(model, p, True) == ref_sampled[i] for i,p in enumerate(PROMPTS))

gp = 100.0*g_match/len(PROMPTS); sp = 100.0*s_match/len(PROMPTS)
safe = (gp==100.0 and sp==100.0)
print(f"\n[FFN fine-grain per-chan-w + per-tok-act] greedy {gp:.1f}% / sampled {sp:.1f}% token-identical -> {'FP8-SAFE' if safe else 'DECLINE'}", flush=True)
print(f"CRUDE per-tensor baseline (M3) was: greedy 66.0% / sampled 64.0%", flush=True)
import json
json.dump({"granularity":"per_channel_weight+per_token_activation","class":"FFN","n_layers":len(ffn),
           "greedy_pct":gp,"sampled_pct":sp,"FP8_SAFE":safe,
           "crude_per_tensor_greedy":66.0,"crude_per_tensor_sampled":64.0},
          open("/home/ubuntu/d9_s1b_ffn_finegrain.json","w"), indent=2)
print("WROTE d9_s1b_ffn_finegrain.json", flush=True)
