#!/usr/bin/env python3
# R.A E2E product test — shared model/forward/detector machinery.
# READ-ONLY w.r.t. the production .so; this is a standalone PyTorch measurement harness,
# NOT injected with the CIPHER runtime (no LD_PRELOAD / CUDA_INJECTION64_PATH).
# Real Mistral-7B-v0.1 weights, fp16 GEMMs (matches the settled Step-A fault model:
# fp16 top-exponent bit flips, |delta|>=2.76). Attention/KV deliberately EXCLUDED from the
# cost loop (weight-streaming linears only) — the detector validates linear GEMM outputs;
# attention-internal QK^T/PV is the stated blind spot. Omitting attention shrinks the step
# denominator => OVERSTATES detector overhead % => conservative for the throughput headline.
import os, glob, json, statistics, time, ctypes
import torch
import torch.nn.functional as F

DEV = 'cuda'
H, I, KV, V, L = 4096, 14336, 1024, 32000, 32
MIST = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/'))[0]

# The 7 distinct linear GEMM shapes per layer + lm_head. (out_features, in_features).
# F.linear(x[*,in], W[out,in]) -> y[*,out].
LIN_SHAPES = {
    'q':  (H,  H),   'k':  (KV, H),  'v':  (KV, H),  'o':  (H,  H),
    'gate': (I, H),  'up': (I, H),   'down': (H, I),
}
LM_SHAPE = (V, H)


def load_real_weights():
    """Load real Mistral-7B-v0.1 linear weights from safetensors, cast to fp16 on GPU."""
    from safetensors.torch import load_file
    shards = sorted(glob.glob(MIST + '*.safetensors'))
    sd = {}
    for s in shards:
        sd.update(load_file(s, device='cpu'))
    def g(k):
        return sd[k].to(DEV, dtype=torch.float16, non_blocking=True).contiguous()
    W = {'q': [], 'k': [], 'v': [], 'o': [], 'gate': [], 'up': [], 'down': []}
    for i in range(L):
        p = f'model.layers.{i}.'
        W['q'].append(g(p + 'self_attn.q_proj.weight'))
        W['k'].append(g(p + 'self_attn.k_proj.weight'))
        W['v'].append(g(p + 'self_attn.v_proj.weight'))
        W['o'].append(g(p + 'self_attn.o_proj.weight'))
        W['gate'].append(g(p + 'mlp.gate_proj.weight'))
        W['up'].append(g(p + 'mlp.up_proj.weight'))
        W['down'].append(g(p + 'mlp.down_proj.weight'))
    Wlm = g('lm_head.weight')
    torch.cuda.synchronize()
    return W, Wlm


def step_linears(x0, W, Wlm, cache=None):
    """One weight-streaming step over all 32 layers' linears + lm_head at batch/M = x0.shape[0].
    Mirrors the real per-step GEMM byte traffic (reads ~14.3GB of weights). RMSNorm/SiLU keep
    activations in realistic fp16 magnitude. Attention is replaced by an identity pass-through
    on the o_proj input (cost-excluded, conservative; see module docstring).
    If cache is a dict, every linear output is stored into it keyed by (layer, name) for the
    detector to compare against. Returns the lm_head logits."""
    x = x0
    eps = 1e-5
    def rms(t):
        return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps).half()
    for i in range(L):
        h = rms(x)
        q = F.linear(h, W['q'][i]); k = F.linear(h, W['k'][i]); v = F.linear(h, W['v'][i])
        # attention excluded: feed q straight into o_proj (shape-correct, cost = o_proj GEMM)
        o = F.linear(q, W['o'][i])
        x = x + o
        h2 = rms(x)
        gt = F.linear(h2, W['gate'][i]); up = F.linear(h2, W['up'][i])
        inter = F.silu(gt) * up
        dn = F.linear(inter, W['down'][i])
        x = x + dn
        if cache is not None:
            cache[(i, 'q')] = q; cache[(i, 'k')] = k; cache[(i, 'v')] = v; cache[(i, 'o')] = o
            cache[(i, 'gate')] = gt; cache[(i, 'up')] = up; cache[(i, 'down')] = dn
    lg = F.linear(rms(x), Wlm)
    if cache is not None:
        cache[(-1, 'lm')] = lg
    return lg


def recompute_residual(x0, W, Wlm, cached):
    """Detector: re-run every linear with a SECOND independent cuBLAS call (same X/W/shape/dtype)
    and return the max |recompute - cached| over all GEMM outputs. fp32 abs-diff. Same-algo
    cuBLAS is deterministic => clean residual is exactly 0 (T=0). A Step-A output bit flip in
    `cached` makes the residual >= |delta| >= 2.76 => caught."""
    maxres = torch.zeros((), device=DEV, dtype=torch.float32)
    x = x0
    eps = 1e-5
    def rms(t):
        return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps).half()
    for i in range(L):
        h = rms(x)
        for name, src in (('q', h), ('k', h), ('v', h)):
            y = F.linear(src, W[name][i])
            maxres = torch.maximum(maxres, (y.float() - cached[(i, name)].float()).abs().max())
        q = F.linear(h, W['q'][i])
        o = F.linear(q, W['o'][i])
        maxres = torch.maximum(maxres, (o.float() - cached[(i, 'o')].float()).abs().max())
        x = x + o
        h2 = rms(x)
        gt = F.linear(h2, W['gate'][i]); up = F.linear(h2, W['up'][i])
        maxres = torch.maximum(maxres, (gt.float() - cached[(i, 'gate')].float()).abs().max())
        maxres = torch.maximum(maxres, (up.float() - cached[(i, 'up')].float()).abs().max())
        inter = F.silu(gt) * up
        dn = F.linear(inter, W['down'][i])
        maxres = torch.maximum(maxres, (dn.float() - cached[(i, 'down')].float()).abs().max())
        x = x + dn
    lg = F.linear(rms(x), Wlm)
    maxres = torch.maximum(maxres, (lg.float() - cached[(-1, 'lm')].float()).abs().max())
    return maxres


def lock_clock(mhz=1980):
    os.system(f'sudo nvidia-smi -lgc {mhz},{mhz} >/dev/null 2>&1')

def reset_clock():
    os.system('sudo nvidia-smi -rgc >/dev/null 2>&1')

def gpu_sample():
    import subprocess
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=clocks.sm,power.draw,utilization.gpu,utilization.memory',
         '--format=csv,noheader,nounits']).decode().strip()
    sm, pw, ug, um = [x.strip() for x in out.split(',')]
    return dict(sm_mhz=float(sm), power_w=float(pw), util_gpu=float(ug), util_mem=float(um))
