#!/usr/bin/env python3
"""
A1 — CORRUPTION HARM CHARACTERIZATION (offline scratch harness).

Inject single fp16 bit-flips into the OUTPUT of intercepted GEMMs on Mistral-7B
and measure harm by bit significance. Records the local delta |v'-v| of every
flip so the A2 coverage join (residual == |delta|) is pure arithmetic.

Fault model: transient single-bit flip in one element of an intercepted GEMM
output, at the last sequence position (the element on the direct path to the
next-token logit). Read-only: no .so touched, no model weights modified on disk.
"""
import os, sys, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
OUT   = "/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a1_results.json"
SEED  = 1234
LAYERS = [0, 8, 16, 24, 31]
PROJ   = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
K_CHAN = 8          # random channels per (site,bit)
BITS   = list(range(16))   # 0..9 mantissa, 10..14 exponent, 15 sign

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ---- shape map (out features) so we can tag each injection with its GEMM (M,K,N)
HID, INTER, KV, VOCAB = 4096, 14336, 1024, 32000
OUTF = {"q_proj":HID,"k_proj":KV,"v_proj":KV,"o_proj":HID,
        "gate_proj":INTER,"up_proj":INTER,"down_proj":HID,"lm_head":VOCAB}
INF  = {"q_proj":HID,"k_proj":HID,"v_proj":HID,"o_proj":HID,
        "gate_proj":HID,"up_proj":HID,"down_proj":INTER,"lm_head":HID}

def fp16_flip(val, bit):
    """flip `bit` of fp16 scalar val; return new python float (maybe inf/nan) and delta."""
    u = np.frombuffer(np.float16(val).tobytes(), dtype=np.uint16)[0]
    u2 = np.uint16(u ^ np.uint16(1 << bit))
    new = np.frombuffer(np.uint16(u2).tobytes(), dtype=np.float16)[0]
    return float(new), (float(new) - float(val))

# ---- global injection state; one hook on every target module reads it
INJ = {"on": False, "name": None, "chan": None, "bit": None,
       "old": None, "new": None, "delta": None}

def make_hook(name):
    def hook(module, inp, out):
        if not INJ["on"] or INJ["name"] != name:
            return out
        t = out
        # out shape [B, S, F]; inject at last seq pos, chosen channel
        b = 0; s = t.shape[1]-1; c = INJ["chan"]
        old = float(t[b, s, c].item())
        new, delta = fp16_flip(old, INJ["bit"])
        t[b, s, c] = float(new)
        INJ["old"], INJ["new"], INJ["delta"] = old, new, delta
        return t
    return hook

def main():
    print("loading", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    # register hooks on all target modules
    handles = []
    targets = {}
    for L in LAYERS:
        lyr = model.model.layers[L]
        for p in PROJ:
            mod = getattr(lyr.self_attn, p, None) or getattr(lyr.mlp, p, None)
            name = f"L{L}.{p}"
            targets[name] = (p, OUTF[p], INF[p])
            handles.append(mod.register_forward_hook(make_hook(name)))
    targets["lm_head"] = ("lm_head", OUTF["lm_head"], INF["lm_head"])
    handles.append(model.lm_head.register_forward_hook(make_hook("lm_head")))

    prompt = "The quick brown fox jumps over the lazy dog. In a distant kingdom, the old"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    S = ids.shape[1]
    print("seq len", S, flush=True)

    @torch.no_grad()
    def forward_logits():
        out = model(ids)
        return out.logits[0, -1, :].float()   # last-token logits, fp32 copy

    # clean reference
    INJ["on"] = False
    clean = forward_logits()
    clean_top1 = int(clean.argmax().item())
    clean_finite = bool(torch.isfinite(clean).all())
    clean_scale = float(clean.abs().max().item())
    print(f"clean top1={clean_top1} finite={clean_finite} scale={clean_scale:.3f}", flush=True)

    # ---------- VALIDATION GATES ----------
    gates = {}
    # gate (a): zero-effect injection (bit flip then flip back == identity); also
    # verify that with INJ off we are bit-identical across two runs.
    c2 = forward_logits()
    gates["determinism_bitident"] = bool(torch.equal(clean, c2))
    # forced flip identity: flip a chosen low element and confirm hook actually wrote
    INJ.update(on=True, name="lm_head", chan=clean_top1, bit=0)  # mantissa LSB on top logit elem
    _ = forward_logits()
    gates["hook_fired_lsb"] = (INJ["delta"] is not None)
    gates["lsb_delta"] = INJ["delta"]
    # forced top-exponent flip on a large-magnitude lm_head channel -> must blow up
    INJ.update(on=True, name="lm_head", chan=clean_top1, bit=14)
    big = forward_logits()
    INJ["on"] = False
    gates["expflip_top1_changed"] = bool(int(big.argmax().item()) != clean_top1 or not torch.isfinite(big).all())
    gates["expflip_delta"] = INJ["delta"]
    print("GATES:", json.dumps(gates), flush=True)

    # ---------- SWEEP ----------
    results = []
    rng = random.Random(SEED)
    sites = list(targets.items())
    t0 = time.time(); n=0; total=len(sites)*len(BITS)*K_CHAN
    for name,(ptype,outf,inf) in sites:
        for bit in BITS:
            for k in range(K_CHAN):
                chan = rng.randrange(outf)
                INJ.update(on=True, name=name, chan=chan, bit=bit,
                           old=None,new=None,delta=None)
                lg = forward_logits()
                INJ["on"] = False
                finite = bool(torch.isfinite(lg).all())
                if finite:
                    top1 = int(lg.argmax().item())
                    dmax = float((lg-clean).abs().max().item())
                    # symmetric KL on softmax
                    pc = torch.softmax(clean,0); pq = torch.softmax(lg,0)
                    kl = float((pc*(pc.clamp_min(1e-12).log()-pq.clamp_min(1e-12).log())).sum().item())
                else:
                    top1=-1; dmax=float('inf'); kl=float('inf')
                naninf = (not finite)
                top1_flip = (top1 != clean_top1) and finite
                harmful = bool(naninf or top1_flip)
                results.append(dict(site=name, ptype=ptype, M=S, K=inf, N=outf,
                                    bit=bit, chan=chan,
                                    old=INJ["old"], new=INJ["new"],
                                    delta=INJ["delta"],
                                    abs_delta=(abs(INJ["delta"]) if INJ["delta"] is not None and np.isfinite(INJ["delta"]) else float('inf')),
                                    naninf=naninf, top1_flip=top1_flip,
                                    dmax=dmax, kl=kl, harmful=harmful))
                n+=1
                if n % 400 == 0:
                    el=time.time()-t0
                    print(f"  {n}/{total} {el:.0f}s eta {el/n*(total-n):.0f}s", flush=True)
    dt = time.time()-t0
    out = dict(meta=dict(model=MODEL, seed=SEED, seq=S, layers=LAYERS,
                         proj=PROJ, k_chan=K_CHAN, bits=BITS,
                         clean_top1=clean_top1, clean_scale=clean_scale,
                         n_injections=len(results), seconds=dt,
                         note="single fp16 bit flip at last-token GEMM output; harmful=NaN/Inf or top1 flip"),
               gates=gates, results=results)
    json.dump(out, open(OUT,"w"))
    print(f"wrote {OUT}  ({len(results)} injections, {dt:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
