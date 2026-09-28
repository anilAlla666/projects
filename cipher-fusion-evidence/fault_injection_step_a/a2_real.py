#!/usr/bin/env python3
"""
A2 (real-model) — three things the synthetic microbench cannot give:
  (A) THRESHOLD: clean checksum residual noise floor per shape, on REAL Mistral-7B
      GEMM outputs, in fp16 vs fp32 accumulate. This T is on the same scale as the
      A1 |delta| values, so the coverage join (residual==|delta|) is apples-to-apples.
  (B) END-TO-END EAGER OVERHEAD: prefill(2048) and decode-step latency WITH vs
      WITHOUT verify on all 225 linears. Exposes the launch-bound cost the per-kernel
      ratio hides (the decode verdict-flipper the advisor flagged).
  (C) CAPTURED VERIFY COST: the 225 verify-compute ops captured into ONE CUDA graph
      (pure tensor ops -> legal) and replayed. The in-graph floor; the residual flag
      (.item host sync) is left OUT of the graph by construction = capture-illegal.
"""
import json, time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
OUT="/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a2_real.json"
HID,INTER,KV,VOCAB=4096,14336,1024,32000

VERIFY={"on":False,"flag":False}
captured_io={}   # op type -> (A,W,C) one real example

def vhook_factory(name, capture_thr=False):
    def hook(module, inp, out):
        A=inp[0]; W=module.weight
        if capture_thr and name not in captured_io:
            captured_io[name]=(A.detach().float().reshape(-1,A.shape[-1]).clone(),
                               W.detach().float().clone(),
                               out.detach().float().reshape(-1,out.shape[-1]).clone())
        if not VERIFY["on"]: return out
        if not hasattr(module,"_wcol"): module._wcol=W.float().sum(0)
        r=A.float().reshape(-1,A.shape[-1])@module._wcol
        s=out.float().reshape(-1,out.shape[-1]).sum(1)
        resid=(r-s).abs()
        if VERIFY["flag"]:
            _=bool((resid.max()>1e30).item())   # host sync; threshold huge so never branches
        return out
    return hook

def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    # sdpa = production-realistic attention. It gives a FASTER decode baseline than
    # eager, so the verify-overhead % is measured against the conservative (smaller)
    # denominator -> we do not flatter the 3% verdict. (A1 used eager for hook
    # determinism; A2 is timing-only, so the production path is the right denominator.)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,
            attn_implementation="sdpa").cuda().eval()
    linmods=[]
    for L in range(32):
        ly=model.model.layers[L]
        for p in ["q_proj","k_proj","v_proj","o_proj"]: linmods.append((f"L{L}.{p}",getattr(ly.self_attn,p)))
        for p in ["gate_proj","up_proj","down_proj"]:   linmods.append((f"L{L}.{p}",getattr(ly.mlp,p)))
    linmods.append(("lm_head",model.lm_head))
    print("linear modules hooked:",len(linmods),flush=True)
    # capture real IO from layer 16 representative of each shape, during a prefill
    rep={"L16.q_proj":"q/o","L16.k_proj":"k/v","L16.gate_proj":"gate/up","L16.down_proj":"down","lm_head":"lm_head"}
    for name,mod in linmods:
        mod.register_forward_hook(vhook_factory(name, capture_thr=(name in rep)))

    # ---- realistic-length prefill for threshold capture & prefill timing
    txt=("In the field of computer architecture, error detection through algorithm "
         "based fault tolerance has been studied since Huang and Abraham. ")*40
    ids=tok(txt,return_tensors="pt").input_ids[:, :2048].cuda()
    M_pf=ids.shape[1]
    print("prefill tokens",M_pf,flush=True)

    # --- (A) threshold: run one clean prefill, capture IO, compute residual floor
    VERIFY["on"]=False
    with torch.no_grad(): _=model(ids)
    thr={}
    for name,tag in rep.items():
        A,W,C=captured_io[name]      # A[M,K] fp32, W[N,K] fp32, C[M,N] fp32
        wcol=W.sum(0)
        # fp32 accumulate
        r32=A@wcol; s32=C.sum(1); res32=(r32-s32).abs()
        # fp16 accumulate (emulate fp16 reduction)
        r16=(A.half()@wcol.half()).float(); s16=C.half().sum(1).float(); res16=(r16-s16).abs()
        Cscale=float(C.abs().mean().item()); rowsumscale=float(s32.abs().mean().item())
        thr[tag]=dict(M=A.shape[0],K=A.shape[1],N=C.shape[1],
            T_fp32_max=float(res32.max()), T_fp32_p999=float(res32.float().quantile(0.999)),
            T_fp16_max=float(res16.max()), T_fp16_p999=float(res16.float().quantile(0.999)),
            C_abs_mean=Cscale, rowsum_abs_mean=rowsumscale)
        print(f"thr {tag:9s} N={C.shape[1]:5d} T_fp32_max={float(res32.max()):.4f} "
              f"T_fp16_max={float(res16.max()):.3f} rowsum~{rowsumscale:.1f}",flush=True)

    # --- (B) end-to-end eager overhead
    def time_prefill(iters=8):
        for _ in range(2):
            with torch.no_grad(): _=model(ids)
        torch.cuda.synchronize(); ts=[]
        for _ in range(iters):
            t=time.perf_counter()
            with torch.no_grad(): _=model(ids)
            torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e3)
        return statistics.median(ts)
    def time_decode(nstep=32,iters=5):
        with torch.no_grad():
            o=model(ids[:, :32], use_cache=True); pkv=o.past_key_values
            nxt=o.logits[:, -1:].argmax(-1)
        # warm
        for _ in range(2):
            with torch.no_grad():
                oo=model(nxt, past_key_values=pkv, use_cache=True)
        torch.cuda.synchronize(); per=[]
        for _ in range(iters):
            with torch.no_grad():
                o=model(ids[:, :32], use_cache=True); pkv=o.past_key_values
                nxt=o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); t=time.perf_counter()
            with torch.no_grad():
                for _ in range(nstep):
                    o=model(nxt,past_key_values=pkv,use_cache=True)
                    pkv=o.past_key_values; nxt=o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); per.append((time.perf_counter()-t)*1e3/nstep)
        return statistics.median(per)

    VERIFY["on"]=False; pf_off=time_prefill(); dc_off=time_decode()
    VERIFY["on"]=True;  VERIFY["flag"]=False; pf_vc=time_prefill(); dc_vc=time_decode()
    VERIFY["on"]=True;  VERIFY["flag"]=True;  pf_vf=time_prefill(); dc_vf=time_decode()
    VERIFY["on"]=False
    e2e=dict(prefill_off_ms=pf_off, prefill_verify_compute_ms=pf_vc, prefill_verify_full_ms=pf_vf,
             decode_off_ms=dc_off, decode_verify_compute_ms=dc_vc, decode_verify_full_ms=dc_vf,
             prefill_vc_ovh=(pf_vc-pf_off)/pf_off, prefill_vf_ovh=(pf_vf-pf_off)/pf_off,
             decode_vc_ovh=(dc_vc-dc_off)/dc_off, decode_vf_ovh=(dc_vf-dc_off)/dc_off)
    print("E2E eager:",json.dumps({k:round(v,4) for k,v in e2e.items()}),flush=True)

    # --- (C) captured verify cost: 225 verify-compute ops in ONE cuda graph
    counts={(HID,HID):64,(HID,KV):64,(HID,INTER):64,(INTER,HID):32,(HID,VOCAB):1}  # (K,N):count
    ins=[]
    for (K,N),cnt in counts.items():
        for _ in range(cnt):
            A=torch.randn(1,K,device="cuda")
            wcol=torch.randn(K,device="cuda")
            C=torch.randn(1,N,device="cuda")
            res=torch.empty(1,device="cuda")
            ins.append((A,wcol,C,res))
    def run_verify():
        for A,wcol,C,res in ins:
            res.copy_(((A@wcol)-C.sum(1)).abs())
    cap=None
    try:
        for _ in range(3): run_verify()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g): run_verify()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        # time graph replay
        ev=lambda f,it=200: (lambda a,b:[ (a.record(),f(),b.record(),torch.cuda.synchronize())[0] for _ in range(it)])(torch.cuda.Event(True),torch.cuda.Event(True))
        # simpler timing:
        def t_graph(it=200):
            for _ in range(10): g.replay()
            torch.cuda.synchronize(); a=torch.cuda.Event(True);b=torch.cuda.Event(True)
            a.record()
            for _ in range(it): g.replay()
            b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it
        def t_eager_verify(it=50):
            for _ in range(5): run_verify()
            torch.cuda.synchronize(); a=torch.cuda.Event(True);b=torch.cuda.Event(True)
            a.record()
            for _ in range(it): run_verify()
            b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it
        cap=dict(captured_verify_ms=t_graph(), eager_verify_ms=t_eager_verify(),
                 n_ops=len(ins),
                 note="225 row-sum verify-compute ops; captured graph replay vs eager loop")
        print("CAPTURED:",json.dumps({k:round(v,4) if isinstance(v,float) else v for k,v in cap.items()}),flush=True)
    except Exception as ex:
        cap=dict(error=repr(ex)); print("graph capture failed:",ex,flush=True)

    json.dump(dict(threshold=thr,e2e=e2e,captured=cap,
        meta=dict(model=MODEL,prefill_M=M_pf,n_linears=len(linmods),
                  decode_note="32-tok prefill then 32 decode steps, per-step median")),
        open(OUT,"w"))
    print("wrote",OUT,flush=True)

if __name__=="__main__": main()
