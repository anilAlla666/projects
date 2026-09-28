#!/usr/bin/env python3
# D.9 §1 BASELINE RE-MEASUREMENT — measure-only, CIPHER observe-only/ops-OFF.
# Establishes the honest current MFU on the DEPLOYED cipher_rt_phase4 substrate.
# MFU denominator = bf16/fp16 989 TFLOPS H100 reference peak (stated on every number).
# 3 parts: (A) clocks-under-load + 4096^2 GEMM proxy, (B) component-resolved prefill
# forward (Mistral-7B), (C) FP8 delivered-throughput probe (torch._scaled_mm).
import os, sys, json, time, subprocess, statistics

PEAK_BF16 = 989.0e12          # H100 SXM5 BF16/FP16 dense peak (the 67% anchor denominator)
PEAK_FP8  = 1979.0e12         # H100 FP8 E4M3 TC peak
MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"

def smi_clock():
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        sm, pw = out.split(",")
        return float(sm), float(pw)
    except Exception as e:
        return -1.0, -1.0

# ---------- PART A: clocks-under-load + 4096^2 GEMM proxy (CIPHER ops OFF) ----------
def part_a():
    import torch
    N = 4096
    a = torch.randn(N, N, dtype=torch.float16, device="cuda")
    b = torch.randn(N, N, dtype=torch.float16, device="cuda")
    for _ in range(20): c = a @ b
    torch.cuda.synchronize()
    flops_per = 2.0 * N * N * N
    samples = []                      # (t, clock_mhz, power_w, inst_tflops)
    t_end = time.time() + 60.0
    last_sample = 0.0
    iters = 0
    win_start = time.time(); win_iters = 0
    while time.time() < t_end:
        for _ in range(50): c = a @ b
        torch.cuda.synchronize()
        iters += 50; win_iters += 50
        now = time.time()
        if now - last_sample >= 1.0:
            dt = now - win_start
            inst_tflops = (win_iters * flops_per / dt) / 1e12
            cl, pw = smi_clock()
            samples.append({"t": round(now - (t_end-60.0),1), "clock_mhz": cl, "power_w": pw, "tflops": round(inst_tflops,1)})
            last_sample = now; win_start = now; win_iters = 0
    # sustained = median of the last 30s of instantaneous tflops
    tail = [s["tflops"] for s in samples if s["t"] >= 30.0]
    sustained_tflops = statistics.median(tail) if tail else (samples[-1]["tflops"] if samples else 0)
    sustained_clock = statistics.median([s["clock_mhz"] for s in samples if s["t"] >= 30.0] or [-1])
    def at(tt):
        c = min(samples, key=lambda s: abs(s["t"]-tt)) if samples else {}
        return c
    res = {"gemm_N": N, "sustained_tflops": round(sustained_tflops,1),
           "sustained_mfu_pct": round(100*sustained_tflops*1e12/PEAK_BF16,1),
           "sustained_clock_mhz": sustained_clock,
           "clock_at_0s": at(1).get("clock_mhz"), "clock_at_30s": at(30).get("clock_mhz"), "clock_at_60s": at(59).get("clock_mhz"),
           "power_at_30s": at(30).get("power_w"), "samples": samples}
    print(f"[A 4096^2 GEMM proxy] sustained {res['sustained_tflops']} TFLOPS = {res['sustained_mfu_pct']}% MFU(vs989) "
          f"| clock 0s={res['clock_at_0s']} 30s={res['clock_at_30s']} 60s={res['clock_at_60s']} MHz | pwr~{res['power_at_30s']}W", flush=True)
    return res

# ---------- PART B: component-resolved prefill forward (Mistral-7B) ----------
def part_b():
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    P_linear = sum(p.numel() for n,p in model.named_parameters() if "weight" in n and p.dim()==2)
    L, H, hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size//cfg.num_attention_heads
    shapes = [(1,2048),(1,4096),(4,2048),(8,2048),(8,4096),(16,2048)]
    rows = []
    for (B,S) in shapes:
        try:
            ids = torch.randint(0, cfg.vocab_size, (B,S), device="cuda")
            with torch.no_grad():
                for _ in range(2): model(ids)             # warmup
                torch.cuda.synchronize()
                nrep = 5
                t0 = time.time()
                for _ in range(nrep): model(ids)
                torch.cuda.synchronize()
                wall = (time.time()-t0)/nrep
            T = B*S
            lin = 2.0*P_linear*T
            attn = 2.0*L*B*H*(S**2)*hd                     # causal QK^T+AV
            tot = lin+attn
            tflops = tot/wall/1e12
            mfu = 100*tflops*1e12/PEAK_BF16
            cl,pw = smi_clock()
            # component breakdown via profiler (one rep)
            comp = profile_components(model, ids)
            rows.append({"B":B,"S":S,"tokens":T,"wall_ms":round(wall*1e3,2),
                         "linear_TFLOP":round(lin/1e12,1),"attn_TFLOP":round(attn/1e12,1),
                         "achieved_TFLOPS":round(tflops,1),"MFU_pct_vs989":round(mfu,1),
                         "clock_mhz":cl,"power_w":pw,"components":comp})
            print(f"[B B={B} S={S} T={T}] {round(wall*1e3,1)}ms  {round(tflops,1)} TFLOPS = {round(mfu,1)}% MFU(vs989)  "
                  f"comp: gemm={comp.get('gemm_pct')}% attn={comp.get('attn_pct')}% epi={comp.get('other_pct')}%  clk={cl}", flush=True)
        except Exception as e:
            rows.append({"B":B,"S":S,"error":str(e)}); print(f"[B B={B} S={S}] ERR {e}", flush=True)
    return {"P_linear_B": round(P_linear/1e9,3), "n_layers":L,"n_heads":H,"head_dim":hd, "shapes": rows}

def profile_components(model, ids):
    import torch
    from torch.profiler import profile, ProfilerActivity
    def cuda_us(e):
        for attr in ("self_device_time_total","self_cuda_time_total","device_time_total","cuda_time_total"):
            v = getattr(e, attr, 0) or 0
            if v: return float(v)
        return 0.0
    # categorize CUDA KERNELS by name (attn checked first — some attn kernels contain 'gemm').
    GEMM = ["gemm","cutlass","cublas","wgmma","xmma","nvjet","sm90_","sm80_","tensorop","implicit_gemm",
            "ampere","volta","s16816","s1688","hgemm","sgemm","gett","128x128","256x128","cublaslt"]
    ATTN = ["flash","fmha","attention","softmax","sdpa","mha","cudnn","fwd_kernel","_attn"]
    def bucket(nm):
        n = nm.lower()
        if any(k in n for k in ATTN): return "attn"
        if any(k in n for k in GEMM): return "gemm"
        return "other"
    try:
        with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(ids); torch.cuda.synchronize()
        kerns = []
        for e in prof.key_averages():
            cu = cuda_us(e)
            if cu>0: kerns.append((e.key, cu))
        gemm=attn=other=0.0
        for nm,cu in kerns:
            b=bucket(nm); gemm+=cu*(b=="gemm"); attn+=cu*(b=="attn"); other+=cu*(b=="other")
        tot=gemm+attn+other
        if tot<=0: return {}
        kerns.sort(key=lambda x:-x[1])
        top = [{"k":nm[:70],"ms":round(cu/1e3,2),"bucket":bucket(nm)} for nm,cu in kerns[:15]]
        return {"gemm_pct":round(100*gemm/tot,1),"attn_pct":round(100*attn/tot,1),"other_pct":round(100*other/tot,1),
                "total_cuda_ms":round(tot/1e3,2),"top_kernels":top}
    except Exception as e:
        return {"profiler_error": str(e)}

# ---------- PART C: FP8 delivered-throughput probe (torch._scaled_mm vs fp16) ----------
def part_c():
    import torch
    shapes = []
    for M in [2048,4096,8192,16384,32768]:
        for (N,K,tag) in [(4096,4096,"attn_proj"),(14336,4096,"ffn_up"),(4096,14336,"ffn_down")]:
            shapes.append((M,N,K,tag))
    rows = []
    sa = torch.tensor(1.0, device="cuda"); sb = torch.tensor(1.0, device="cuda")
    for (M,N,K,tag) in shapes:
        try:
            a16 = torch.randn(M,K,dtype=torch.float16,device="cuda")
            b16 = torch.randn(N,K,dtype=torch.float16,device="cuda")  # we use b16.t() => [K,N]
            # fp16
            for _ in range(5): c = a16 @ b16.t()
            torch.cuda.synchronize()
            t0=time.time()
            for _ in range(20): c = a16 @ b16.t()
            torch.cuda.synchronize(); t16=(time.time()-t0)/20
            # fp8 e4m3
            a8 = a16.to(torch.float8_e4m3fn); b8 = b16.to(torch.float8_e4m3fn)
            for _ in range(5): c8 = torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float16)
            torch.cuda.synchronize()
            t0=time.time()
            for _ in range(20): c8 = torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float16)
            torch.cuda.synchronize(); t8=(time.time()-t0)/20
            f = 2.0*M*N*K
            tf16=f/t16/1e12; tf8=f/t8/1e12
            rows.append({"M":M,"N":N,"K":K,"tag":tag,"fp16_TFLOPS":round(tf16,1),"fp16_MFU_vs989":round(100*tf16*1e12/PEAK_BF16,1),
                         "fp8_TFLOPS":round(tf8,1),"fp8_MFU_vs989":round(100*tf8*1e12/PEAK_BF16,1),"fp8_MFU_vs_fp8peak":round(100*tf8*1e12/PEAK_FP8,1),
                         "fp8_over_fp16_ratio":round(tf8/tf16,3)})
            print(f"[C M={M} {tag} N={N} K={K}] fp16 {round(tf16,0)} / fp8 {round(tf8,0)} TFLOPS  ratio {round(tf8/tf16,2)}x  (fp8 {round(100*tf8*1e12/PEAK_BF16,0)}% vs989)", flush=True)
        except Exception as e:
            rows.append({"M":M,"N":N,"K":K,"tag":tag,"error":str(e)}); print(f"[C M={M} {tag}] ERR {e}", flush=True)
    return {"shapes": rows}

if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv)>1 else "all"
    out = {}
    try: out = json.load(open("/home/ubuntu/d9_s1_result.json"))   # merge, don't clobber prior parts
    except Exception: out = {}
    if part in ("all","A"): out["A_gemm_proxy"] = part_a()
    if part in ("all","B"): out["B_prefill_forward"] = part_b()
    if part in ("all","C"): out["C_fp8_probe"] = part_c()
    out["peak_bf16_TFLOPS"]=PEAK_BF16/1e12; out["peak_fp8_TFLOPS"]=PEAK_FP8/1e12
    json.dump(out, open("/home/ubuntu/d9_s1_result.json","w"), indent=2)
    print("WROTE /home/ubuntu/d9_s1_result.json", flush=True)
