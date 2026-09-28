#!/usr/bin/env python3
# READ-ONLY MFU batch-sweep harness. Tests whether driver-level BATCHING raises MFU as the roofline predicts.
# - LOCKED SM clock (the documented DVFS trap: natural-clock + post-hoc scaling can manufacture/hide an MFU climb).
#   Verify the lock held (min==max) at EVERY point incl the largest batch. Also report natural-clock throughput context.
# - Real Mistral-7B-v0.1 bf16 forward. lm_head last-token only (logits_to_keep=1): serving-realistic prefill, avoids the
#   16.8GB-at-M=128 logits OOM. lm_head FLOP counted as batch*2HV (one sampled token per sequence).
# - Three regimes (model loaded ONCE): prefill SEQ=2048 (the required compute-bound spine), prefill SEQ=16 (memory-bound
#   POSITIVE CONTROL: proves the instrument detects an MFU climb when one exists), decode (engine-realistic, 1 tok/seq).
# - FP8 (Q2): same script under LD_PRELOAD=$SO with CIPHER_FP8 toggled; reads the .so counters to confirm handled>0.
# Peak: H100 SXM bf16 dense 989.4 TFLOP/s @1980 (scaled to locked clock); FP8 dense 1978.9; HBM3 BW 3.35 TB/s.
import os, sys, ctypes, time, json, subprocess, gc
import torch, pynvml

SO       = os.environ.get("CIPHER_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")  # Jun-01 anchor (md5 2edba0d2)
MODEL    = os.environ.get("GO3_MODEL", "/home/ubuntu/models/Mistral-7B-v0.1")
MODE     = os.environ.get("MODE", "bf16")          # bf16 | fp8 (fp8 requires LD_PRELOAD=$SO CIPHER_FP8=1)
LOCK_MHZ = int(os.environ.get("LOCK_MHZ", "1200")) # 0 => natural clock
ITERS    = int(os.environ.get("ITERS", "30"))
WARM     = int(os.environ.get("WARM", "8"))
PLAN     = os.environ.get("PLAN", "prefill2048,prefill16,decode")  # comma list of regimes to run
PREF_BATCHES = [int(x) for x in os.environ.get("PREF_BATCHES", "1,8,32,64,128").split(",")]
DEC_BATCHES  = [int(x) for x in os.environ.get("DEC_BATCHES",  "1,8,32,64,128,256").split(",")]
DECODE_CTX   = int(os.environ.get("DECODE_CTX", "512"))
TAG          = os.environ.get("TAG", "run")

PEAK_BF16_BOOST = 989.4     # H100 SXM5 BF16/FP16 tensor-core DENSE TFLOP/s @ 1980 MHz (NVIDIA datasheet)
PEAK_FP8_BOOST  = 1978.9    # H100 SXM5 FP8 tensor-core DENSE TFLOP/s @ 1980 MHz
HBM_BW_TBs      = 3.35      # H100 80GB HBM3 SXM bandwidth (TB/s)
BOOST_MHZ       = 1980

pynvml.nvmlInit(); DEV = pynvml.nvmlDeviceGetHandleByIndex(0)
def smclk(): return pynvml.nvmlDeviceGetClockInfo(DEV, pynvml.NVML_CLOCK_SM)
def powW():  return pynvml.nvmlDeviceGetPowerUsage(DEV) / 1000.0
def memMB(): return pynvml.nvmlDeviceGetMemoryInfo(DEV).used / 1024 / 1024

def lock_clock(mhz):
    if mhz > 0:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-lgc", f"{mhz},{mhz}"], capture_output=True)
        time.sleep(0.4)
def reset_clock():
    subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)

# FP8 actuator counters (read-only telemetry; interception itself happens via LD_PRELOAD at the cublasGemmEx symbol)
lib = ctypes.CDLL(SO)
for s in ("cipher_rt_fp8_calls_handled", "cipher_rt_fp8_weights_quantized", "cipher_rt_fp8_max_n", "cipher_rt_fp8_is_active"):
    try: getattr(lib, s).restype = ctypes.c_ulong
    except Exception: pass
def fp8_handled():
    try: return int(lib.cipher_rt_fp8_calls_handled())
    except Exception: return -1
def fp8_wq():
    try: return int(lib.cipher_rt_fp8_weights_quantized())
    except Exception: return -1
def fp8_maxn():
    try: return int(lib.cipher_rt_fp8_max_n())
    except Exception: return -1
def fp8_active():
    try: return int(lib.cipher_rt_fp8_is_active())
    except Exception: return -1

from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
print(f"[load] MODE={MODE} fp8_active={fp8_active()} model={os.path.basename(MODEL)} so={os.path.basename(SO)}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
c = m.config
H, I_, L = c.hidden_size, c.intermediate_size, c.num_hidden_layers
nh = c.num_attention_heads; nkv = getattr(c, "num_key_value_heads", nh); hd = H // nh; V = c.vocab_size
print(f"[cfg] H={H} I={I_} L={L} nh={nh} nkv={nkv} hd={hd} V={V}", flush=True)

# --- analytic FLOP & arithmetic intensity from REAL layer dims -----------------------------------------------------
# Linear GEMMs per layer as (k_in, n_out): q,k,v,o + gate,up,down. (2*k*n MACs per token each.)
GEMMS = [(H, H), (H, nkv * hd), (H, nkv * hd), (H, H), (H, I_), (H, I_), (I_, H)]
lin_flop_per_tok = sum(2 * k * n for (k, n) in GEMMS)         # linear-layer FLOP per token per layer
def attn_flop_per_tok(ctx):  return 4 * nh * hd * ctx          # QK^T + AV per token per layer (causal upper bound: ctx)
LMHEAD_FLOP = 2 * H * V                                        # per sampled token (last position)

def gemm_AI(m_tokens):
    """Aggregate arithmetic intensity (FLOP/byte, bf16 2B) over all linear GEMMs at activation m=m_tokens.
       weights read once, activations read+written. This is the roofline x-axis I(M)."""
    num = sum(2 * m_tokens * k * n for (k, n) in GEMMS)
    den = sum(2 * (k * n + m_tokens * k + m_tokens * n) for (k, n) in GEMMS)  # 2 bytes/elem
    return num / den
RIDGE = PEAK_BF16_BOOST * 1e12 / (HBM_BW_TBs * 1e12)           # I* = peak/BW (FLOP/byte), bf16

base = tok("The history of artificial intelligence and high performance computing spans decades of research. ",
           return_tensors="pt").input_ids

def peak_at(clk_mhz, fp8=False):
    base_peak = PEAK_FP8_BOOST if fp8 else PEAK_BF16_BOOST
    return base_peak * (clk_mhz / BOOST_MHZ)

def time_loop(run_one, iters, warm):
    with torch.no_grad():
        for _ in range(warm): run_one()
        torch.cuda.synchronize()
    clks = []; pws = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(iters):
            run_one()
            if i % 3 == 0: clks.append(smclk()); pws.append(powW())
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, clks, pws

def measure_prefill(seq, batch):
    ids = base.repeat(1, (seq // base.shape[1]) + 1)[:, :seq].repeat(batch, 1).cuda()
    n_tok = ids.numel()
    def run_one(): m(ids, logits_to_keep=1)
    dt, clks, pws = time_loop(run_one, ITERS, WARM)
    tps = n_tok * ITERS / dt
    flop_per_fwd = n_tok * L * (lin_flop_per_tok + attn_flop_per_tok(seq)) + batch * LMHEAD_FLOP
    ach = flop_per_fwd * ITERS / dt / 1e12
    clk = sum(clks) / len(clks); pk = peak_at(clk, fp8=(MODE == "fp8"))
    AI = gemm_AI(n_tok)   # prefill: m = batch*seq
    return dict(regime=f"prefill{seq}", batch=batch, seq=seq, n_tok=n_tok, tok_per_fwd=n_tok,
                tps=round(tps, 1), tflops=round(ach, 1), mfu=round(ach / pk * 100, 2),
                AI=round(AI, 1), AI_over_Istar=round(AI / RIDGE, 3), roofline_pred=round(min(1.0, AI / RIDGE), 3),
                clk_avg=round(clk), clk_min=min(clks), clk_max=max(clks), powW=round(sum(pws) / len(pws)),
                memMB=round(memMB()), peak_used=round(pk), fp8_handled=fp8_handled(), fp8_wq=fp8_wq(), fp8_maxn=fp8_maxn())

def measure_decode(batch, ctx):
    # Build a real prefilled KV cache of length `ctx`, then time single-token decode steps (1 token/sequence).
    prompt = base.repeat(1, (ctx // base.shape[1]) + 1)[:, :ctx].repeat(batch, 1).cuda()
    with torch.no_grad():
        out = m(prompt, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:, :].argmax(-1)   # [batch,1]
    def run_one():
        nonlocal past, nxt
        o = m(nxt, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:, :].argmax(-1)
    # warmup grows cache a little; that's fine — the roofline-relevant dim is m=batch (1 tok/seq), context ~ctx.
    dt, clks, pws = time_loop(run_one, ITERS, WARM)
    tps = batch * ITERS / dt
    avg_ctx = ctx + WARM + ITERS // 2
    flop_per_step = batch * L * (lin_flop_per_tok + attn_flop_per_tok(avg_ctx)) + batch * LMHEAD_FLOP
    ach = flop_per_step * ITERS / dt / 1e12
    clk = sum(clks) / len(clks); pk = peak_at(clk, fp8=(MODE == "fp8"))
    AI = gemm_AI(batch)   # decode: m = batch (one token per sequence)
    return dict(regime="decode", batch=batch, seq=1, ctx=avg_ctx, n_tok=batch, tok_per_fwd=batch,
                tps=round(tps, 1), tflops=round(ach, 2), mfu=round(ach / pk * 100, 2),
                AI=round(AI, 1), AI_over_Istar=round(AI / RIDGE, 3), roofline_pred=round(min(1.0, AI / RIDGE), 3),
                clk_avg=round(clk), clk_min=min(clks), clk_max=max(clks), powW=round(sum(pws) / len(pws)),
                memMB=round(memMB()), peak_used=round(pk), fp8_handled=fp8_handled(), fp8_wq=fp8_wq(), fp8_maxn=fp8_maxn())

# Lock the clock once, after load, before the first timed window; keep GPU busy so it holds across all points.
lock_clock(LOCK_MHZ)
print(f"[clock] requested lock={LOCK_MHZ}MHz  RIDGE_Istar={RIDGE:.0f} FLOP/byte  lin_flop/tok={lin_flop_per_tok/1e9:.3f}G  peak_bf16@{LOCK_MHZ}={peak_at(LOCK_MHZ):.0f} TFLOP/s", flush=True)

results = []
def do(fn, *a):
    try:
        r = fn(*a); results.append(r)
        lockok = "LOCK_OK" if r["clk_min"] == r["clk_max"] == LOCK_MHZ else f"LOCK_DRIFT[{r['clk_min']}-{r['clk_max']}]"
        print(f"[{r['regime']:>11} M={r['batch']:>4}] {r['tps']:>10,.0f} tok/s | {r['tflops']:>7.1f} TF/s | "
              f"MFU={r['mfu']:>5.1f}% | AI={r['AI']:>7.1f} I/I*={r['AI_over_Istar']:>5.2f} pred={r['roofline_pred']:>4.2f} | "
              f"{r['clk_avg']}MHz {lockok} | {r['powW']}W | {r['memMB']}MB | fp8h={r['fp8_handled']} wq={r['fp8_wq']}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[{fn.__name__} {a}] OOM -> HBM CAP hit, batch skipped: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[{fn.__name__} {a}] OOM -> HBM CAP hit, batch skipped", flush=True)
            torch.cuda.empty_cache(); gc.collect()
        else:
            raise

if "prefill2048" in PLAN:
    for b in PREF_BATCHES: do(measure_prefill, 2048, b)
if "prefill16" in PLAN:
    for b in PREF_BATCHES: do(measure_prefill, 16, b)
if "decode" in PLAN:
    for b in DEC_BATCHES: do(measure_decode, b, DECODE_CTX)

reset_clock()
print("MFU_JSON " + json.dumps(dict(mode=MODE, lock_mhz=LOCK_MHZ, tag=TAG, ridge=RIDGE,
      peak_bf16_boost=PEAK_BF16_BOOST, peak_fp8_boost=PEAK_FP8_BOOST, hbm_bw_tbs=HBM_BW_TBs,
      lin_flop_per_tok=lin_flop_per_tok, results=results)), flush=True)
sys.stdout.flush(); os._exit(0)
