#!/usr/bin/env python3
"""
R.A END-TO-END PRODUCT TEST — one configured detector vs a realistic serving trace.

Detector under test: periodic FULL recompute over cuBLAS (gap-free). Every N steps, re-run
all linear GEMMs of the current step with a second independent cuBLAS call and compare to the
step's cached output (fp32 abs-diff, threshold T). N chosen empirically = smallest N with
measured trace-weighted overhead < 3%.

Workload: custom SATURATED CUDA-graph serving loop (vLLM rejected on principle: full recompute
needs in-context activations+weights -> no separate process can host it; in-engine insertion is
a forbidden prod-wiring change that also breaks graph capture). Real Mistral-7B-v0.1 fp16 weights,
real shapes/byte-traffic. Decode batch B held near capacity = saturation; chunked prefill M_chunk.

Phases: 0 calibrate (base/recompute step times, saturation SM%/MEM%, clean residual->T)
        1 trace gen + scheduler (mixed prompts/decodes, arrivals over time)
        2 throughput (real wall-clock, with vs without detector)
        3 detection (eager, value-exact): transient + persistent fault injection, latency, FP count

READ-ONLY w.r.t. production .so. NOT injected (no LD_PRELOAD / CUDA_INJECTION64_PATH).
"""
import os, sys, glob, json, time, statistics, subprocess, random, argparse
import torch
import torch.nn.functional as F

DEV = 'cuda'
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='2 layers, random weights, tiny trace')
ap.add_argument('--B', type=int, default=128, help='saturated decode batch')
ap.add_argument('--Mchunk', type=int, default=512, help='chunked-prefill tokens/step')
ap.add_argument('--requests', type=int, default=1500)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--phase', default='all', help='calib|trace|throughput|detect|all')
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
H, I, KV, V = 4096, 14336, 1024, 32000
L = 2 if args.smoke else 32
B = args.B
MCHUNK = args.Mchunk
EPS = 1e-5

# ---------------------------------------------------------------- weights
def rand_W(n, k):
    return (torch.randn(n, k, device=DEV) * 0.02).half()

def load_weights():
    if args.smoke:
        W = {n: [rand_W(*s) for _ in range(L)] for n, s in
             dict(q=(H,H), k=(KV,H), v=(KV,H), o=(H,H), gate=(I,H), up=(I,H), down=(H,I)).items()}
        return W, rand_W(V, H)
    from safetensors.torch import load_file
    mist = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/*/'))[0]
    sd = {}
    for s in sorted(glob.glob(mist + '*.safetensors')):
        sd.update(load_file(s, device='cpu'))
    g = lambda k: sd[k].to(DEV, dtype=torch.float16, non_blocking=True).contiguous()
    W = {n: [] for n in ('q','k','v','o','gate','up','down')}
    for i in range(L):
        p = f'model.layers.{i}.'
        W['q'].append(g(p+'self_attn.q_proj.weight')); W['k'].append(g(p+'self_attn.k_proj.weight'))
        W['v'].append(g(p+'self_attn.v_proj.weight')); W['o'].append(g(p+'self_attn.o_proj.weight'))
        W['gate'].append(g(p+'mlp.gate_proj.weight')); W['up'].append(g(p+'mlp.up_proj.weight'))
        W['down'].append(g(p+'mlp.down_proj.weight'))
    Wlm = g('lm_head.weight')
    torch.cuda.synchronize()
    return W, Wlm

def rms(t):
    return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + EPS).half()

# ---------------------------------------------------------------- forwards
# Base serving step: weight-streaming forward over all linears + lm_head. Attention replaced by
# identity q->o_proj (cost-excluded blind spot; shrinks denominator => conservative overhead %).
# x clamped to keep the synthetic 32-layer residual stream finite (does not affect GEMM cost).
def forward(xin, W, Wlm):
    x = xin
    for i in range(L):
        h = rms(x)
        q = F.linear(h, W['q'][i]); k = F.linear(h, W['k'][i]); v = F.linear(h, W['v'][i])
        o = F.linear(q, W['o'][i])
        x = (x + o).clamp_(-30, 30)
        h2 = rms(x)
        gt = F.linear(h2, W['gate'][i]); up = F.linear(h2, W['up'][i])
        dn = F.linear(F.silu(gt) * up, W['down'][i])
        x = (x + dn).clamp_(-30, 30)
    return F.linear(rms(x), Wlm)

# Detector recompute+compare (COST graph): same forward + fp16 max|out - cache| reductions into a
# static scalar. Comparison is fp16 (the configured minimal exact check): clean recompute is
# bit-identical to the cache => diff is exactly 0; a Step-A bit flip gives a large fp16-representable
# diff (or inf/nan), so fp16 separates clean(0) from corrupt cleanly without the fp32-upcast traffic
# tax. cache contents irrelevant for cost; reduction cost is real.
def forward_recompute(xin, W, Wlm, cache, maxbuf):
    x = xin
    m = torch.zeros((), device=DEV, dtype=torch.float32)
    def cmp(y, key):
        return torch.maximum(m, (y - cache[key]).abs().amax().float())
    for i in range(L):
        h = rms(x)
        q = F.linear(h, W['q'][i]); m = cmp(q, 'H')
        k = F.linear(h, W['k'][i]); m = cmp(k, 'KV')
        v = F.linear(h, W['v'][i]); m = cmp(v, 'KV')
        o = F.linear(q, W['o'][i]); m = cmp(o, 'H')
        x = (x + o).clamp_(-30, 30)
        h2 = rms(x)
        gt = F.linear(h2, W['gate'][i]); m = cmp(gt, 'I')
        up = F.linear(h2, W['up'][i]);   m = cmp(up, 'I')
        dn = F.linear(F.silu(gt) * up, W['down'][i]); m = cmp(dn, 'H')
        x = (x + dn).clamp_(-30, 30)
    lg = F.linear(rms(x), Wlm); m = cmp(lg, 'V')
    maxbuf.copy_(m)
    return lg

# ---------------------------------------------------------------- graph capture
def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g

def med_ms(g, iters=50, warm=15):
    for _ in range(warm): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

def cache_buffers(M):
    return {'H': torch.zeros(M, H, device=DEV, dtype=torch.float16),
            'KV': torch.zeros(M, KV, device=DEV, dtype=torch.float16),
            'I': torch.zeros(M, I, device=DEV, dtype=torch.float16),
            'V': torch.zeros(M, V, device=DEV, dtype=torch.float16)}

def lock_clock(mhz=1980): os.system(f'sudo nvidia-smi -lgc {mhz},{mhz} >/dev/null 2>&1')
def reset_clock():        os.system('sudo nvidia-smi -rgc >/dev/null 2>&1')

def gpu_now():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=clocks.sm,power.draw,utilization.gpu,utilization.memory',
         '--format=csv,noheader,nounits']).decode().strip().split(',')
    return dict(sm=float(out[0]), pw=float(out[1]), sm_util=float(out[2]), mem_util=float(out[3]))

def sample_under_load(g, secs=3.0):
    """Replay g continuously for `secs`, sampling SM%/MEM%/clock/power. Returns medians."""
    s = []
    t_end = None
    # warm
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    import time as _t
    start = _t.perf_counter()
    while _t.perf_counter() - start < secs:
        for _ in range(30): g.replay()
        s.append(gpu_now())   # sampled while GPU is busy with the queued replays
    torch.cuda.synchronize()
    med = lambda k: statistics.median([x[k] for x in s]) if s else 0.0
    return dict(sm_mhz=med('sm'), power_w=med('pw'), sm_util=med('sm_util'),
                mem_util=med('mem_util'), n_samples=len(s))

print(f"[setup] L={L} B={B} Mchunk={MCHUNK} smoke={args.smoke}", flush=True)
t0 = time.time()
W, Wlm = load_weights()
print(f"[setup] weights loaded {torch.cuda.memory_allocated()/1e9:.1f} GB in {time.time()-t0:.1f}s", flush=True)

# persistent input buffers + caches + maxbuf
xin_dec = (torch.randn(B, H, device=DEV) * 0.1).half()
xin_pre = (torch.randn(MCHUNK, H, device=DEV) * 0.1).half()
cache_dec = cache_buffers(B); cache_pre = cache_buffers(MCHUNK)
maxbuf = torch.zeros((), device=DEV, dtype=torch.float32)

results = {}

# ============================================================ PHASE 0: CALIBRATE
def phase_calib():
    print("[calib] capturing graphs...", flush=True)
    g_base_dec = capture(lambda: forward(xin_dec, W, Wlm))
    g_base_pre = capture(lambda: forward(xin_pre, W, Wlm))
    g_rec_dec  = capture(lambda: forward_recompute(xin_dec, W, Wlm, cache_dec, maxbuf))
    g_rec_pre  = capture(lambda: forward_recompute(xin_pre, W, Wlm, cache_pre, maxbuf))
    print("[calib] timing...", flush=True)
    t_base_dec = med_ms(g_base_dec); t_base_pre = med_ms(g_base_pre)
    t_rec_dec  = med_ms(g_rec_dec);  t_rec_pre  = med_ms(g_rec_pre)
    print("[calib] sampling saturation (decode)...", flush=True)
    sat_dec = sample_under_load(g_base_dec, secs=3.0)
    print("[calib] sampling saturation (prefill)...", flush=True)
    sat_pre = sample_under_load(g_base_pre, secs=3.0)
    cal = dict(
        B=B, Mchunk=MCHUNK, L=L,
        t_base_decode_ms=t_base_dec, t_base_prefill_ms=t_base_pre,
        t_recompute_decode_ms=t_rec_dec, t_recompute_prefill_ms=t_rec_pre,
        r_decode=t_rec_dec / t_base_dec, r_prefill=t_rec_pre / t_base_pre,
        decode_tok_s=B / (t_base_dec/1e3), prefill_tok_s=MCHUNK / (t_base_pre/1e3),
        saturation_decode=sat_dec, saturation_prefill=sat_pre)
    results['calib'] = cal
    json.dump(cal, open(f"{HERE}/calib.json", 'w'), indent=1)
    print(f"[calib] base dec {t_base_dec:.3f}ms ({cal['decode_tok_s']:.0f} tok/s) "
          f"pre {t_base_pre:.3f}ms ({cal['prefill_tok_s']:.0f} tok/s)", flush=True)
    print(f"[calib] recompute dec {t_rec_dec:.3f}ms (r={cal['r_decode']:.3f}) "
          f"pre {t_rec_pre:.3f}ms (r={cal['r_prefill']:.3f})", flush=True)
    print(f"[calib] decode SM%={sat_dec['sm_util']:.0f} MEM%={sat_dec['mem_util']:.0f} "
          f"clk={sat_dec['sm_mhz']:.0f} pw={sat_dec['power_w']:.0f}W", flush=True)
    print(f"[calib] prefill SM%={sat_pre['sm_util']:.0f} MEM%={sat_pre['mem_util']:.0f} "
          f"clk={sat_pre['sm_mhz']:.0f} pw={sat_pre['power_w']:.0f}W", flush=True)
    return g_base_dec, g_base_pre, g_rec_dec, g_rec_pre

# ============================================================ PHASE 1: TRACE
def gen_trace():
    """Realistic request trace -> continuous-batching scheduler -> ordered step list.
    Each request: prompt_len (mixed), decode_len (mixed), arrival_time (Poisson over wall).
    Scheduler: chunked prefill (MCHUNK), decode batch held near B (saturation). Returns the
    ordered list of ('decode'|'prefill', n_real_tokens) plus stats."""
    cal = results['calib']
    tb_dec = cal['t_base_decode_ms']; tb_pre = cal['t_base_prefill_ms']
    R = 8 if args.smoke else args.requests
    rng = random.Random(args.seed)
    def prompt_len():
        b = rng.random()
        if b < 0.50:  return rng.randint(64, 512)      # short
        if b < 0.85:  return rng.randint(512, 1536)    # medium
        return rng.randint(1536, 2048)                 # long-context
    def decode_len():
        b = rng.random()
        if b < 0.30:  return rng.randint(16, 64)
        if b < 0.70:  return rng.randint(64, 256)
        return rng.randint(256, 768)
    reqs = []
    # arrivals: Poisson; rate chosen to oversaturate (mean gap << decode step) so batch stays full
    t = 0.0
    mean_gap_ms = (tb_dec * 0.15) if not args.smoke else 1.0   # heavy backlog -> saturation
    for j in range(R):
        t += rng.expovariate(1.0 / mean_gap_ms)
        reqs.append(dict(id=j, arr=t, plen=prompt_len(), dlen=decode_len()))
    reqs.sort(key=lambda r: r['arr'])

    waiting = []           # arrived, prefill not finished: dict with prefilled tokens so far
    running = []           # decoding: dict with remaining decode tokens
    steps = []             # (type, n_real_tokens)
    wall = 0.0
    nxt = 0                # index into reqs by arrival
    pre_tokens = dec_tokens = 0
    while nxt < len(reqs) or waiting or running:
        # admit arrivals up to current wall
        while nxt < len(reqs) and reqs[nxt]['arr'] <= wall:
            r = reqs[nxt]; r['pre'] = 0; waiting.append(r); nxt += 1
        # if nothing admitted yet and nothing to do, jump wall to next arrival
        if not waiting and not running and nxt < len(reqs):
            wall = reqs[nxt]['arr']; continue
        # scheduler: prefill-prioritized chunked prefill while running batch has headroom or queue
        # has work; otherwise decode. Keeps running batch near capacity = saturation.
        do_prefill = bool(waiting) and (len(running) < B)
        if do_prefill:
            r = waiting[0]
            chunk = min(MCHUNK, r['plen'] - r['pre'])
            r['pre'] += chunk
            pre_tokens += chunk
            steps.append(('prefill', chunk))
            wall += tb_pre
            if r['pre'] >= r['plen']:
                r['rem'] = r['dlen']; running.append(r); waiting.pop(0)
        else:
            if not running:
                # no decode work; if queue empty wait for arrival
                if nxt < len(reqs): wall = reqs[nxt]['arr']; continue
                else: break
            n_real = len(running)
            dec_tokens += n_real
            steps.append(('decode', n_real))
            wall += tb_dec
            for r in running: r['rem'] -= 1
            running = [r for r in running if r['rem'] > 0]
        # admit arrivals that came due during this step
    n_dec = sum(1 for s in steps if s[0] == 'decode')
    n_pre = sum(1 for s in steps if s[0] == 'prefill')
    pre_wall = n_pre * tb_pre; dec_wall = n_dec * tb_dec; tot_wall = pre_wall + dec_wall
    tr = dict(requests=R, n_steps=len(steps), n_decode_steps=n_dec, n_prefill_steps=n_pre,
              prefill_tokens=pre_tokens, decode_tokens=dec_tokens,
              prefill_wall_ms=pre_wall, decode_wall_ms=dec_wall,
              prefill_wall_share=pre_wall/tot_wall, decode_wall_share=dec_wall/tot_wall,
              est_total_wall_s=tot_wall/1e3)
    results['trace'] = tr
    json.dump({**tr, 'steps': steps}, open(f"{HERE}/trace.json", 'w'))
    print(f"[trace] R={R} steps={len(steps)} (dec={n_dec} pre={n_pre}) "
          f"prefill-wall-share={tr['prefill_wall_share']*100:.1f}% "
          f"decode-wall-share={tr['decode_wall_share']*100:.1f}% est {tr['est_total_wall_s']:.1f}s",
          flush=True)
    return steps

def pick_N(steps):
    """Smallest N whose trace-weighted overhead < 3%. overhead(N) = sum_{check} t_recompute /
    sum_all t_base, checks every N steps (i = N-1, 2N-1, ...)."""
    cal = results['calib']
    tb = {'decode': cal['t_base_decode_ms'], 'prefill': cal['t_base_prefill_ms']}
    tr_ = {'decode': cal['t_recompute_decode_ms'], 'prefill': cal['t_recompute_prefill_ms']}
    base_wall = sum(tb[t] for t, _ in steps)
    table = {}
    chosen = None
    for N in range(1, 200):
        extra = sum(tr_[steps[i][0]] for i in range(N-1, len(steps), N))
        ov = 100.0 * extra / base_wall
        table[N] = ov
        if chosen is None and ov < 3.0:
            chosen = N
    results['N_overhead_table'] = {k: table[k] for k in sorted(table) if k <= 64}
    results['N_chosen'] = chosen
    print(f"[N] overhead: N=1 {table[1]:.1f}% N=8 {table[8]:.2f}% N=16 {table[16]:.2f}% "
          f"N=32 {table[32]:.2f}% N=34 {table[34]:.2f}% N=48 {table[48]:.2f}%", flush=True)
    print(f"[N] chosen (smallest <3%): N={chosen} -> {table[chosen]:.2f}%", flush=True)
    return chosen

# ============================================================ PHASE 2: THROUGHPUT (real wall)
def phase_throughput(steps, N, graphs):
    g_base_dec, g_base_pre, g_rec_dec, g_rec_pre = graphs
    cal = results['calib']
    def run(detector):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, (typ, _) in enumerate(steps):
            if typ == 'decode': g_base_dec.replay()
            else:               g_base_pre.replay()
            if detector and ((i+1) % N == 0):
                if typ == 'decode': g_rec_dec.replay()
                else:               g_rec_pre.replay()
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    # warm the queue
    for _ in range(10): g_base_dec.replay()
    torch.cuda.synchronize()
    sat = []  # sample clock/power during the real run
    w_off = run(False)
    clk_off = gpu_now()
    w_on  = run(True)
    clk_on = gpu_now()
    tr = results['trace']
    toks = tr['prefill_tokens'] + tr['decode_tokens']
    tps_off = toks / w_off; tps_on = toks / w_on
    delta = 100.0 * (w_on - w_off) / w_off
    out = dict(N=N, wall_off_s=w_off, wall_on_s=w_on,
               tok_s_off=tps_off, tok_s_on=tps_on,
               throughput_delta_pct=delta,
               decode_tok_s_off=tr['decode_tokens']/w_off, decode_tok_s_on=tr['decode_tokens']/w_on,
               clk_off=clk_off, clk_on=clk_on,
               n_check_steps=sum(1 for i in range(len(steps)) if (i+1)%N==0))
    results['throughput'] = out
    json.dump(out, open(f"{HERE}/throughput.json", 'w'), indent=1)
    print(f"[tput] OFF {w_off:.2f}s {tps_off:.0f} tok/s | ON(N={N}) {w_on:.2f}s {tps_on:.0f} tok/s "
          f"| delta {delta:+.2f}% | clk_on {clk_on['sm']:.0f}MHz {clk_on['pw']:.0f}W", flush=True)
    return out

# ============================================================ PHASE 3: DETECTION (eager, exact)
def eager_forward_cache(xin, fault=None):
    """Eager forward; store real per-(layer,name) outputs. `fault`=(layer,name) injects a Step-A
    top-exponent-bit flip into that cached output (models a corrupted STORED activation)."""
    x = xin; cache = {}
    def flip(y, key):
        if fault is not None and key == fault:
            flat = y.view(-1)
            idx = int(flat.shape[0]) // 2
            u = flat.view(torch.int16)
            before = float(flat[idx])
            u[idx] ^= (1 << 14)           # top-exponent bit (Step-A harmful class)
            after = float(flat[idx])
            eager_forward_cache.last_delta = abs(after - before)
        return y
    for i in range(L):
        h = rms(x)
        q = flip(F.linear(h, W['q'][i]), (i,'q')); k = flip(F.linear(h, W['k'][i]), (i,'k'))
        v = flip(F.linear(h, W['v'][i]), (i,'v')); o = flip(F.linear(q, W['o'][i]), (i,'o'))
        cache[(i,'q')]=q; cache[(i,'k')]=k; cache[(i,'v')]=v; cache[(i,'o')]=o
        x = (x + o).clamp_(-30, 30)
        h2 = rms(x)
        gt = flip(F.linear(h2, W['gate'][i]), (i,'gate')); up = flip(F.linear(h2, W['up'][i]), (i,'up'))
        dn = flip(F.linear(F.silu(gt)*up, W['down'][i]), (i,'down'))
        cache[(i,'gate')]=gt; cache[(i,'up')]=up; cache[(i,'down')]=dn
        x = (x + dn).clamp_(-30, 30)
    lg = flip(F.linear(rms(x), Wlm), (-1,'lm')); cache[(-1,'lm')]=lg
    return cache

def eager_recompute_residual(xin, cache):
    """Independent CLEAN recompute -> nan/inf-aware max |recompute - cache| (fp16 diff). Returns
    +inf if any output differs in finiteness (a flip into the fp16 inf/nan range is a real, caught
    corruption); otherwise the finite max abs diff. Clean (bit-identical) => exactly 0."""
    import math
    x = xin; m = 0.0; nonfinite = False
    def cmp(y, key):
        nonlocal nonfinite
        d = (y - cache[key]).abs()
        # a nan diff arises iff exactly one of (y, cache) is nan/inf -> corruption present
        if not bool(torch.isfinite(d).all()):
            nonfinite = True
            return float(torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).amax())
        return float(d.amax())
    for i in range(L):
        h = rms(x)
        q = F.linear(h, W['q'][i]); m=max(m,cmp(q,(i,'q')))
        k = F.linear(h, W['k'][i]); m=max(m,cmp(k,(i,'k')))
        v = F.linear(h, W['v'][i]); m=max(m,cmp(v,(i,'v')))
        o = F.linear(q, W['o'][i]); m=max(m,cmp(o,(i,'o')))
        x = (x + o).clamp_(-30, 30)
        h2 = rms(x)
        gt = F.linear(h2, W['gate'][i]); m=max(m,cmp(gt,(i,'gate')))
        up = F.linear(h2, W['up'][i]);   m=max(m,cmp(up,(i,'up')))
        dn = F.linear(F.silu(gt)*up, W['down'][i]); m=max(m,cmp(dn,(i,'down')))
        x = (x + dn).clamp_(-30, 30)
    lg = F.linear(rms(x), Wlm); m=max(m,cmp(lg,(-1,'lm')))
    return float('inf') if nonfinite else m

def phase_detect(steps, N):
    rng = random.Random(args.seed + 7)
    names = ['q','k','v','o','gate','up','down']
    # ---- 3a clean residual over many steps -> T, and FALSE POSITIVES over the whole clean trace
    print("[detect] clean residual sweep (T calibration + FP count)...", flush=True)
    clean_res = []
    n_clean_checks = sum(1 for i in range(len(steps)) if (i+1)%N==0)
    # sample clean residuals: run recompute on a sample of clean check-steps (eager is slow; sample)
    n_sample = 80 if not args.smoke else 4
    for s in range(n_sample):
        M = B if rng.random() < 0.7 else MCHUNK
        xin = (torch.randn(M, H, device=DEV)*0.1).half()
        cache = eager_forward_cache(xin, fault=None)
        clean_res.append(eager_recompute_residual(xin, cache))
    T_max_clean = max(clean_res)
    T = T_max_clean  # 0 FP by construction: trigger iff residual > T
    fp = sum(1 for r in clean_res if r > T)  # by definition 0 at T=max; report the distribution
    # extrapolated FP over the full clean trace: all clean check-steps have residual <= T_max_clean
    # ---- 3b TRANSIENT single-step faults: caught iff the faulted step is a check step (latency 0)
    print("[detect] transient single-step injections...", flush=True)
    n_inj = 6 if args.smoke else 60
    trans = []
    for _ in range(n_inj):
        M = B if rng.random() < 0.7 else MCHUNK
        layer = rng.randrange(L); name = rng.choice(names)
        xin = (torch.randn(M, H, device=DEV)*0.1).half()
        cache = eager_forward_cache(xin, fault=(layer, name))
        delta = getattr(eager_forward_cache, 'last_delta', 0.0)
        res = eager_recompute_residual(xin, cache)   # recompute is clean -> residual = |delta|
        caught_if_checked = res > T
        trans.append(dict(layer=layer, name=name, delta=delta, residual=res,
                          caught_if_check_step=bool(caught_if_checked),
                          margin=(res/T if T > 0 else float('inf'))))
    caught_when_checked = sum(1 for t in trans if t['caught_if_check_step'])
    n_finite_harmful = sum(1 for t in trans if t['residual'] != float('inf') and t['residual'] >= 2.76)
    n_inf = sum(1 for t in trans if t['residual'] == float('inf'))
    # transient coverage over a random check schedule: P(faulted step is a check step) = 1/N
    transient_coverage = 1.0 / N
    # ---- 3c PERSISTENT (stuck) fault: corrupts every step -> caught at first check within N steps
    print("[detect] persistent stuck-fault latency...", flush=True)
    n_pers = 4 if args.smoke else 30
    pers = []
    for _ in range(n_pers):
        layer = rng.randrange(L); name = rng.choice(names)
        inj_step = rng.randrange(0, 3*N)           # fault turns on at this trace step (covers full rem range)
        # next check step at index k where (k+1)%N==0 and k>=inj_step
        first_check = ((inj_step // N) + 1) * N - 1
        latency = first_check - inj_step           # = N-1-(inj_step % N): every corrupted step is a check candidate
        # verify the check at that step catches it (residual>T) on a representative corrupted step
        M = B if rng.random() < 0.7 else MCHUNK
        xin = (torch.randn(M, H, device=DEV)*0.1).half()
        cache = eager_forward_cache(xin, fault=(layer, name))  # persistent => this step corrupted too
        res = eager_recompute_residual(xin, cache)
        pers.append(dict(layer=layer, name=name, inj_step=inj_step, caught_step=first_check,
                         latency_steps=latency, residual=res, caught=bool(res > T)))
    out = dict(
        N=N, T=T, T_max_clean=T_max_clean, clean_residuals=clean_res,
        clean_FP_on_sampled_checks=fp, n_clean_checks_full_trace=n_clean_checks,
        transient=dict(n=n_inj, caught_when_step_is_check=caught_when_checked,
                       all_detectable_when_checked=(caught_when_checked==n_inj),
                       n_finite_harmful_ge_2p76=n_finite_harmful, n_inf_nan=n_inf,
                       coverage_per_pass=transient_coverage, latency_steps_when_caught=0,
                       detail=trans[:12]),
        persistent=dict(n=n_pers, all_caught=all(p['caught'] for p in pers),
                        latency_min=min(p['latency_steps'] for p in pers),
                        latency_max=max(p['latency_steps'] for p in pers),
                        latency_mean=statistics.mean(p['latency_steps'] for p in pers),
                        latency_bound=N-1, detail=pers))
    results['detect'] = out
    json.dump(out, open(f"{HERE}/detect.json", 'w'), indent=1)
    print(f"[detect] T={T:.3g} (max clean residual) | clean FP on sampled checks={fp}/{n_sample}", flush=True)
    print(f"[detect] transient: {caught_when_checked}/{n_inj} caught when step IS a check "
          f"(coverage 1/N={transient_coverage*100:.1f}%, latency 0); "
          f"min residual margin {min(t['margin'] for t in trans):.0f}x", flush=True)
    print(f"[detect] persistent: all_caught={out['persistent']['all_caught']} "
          f"latency steps min/mean/max {out['persistent']['latency_min']}/"
          f"{out['persistent']['latency_mean']:.1f}/{out['persistent']['latency_max']} (bound N-1={N-1})",
          flush=True)
    return out

# ============================================================ DRIVER
graphs = phase_calib()
steps = gen_trace()
N = pick_N(steps)
phase_throughput(steps, N, graphs)
phase_detect(steps, N)

json.dump(results, open(f"{HERE}/run_e2e_summary.json", 'w'), indent=1, default=str)
print("[done] wrote run_e2e_summary.json", flush=True)
