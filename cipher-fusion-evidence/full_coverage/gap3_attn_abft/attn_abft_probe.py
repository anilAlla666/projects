#!/usr/bin/env python3
# GAP 3 — ABFT for attention-internal QK^T / PV SDC. STANDALONE PROBE.
# ============================================================================
# READ-ONLY w.r.t. production .so. No LD_PRELOAD / CUDA_INJECTION64_PATH, no vLLM,
# no monkeypatch. Pure-PyTorch measurement of two algebraic checksums for the
# attention block, which the periodic-recompute linear-GEMM detector is BLIND to
# (attention produces no linear-GEMM output; QK^T/PV are the named Step-A blind spot).
#
# Fault model = settled Step-A harmful class: fp16 top-exponent bit-14 flip into a
# stored value -> NaN/inf, or finite |delta| >= 2.76. (bit13 is 3/200; we use bit14.)
#
# TWO CHECKS (math verified before drafting, see GAP3 writeup):
#   O-row-checksum: rowsum_V = V @ 1_d ([sk]);  ref_O = P @ rowsum_V  ==  O @ 1_d  (== O row-sums)
#       catches PV-GEMM output flips. BLIND to score flips (ref uses the SAME corrupted P).
#   S-row-checksum: colsum_K = K^T @ 1_sk ([d]); ref_S = (Q @ colsum_K)*scale  ==  S @ 1_sk (raw, pre-mask)
#       catches QK^T score flips. Computed on RAW pre-mask S (post-mask row-sum is -inf -> dead).
#
# RESIDUAL BLIND CLASS (proven): finite softmax-INTERNAL corruption (post-S, pre-PV):
#   a finite flip in P, or in the online-softmax bookkeeping (m_i, l_i, exp, correction),
#   scales/shifts O and its checksum reference IDENTICALLY -> invisible to BOTH.
#   (NaN/inf softmax-internal IS caught by the explicit isnan/isinf guard on O.)
#   Also blind by construction (caught upstream, not here): Q/K/V *input* corruption ->
#   q/k/v_proj LINEAR checksum (Step-A). A flip in stored V is blind to the O-check
#   (rowsum_V derives from the same V), exactly analogous to score-flip-blind-to-O.
# ============================================================================
import json, math, time, statistics
import torch

torch.manual_seed(1234)
dev = 'cuda'
assert torch.cuda.is_available()

# Mistral-7B shapes. GQA derived from shapes (NOT the "8:1" label):
#   q heads = H/head_dim = 4096/128 = 32 ; KV heads = KV/head_dim = 1024/128 = 8 ; group = 32/8 = 4.
HEAD_DIM = 128
N_Q_HEADS = 4096 // HEAD_DIM   # 32
N_KV_HEADS = 1024 // HEAD_DIM  # 8
GROUP = N_Q_HEADS // N_KV_HEADS  # 4
SCALE = 1.0 / math.sqrt(HEAD_DIM)
OUT = '/home/ubuntu/cipher-fusion-evidence/full_coverage/gap3_attn_abft'

# ---- Step-A harmful-class injector: fp16 top-exponent bit-14 flip ----
# The fault model is a bit flip in a STORED value. Attention intermediates (S,P,O) are
# stored in fp16 in real kernels, so we cast the targeted scalar to fp16, flip bit-14 of its
# int16 bit-pattern, and write the (possibly NaN/inf or large-finite) fp16 value back. The
# host tensors here are fp32 (fp32-accum discipline); we round-trip the single element through
# fp16 so the injected delta exactly matches the Step-A harmful class.
def flip_bit14_fp16(v_fp16_scalar):
    i = v_fp16_scalar.view(torch.int16)
    i = i ^ torch.tensor(1 << 14, dtype=torch.int16, device=i.device)
    return i.view(torch.float16)

def inject_flip(t_fp32, idx):
    """Flip bit-14 of the fp16 representation of t_fp32[idx], write back. Returns (delta, naninf)."""
    old = float(t_fp32[idx])
    v16 = t_fp32[idx].to(torch.float16).reshape(1)      # 1-dim so .view(int16) is legal
    new16 = flip_bit14_fp16(v16)[0]
    naninf = bool(torch.isnan(new16) or torch.isinf(new16))
    t_fp32[idx] = new16.to(torch.float32)
    delta = float('inf') if naninf else abs(float(new16) - old)
    return delta, naninf


# ---- realistic per-head Q,K,V at Mistral activation magnitude ----
# Step-A used activations of ~0.1 std post-RMSNorm-scale; attention Q/K ~ N(0, ~1) after proj
# then scaled by 1/sqrt(d). We use std that produces score magnitudes in the realistic range.
def make_qkv(seqq, seqk, n_kv_heads=N_KV_HEADS, dtype=torch.float16):
    # one tensor per KV head; q-heads in a group share that head's K,V.
    Q = (torch.randn(N_Q_HEADS, seqq, HEAD_DIM, device=dev) * 1.0).to(dtype)
    K = (torch.randn(n_kv_heads, seqk, HEAD_DIM, device=dev) * 1.0).to(dtype)
    Vv = (torch.randn(n_kv_heads, seqk, HEAD_DIM, device=dev) * 1.0).to(dtype)
    return Q, K, Vv


def attn_forward(Q, K, Vv, causal):
    """Dense reference attention per head with GQA broadcast. Returns S_raw, P, O (all per q-head).
    fp16 matmuls (as deployed); softmax in fp32 for stability (HF/FA do exp in fp32)."""
    seqq = Q.shape[1]; seqk = K.shape[2] if K.dim() == 3 else K.shape[1]
    Os, Ss, Ps = [], [], []
    for hq in range(N_Q_HEADS):
        hkv = hq // GROUP
        q = Q[hq]; k = K[hkv]; v = Vv[hkv]
        S_raw = (q.float() @ k.float().t()) * SCALE          # [sq,sk] fp32-accum of fp16 inputs
        S = S_raw.clone()
        if causal:
            seqq_, seqk_ = S.shape
            # causal: query position i (offset seqk-seqq) attends to key <= i
            qpos = torch.arange(seqq_, device=dev)[:, None] + (seqk_ - seqq_)
            kpos = torch.arange(seqk_, device=dev)[None, :]
            S = S.masked_fill(kpos > qpos, float('-inf'))
        P = torch.softmax(S, dim=-1)                          # [sq,sk] fp32
        O = P @ v.float()                                     # [sq,d] fp32
        Ss.append(S_raw); Ps.append(P); Os.append(O)
    return torch.stack(Ss), torch.stack(Ps), torch.stack(Os)  # [Hq,sq,sk],[Hq,sq,sk],[Hq,sq,d]


# ---- the two checks (fp32 accum, Step-A discipline) ----
def s_check_signal(S_raw, Q, K):
    """|rowsum(S_raw) - (Q@colsum_K)*SCALE| per q-head, max over rows. RAW pre-mask S."""
    sig = torch.zeros((), device=dev)
    for hq in range(N_Q_HEADS):
        hkv = hq // GROUP
        colsum_K = K[hkv].float().sum(dim=0)                  # [d]
        ref = (Q[hq].float() @ colsum_K) * SCALE              # [sq]
        chk = S_raw[hq].sum(dim=1)                            # [sq]
        d = (chk - ref).abs()
        # NaN-safe: any nan/inf in chk counts as caught (huge signal)
        if torch.isnan(d).any() or torch.isinf(d).any():
            return float('inf')
        sig = torch.maximum(sig, d.max())
    return float(sig)

def o_check_signal(O, P, Vv):
    """|rowsum(O) - P@rowsum_V| per q-head, max over rows."""
    sig = torch.zeros((), device=dev)
    for hq in range(N_Q_HEADS):
        hkv = hq // GROUP
        rowsum_V = Vv[hkv].float().sum(dim=1)                 # [sk]
        ref = P[hq].float() @ rowsum_V                        # [sq]
        chk = O[hq].sum(dim=1)                                # [sq]
        d = (chk - ref).abs()
        if torch.isnan(d).any() or torch.isinf(d).any():
            return float('inf')
        sig = torch.maximum(sig, d.max())
    return float(sig)


# ---- calibrate thresholds on CLEAN attention (0 clean false positives) ----
def calibrate_T(seqq, seqk, causal, reps=24):
    Ts, To = 0.0, 0.0
    for _ in range(reps):
        Q, K, Vv = make_qkv(seqq, seqk)
        S_raw, P, O = attn_forward(Q, K, Vv, causal)
        Ts = max(Ts, s_check_signal(S_raw, Q, K))
        To = max(To, o_check_signal(O, P, Vv))
    return Ts, To


# ============================================================================
# DETECTION EXPERIMENT: inject Step-A flips at 3 sites, report per-site rate.
# ============================================================================
def run_regime(name, seqq, seqk, causal, n_trials=200):
    Ts_raw, To_raw = calibrate_T(seqq, seqk, causal)
    # attention checksum ref uses a DIFFERENT fp evaluation order than the output (chk=sum(P@V) vs
    # ref=P@sum(V)), so clean roundoff is NONZERO (unlike the bit-exact GEMM recompute where T=0).
    # Operating T = SAFETY * max-clean-roundoff so clean FP=0; sites A/B margins are ~1e4-1e7x so this
    # does NOT dent designed coverage; it only shrinks the incidental site-C roundoff coverage (the
    # pre-registered softmax-internal BLIND residual), which is correct to surrender for 0 FP.
    SAFETY = 8.0
    Ts, To = Ts_raw * SAFETY, To_raw * SAFETY
    res = {'regime': name, 'seqq': seqq, 'seqk': seqk, 'causal': causal,
           'T_s_raw': Ts_raw, 'T_o_raw': To_raw, 'safety': SAFETY, 'T_s': Ts, 'T_o': To, 'n_trials': n_trials}
    print(f"\n=== {name}  seqq={seqq} seqk={seqk} causal={causal} ===")
    print(f"  T_s(roundoff floor x{SAFETY:.0f}) = {Ts:.4e}   T_o = {To:.4e}  (raw {Ts_raw:.2e}/{To_raw:.2e})")

    # ---- clean false-positive check (100 fresh clean runs) ----
    fp_s = fp_o = 0
    for _ in range(100):
        Q, K, Vv = make_qkv(seqq, seqk); S_raw, P, O = attn_forward(Q, K, Vv, causal)
        if s_check_signal(S_raw, Q, K) > Ts: fp_s += 1
        if o_check_signal(O, P, Vv) > To: fp_o += 1
    res['clean_fp_s'] = fp_s; res['clean_fp_o'] = fp_o
    print(f"  clean false positives (100 runs): S-check={fp_s}  O-check={fp_o}")

    def caught(sig_s, sig_o):
        # naninf -> inf signal -> caught; finite -> threshold
        return (sig_s > Ts) or (sig_o > To) or math.isinf(sig_s) or math.isinf(sig_o)

    sites = {'A_score_S': 0, 'B_pv_output_O': 0, 'C_softmax_internal_P': 0}
    site_naninf = {'A_score_S': 0, 'B_pv_output_O': 0, 'C_softmax_internal_P': 0}
    margins = {'A_score_S': [], 'B_pv_output_O': [], 'C_softmax_internal_P': []}

    for t in range(n_trials):
        Q, K, Vv = make_qkv(seqq, seqk)
        S_raw, P, O = attn_forward(Q, K, Vv, causal)
        hq = torch.randint(0, N_Q_HEADS, (1,)).item()
        ri = torch.randint(0, seqq, (1,)).item()

        # ---- SITE A: flip a SCORE in raw S (post-QK^T, pre-softmax), valid (unmasked) col ----
        # valid col for query ri: <= ri + (seqk-seqq)
        maxc = ri + (seqk - seqq) if causal else seqk - 1
        ci = torch.randint(0, max(1, maxc + 1), (1,)).item()
        Sa = S_raw.clone(); delta, naninf = inject_flip(Sa[hq], (ri, ci))
        # re-derive P,O from corrupted S (recompute mask)
        Sm = Sa[hq].clone()
        if causal:
            qpos = torch.arange(seqq, device=dev)[:, None] + (seqk - seqq)
            kpos = torch.arange(seqk, device=dev)[None, :]
            Sm = Sm.masked_fill(kpos > qpos, float('-inf'))
        Pa = P.clone(); Pa[hq] = torch.softmax(Sm, dim=-1)
        Oa = O.clone(); Oa[hq] = Pa[hq] @ Vv[hq // GROUP].float()
        sg = s_check_signal(Sa, Q, K); og = o_check_signal(Oa, Pa, Vv)
        if caught(sg, og):
            sites['A_score_S'] += 1
        if naninf: site_naninf['A_score_S'] += 1
        else: margins['A_score_S'].append(max(sg / Ts, og / To))

        # ---- SITE B: flip a PV OUTPUT element in O (post P@V) ----
        Ob = O.clone(); ci2 = torch.randint(0, HEAD_DIM, (1,)).item()
        Ob16 = Ob[hq].to(torch.float16)
        delta, naninf = inject_flip(Ob16, (ri, ci2))
        Ob[hq] = Ob16.float()
        sg = s_check_signal(S_raw, Q, K); og = o_check_signal(Ob, P, Vv)
        if caught(sg, og):
            sites['B_pv_output_O'] += 1
        if naninf: site_naninf['B_pv_output_O'] += 1
        else: margins['B_pv_output_O'].append(max(sg / Ts, og / To))

        # ---- SITE C: SOFTMAX-INTERNAL: flip a P probability (post-softmax, pre-PV) ----
        Pc = P.clone(); maxc = ri + (seqk - seqq) if causal else seqk - 1
        ci3 = torch.randint(0, max(1, maxc + 1), (1,)).item()
        Pc16 = Pc[hq].to(torch.float16)
        delta, naninf = inject_flip(Pc16, (ri, ci3))
        Pc[hq] = Pc16.float()
        Oc = O.clone(); Oc[hq] = Pc[hq] @ Vv[hq // GROUP].float()
        # O-check ref uses the SAME corrupted Pc -> should MATCH (blind), unless NaN
        sg = s_check_signal(S_raw, Q, K); og = o_check_signal(Oc, Pc, Vv)
        if caught(sg, og):
            sites['C_softmax_internal_P'] += 1
        if naninf: site_naninf['C_softmax_internal_P'] += 1
        else: margins['C_softmax_internal_P'].append(max(sg / Ts, og / To))

    res['detect'] = {k: {'caught': v, 'rate': 100.0 * v / n_trials,
                         'naninf': site_naninf[k],
                         'finite_min_margin': (min(margins[k]) if margins[k] else None),
                         'finite_n': len(margins[k])}
                     for k, v in sites.items()}
    for k, d in res['detect'].items():
        mm = d['finite_min_margin']
        print(f"  site {k:22s}: {d['caught']}/{n_trials} = {d['rate']:6.2f}%   "
              f"naninf={d['naninf']:3d}  finite_min_margin="
              f"{('%.1fx' % mm) if mm is not None else 'n/a'}")
    return res


# ============================================================================
# FlashAttn STREAMING verification: rowsum_V as appended checksum column survives
# online rescaling (PRESERVED); l_i / correction-factor fault is BLIND (subsumed).
# ============================================================================
def fa_streaming_check(seqq=64, seqk=512, B=64):
    Q = torch.randn(seqq, HEAD_DIM, device=dev)
    K = torch.randn(seqk, HEAD_DIM, device=dev)
    Vv = torch.randn(seqk, HEAD_DIM, device=dev)
    rowsum_V = Vv.sum(dim=1, keepdim=True)
    V_aug = torch.cat([Vv, rowsum_V], dim=1)          # [sk, d+1]

    def online(corrupt_l_row=None):
        O = torch.zeros(seqq, HEAD_DIM + 1, device=dev)
        m = torch.full((seqq,), float('-inf'), device=dev)
        l = torch.zeros(seqq, device=dev)
        for j in range(0, seqk, B):
            Kb = K[j:j + B]; Vb = V_aug[j:j + B]
            Sb = (Q @ Kb.t()) * SCALE
            mb = Sb.max(dim=1).values; m_new = torch.maximum(m, mb)
            corr = torch.exp(m - m_new)
            p = torch.exp(Sb - m_new[:, None])
            l = l * corr + p.sum(dim=1)
            O = O * corr[:, None] + p @ Vb
            m = m_new
        if corrupt_l_row is not None:
            l[corrupt_l_row] *= 1.5                    # denominator (l_i) fault
        return O / l[:, None]

    Oa = online()
    clean = float((Oa[:, HEAD_DIM] - Oa[:, :HEAD_DIM].sum(1)).abs().max())
    Ob = online(corrupt_l_row=2)
    blind = float((Ob[:, HEAD_DIM] - Ob[:, :HEAD_DIM].sum(1)).abs().max())
    # confirm O genuinely wrong despite blind check
    wrong = float((Ob[2, :HEAD_DIM] - Oa[2, :HEAD_DIM]).abs().max())
    print("\n=== FlashAttn streaming O-checksum (appended col) ===")
    print(f"  clean |chkcol - O.sum| = {clean:.3e}  <- rescaling cancels (PRESERVED)")
    print(f"  l_i-fault |chkcol - O.sum| = {blind:.3e}  <- BLIND (scales O and col identically)")
    print(f"  but O genuinely wrong (row2) = {wrong:.3e}  <- finite softmax-internal residual")
    return {'clean': clean, 'l_fault_blind': blind, 'o_wrong': wrong}


# ============================================================================
# FLOP FLOOR (analytical) — STANDALONE-PROBE estimate; realizable floor needs FA-epilogue fusion.
# ============================================================================
def flop_floor():
    # per q-head, attention FLOP: S=QK^T = 2*sq*sk*d ; PV = 2*sq*sk*d ; total ~ 4*sq*sk*d
    # O-check ref P@rowsum_V = 2*sq*sk  ; S-check ref Q@colsum_K = 2*sq*d
    # colsum_K/rowsum_V reductions: O(sk*d) each, amortized/incremental in decode.
    out = {}
    for nm, sq, sk in [('prefill_seq512', 512, 512), ('decode', 1, 512)]:
        attn = 4 * sq * sk * HEAD_DIM
        o_ref = 2 * sq * sk
        s_ref = 2 * sq * HEAD_DIM
        # reductions counted once (recompute-from-scratch upper bound)
        reduce = 2 * sk * HEAD_DIM * 2  # colsum_K + rowsum_V
        out[nm] = {
            'attn_flop': attn,
            'o_ref_pct': 100.0 * o_ref / attn,
            's_ref_pct': 100.0 * s_ref / attn,
            'combined_check_pct': 100.0 * (o_ref + s_ref) / attn,
            'with_scratch_reductions_pct': 100.0 * (o_ref + s_ref + reduce) / attn,
        }
    return out


if __name__ == '__main__':
    results = {'shapes': {'HEAD_DIM': HEAD_DIM, 'N_Q_HEADS': N_Q_HEADS,
                          'N_KV_HEADS': N_KV_HEADS, 'GROUP': GROUP}}
    results['regimes'] = []
    # prefill: causal, sq=sk=512 ; decode: 1 query against KV-len 512 (non-causal, all keys valid)
    results['regimes'].append(run_regime('prefill', 512, 512, causal=True, n_trials=200))
    results['regimes'].append(run_regime('decode', 1, 512, causal=False, n_trials=200))
    results['fa_streaming'] = fa_streaming_check()
    results['flop_floor'] = flop_floor()
    print("\n=== FLOP FLOOR (STANDALONE-PROBE estimate; realizable needs FA-epilogue fusion) ===")
    for nm, d in results['flop_floor'].items():
        print(f"  {nm}: O-check {d['o_ref_pct']:.3f}%  S-check {d['s_ref_pct']:.3f}%  "
              f"combined {d['combined_check_pct']:.3f}%  (+scratch-reductions {d['with_scratch_reductions_pct']:.2f}%)")
    json.dump(results, open(f'{OUT}/attn_abft_result.json', 'w'), indent=1)
    print(f"\nwrote {OUT}/attn_abft_result.json")
