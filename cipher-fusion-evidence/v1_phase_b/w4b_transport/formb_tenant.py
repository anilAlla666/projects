"""W.4b.2 Form B' prototype — TENANT process (injected) (§5 items 1,3,6).

Launched INJECTED (CUDA_INJECTION64_PATH=libcipher_rt.so) so it (a) produces a
real W.6 sub-B fingerprint and (b) auto-registers into the NR 30/31 co-residence
cohort once warmed. It is a real, independent GPU process running the model —
this is what gives the eligibility gate genuine admission work (memo §0).

Steady state it does NOT run the transformer body: it embeds the previous token,
ships the [1,hidden] activation to the vanilla executor via cuIPC, blocks on the
per-token barrier, receives its [1,vocab] logit row, argmaxes, ships the next
embed. The executor runs the batched body (the lift). If the gate denies (solo),
the tenant runs its own full forward locally (negative-control / blocked path).

Correctness (§5 item 6): after the batched run the tenant runs its own solo B=1
greedy on the same prompt and computes per-step logit-KL(batched-row || solo)
plus a token-match check (CP 5.6 gate, tol ~5e-5).
"""
import os
import sys
import json
import time
import socket
import ctypes

MODEL = os.environ["WL_MODEL"]
SOCK = os.environ["FORMB_SOCK"]
ROW = int(os.environ["TENANT_ROW"])
PLEN = int(os.environ["PLEN"])
GEN = int(os.environ["GEN_LEN"])
PROMPT = os.environ["PROMPT"]
RESULT = os.environ["RESULT_JSON"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402
import formb_ipc as ipc            # noqa: E402

DEV = "cuda"
# W.4b.7 item 1: env-gated per-phase decomposition (tenant side). OFF by default.
PROFILE = os.environ.get("CIPHER_FORMB_PROFILE", "0") == "1"


_RT = os.environ.get("CIPHER_RT_LIB", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")


def own_fingerprint():
    try:
        # injected lib is RTLD_LOCAL; load by path to read in-process g_signals
        libc = ctypes.CDLL(_RT, mode=os.RTLD_LOCAL | 0x0001)
        libc.cipher_workload_model_fingerprint.restype = ctypes.c_uint64
        return int(libc.cipher_workload_model_fingerprint())
    except Exception as e:
        print("own_fingerprint err: %s" % e, file=sys.stderr, flush=True)
        return 0


def solo_greedy(m, embed, ids):
    """B=1 greedy reference; returns (tokens, per_step_logits[gen, vocab])."""
    toks, logs = [], []
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
        past = out.past_key_values
        cur = ids.shape[1]
        lg = out.logits[:, -1, :]
        for _ in range(GEN):
            logs.append(lg[0].float().clone())
            nt = lg.argmax(-1)
            toks.append(int(nt))
            cp = torch.tensor([cur], device=DEV)
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past,
                    use_cache=True, cache_position=cp)
            past = out.past_key_values
            lg = out.logits[:, -1, :]
            cur += 1
    return toks, torch.stack(logs)


def main():
    ipc.bridge_init()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    embed = m.get_input_embeddings()
    hidden = m.config.hidden_size

    # fixed-length prompt to EXACTLY PLEN (right-pad if short; tokenizers vary by
    # model). batched and solo see the same padded ids, so the comparison holds.
    enc = tok(PROMPT, return_tensors="pt").input_ids[0]
    if enc.shape[0] >= PLEN:
        ids = enc[:PLEN].unsqueeze(0).cuda()
    else:
        padv = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.cat([enc, torch.full((PLEN - enc.shape[0],), padv,
                                         dtype=enc.dtype)]).unsqueeze(0).cuda()

    # warmup: enough forwards for the substrate to observe LM-head/FFN and the
    # fp to stabilize (registers into NR 30/31 via the classifier heartbeat)
    with torch.no_grad():
        for _ in range(8):
            m(ids)
    torch.cuda.synchronize()
    fp = own_fingerprint()

    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(100):
        try:
            c.connect(SOCK); break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    ipc.send_msg(c, {"hello": 1, "tgid": os.getpid(), "row": ROW,
                     "model": MODEL, "fp": fp, "plen": PLEN})
    gate = ipc.recv_msg(c)            # {"mode": "batch"|"solo", ...}
    mode = gate["mode"]

    res = {"row": ROW, "tgid": os.getpid(), "fp": fp, "model": MODEL, "mode": mode}

    if mode == "solo":
        # gate denied (distinct fp / not eligible): run own full forward
        toks, _ = solo_greedy(m, embed, ids)
        res.update({"tokens": toks, "note": "ran solo (gate denied / blocked)"})
        json.dump(res, open(RESULT, "w"), indent=2)
        print("TENANT row=%d SOLO tokens0..4=%s" % (ROW, toks[:5]),
              file=sys.stderr, flush=True)
        c.close(); return

    # ---- solo FP16 gold trajectory FIRST (the teacher-forcing reference) ----
    solo_toks, solo_logs = solo_greedy(m, embed, ids)

    # ---- batched path: cuIPC setup ----
    embeds_out = ipc.IpcOut([PLEN, hidden], torch.float16, tenant_id=ROW + 100)
    ipc.send_buffer(c, embeds_out, extra={"role": "embeds", "row": ROW})
    logit_in, _ = ipc.recv_buffer(c)       # executor's [1,vocab] logit buffer for us
    vocab = m.config.vocab_size

    # TEACHER-FORCED (CP 5.6 gate, memo §5 item 6): ship the GOLD token each step
    # so both paths see the identical input trajectory -> per-step logit KL is a
    # pure batched-vs-unbatched FP-reduction-order difference (no free-running KV
    # drift accumulation). batched_argmax[t] must still equal the gold token.
    batched_argmax, batched_logs = [], []
    with torch.no_grad():
        emb = embed(ids)[0]                 # [plen, hidden]
        embeds_out.write(emb)
        ipc.send_msg(c, {"phase": "prefill"})
        resp = ipc.recv_msg(c)              # {"done_step":0} OR {"fallback":"solo"}
        if resp and resp.get("fallback"):
            # gate admitted then the first-coalesce correctness backstop BLOCKED
            # the group -> abandon batch, emit our own correct solo output.
            ipc.send_msg(c, {"done": 1}); c.close()
            res.update({"mode": "solo_fallback", "tokens": solo_toks,
                        "tok_match": len(solo_toks), "tok_total": len(solo_toks),
                        "kl_max": 0.0, "kl_mean": 0.0,
                        "note": "correctness BLOCK -> ran solo (no corruption)"})
            json.dump(res, open(RESULT, "w"), indent=2)
            print("TENANT row=%d SOLO_FALLBACK (BLOCK) tokens0..4=%s"
                  % (ROW, solo_toks[:5]), file=sys.stderr, flush=True)
            return
        lg = logit_in.read([vocab]).float()
        batched_logs.append(lg.clone())
        batched_argmax.append(int(lg.argmax(-1)))
        tpacc = {"embed": 0.0, "write_sync": 0.0, "send": 0.0,
                 "recv_wait": 0.0, "logit_read": 0.0}
        tpcnt = 0
        for t in range(1, GEN):
            _t0 = time.perf_counter() if PROFILE else 0.0
            forced = solo_toks[t - 1]       # gold input, NOT batched argmax
            e = embed(torch.tensor([[forced]], device=DEV))[0]   # [1, hidden]
            if PROFILE:
                torch.cuda.synchronize(); _te = time.perf_counter()
                tpacc["embed"] += _te - _t0
            embeds_out.write(e)                              # sync=True (host sync)
            if PROFILE:
                _tw = time.perf_counter(); tpacc["write_sync"] += _tw - _te
            ipc.send_msg(c, {"phase": "decode", "t": t})
            if PROFILE:
                _ts = time.perf_counter(); tpacc["send"] += _ts - _tw
            ipc.recv_msg(c)
            if PROFILE:
                _tr = time.perf_counter(); tpacc["recv_wait"] += _tr - _ts
            lg = logit_in.read([vocab]).float()
            batched_logs.append(lg.clone())
            batched_argmax.append(int(lg.argmax(-1)))
            if PROFILE:
                torch.cuda.synchronize()
                tpacc["logit_read"] += time.perf_counter() - _tr; tpcnt += 1
    if PROFILE and tpcnt:
        tms = {k: round(1000.0 * v / tpcnt, 3) for k, v in tpacc.items()}
        print("TENANT row=%d PROFILE (ms/step): %s" % (ROW,
              " ".join("%s=%.2f" % (k, tms[k]) for k in
                       ("embed", "write_sync", "send", "recv_wait", "logit_read"))),
              file=sys.stderr, flush=True)
        res["tenant_profile_ms"] = tms
    ipc.send_msg(c, {"done": 1})
    c.close()

    bl = torch.stack(batched_logs)          # [GEN, vocab]
    n = min(bl.shape[0], solo_logs.shape[0])
    lp = F.log_softmax(solo_logs[:n], dim=-1)
    lq = F.log_softmax(bl[:n], dim=-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)     # [n]
    match = sum(int(a == b) for a, b in zip(batched_argmax[:n], solo_toks[:n]))
    # Memory #11 rigor: any token divergence is logged with the SOLO top1-top2
    # logit gap at that step. A near-tie gap (<~0.05) => benign batched-vs-solo
    # FP-reduction-order greedy flip (KL stays ~1e-5); a large gap => real
    # corruption => HARD STOP signal.
    divergences = []
    for t in range(n):
        if batched_argmax[t] != solo_toks[t]:
            top2 = torch.topk(solo_logs[t], 2).values
            divergences.append({"step": t, "batched_tok": batched_argmax[t],
                                "solo_tok": solo_toks[t],
                                "solo_top1_top2_gap": float(top2[0] - top2[1]),
                                "step_kl": float(kl[t])})
    res.update({"batched_argmax": batched_argmax, "solo_tokens": solo_toks,
                "tok_match": match, "tok_total": n, "divergences": divergences,
                "kl_mean": float(kl.mean()), "kl_max": float(kl.max())})
    json.dump(res, open(RESULT, "w"), indent=2)
    print("TENANT row=%d BATCH tf_match=%d/%d kl_max=%.3e (teacher-forced) gold0..4=%s"
          % (ROW, match, n, float(kl.max()), solo_toks[:5]),
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
