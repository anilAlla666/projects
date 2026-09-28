"""W.4b.2 Form B' prototype — EXECUTOR process (VANILLA) (§5 items 3,4).

Runs NO injection (zero cuBLAS-shim tax — the W.4b.1 finding). It:
  1. reads its deployment-match fp from the one-time injected warmup (§2 #2a);
  2. reads the live NR 31 cohort (query-only) + runs the AUDITED W.4a guard
     cipher_rt_pool_partition (formb_cohort.partition) to admit ONLY co-resident
     tenants whose fp == its own deployed-model fp;
  3. if >=2 admitted, runs the batched B=N transformer step over cuIPC-gathered
     per-token embeds with EXECUTOR-OWNED batched KV (HF forward), and scatters
     each logit row back via cuIPC. Per-sequence attention over cuIPC-MAPPED PEER
     KV is the deferred R-4 surface (W.4b.2b) — here the KV is executor-owned.

This measures the per-token rendezvous + cuIPC gather/scatter plumbing gap vs the
in-process ceiling (the activation-transport half of R-2).
"""
import os
import sys
import json
import time
import socket

MODEL = os.environ["WL_MODEL"]
SOCK = os.environ["FORMB_SOCK"]
N = int(os.environ["N_TENANTS"])
FP_FILE = os.environ["FP_FILE"]
GEN = int(os.environ["GEN_LEN"])
RESULT = os.environ["RESULT_JSON"]
SENTINEL = os.environ.get("SENTINEL", "/tmp/formb.decode_window")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                       # noqa: E402
import numpy as np                 # noqa: E402
import formb_ipc as ipc            # noqa: E402
import formb_cohort as cohort      # noqa: E402

DEV = "cuda"
# correctness-backstop tolerance on the top-K-projected rel-diff (catastrophe
# guard, not a precision gate — the token-match is the precision signal). Legit
# batched-vs-solo on the top-K slice is ~1e-3; a wrong-model / forced-fault row is
# >> tol. TOPK keeps the comparison on large-magnitude logits where rel-diff is
# well-behaved (raw-logit rel-diff is dominated by near-zero entries).
TOL = float(os.environ.get("CIPHER_FORMB_TOL", "0.15"))
TOPK = int(os.environ.get("CIPHER_FORMB_TOPK", "64"))
FAULT_ROW = int(os.environ.get("CIPHER_FORMB_FAULT", "-1"))  # 3.3 forced-fault test
# W.4b.7 item 1: env-gated per-phase decomposition of the per-token rendezvous.
# Pure measurement (perf_counter for CPU phases + reused cuda Events for the GPU
# split); OFF by default ⇒ zero behavior change. The ground-truth step time is the
# un-profiled decode_wall; this only ATTRIBUTES it.
PROFILE = os.environ.get("CIPHER_FORMB_PROFILE", "0") == "1"


def main():
    ipc.bridge_init()
    executor_fp = int(open(FP_FILE).read().strip())
    self_tgid = os.getpid()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    hidden = m.config.hidden_size
    vocab = m.config.vocab_size

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(64)
    print("EXECUTOR listening %s fp=0x%016x" % (SOCK, executor_fp),
          file=sys.stderr, flush=True)

    conns, hellos = {}, {}
    for _ in range(N):
        conn, _a = srv.accept()
        h = ipc.recv_msg(conn)
        conns[h["row"]] = conn
        hellos[h["row"]] = h
    print("EXECUTOR all %d tenants connected; rows=%s" % (N, sorted(hellos)),
          file=sys.stderr, flush=True)

    # Wait for the cohort to reflect both warmed tenants AND for their fps to
    # CONVERGE. Registration is classifier-throttled (~1/s) and the fp is
    # unstable during warmup (it settles only once the LM-head shape is
    # observed), so "nonzero" is not enough — we must wait for the deployed-model
    # fp. A settle grace lets the negative control (a genuinely different model,
    # which will never converge to executor_fp) proceed without hanging.
    want_tgids = {hellos[r]["tgid"] for r in hellos}
    peers = []
    t0 = time.time()
    all_present_since = None
    while time.time() - t0 < 25:
        peers = cohort.query_cohort()
        present = {tg: fp for tg, fp in peers if tg in want_tgids}
        if len(present) == len(want_tgids):
            if all_present_since is None:
                all_present_since = time.time()
            matched = sum(1 for fp in present.values() if fp == executor_fp)
            if matched == len(want_tgids) or time.time() - all_present_since > 6:
                break
        time.sleep(0.5)

    # ---- the gate: audited W.4a partition over the live cohort ----
    dec = cohort.partition(self_tgid, executor_fp, peers)
    matched_tgids = set(dec["peer_tgids"])
    matched_rows = sorted(r for r in hellos if hellos[r]["tgid"] in matched_tgids)
    eligible = len(matched_rows) >= 2

    gate = {"executor_fp": executor_fp, "cohort": peers,
            "partition": dec, "matched_rows": matched_rows,
            "eligible_for_batch": eligible,
            "tenant_fps": {r: hellos[r]["fp"] for r in hellos}}
    print("EXECUTOR gate: eligible=%s matched_rows=%s n_in_group=%d "
          "distinct_rejected=%d" % (eligible, matched_rows, dec["n_in_group"],
                                    dec["distinct_rejected"]),
          file=sys.stderr, flush=True)

    if not eligible:
        for r in conns:
            ipc.send_msg(conns[r], {"mode": "solo",
                                    "reason": "gate not eligible (need >=2 same-fp)"})
        for r in conns:
            conns[r].close()
        srv.close(); os.unlink(SOCK)
        json.dump({"role": "executor", "gate": gate, "ran_batch": False},
                  open(RESULT, "w"), indent=2, default=str)
        return

    # admit matched rows to the batch; deny the rest (negative-control path)
    for r in conns:
        ipc.send_msg(conns[r], {"mode": "batch" if r in matched_rows else "solo"})

    # ---- cuIPC channels for the admitted group ----
    R = len(matched_rows)
    plen = hellos[matched_rows[0]]["plen"]
    embeds_in = {}
    logit_out = {}
    for r in matched_rows:
        ein, _mani = ipc.recv_buffer(conns[r])          # tenant's [plen,hidden] embeds
        embeds_in[r] = ein
        lo = ipc.IpcOut([1, vocab], torch.float16, tenant_id=r + 200)
        ipc.send_buffer(conns[r], lo, extra={"role": "logits", "row": r})
        logit_out[r] = lo

    def gather(decode):
        rows = []
        for r in matched_rows:
            v = embeds_in[r].view                        # [plen, hidden] read-only
            rows.append((v[0:1] if decode else v).clone())
        if decode:
            return torch.stack([x.reshape(1, hidden) for x in rows], 0)  # [R,1,h]
        return torch.stack(rows, 0)                                       # [R,plen,h]

    def scatter(logits_step):                            # [R, vocab]
        # W.4b.4: copy all R rows, then ONE sync (done by the caller) instead of
        # one sync per tenant buffer — collapses R syncs/step to 1.
        for i, r in enumerate(matched_rows):
            logit_out[r].write(logits_step[i].reshape(1, vocab), sync=False)

    # ---- batched decode (executor-owned KV; per-seq attn via HF mask) ----
    with open(SENTINEL, "w") as s:
        s.write("SETUP_DONE %.3f\n" % time.time())

    torch.cuda.synchronize()
    with torch.no_grad():
        # prefill
        for r in matched_rows:
            ipc.recv_msg(conns[r])                       # {"phase":"prefill"}
        emb = gather(decode=False)                       # [R, plen, hidden]
        attn = torch.ones((R, plen), device=DEV, dtype=torch.long)
        out = m(inputs_embeds=emb, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        prefill_logits = out.logits[:, -1, :]            # [R, vocab]

        # ===== W.4b.3 first-coalesce correctness backstop (audited W.4a guard) =====
        # Compare each tenant's batched-path row to its single-tenant solo B=1
        # reference (same gathered embeds). Catastrophe (wrong-model coalesce) or a
        # forced fault => fail => mark_blocked => the whole group falls to solo.
        group_fp, gK, gN, gdt = executor_fp, hidden, R, 1
        blocked = cohort.is_blocked(group_fp, gK, gN, gdt) == 1
        n_fail, per_pass, mad, mrd, argmax_div = 0, [1] * R, 0.0, 0.0, 0
        if not blocked:
            solo_rows, batched_rows = [], []
            for i, r in enumerate(matched_rows):
                so = m(inputs_embeds=emb[i:i + 1], attention_mask=attn[i:i + 1],
                       use_cache=True)
                solo_rows.append(so.logits[0, -1, :].float().cpu().numpy())
                batched_rows.append(prefill_logits[i].float().cpu().numpy())
            if 0 <= FAULT_ROW < R:                       # 3.3 synthetic divergence
                rng = np.random.RandomState(0)
                batched_rows[FAULT_ROW] = (rng.randn(batched_rows[FAULT_ROW].shape[0])
                                           .astype(np.float32) * 50.0)
            # Project each row onto the SOLO reference's top-K token indices before
            # the audited rel-diff guard. Raw-logit element-wise rel-diff is
            # dominated by near-zero entries (legit max_rel ~122 at max_abs ~0.02);
            # the top-K (large-magnitude) slice makes rel-diff meaningful and
            # catastrophe-sensitive (legit ~1e-3; wrong-model / fault row >> tol).
            proj_out, proj_ref = [], []
            for b, s in zip(batched_rows, solo_rows):
                idx = np.argsort(s)[-TOPK:]
                proj_ref.append(np.ascontiguousarray(s[idx]))
                proj_out.append(np.ascontiguousarray(b[idx]))
                if int(np.argmax(b)) != int(np.argmax(s)):
                    argmax_div += 1
            mad = max(float(np.max(np.abs(b - s)))
                      for b, s in zip(proj_out, proj_ref))
            mrd = max(float(np.max(np.abs(b - s) / np.maximum(np.abs(s), 1e-12)))
                      for b, s in zip(proj_out, proj_ref))
            n_fail, per_pass = cohort.correctness_check(proj_out, proj_ref, TOL)
            print("FIRSTCOALESCE topk=%d max_abs=%.4f max_rel=%.4f tol=%.3f "
                  "fails=%d pass=%s argmax_div=%d"
                  % (TOPK, mad, mrd, TOL, n_fail, per_pass, argmax_div),
                  file=sys.stderr, flush=True)
            if n_fail > 0:
                cohort.mark_blocked(group_fp, gK, gN, gdt)
                blocked = True
        if blocked:
            # 3.2 ratio auto-disable: 1 group offered, 1 blocked => 100% > 50%
            # session-wide disable -> every tenant in the group falls solo.
            for r in matched_rows:
                ipc.send_msg(conns[r], {"fallback": "solo", "reason": "correctness BLOCK"})
            for r in matched_rows:
                ipc.recv_msg(conns[r]); conns[r].close()
            srv.close(); os.unlink(SOCK)
            json.dump({"role": "executor", "gate": gate, "ran_batch": False,
                       "blocked": True,
                       "block": {"fp": group_fp, "K": gK, "N": gN, "dtype": gdt,
                                 "n_fail": n_fail, "per_pass": per_pass,
                                 "max_abs": round(mad, 4), "max_rel": round(mrd, 4),
                                 "tol": TOL, "topk": TOPK, "argmax_div": argmax_div,
                                 "fault_row": FAULT_ROW},
                       "ratio_disable": {"groups_offered": 1, "groups_blocked": 1,
                                         "pct": 100.0, "session_disabled": True},
                       "counters": cohort.counters()},
                      open(RESULT, "w"), indent=2, default=str)
            print("EXECUTOR BLOCK fired -> group falls solo (session disabled); "
                  "counters=%s" % cohort.counters(), file=sys.stderr, flush=True)
            return
        # ===== passed: commit the group to batched decode =====
        scatter(prefill_logits)
        torch.cuda.synchronize()
        for r in matched_rows:
            ipc.send_msg(conns[r], {"done_step": 0})
        cur = plen

        # W.4b.7 item 1: per-phase accumulators (CPU phases via perf_counter; GPU
        # split via reused Events read AFTER the existing per-step sync at line ~253,
        # where they are guaranteed complete). Profiling adds only cheap timers.
        pacc = {"recv": 0.0, "enqueue": 0.0, "sync_wait": 0.0, "send": 0.0,
                "gpu_gather": 0.0, "gpu_forward": 0.0, "gpu_scatter": 0.0}
        pcnt = 0
        if PROFILE:
            ev_g0 = torch.cuda.Event(enable_timing=True)
            ev_g1 = torch.cuda.Event(enable_timing=True)
            ev_f1 = torch.cuda.Event(enable_timing=True)
            ev_s1 = torch.cuda.Event(enable_timing=True)

        with open(SENTINEL, "a") as s:
            s.write("DECODE_START %.3f\n" % time.time())
        td0 = time.perf_counter()
        for t in range(1, GEN):
            _t0 = time.perf_counter() if PROFILE else 0.0
            for r in matched_rows:
                ipc.recv_msg(conns[r])                   # {"phase":"decode","t":t}
            if PROFILE:
                _trecv = time.perf_counter(); pacc["recv"] += _trecv - _t0
                ev_g0.record()
            emb = gather(decode=True)                    # [R,1,hidden]
            if PROFILE:
                ev_g1.record()
            attn = torch.cat([attn, torch.ones((R, 1), device=DEV,
                                               dtype=torch.long)], 1)
            cp = torch.tensor([cur], device=DEV)
            out = m(inputs_embeds=emb, past_key_values=past, attention_mask=attn,
                    use_cache=True, cache_position=cp)
            if PROFILE:
                ev_f1.record()
            past = out.past_key_values
            scatter(out.logits[:, -1, :])
            if PROFILE:
                ev_s1.record()
                _tq = time.perf_counter(); pacc["enqueue"] += _tq - _trecv
            torch.cuda.synchronize()
            if PROFILE:
                _ts = time.perf_counter(); pacc["sync_wait"] += _ts - _tq
                pacc["gpu_gather"] += ev_g0.elapsed_time(ev_g1) / 1000.0
                pacc["gpu_forward"] += ev_g1.elapsed_time(ev_f1) / 1000.0
                pacc["gpu_scatter"] += ev_f1.elapsed_time(ev_s1) / 1000.0
            for r in matched_rows:
                ipc.send_msg(conns[r], {"done_step": t})
            if PROFILE:
                pacc["send"] += time.perf_counter() - _ts; pcnt += 1
            cur += 1
        decode_wall = time.perf_counter() - td0
        with open(SENTINEL, "a") as s:
            s.write("DECODE_END %.3f\n" % time.time())

    for r in matched_rows:
        ipc.recv_msg(conns[r])                           # {"done":1}
        conns[r].close()
    srv.close(); os.unlink(SOCK)

    decode_tokens = R * (GEN - 1)
    prof = None
    if PROFILE and pcnt:
        ms = {k: round(1000.0 * v / pcnt, 3) for k, v in pacc.items()}
        ms["step_total_wall"] = round(1000.0 * decode_wall / pcnt, 3)
        ms["disable_auto_init"] = os.environ.get("CIPHER_RT_DISABLE_AUTO_INIT", "0")
        prof = {"per_step_ms": ms, "steps": pcnt}
        print("EXECUTOR PROFILE (ms/step, N=%d auto_init_disabled=%s): %s"
              % (R, ms["disable_auto_init"],
                 " ".join("%s=%.2f" % (k, ms[k]) for k in
                          ("recv", "enqueue", "sync_wait", "send", "gpu_gather",
                           "gpu_forward", "gpu_scatter", "step_total_wall"))),
              file=sys.stderr, flush=True)
    json.dump({"role": "executor", "gate": gate, "ran_batch": True,
               "R": R, "gen_len": GEN, "plen": plen,
               "decode_wall_s": round(decode_wall, 4),
               "decode_tokens": decode_tokens,
               "decode_tok_s": round(decode_tokens / decode_wall, 3)
               if decode_wall else None,
               "profile": prof},
              open(RESULT, "w"), indent=2, default=str)
    print("EXECUTOR batched R=%d decode_tok/s=%.1f wall=%.3fs"
          % (R, decode_tokens / decode_wall, decode_wall),
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
