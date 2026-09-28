"""W.4b.6 EXECUTOR (VANILLA weight-owner + W.4a gate + executor-internal decode).

The dedicated batched-decode executor (CP 5.6 §2.2A). It:
  1. loads the model, packs weights into a cuIPC arena (Track-2 owner), serves the
     fd to each tenant so N tenants share ONE physical weight copy (unlocks N=8
     Mistral on 80 GB);
  2. runs the AUDITED W.4a gate (cipher_rt_pool_partition over the live NR 31
     cohort) — admits same-fp tenants, rejects distinct-fp (Memory #11);
  3. first-coalesce correctness backstop (W.4b.3, top-K-projected audited guard);
  4. runs the batched B=N decode INTERNALLY (no per-token round-trip — the
     multi-token-window-at-K=all endpoint; the option-i ~79 ms/step rendezvous is
     banked R-4 debt) and scatters each tenant's tokens back.

Executor is VANILLA (no CUDA_INJECTION64_PATH → zero shim tax). Reads its
deployment-match fp from the one-time injected warmup (FP_FILE).
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
SENTINEL = os.environ.get("SENTINEL", "/tmp/formb6.decode_window")
SHARE = os.environ.get("CIPHER_FORMB_SHARE", "1") == "1"
TOL = float(os.environ.get("CIPHER_FORMB_TOL", "0.15"))
TOPK = int(os.environ.get("CIPHER_FORMB_TOPK", "64"))
FAULT_ROW = int(os.environ.get("CIPHER_FORMB_FAULT", "-1"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                       # noqa: E402
import numpy as np                 # noqa: E402
import formb_ipc as ipc            # noqa: E402
import formb_cohort as cohort      # noqa: E402
import formb6_weights as W         # noqa: E402

DEV = "cuda"


def main():
    ipc.bridge_init()
    executor_fp = int(open(FP_FILE).read().strip())
    self_tgid = os.getpid()
    from transformers import AutoModelForCausalLM
    # bind+listen BEFORE the slow 14 GB load so tenants' connect() succeeds
    # immediately and blocks on recv (load can exceed the connect-retry window).
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(64)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    vocab = m.config.vocab_size
    manifest = None
    if SHARE:
        _arena, manifest = W.pack_arena(m, tenant_id=1)   # executor params now arena-backed
    print("EXECUTOR listening %s fp=0x%016x share=%d" % (SOCK, executor_fp, SHARE),
          file=sys.stderr, flush=True)

    conns, hellos, tgids = {}, {}, {}
    for _ in range(N):
        conn, _a = srv.accept()
        h = ipc.recv_msg(conn)                            # HELLO {tgid,row,model}
        r = h["row"]; conns[r] = conn; tgids[r] = h["tgid"]
        # serve shared weights ONLY to same-model tenants; a distinct-model
        # (negative-control) tenant loads its own weights (SHARE=0) and is
        # rejected by the audited gate, not by a manifest mismatch.
        if SHARE and h.get("model") == MODEL:
            fd = _arena.export_fd()
            ipc.send_arena_fd(conn, manifest, fd); os.close(fd)   # length-prefixed
    for r in conns:
        w = ipc.recv_msg(conns[r])                        # {warmed,fp,plen}
        hellos[r] = {"tgid": tgids[r], "fp": w["fp"], "plen": w["plen"]}
    print("EXECUTOR all %d tenants warmed" % N, file=sys.stderr, flush=True)

    # wait for cohort convergence to the deployed-model fp
    want = set(tgids.values())
    peers = []
    t0 = time.time(); allp = None
    while time.time() - t0 < 30:
        peers = cohort.query_cohort()
        present = {tg: fp for tg, fp in peers if tg in want}
        if len(present) == len(want):
            if allp is None:
                allp = time.time()
            if sum(1 for fp in present.values() if fp == executor_fp) == len(want) \
               or time.time() - allp > 6:
                break
        time.sleep(0.5)

    dec = cohort.partition(self_tgid, executor_fp, peers)
    matched_tgids = set(dec["peer_tgids"])
    matched_rows = sorted(r for r in hellos if tgids[r] in matched_tgids)
    eligible = len(matched_rows) >= 2
    gate = {"executor_fp": executor_fp, "cohort": peers, "partition": dec,
            "matched_rows": matched_rows, "eligible_for_batch": eligible,
            "tenant_fps": {r: hellos[r].get("fp") for r in hellos}}
    print("EXECUTOR gate eligible=%s matched=%s distinct_rejected=%d"
          % (eligible, matched_rows, dec["distinct_rejected"]), file=sys.stderr, flush=True)

    if not eligible:
        for r in conns:
            ipc.send_msg(conns[r], {"mode": "solo"})
        for r in conns:
            conns[r].close()
        srv.close(); os.unlink(SOCK)
        json.dump({"role": "executor", "gate": gate, "ran_batch": False},
                  open(RESULT, "w"), indent=2, default=str)
        return
    for r in conns:
        ipc.send_msg(conns[r], {"mode": "batch" if r in matched_rows else "solo"})

    # admitted tenants submit prompt ids
    subs = {}
    for r in matched_rows:
        s = ipc.recv_msg(conns[r]); subs[r] = s["ids"]
    R = len(matched_rows)
    plen = len(subs[matched_rows[0]])
    ids_batch = torch.tensor([subs[r] for r in matched_rows], device=DEV)  # [R, plen]

    with torch.no_grad():
        out = m(input_ids=ids_batch, use_cache=True)
        past = out.past_key_values
        prefill_logits = out.logits[:, -1, :]            # [R, vocab]

        # ---- W.4b.3 first-coalesce correctness backstop (audited guard) ----
        gfp, gK, gN, gdt = executor_fp, m.config.hidden_size, R, 1
        blocked = cohort.is_blocked(gfp, gK, gN, gdt) == 1
        n_fail, per_pass, mad, mrd, adiv = 0, [1] * R, 0.0, 0.0, 0
        if not blocked:
            proj_out, proj_ref = [], []
            for i in range(R):
                so = m(input_ids=ids_batch[i:i + 1], use_cache=True)
                s = so.logits[0, -1, :].float().cpu().numpy()
                b = prefill_logits[i].float().cpu().numpy()
                if 0 <= FAULT_ROW < R and i == FAULT_ROW:
                    b = np.random.RandomState(0).randn(b.shape[0]).astype(np.float32) * 50.0
                idx = np.argsort(s)[-TOPK:]
                proj_ref.append(np.ascontiguousarray(s[idx]))
                proj_out.append(np.ascontiguousarray(b[idx]))
                if int(np.argmax(b)) != int(np.argmax(s)):
                    adiv += 1
            mad = max(float(np.max(np.abs(b - s))) for b, s in zip(proj_out, proj_ref))
            mrd = max(float(np.max(np.abs(b - s) / np.maximum(np.abs(s), 1e-12)))
                      for b, s in zip(proj_out, proj_ref))
            n_fail, per_pass = cohort.correctness_check(proj_out, proj_ref, TOL)
            print("FIRSTCOALESCE topk=%d max_abs=%.4f max_rel=%.4f tol=%.3f fails=%d "
                  "argmax_div=%d" % (TOPK, mad, mrd, TOL, n_fail, adiv),
                  file=sys.stderr, flush=True)
            if n_fail > 0:
                cohort.mark_blocked(gfp, gK, gN, gdt); blocked = True
        if blocked:
            for r in matched_rows:
                ipc.send_msg(conns[r], {"fallback": "solo"})
            for r in matched_rows:
                conns[r].close()
            srv.close(); os.unlink(SOCK)
            json.dump({"role": "executor", "gate": gate, "ran_batch": False,
                       "blocked": True, "block": {"n_fail": n_fail, "max_rel": round(mrd, 4),
                       "tol": TOL, "fault_row": FAULT_ROW, "argmax_div": adiv},
                       "counters": cohort.counters()}, open(RESULT, "w"),
                      indent=2, default=str)
            print("EXECUTOR BLOCK -> all solo", file=sys.stderr, flush=True)
            return

        # ---- executor-internal batched decode (no per-token round-trip) ----
        # Accumulate the per-step argmax tensors ON-GPU and transfer ONCE after
        # the loop. A per-token int(nt[i]) inside the timed window forces a
        # device->host sync every step and serializes the decode (~2.4x slower
        # at TinyLlama) — a measurement artifact, not a substrate cost.
        nt = prefill_logits.argmax(-1)                   # [R]
        steps = [nt]
        cur = plen
        with open(SENTINEL, "w") as s:
            s.write("DECODE_START %.3f\n" % time.time())
        td0 = time.perf_counter()
        for _ in range(1, GEN):
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values
            nt = out.logits[:, -1, :].argmax(-1)
            steps.append(nt)
            cur += 1
        torch.cuda.synchronize()
        decode_wall = time.perf_counter() - td0
        with open(SENTINEL, "a") as s:
            s.write("DECODE_END %.3f\n" % time.time())
        tok_mat = torch.stack(steps, 1).cpu().tolist()   # [R, GEN], one transfer
        outs = tok_mat

    for i, r in enumerate(matched_rows):
        ipc.send_msg(conns[r], {"tokens": outs[i]})
        conns[r].close()
    srv.close(); os.unlink(SOCK)
    dtok = R * (GEN - 1)
    json.dump({"role": "executor", "gate": gate, "ran_batch": True, "R": R,
               "gen_len": GEN, "plen": plen, "share": SHARE,
               "peak_hbm_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "decode_wall_s": round(decode_wall, 4), "decode_tokens": dtok,
               "decode_tok_s": round(dtok / decode_wall, 3)},
              open(RESULT, "w"), indent=2, default=str)
    print("EXECUTOR batched R=%d decode_tok/s=%.1f wall=%.3fs peakHBM=%.1fGiB"
          % (R, dtok / decode_wall, decode_wall,
             torch.cuda.max_memory_allocated() / 2**30), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
