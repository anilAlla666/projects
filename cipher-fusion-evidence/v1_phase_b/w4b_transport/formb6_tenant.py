"""W.4b.6 TENANT (injected, weight-shared, executor-internal-decode).

Real independent GPU process: imports the executor-owner's shared Mistral weights
(Track-2 SC3 read-only arena), warms up to materialize its W.6 fp (NR 30/31
registration), then HANDS OFF its decode to the executor (submits prompt + gen_len,
receives the batched tokens) — no per-token round-trip (the option-i ~79 ms/step
rendezvous is banked R-4 debt; this is the multi-token-window-at-K=all endpoint).
Correctness = exact greedy token match vs its OWN solo B=1 reference (run locally,
off the executor's decode critical path); divergences classified by the solo
top1-top2 gap (near-tie ⇒ benign FP-order flip).
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
SHARE = os.environ.get("CIPHER_FORMB_SHARE", "1") == "1"   # weight-share on/off

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                       # noqa: E402
import formb_ipc as ipc            # noqa: E402
import formb6_weights as W         # noqa: E402

DEV = "cuda"
_RT = os.environ.get("CIPHER_RT_LIB", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")


def own_fp():
    try:
        libc = ctypes.CDLL(_RT, mode=os.RTLD_LOCAL | 0x0001)
        libc.cipher_workload_model_fingerprint.restype = ctypes.c_uint64
        return int(libc.cipher_workload_model_fingerprint())
    except Exception as e:
        print("own_fp err %s" % e, file=sys.stderr, flush=True); return 0


def enc_ids(tok):
    t = tok(PROMPT, return_tensors="pt").input_ids[0]
    if t.shape[0] >= PLEN:
        return t[:PLEN].unsqueeze(0).cuda()
    padv = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return torch.cat([t, torch.full((PLEN - t.shape[0],), padv,
                                    dtype=t.dtype)]).unsqueeze(0).cuda()


def solo_greedy(m, ids):
    toks, logs = [], []
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True); past = out.past_key_values
        lg = out.logits[:, -1, :]; cur = ids.shape[1]
        for _ in range(GEN):
            logs.append(lg[0].float().clone()); nt = lg.argmax(-1); toks.append(int(nt))
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; lg = out.logits[:, -1, :]; cur += 1
    return toks, torch.stack(logs)


def main():
    ipc.bridge_init()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    tok = AutoTokenizer.from_pretrained(MODEL)

    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(300):
        try:
            c.connect(SOCK); break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    ipc.send_msg(c, {"hello": 1, "tgid": os.getpid(), "row": ROW, "model": MODEL})

    if SHARE:
        manifest, fd = ipc.recv_arena_fd(c)        # length-prefixed manifest + fd
        cfg = AutoConfig.from_pretrained(MODEL)
        with init_empty_weights(include_buffers=False):
            m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
        # HOLD the arena: its WeightArena dtor unmaps the VMM and the model's
        # from_blob param views use a no-op deleter — dropping it => UAF.
        _arena = W.import_rebind(m, fd, manifest); os.close(fd); m.eval()
    else:
        _arena = None
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    ids = enc_ids(tok)
    with torch.no_grad():                      # warmup -> fp registers (NR 30/31)
        for _ in range(8):
            m(ids)
    torch.cuda.synchronize()
    fp = own_fp()
    ipc.send_msg(c, {"warmed": 1, "fp": fp, "plen": PLEN})
    gate = ipc.recv_msg(c)
    res = {"row": ROW, "tgid": os.getpid(), "fp": fp, "model": MODEL,
           "mode": gate["mode"]}

    if gate["mode"] != "batch":
        toks, _ = solo_greedy(m, ids)
        res.update({"tokens": toks, "note": "gate denied -> solo"})
        json.dump(res, open(RESULT, "w"), indent=2)
        print("TENANT row=%d SOLO (gate denied) tok0..4=%s" % (ROW, toks[:5]),
              file=sys.stderr, flush=True); c.close(); return

    # hand off decode: submit prompt ids, receive batched tokens
    ipc.send_msg(c, {"submit": 1, "ids": ids[0].tolist(), "gen": GEN})
    resp = ipc.recv_msg(c)
    if resp.get("fallback"):                   # first-coalesce BLOCK -> solo
        toks, _ = solo_greedy(m, ids)
        res.update({"mode": "solo_fallback", "tokens": toks,
                    "note": "correctness BLOCK -> solo (no corruption)"})
        json.dump(res, open(RESULT, "w"), indent=2)
        print("TENANT row=%d SOLO_FALLBACK (BLOCK)" % ROW, file=sys.stderr, flush=True)
        c.close(); return
    batched_toks = resp["tokens"]
    c.close()

    # correctness: exact greedy token match vs own solo (off critical path)
    solo_toks, solo_logs = solo_greedy(m, ids)
    n = min(len(batched_toks), len(solo_toks))
    divs = []
    for t in range(n):
        if batched_toks[t] != solo_toks[t]:
            top2 = torch.topk(solo_logs[t], 2).values
            divs.append({"step": t, "batched": batched_toks[t], "solo": solo_toks[t],
                         "solo_top1_top2_gap": float(top2[0] - top2[1])})
    match = sum(int(batched_toks[t] == solo_toks[t]) for t in range(n))
    res.update({"tokens": batched_toks, "solo_tokens": solo_toks,
                "tok_match": match, "tok_total": n, "divergences": divs})
    json.dump(res, open(RESULT, "w"), indent=2)
    print("TENANT row=%d BATCH match=%d/%d divs=%d tok0..4=%s"
          % (ROW, match, n, len(divs), batched_toks[:5]), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
