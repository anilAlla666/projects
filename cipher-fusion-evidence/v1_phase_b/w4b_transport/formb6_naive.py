"""W.4b.6 Step 3 — NAIVE denominator, weight-share held CONSTANT.

N concurrent B=1 solo decoders that SHARE one physical weight copy (Step-1 cuIPC
arena) but do NOT coalesce — each runs its own greedy decode loop. This is the
honest "naive N-concurrent" baseline for substrate-attribution: it isolates the
*batching/coalescing* lever (full-handoff vs this) from the *weight-sharing* lever
(which both hold constant). 8 independent loaders would OOM at N=8 Mistral and
would also conflate the two levers — so the denominator must share weights too.

Owner forks N workers BEFORE CUDA init, loads+packs the model into a RW arena,
serves the fd to each worker; workers import (READ-only view, held alive), warm,
barrier, then decode GEN tokens concurrently. The decode window (max worker wall)
is bracketed by the SENTINEL for the external power sampler.
"""
import os
import sys
import json
import time
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ["WL_MODEL"]
N = int(os.environ["N_WORKERS"])
GEN = int(os.environ["GEN_LEN"])
PLEN = int(os.environ["PLEN"])
SENT = os.environ["SENTINEL"]
RESULT = os.environ["RESULT_JSON"]
SOCK = os.environ["FORMB_SOCK"]
DEV = "cuda"
PROMPTS = [
    "The history of computing spans several distinct technological eras that",
    "Renewable energy adoption worldwide has accelerated rapidly over the recent",
    "Modern distributed systems must carefully balance consistency availability and partition",
    "Advances in materials science have enabled lighter stronger and more durable",
    "The global supply chain experienced unprecedented disruption during the pandemic which",
    "Machine learning models trained on large corpora can exhibit surprising emergent",
    "Urban planners increasingly rely on data driven simulations to forecast traffic",
    "Ocean currents play a critical role in regulating the planet climate by",
]


def enc_ids(tok, prompt):
    t = tok(prompt, return_tensors="pt").input_ids[0]
    if t.shape[0] >= PLEN:
        return t[:PLEN].unsqueeze(0).cuda()
    padv = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return torch.cat([t, torch.full((PLEN - t.shape[0],), padv,
                                    dtype=t.dtype)]).unsqueeze(0).cuda()


def worker(row):
    import torch  # noqa
    globals()["torch"] = torch
    import cipher_kv_bridge as kvb
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    import formb6_weights as W
    import formb_ipc as ipc
    kvb.init(64 * 1024 * 1024)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(600):
        try:
            c.connect(SOCK); break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    manifest, fd = ipc.recv_arena_fd(c)             # length-prefixed manifest + fd
    cfg = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights(include_buffers=False):
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    _arena = W.import_rebind(m, fd, manifest); os.close(fd); m.eval()  # HOLD
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = enc_ids(tok, PROMPTS[row % len(PROMPTS)])
    with torch.no_grad():
        for _ in range(4):
            m(ids)
    torch.cuda.synchronize()
    c.sendall(b"r")                       # ready
    c.recv(1)                             # go barrier
    # concurrent solo greedy decode
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True); past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1); cur = ids.shape[1]
        for _ in range(1, GEN):
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; nt = out.logits[:, -1, :].argmax(-1); cur += 1
        torch.cuda.synchronize()
    c.sendall(b"d"); assert _arena.base; c.close(); sys.exit(0)


def owner(child_pids):
    import torch
    globals()["torch"] = torch
    import cipher_kv_bridge as kvb
    from transformers import AutoModelForCausalLM
    import formb6_weights as W
    kvb.init(64 * 1024 * 1024)
    # bind BEFORE the slow load so workers' connect() succeeds immediately
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(N + 1)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    arena, manifest = W.pack_arena(m, tenant_id=1)
    import formb_ipc as ipc
    conns = []
    for _ in range(N):
        conn, _a = srv.accept()
        fd = arena.export_fd()
        ipc.send_arena_fd(conn, manifest, fd); os.close(fd)   # length-prefixed
        conns.append(conn)
    for conn in conns:                    # wait all ready
        conn.recv(1)
    with open(SENT, "w") as s:
        s.write("DECODE_START %.3f\n" % time.time())
    t0 = time.perf_counter()
    for conn in conns:                    # broadcast go
        conn.sendall(b"g")
    for conn in conns:                    # wait all done
        conn.recv(1)
    wall = time.perf_counter() - t0
    with open(SENT, "a") as s:
        s.write("DECODE_END %.3f\n" % time.time())
    for conn in conns:
        conn.close()
    srv.close(); os.remove(SOCK)
    dtok = N * (GEN - 1)
    json.dump({"mode": "naive_concurrent_shareweight", "N": N, "gen_len": GEN,
               "decode_wall_s": round(wall, 4), "decode_tokens": dtok,
               "decode_tok_s": round(dtok / wall, 3),
               "peak_hbm_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
              open(RESULT, "w"), indent=2)
    print("NAIVE N=%d concurrent-solo share-weight decode_tok/s=%.1f wall=%.3fs peakHBM=%.1fGiB"
          % (N, dtok / wall, wall, torch.cuda.max_memory_allocated() / 2**30),
          file=sys.stderr, flush=True)
    for p in child_pids:
        os.waitpid(p, 0)


if __name__ == "__main__":
    pids = []
    for r in range(N):
        pid = os.fork()
        if pid == 0:
            worker(r)
        pids.append(pid)
    owner(pids)
