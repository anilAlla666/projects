"""W.4b.6 faithful control: B=N decode with N WARMED co-resident tenants.

Mirrors the cross run's GPU state exactly minus the per-step transport: owner
packs the shared arena + serves fds; N holders import the shared weights, run the
same 8-forward warmup as a real tenant, then block (idle) — leaving N warmed
Mistral CUDA contexts resident; the owner then runs B=N decode, power-windowed.
If owner decode ~= cross executor decode (~60 tok/s) and << inproc-alone /
inproc+bare-idle (~156), the architecture tax is the co-resident WARMED tenants,
not transport, not bare context count, not arena weights, not the backstop.
"""
import os
import sys
import json
import time
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ["WL_MODEL"]
N = int(os.environ["N_HOLD"])
GEN = int(os.environ["GEN_LEN"])
PLEN = int(os.environ["PLEN"])
SENT = os.environ["SENTINEL"]
RESULT = os.environ["RESULT_JSON"]
SOCK = os.environ["FORMB_SOCK"]
DEV = "cuda"


def holder(row):
    # optionally inject the substrate shim into the holder (mirror real tenants),
    # set BEFORE any CUDA init so the driver picks it up.
    if os.environ.get("CIPHER_INJECT_HOLDERS") == "1":
        os.environ["CUDA_INJECTION64_PATH"] = \
            "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
    import torch
    globals()["torch"] = torch
    import cipher_kv_bridge as kvb
    from transformers import AutoConfig, AutoModelForCausalLM
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
    manifest, fd = ipc.recv_arena_fd(c)
    cfg = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights(include_buffers=False):
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    _arena = W.import_rebind(m, fd, manifest); os.close(fd); m.eval()
    ids = torch.full((1, PLEN), 1 + row, dtype=torch.long, device=DEV)
    with torch.no_grad():
        for _ in range(8):                  # same warmup as a real tenant
            m(ids)
    torch.cuda.synchronize()
    c.sendall(b"r")                          # warmed + idle
    c.recv(1)                                # block until owner done
    assert _arena.base; sys.exit(0)


def owner(pids):
    import torch
    globals()["torch"] = torch
    import cipher_kv_bridge as kvb
    from transformers import AutoModelForCausalLM
    import formb6_weights as W
    import formb_ipc as ipc
    kvb.init(64 * 1024 * 1024)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(N + 1)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    arena, manifest = W.pack_arena(m, tenant_id=1)
    conns = []
    for _ in range(N):
        conn, _a = srv.accept()
        fd = arena.export_fd()
        ipc.send_arena_fd(conn, manifest, fd); os.close(fd)
        conns.append(conn)
    for conn in conns:
        conn.recv(1)                          # all holders warmed + idle
    ids = torch.stack([torch.full((PLEN,), 1 + i, dtype=torch.long) for i in range(N)], 0).cuda()
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True); past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1); cur = PLEN; torch.cuda.synchronize()
        with open(SENT, "w") as s:
            s.write("DECODE_START %.3f\n" % time.time())
        t0 = time.perf_counter()
        for _ in range(1, GEN):
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; nt = out.logits[:, -1, :].argmax(-1); cur += 1
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        with open(SENT, "a") as s:
            s.write("DECODE_END %.3f\n" % time.time())
    dtok = N * (GEN - 1)
    json.dump({"mode": "inproc_with_%d_WARMED_tenants" % N, "B": N, "gen_len": GEN,
               "decode_wall_s": round(wall, 4), "decode_tokens": dtok,
               "decode_tok_s": round(dtok / wall, 3),
               "peak_hbm_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
              open(RESULT, "w"), indent=2)
    print("WARMHOLD owner B=%d decode_tok/s=%.1f wall=%.3fs with %d warmed tenants"
          % (N, dtok / wall, wall, N), file=sys.stderr, flush=True)
    for conn in conns:
        conn.sendall(b"x")
    for p in pids:
        os.waitpid(p, 0)
    srv.close(); os.remove(SOCK)


if __name__ == "__main__":
    pids = []
    for r in range(N):
        pid = os.fork()
        if pid == 0:
            holder(r)
        pids.append(pid)
    owner(pids)
