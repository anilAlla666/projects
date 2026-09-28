"""W.4b.6 Step 1.3 — weight-sharing lossless smoke (fork-first, CUDA-safe).
Owner packs WL_MODEL into a cuIPC arena; importer meta-loads + rebinds to
read-only views; both run the same forward; assert bit-identical / KL<=5e-5.
Run: WL_MODEL=/home/ubuntu/models/TinyLlama-1.1B python3 formb6_ws_smoke.py
"""
import os
import sys
import json
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SOCK = "/tmp/formb6_ws_smoke.sock"
MODEL = os.environ["WL_MODEL"]
PROMPT = "The history of computing spans several distinct eras each defined by"
HERE = os.path.dirname(os.path.abspath(__file__))


def importer():
    import torch
    import torch.nn.functional as F
    import cipher_kv_bridge as kvb
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    import formb6_weights as W
    import formb_ipc as ipc
    kvb.init(64 * 1024 * 1024)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    import time
    for _ in range(600):
        try:
            c.connect(SOCK); break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    manifest, fd = ipc.recv_arena_fd(c)   # length-prefixed manifest + fd
    cfg = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights(include_buffers=False):
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    # HOLD the returned arena alive: its WeightArena dtor unmaps/frees the VMM
    # (cipher_kv_bridge.cpp:95) and the model's from_blob param views use a
    # no-op deleter — dropping the arena => use-after-free => illegal access.
    arena = W.import_rebind(m, fd, manifest); os.close(fd)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        y = m(ids).logits.float().cpu()
    torch.save(y, HERE + "/.ws_importer.pt")
    assert arena.base                      # keep `arena` referenced past forward
    c.sendall(b"done"); c.close()
    sys.exit(0)


def owner():
    import torch
    import cipher_kv_bridge as kvb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import formb6_weights as W
    kvb.init(64 * 1024 * 1024)
    # bind+listen BEFORE the (slow) load so the importer's connect() succeeds
    # immediately and blocks harmlessly on recv_fds — a 14 GB Mistral load
    # otherwise exceeds the importer's connect-retry window and hangs accept().
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(1)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    arena, manifest = W.pack_arena(m, tenant_id=1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        y = m(ids).logits.float().cpu()
    torch.save(y, HERE + "/.ws_owner.pt")
    import formb_ipc as ipc
    conn, _ = srv.accept()
    fd = arena.export_fd()
    ipc.send_arena_fd(conn, manifest, fd); os.close(fd)
    conn.recv(16)
    print("OWNER peak_HBM=%.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30),
          file=sys.stderr, flush=True)
    conn.close(); srv.close(); os.remove(SOCK)


if __name__ == "__main__":
    child = os.fork()
    if child == 0:
        importer()
    else:
        owner()
        os.waitpid(child, 0)
        import torch
        import torch.nn.functional as F
        yo = torch.load(HERE + "/.ws_owner.pt", weights_only=True)
        yc = torch.load(HERE + "/.ws_importer.pt", weights_only=True)
        identical = bool(torch.equal(yo, yc))
        lp = F.log_softmax(yo[0, -1].float(), -1)
        lq = F.log_softmax(yc[0, -1].float(), -1)
        kl = float((lp.exp() * (lp - lq)).sum())
        print("WS-SMOKE bit_identical=%s last_tok_KL=%.3e -> %s"
              % (identical, kl, "PASS" if (identical or kl <= 5e-5) else "FAIL"),
              file=sys.stderr, flush=True)
