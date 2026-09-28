"""Phase B batched-executor thin client — sends all 5 WL01 prompts up front,
then collects 5 responses (output ids + teacher-forced KL stats)."""
import os
import sys
import json
import time
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_ipc import send_msg, recv_msg

SOCK = os.environ.get("BATCH_SOCK", "/tmp/cipher_batch_exec.sock")
TENANT = os.environ["TENANT"]
GEN_LEN = int(os.environ.get("GEN_LEN", "128"))
OUT = os.environ["OUT_JSON"]
PROMPTS = [
    "The Pacific Ocean is the largest ocean on Earth, covering approximately",
    "Sarah opened the old letter with trembling hands. The handwriting was "
    "her grandmother's, and the date read",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot =",
    "If a train leaves Chicago at 3 PM traveling east at 60 mph and another "
    "leaves New York at 4 PM traveling west at 80 mph,",
    "User: What are the main differences between supervised and unsupervised "
    "learning?\nAssistant: The main differences are",
]

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
for _ in range(900):
    try:
        s.connect(SOCK)
        break
    except (FileNotFoundError, ConnectionRefusedError):
        time.sleep(0.2)
else:
    print("CLIENT %s could not connect" % TENANT, file=sys.stderr)
    sys.exit(1)

for i, p in enumerate(PROMPTS):
    send_msg(s, {"tenant": TENANT, "prompt_idx": i, "text": p, "gen_len": GEN_LEN})
res = []
for i in range(len(PROMPTS)):
    rep = recv_msg(s)
    if rep is None:
        print("CLIENT %s lost executor at %d" % (TENANT, i), file=sys.stderr)
        break
    res.append({"prompt": i, "output_ids": rep["output_ids"], "gen": rep["gen"],
                "kl_mean": rep["kl_mean"], "kl_max": rep["kl_max"]})
    print("CLIENT %s prompt=%d gen=%d kl_mean=%.6f kl_max=%.6f"
          % (TENANT, i, rep["gen"], rep["kl_mean"], rep["kl_max"]),
          file=sys.stderr, flush=True)
json.dump({"tenant": TENANT, "prompts": res}, open(OUT, "w"), indent=2)
s.close()
