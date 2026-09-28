#!/usr/bin/env python3
"""CP 5.4 Step 1.6B-1 — sustained-decode PARTITION tenant (THROWAWAY).

A real PARTITION-class tenant for the Step 1.6 mixed-deployment runs. Runs
under `libcipher_rt` LD_PRELOAD with CIPHER_QOS_CLASS=partition +
CIPHER_SM_COUNT=<n> (set by the orchestrator) — libcipher_rt allocates the
kmod partition and builds the green context transparently; this script is an
ordinary PyTorch decoder.

Captures, per the Step 1.6 design memo §11.2 / §6:
  - per-decode-step wall latency (prefill step 0 recorded separately, excluded
    from the decode-variance population) — the headline variance metric;
  - per-prompt teacher-forced logit-KL vs the WL01 clean-FP16 gold — the
    correctness gate (<= 0.1);
  - a per-round `%smid` probe on its own (green-ctx) stream -> the observed
    physical SM set — the runtime-disjointness probe input.

The tenant's allocated SM set is read authoritatively from the kmod ledger
(`CIPHER_CP54_QUERY` ioctl on /dev/cipher — the calling PID's own `my_grp_mask`),
NOT from libcipher_rt's `cipher_rt_green_ctx_sm_count()`: under the CUDA-injection
load that symbol's static state is not reliably visible to a `ctypes` reader.
The `%smid` probe gives the *observed* SM set; clause-1 self-check is
`observed ⊆ allocated`.

Loader: launch with `CUDA_INJECTION64_PATH=<libcipher_rt.so>` — libcipher_rt's
green-context init runs in `InitializeInjection`, which the CUDA driver fires
only for the injection path (CP 2.5 is LD_PRELOAD-free). Plain LD_PRELOAD does
NOT activate the partition.

Orchestrator coordination (Phase B stdin/stdout pattern):
  stdout: `READY <pid> grp_mask=<hex>`  then  `ROUND <r> SMS <csv>` each round,
          then `DONE`.
  stdin : blocks for a line `START` before the timed rounds (skip with
          NOSIGNAL=1 for standalone sanity).
Writes the full result JSON to $OUT_JSON.

Env: TENANT OUT_JSON [ROUNDS=10] [NOSIGNAL=0] [WL_MODEL] [GOLD_JSON]
     [GOLD_LOGITS]   plus CIPHER_QOS_CLASS / CIPHER_SM_COUNT (read by
     libcipher_rt, not by this script).
"""
import fcntl
import json
import os
import struct
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402
from torch.utils.cpp_extension import load_inline                  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer       # noqa: E402

WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
TENANT = os.environ.get("TENANT", "p0")
OUT_JSON = os.environ["OUT_JSON"]
ROUNDS = int(os.environ.get("ROUNDS", "10"))
NOSIGNAL = os.environ.get("NOSIGNAL", "0") == "1"
BARRIER_TAG = os.environ.get("BARRIER_TAG")        # set => per-round lockstep
MODEL = os.environ.get("WL_MODEL", "/home/ubuntu/models/TinyLlama-1.1B")
GOLD_JSON = os.environ.get("GOLD_JSON", WL01 + "/gold.json")
GOLD_LOGITS = os.environ.get("GOLD_LOGITS", WL01 + "/gold_logits.pt")


def log(m):
    print("[s16-tenant %s] %s" % (TENANT, m), file=sys.stderr, flush=True)


# ---- CP 5.4 ioctl ABI (kmod cipher_ioctl.h, nr 15) + group->SM map ---------
_TYPE = ord('C')
_CP54_STRUCT = 32
CP54_QUERY = (2 << 30) | (_CP54_STRUCT << 16) | (_TYPE << 8) | 15   # _IOR

# group g -> physical SM set (PHASE_1_3A_PROBE.md; same map as cp54_pool.py)
GROUP_SMS = {
    0:  {0, 1, 16, 17, 32, 33, 48, 49},   1:  {2, 3, 18, 19, 34, 35, 50, 51},
    2:  {4, 5, 20, 21, 36, 37, 52, 53},   3:  {6, 7, 22, 23, 38, 39, 54, 55},
    4:  {8, 9, 24, 25, 40, 41, 56, 57},   5:  {10, 11, 26, 27, 42, 43, 58, 59},
    6:  {12, 13, 28, 29, 44, 45, 60, 61}, 7:  {14, 15, 30, 31, 46, 47, 62, 63},
    8:  {64, 65, 78, 79, 92, 93, 106, 107}, 9: {66, 67, 80, 81, 94, 95, 108, 109},
    10: {68, 69, 82, 83, 96, 97, 110, 111}, 11: {70, 71, 84, 85, 98, 99, 112, 113},
    12: {72, 73, 86, 87, 100, 101, 114, 115}, 13: {74, 75, 88, 89, 102, 103, 116, 117},
    14: {76, 77, 90, 91, 104, 105, 118, 119},
}


def query_ledger():
    """Read this PID's CP 5.4 allocation from the kmod ledger — authoritative.
    Returns (my_grp_mask, my_qos, n_partitions, pool_cnt, free_cnt)."""
    fd = os.open("/dev/cipher", os.O_RDWR)
    try:
        buf = bytearray(_CP54_STRUCT)
        fcntl.ioctl(fd, CP54_QUERY, buf, True)
        n_part, pool_cnt, free_cnt, my_mask, my_qos = struct.unpack_from(
            "<IIIII", buf, 0)
        return my_mask, my_qos, n_part, pool_cnt, free_cnt
    finally:
        os.close(fd)


def mask_to_sms(grp_mask):
    """The set of physical SM ids covered by an 8-SM-group bitmask."""
    s = set()
    for g in range(15):
        if grp_mask & (1 << g):
            s |= GROUP_SMS[g]
    return s


_SMID = None


def _smid_ext():
    global _SMID
    if _SMID is None:
        cpp = "void probe_smid(torch::Tensor hit);"
        cu = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void probe_smid_k(int* hit){
    unsigned s; asm volatile("mov.u32 %0, %%smid;":"=r"(s));
    if (s < 256) hit[s] = 1;
}
void probe_smid(torch::Tensor hit){
    probe_smid_k<<<4096, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        hit.data_ptr<int>());
}
'''
        _SMID = load_inline(name="cp54_s16_smid", cpp_sources=[cpp],
                            cuda_sources=[cu], functions=["probe_smid"],
                            verbose=False)
    return _SMID


def probe_sms():
    """Run the %smid kernel on the current (green-ctx) stream; return the
    sorted set of physical SM ids the tenant's kernels actually used."""
    hit = torch.zeros(256, dtype=torch.int32, device="cuda")
    _smid_ext().probe_smid(hit)
    torch.cuda.synchronize()
    return sorted((hit == 1).nonzero().flatten().tolist())


def barrier(r, last):
    """CP 5.4 Step 1.6B-2A per-round lockstep barrier. After round r: drop this
    member's done sentinel; block for the orchestrator's go sentinel before
    round r+1, so every tenant advances round-by-round together (probes are
    contemporaneous; no tenant exits early). No-op when BARRIER_TAG is unset."""
    if not BARRIER_TAG:
        return
    open("/tmp/cp54_s16_%s_%s_r%d.done" % (BARRIER_TAG, TENANT, r), "w").close()
    if last:
        return
    go = "/tmp/cp54_s16_%s_all_r%d.go" % (BARRIER_TAG, r)
    while not os.path.exists(go):
        time.sleep(0.02)


def main():
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)                                    # inference mode
    torch.cuda.synchronize()
    log("model loaded in %.1fs" % (time.perf_counter() - t0))

    # Authoritative allocation from the kmod ledger (this PID's own mask).
    grp_mask, my_qos, n_part, pool_cnt, free_cnt = query_ledger()
    alloc_sms = sorted(mask_to_sms(grp_mask))
    log("kmod ledger: my grp_mask=0x%04x (%d groups / %d SMs) qos=%d  "
        "ledger[n_part=%d pool=%d free=%d]  (CIPHER_QOS_CLASS=%s SM_COUNT=%s)"
        % (grp_mask, bin(grp_mask).count("1"), len(alloc_sms), my_qos,
           n_part, pool_cnt, free_cnt, os.environ.get("CIPHER_QOS_CLASS"),
           os.environ.get("CIPHER_SM_COUNT")))

    gold = {g["prompt"]: g for g in json.load(open(GOLD_JSON))["prompts"]}
    gold_logits = torch.load(GOLD_LOGITS, map_location="cuda")
    n_prompts = len(gold)

    # ---- teacher-forced correctness gate (cascade-free, the TFGATE method) --
    kl_pc = []
    for pr in range(n_prompts):
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]
        full = torch.tensor([g["full_ids"]], device="cuda")
        with torch.no_grad():
            lo = model(input_ids=full).logits[0]
        seg = lo[plen - 1:plen - 1 + gen, :].float()
        lp = F.log_softmax(gold_logits[pr].float(), dim=-1)
        lq = F.log_softmax(seg, dim=-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        km, kx = float(kl.mean()), float(kl.max())
        kl_pc.append({"prompt": pr, "kl_mean": km, "kl_max": kx})
        log("TFGATE prompt=%d kl_mean=%.6e kl_max=%.6e" % (pr, km, kx))
    kl_max_all = max(x["kl_max"] for x in kl_pc)

    def decode_round(pr, timed):
        """One KV-cached greedy decode of prompt pr. Returns (prefill_ms,
        [decode_ms]) — decode_ms empty when timed=False (warmup)."""
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]
        cur = torch.tensor([g["full_ids"][:plen]], device="cuda")
        past = None
        prefill_ms = None
        decode_ms = []
        with torch.no_grad():
            for st in range(gen):
                torch.cuda.synchronize()
                ts = time.perf_counter()
                out = model(input_ids=cur, past_key_values=past,
                            use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - ts) * 1e3
                if st == 0:
                    prefill_ms = dt           # whole-prompt forward — separate
                elif timed:
                    decode_ms.append(dt)
                past = out.past_key_values
                cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return prefill_ms, decode_ms

    _smid_ext()                       # warm the probe compile before timing
    decode_round(0, timed=False)      # warmup round (round 0 is a cold outlier)
    torch.cuda.synchronize()
    log("warmup round done")

    # ---- rendezvous ----
    print("READY %d grp_mask=0x%04x" % (os.getpid(), grp_mask), flush=True)
    if not NOSIGNAL:
        for line in sys.stdin:
            if line.strip() == "START":
                break
    log("START — %d rounds" % ROUNDS)

    # ---- sustained decode; per-step latency + per-round disjointness probe --
    alloc_set = set(alloc_sms)
    rounds = []
    clause1_fail = 0
    for r in range(ROUNDS):
        pr = r % n_prompts
        prefill_ms, decode_ms = decode_round(pr, timed=True)
        sms = probe_sms()
        # clause-1 self-check: observed SMs must be within the allocation.
        outside = sorted(set(sms) - alloc_set)
        if outside:
            clause1_fail += 1
        rounds.append({"round": r, "prompt": pr, "prefill_ms": prefill_ms,
                       "decode_ms": decode_ms, "observed_sms": sms,
                       "clause1_outside": outside})
        print("ROUND %d SMS %s" % (r, ",".join(map(str, sms))), flush=True)
        mean_d = sum(decode_ms) / len(decode_ms) if decode_ms else 0.0
        log("round %d prompt %d  decode_steps=%d  mean=%.3fms  |SMs|=%d "
            "clause1=%s" % (r, pr, len(decode_ms), mean_d, len(sms),
                            "PASS" if not outside else "FAIL %s" % outside))
        barrier(r, last=(r == ROUNDS - 1))         # per-round lockstep

    all_decode = [x for rd in rounds for x in rd["decode_ms"]]
    res = {
        "tenant": TENANT, "pid": os.getpid(),
        "qos_class": os.environ.get("CIPHER_QOS_CLASS"),
        "sm_count_env": os.environ.get("CIPHER_SM_COUNT"),
        "grp_mask": grp_mask, "allocated_sms": alloc_sms,
        "allocated_sm_count": len(alloc_sms),
        "model": MODEL, "rounds": ROUNDS,
        "n_decode_steps": len(all_decode),
        "kl_per_prompt": kl_pc, "kl_max_all": kl_max_all,
        "kl_gate_pass": kl_max_all <= 0.1,
        "clause1_disjointness_fails": clause1_fail,
        "rounds_detail": rounds,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(res, f)
    print("DONE", flush=True)
    log("DONE  decode_steps=%d  kl_max=%.6e  kl_gate=%s  "
        "clause1_fails=%d/%d  alloc=%d SMs"
        % (len(all_decode), kl_max_all, res["kl_gate_pass"],
           clause1_fail, ROUNDS, len(alloc_sms)))


if __name__ == "__main__":
    main()
