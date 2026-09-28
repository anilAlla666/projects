#!/usr/bin/env python3
# R.A coop-hook E2E PRODUCT TEST — A0 vs A3 on a realistic mixed prefill+decode serving trace,
# real vLLM cudagraph ON. Measures: (1) E2E throughput marginal, (2) coverage decomposition
# (decode-FULL steps the hook checks vs prefill/piecewise steps it does NOT), (3) graph-memory delta.
# Same discipline as coop_driver.py: scratch LD_PRELOAD shim, anchor NOT loaded, no vLLM source edit.
import os, sys, time, json, ctypes, subprocess
from collections import Counter

ARM = os.environ.get("RV_ARM", "a0")        # a0 | a3
N = int(os.environ.get("RV_N", "45"))
OUTJ = os.environ.get("RV_OUTJSON", f"/home/ubuntu/cipher-fusion-evidence/ra_coophook/e2e_{ARM}.json")
MODEL = os.environ.get("RV_MODEL", "mistralai/Mistral-7B-v0.1")
SHIM_PATH = "/home/ubuntu/cipher-fusion-evidence/ra_coophook/rv_coop_shim.so"
CAP_SIZES = [1, 2, 4, 8]                     # bounded decode-graph set (dual-graph doubles each)
SEED = 1234

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch

SHIM = None
if ARM == "a3":
    SHIM = ctypes.CDLL(SHIM_PATH)
    SHIM.rv_read.argtypes = [ctypes.POINTER(ctypes.c_double)]
    SHIM.rv_cap_linears.restype = ctypes.c_long
    SHIM.rv_stats.argtypes = [ctypes.c_char_p, ctypes.c_int]

class COOP:
    step_full = 0          # FULL-decode replay steps (hook-covered)
    checks = 0; detect_checks = 0; clean_checks = 0; max_resid = 0.0
    checked = {}; checked_graph_mem = 0
    modes = Counter()      # (self.runtime_mode, fc mode) histogram across all wrapper calls
    full_decode_replays = 0
    piecewise_replays = 0
    buf = (ctypes.c_double * 3)()

import vllm.compilation.cuda_graph as cg
_orig_call = cg.CUDAGraphWrapper.__call__

def patched_call(self, *args, **kwargs):
    if not cg.is_forward_context_available():
        return _orig_call(self, *args, **kwargs)
    fc = cg.get_forward_context()
    sm, fm = str(self.runtime_mode), str(fc.cudagraph_runtime_mode)
    COOP.modes[(sm, fm)] += 1
    # coverage accounting on REPLAY: count FULL-decode (hook-covered) vs everything-else (uncovered)
    is_full = (self.runtime_mode == cg.CUDAGraphMode.FULL and fc.cudagraph_runtime_mode == self.runtime_mode)
    if is_full: COOP.full_decode_replays += 1
    elif fc.cudagraph_runtime_mode != cg.CUDAGraphMode.NONE or sm == "CUDAGraphMode.PIECEWISE":
        COOP.piecewise_replays += 1   # prefill/mixed wrapper executions the FULL-decode hook does NOT touch

    if ARM != "a3" or self.runtime_mode != cg.CUDAGraphMode.FULL or fc.cudagraph_runtime_mode != self.runtime_mode:
        return _orig_call(self, *args, **kwargs)

    # ARM==a3 and this is a FULL-mode call
    bd = fc.batch_descriptor
    entry = self.concrete_cudagraph_entries.get(bd)
    is_capture = entry is None or entry.cudagraph is None
    if is_capture:
        SHIM.rv_reset_slots(); SHIM.rv_set_inject(0); SHIM.rv_arm(0, 0)
        out = _orig_call(self, *args, **kwargs)         # vanilla FULL capture
        SHIM.rv_reset_slots(); SHIM.rv_arm(0, 1)        # checked capture
        torch.cuda.synchronize(); mem0 = torch.cuda.memory_allocated()
        g_chk = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g_chk, pool=self.graph_pool):
                out_chk = self.runnable(*args, **kwargs)
            torch.cuda.synchronize()
            COOP.checked_graph_mem += (torch.cuda.memory_allocated() - mem0)
            COOP.checked[bd] = (g_chk, out_chk)
        except Exception as e:
            print(f"[e2e] checked capture FAILED for {bd}: {e}", file=sys.stderr, flush=True)
            raise
        finally:
            SHIM.rv_arm(0, 0)
        return out

    # FULL replay (full_decode_replays already incremented at the is_full check above)
    COOP.step_full += 1
    ck = COOP.checked.get(bd)
    if ck is not None and COOP.step_full % N == 0:
        g_chk, out_chk = ck
        g_chk.replay()
        SHIM.rv_read(COOP.buf)
        mx, nz = COOP.buf[0], int(COOP.buf[1])
        COOP.checks += 1
        if nz:
            COOP.detect_checks += 1
            if mx > COOP.max_resid: COOP.max_resid = mx
        else:
            COOP.clean_checks += 1
        return out_chk
    entry.cudagraph.replay()
    return entry.output

cg.CUDAGraphWrapper.__call__ = patched_call

# ---- realistic mixed trace (fixed seed -> identical trace for A0 and A3) ----
import random
rng = random.Random(SEED)
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, enforce_eager=False, gpu_memory_utilization=0.85, max_model_len=2048,
          dtype="float16", disable_log_stats=True,
          compilation_config={"cudagraph_capture_sizes": CAP_SIZES})
# 32 requests, varied prompt lengths (prefill spread) + varied gen lengths (decode spread)
base = "The history of computing began with mechanical calculators and evolved through many stages. "
reqs = []
TRACE = os.environ.get("RV_TRACE", "mixed")
NREQ=int(os.environ.get('RV_NREQ','32'))
for _ in range(NREQ):
    if TRACE == "decode":      # short prompts, long gen -> decode-dominated wall time
        plen = rng.choice([1, 2]); glen = rng.choice([256, 384, 512])
    else:                       # mixed: varied prefill + varied decode
        plen = rng.choice([1, 2, 4, 8, 16, 32]); glen = rng.choice([16, 32, 64, 128, 256])
    reqs.append((base * plen, glen))
prompts = [p for p, _ in reqs]
sps = [SamplingParams(max_tokens=g, min_tokens=g, ignore_eos=True, temperature=0.0) for _, g in reqs]

llm.generate(prompts, sps, use_tqdm=False)             # warmup + capture
torch.cuda.synchronize()
mem_after_capture = torch.cuda.memory_allocated() / 1e6   # MB
COOP.step_full = 0; COOP.checks = 0; COOP.detect_checks = 0; COOP.clean_checks = 0
COOP.full_decode_replays = 0; COOP.piecewise_replays = 0; COOP.modes = Counter()

best = None
for _ in range(2):
    t0 = time.perf_counter(); o = llm.generate(prompts, sps, use_tqdm=False); dt = time.perf_counter() - t0
    gt = sum(len(x.outputs[0].token_ids) for x in o); tps = gt / dt
    best = tps if best is None or tps > best else best

stats = {}
if SHIM is not None:
    sb = ctypes.create_string_buffer(1024); SHIM.rv_stats(sb, 1024); stats = json.loads(sb.value.decode())
clocks = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,memory.used",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
anchor_loaded = True
try:
    ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError:
    anchor_loaded = False

res = dict(arm=ARM, N=N, model=MODEL, cap_sizes=CAP_SIZES, seed=SEED, n_requests=len(reqs), trace=os.environ.get('RV_TRACE','mixed'),
           e2e_tok_s=round(best, 1), total_gen_tokens=gt,
           full_decode_replays=COOP.full_decode_replays, piecewise_replays=COOP.piecewise_replays,
           checks=COOP.checks, detect_checks=COOP.detect_checks, clean_checks=COOP.clean_checks,
           max_residual=COOP.max_resid, mem_after_capture_mb=round(mem_after_capture, 1), checked_graph_mem_mb=round(COOP.checked_graph_mem/1e6,1),
           runtime_modes={f"{k[0]}|{k[1]}": v for k, v in COOP.modes.items()},
           shim_stats=stats, clocks=clocks, anchor_loaded=anchor_loaded,
           vllm_plugins=os.environ.get("VLLM_PLUGINS", "<unset>"))
print("E2E", json.dumps(res))
with open(OUTJ, "w") as f:
    json.dump(res, f, indent=1)
