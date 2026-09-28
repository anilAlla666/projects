#!/usr/bin/env python3
# R.A vLLM-COOP CAPTURE-HOOK driver. Models a one-line vLLM source hook by patching
# vllm.compilation.cuda_graph.CUDAGraphWrapper.__call__ (version-pinned 0.20.2, copied semantics).
# Arms (RV_ARM): a0 = baseline cudagraph (no shim action); a1 = in-frame always-on checks (captured
# into vLLM's FULL decode graph); a2 = periodic sidecar graph replayed every RV_N decode steps.
# RV_INJECT >= 0 captures a persistent bit-14 XOR on linear ordinal RV_INJECT inside the FULL graph.
# Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 (patch + shim must live in the engine process).
import os, sys, time, json, ctypes, hashlib, subprocess

ARM = os.environ.get("RV_ARM", "a0")          # a0 | a1 | a2
N = int(os.environ.get("RV_N", "45"))
INJ = int(os.environ.get("RV_INJECT", "-1"))
OUTJ = os.environ.get("RV_OUTJSON", f"/home/ubuntu/cipher-fusion-evidence/ra_coophook/run_{ARM}.json")
RESTRICT = os.environ.get("RV_RESTRICT", "1") == "1"   # cudagraph_capture_sizes=[8]
NOREAD = os.environ.get("RV_NOREAD", "0") == "1"       # a3 isolating arm: replay checked graph, skip D2H read
MODEL = os.environ.get("RV_MODEL", "mistralai/Mistral-7B-v0.1")
B = int(os.environ.get("RV_BATCH", "8")); OUT = int(os.environ.get("RV_OUT", "64"))
SHIM_PATH = "/home/ubuntu/cipher-fusion-evidence/ra_coophook/rv_coop_shim.so"

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0" or ARM == "a0u", \
    "patch needs in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0)"

import torch

# ---- shim binding (only when shim work is needed) ----
SHIM = None
if ARM in ("a1", "a2", "a3") or INJ >= 0:
    SHIM = ctypes.CDLL(SHIM_PATH)  # already LD_PRELOADed; returns same object
    SHIM.rv_read.argtypes = [ctypes.POINTER(ctypes.c_double)]
    SHIM.rv_emit_sidecar.argtypes = [ctypes.c_void_p]
    SHIM.rv_emit_sidecar.restype = ctypes.c_int
    SHIM.rv_cap_linears.restype = ctypes.c_long
    SHIM.rv_stats.argtypes = [ctypes.c_char_p, ctypes.c_int]

class COOP:
    step = 0; checks = 0; detect_checks = 0; clean_checks = 0
    first_detect = None; first_detect_resid = 0.0; max_resid = 0.0; det_log = []
    sidecars = {}; checked = {}; sidecar_emit_errors = None; cap_linears = None
    buf = (ctypes.c_double * 3)()
    captures = 0

# ---- the coop hook: patched CUDAGraphWrapper.__call__ ----
import vllm.compilation.cuda_graph as cg
_orig_call = cg.CUDAGraphWrapper.__call__

def patched_call(self, *args, **kwargs):
    if not cg.is_forward_context_available():
        return _orig_call(self, *args, **kwargs)
    fc = cg.get_forward_context()
    if (fc.cudagraph_runtime_mode != self.runtime_mode
            or self.runtime_mode != cg.CUDAGraphMode.FULL):
        return _orig_call(self, *args, **kwargs)
    bd = fc.batch_descriptor
    entry = self.concrete_cudagraph_entries.get(bd)
    is_capture = entry is None or entry.cudagraph is None

    if is_capture:
        armed = SHIM is not None
        if armed:
            SHIM.rv_reset_slots()
            # inject armed for the VANILLA capture too (A1/A2 single graph; A3 both graphs) so a
            # persistent fault is present on EVERY production step, not only the periodic checked one.
            SHIM.rv_set_inject(1 if INJ >= 0 else 0)
            SHIM.rv_arm(1 if ARM == "a2" else 0, 1 if ARM == "a1" else 0)
        out = _orig_call(self, *args, **kwargs)      # vLLM captures its own (vanilla, or a1/a2-checked) graph
        COOP.captures += 1
        if armed:
            SHIM.rv_arm(0, 0)
            COOP.cap_linears = int(SHIM.rv_cap_linears())
            if ARM == "a2":                          # cheap sidecar (KNOWN-INCORRECT: stale buffers) — kept for the record
                SHIM.rv_reset_slots()
                g2 = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
                with torch.cuda.graph(g2, stream=s):
                    rc = SHIM.rv_emit_sidecar(ctypes.c_void_p(s.cuda_stream))
                torch.cuda.synchronize()
                COOP.sidecars[bd] = g2; COOP.sidecar_emit_errors = rc
            if ARM == "a3":                          # dual FULL graph: capture a 2nd checked full forward
                SHIM.rv_reset_slots()
                SHIM.rv_set_inject(1 if INJ >= 0 else 0)  # same persistent fault also in the checked graph
                SHIM.rv_arm(0, 1)                    # f_checks=1 inline during this capture
                g_chk = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(g_chk, pool=self.graph_pool):
                        out_chk = self.runnable(*args, **kwargs)
                    torch.cuda.synchronize()
                    COOP.checked[bd] = (g_chk, out_chk, int(SHIM.rv_slot_count()))
                    print(f"[coop-py] a3 checked-full capture OK slots={SHIM.rv_slot_count()}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[coop-py] a3 checked-full capture FAILED: {e}", file=sys.stderr, flush=True)
                    raise
                finally:
                    SHIM.rv_arm(0, 0); SHIM.rv_set_inject(0)
        return out

    # ---- replay path ----
    if ARM == "a3" and SHIM is not None:
        COOP.step += 1
        ck = COOP.checked.get(bd)
        if ck is not None and COOP.step % N == 0:
            g_chk, out_chk, _ = ck
            g_chk.replay()                           # full checked forward (correct output + inline checks)
            COOP.checks += 1
            if NOREAD:                               # isolating arm: skip the D2H readback to price it out
                return out_chk
            SHIM.rv_read(COOP.buf)
            mx, nz, am = COOP.buf[0], int(COOP.buf[1]), int(COOP.buf[2])
            if nz:
                COOP.detect_checks += 1
                if COOP.first_detect is None:
                    COOP.first_detect = COOP.step
                    COOP.first_detect_resid = mx     # residual AT the catch step (not the run max)
                if mx > COOP.max_resid: COOP.max_resid = mx
                if len(COOP.det_log) < 16: COOP.det_log.append([COOP.step, mx, nz, am])
            else:
                COOP.clean_checks += 1
            return out_chk                           # checked graph's output (full correct forward)
        entry.cudagraph.replay()                     # vanilla replay
        return entry.output

    out = _orig_call(self, *args, **kwargs)          # a0/a1/a2 vanilla replay
    COOP.step += 1
    if SHIM is not None and ARM in ("a1", "a2") and COOP.step % N == 0:
        if ARM == "a2":
            g2 = COOP.sidecars.get(bd)
            if g2 is not None: g2.replay()
        SHIM.rv_read(COOP.buf)
        mx, nz, am = COOP.buf[0], int(COOP.buf[1]), int(COOP.buf[2])
        COOP.checks += 1
        if nz:
            COOP.detect_checks += 1
            if COOP.first_detect is None:
                COOP.first_detect = COOP.step; COOP.first_detect_resid = mx
            if mx > COOP.max_resid: COOP.max_resid = mx
            if len(COOP.det_log) < 16: COOP.det_log.append([COOP.step, mx, nz, am])
        else:
            COOP.clean_checks += 1
    return out

cg.CUDAGraphWrapper.__call__ = patched_call

# ---- run ----
from vllm import LLM, SamplingParams
kw = dict(model=MODEL, enforce_eager=False, gpu_memory_utilization=0.85,
          max_model_len=2048, dtype="float16", disable_log_stats=True)
if RESTRICT:
    kw["compilation_config"] = {"cudagraph_capture_sizes": [B]}
llm = LLM(**kw)
prompts = ["The history of computing began " * 4] * B
sp = SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)  # warmup

best = None; tok_hash = None
for _ in range(2):
    t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    gt = sum(len(x.outputs[0].token_ids) for x in o); tps = gt / dt
    best = tps if best is None or tps > best else best
    tok_hash = hashlib.md5(json.dumps([list(x.outputs[0].token_ids) for x in o]).encode()).hexdigest()

stats = {}
if SHIM is not None:
    sb = ctypes.create_string_buffer(1024); SHIM.rv_stats(sb, 1024)
    stats = json.loads(sb.value.decode())

clocks = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.applications.graphics,power.draw",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
# substrate probe: prove the frozen anchor is NOT loaded (scratch shim only)
import ctypes.util
anchor_loaded = True
try:
    ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError:
    anchor_loaded = False

res = dict(arm=ARM, N=N, inject=INJ, restrict=RESTRICT, batch=B, out_tokens=OUT,
           decode_tok_s=round(best, 1), token_ids_md5=tok_hash,
           steps_replayed=COOP.step, full_captures=COOP.captures, cap_linears=COOP.cap_linears,
           checks=COOP.checks, detect_checks=COOP.detect_checks, clean_checks=COOP.clean_checks,
           noread=NOREAD, clean_slot_comparisons=COOP.clean_checks * 128,
           first_detect_step=COOP.first_detect, first_detect_residual=COOP.first_detect_resid,
           max_residual=COOP.max_resid,
           det_log_head=COOP.det_log, sidecar_emit_errors=COOP.sidecar_emit_errors,
           shim_stats=stats, clocks_after=clocks, anchor_loaded=anchor_loaded,
           ld_preload=os.environ.get("LD_PRELOAD", ""), vllm_multiproc=os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
           vllm_plugins=os.environ.get("VLLM_PLUGINS", "<unset>"),
           deep_gemm_warmup=os.environ.get("VLLM_DEEP_GEMM_WARMUP", "<unset>"),
           model=MODEL)
print("RESULT", json.dumps(res))
with open(OUTJ, "w") as f:
    json.dump(res, f, indent=1)
