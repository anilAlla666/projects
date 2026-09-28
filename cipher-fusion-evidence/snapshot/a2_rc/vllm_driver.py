#!/usr/bin/env python3
# R.C v1 vLLM driver: real eager Mistral-7B decode (real KV cache, continuous batching, real FlashAttn)
# + a W.6 cohort-registry HEARTBEAT thread (NEW code; the R.A shim does not heartbeat).
#
# The heartbeat registers THIS process's tgid into the kmod co-residence table (ioctl NR 31) with a
# synthesized model fingerprint (a 64-bit hash of the model name) so the external monitor's query-only
# read sees this GPU tenant. We deliberately do NOT load libcipher_rt.so to obtain the substrate's
# internal cipher_workload_model_fingerprint(), because loading the runtime would activate its CUDA
# interception/actuators and contaminate the eager SDC measurement. The fingerprint is therefore a
# DESCRIPTIVE tenant tag, which is all the registry needs for co-residence accounting here.
import sys, os, time, json, ctypes, fcntl, threading, hashlib, signal

MODE   = sys.argv[1] if len(sys.argv) > 1 else "decode"      # decode | bench
EAGER  = os.environ.get("RV_EAGER", "1") == "1"
MODEL  = os.environ.get("RV_MODEL", "mistralai/Mistral-7B-v0.1")
B      = int(os.environ.get("RV_BATCH", "8"))
OUTTOK = int(os.environ.get("RV_OUT_TOKENS", "64"))
WARM   = int(os.environ.get("RV_WARM_TOKENS", str(OUTTOK)))
UTIL   = float(os.environ.get("RV_UTIL", "0.85"))
MAXLEN = int(os.environ.get("RV_MAXLEN", "2048"))
RESULT = os.environ.get("RV_RESULT", "")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_PLUGINS"] = ""  # CRITICAL substrate-line guard: cipher_vllm_kv/cipher_vllm_kvdedup auto-load via vllm.general_plugins entry points (easy-install.pth -> /home/ubuntu/cipher_vllm_plugin) and pull the cipher_v2 substrate into the engine; empty = load no plugins


# ---- W.6 cohort registry heartbeat (ioctl NR 31 on /dev/cipher) ----
MAXP = 128
class _Entry(ctypes.Structure):
    _fields_ = [("tgid", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("fp", ctypes.c_uint64)]
class _Query(ctypes.Structure):
    _fields_ = [("caller_fp", ctypes.c_uint64), ("max_entries", ctypes.c_uint32),
                ("n_resident", ctypes.c_uint32), ("entries", _Entry * MAXP)]
def _iowr(t, nr, size): return (3 << 30) | (size << 16) | (t << 8) | nr
_NR_QUERY = _iowr(ord('C'), 31, ctypes.sizeof(_Query))
_FP = int.from_bytes(hashlib.sha256(MODEL.encode()).digest()[:8], "little") | 1  # nonzero descriptive fp

_hb_stop = threading.Event()
def _heartbeat():
    try:
        fd = os.open("/dev/cipher", os.O_RDWR)
    except OSError as e:
        sys.stderr.write(f"[driver] heartbeat: cannot open /dev/cipher: {e}\n"); return
    while not _hb_stop.is_set():
        try:
            q = _Query(); q.caller_fp = _FP; q.max_entries = MAXP
            fcntl.ioctl(fd, _NR_QUERY, q)
        except OSError:
            pass
        _hb_stop.wait(1.0)
    os.close(fd)

# By default the W.6 heartbeat is done by the shim INSIDE the GPU-holding worker (so cohort-tgid ==
# SDC-pid == NVML-pid). The parent-process heartbeat is opt-in (RV_DRIVER_HEARTBEAT=1) and OFF here to
# avoid registering the orchestrator parent (which holds no CUDA context under vLLM V1) as a 2nd tenant.
if os.environ.get("RV_DRIVER_HEARTBEAT", "0") == "1":
    hb = threading.Thread(target=_heartbeat, daemon=True); hb.start()
    sys.stderr.write(f"[driver] pid={os.getpid()} parent-heartbeat started fp=0x{_FP:x}\n")
else:
    sys.stderr.write(f"[driver] pid={os.getpid()} model={MODEL} (W.6 heartbeat delegated to worker shim)\n")

from vllm import LLM, SamplingParams

def _relock_clocks():
    # The -lgc 1980 lock is observed to release during vLLM engine init on this box (driver 580.105.08);
    # re-apply right before the measured window so the achieved clock is the locked one. Fail-soft.
    import subprocess
    try: subprocess.run(["sudo","-n","nvidia-smi","-lgc","1980,1980"], capture_output=True, timeout=10)
    except Exception: pass

llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=UTIL, max_model_len=MAXLEN,
          dtype="float16", disable_log_stats=True)
prompts = ["The history of computing began " * 4] * B
sp_warm = SamplingParams(max_tokens=WARM, min_tokens=WARM, ignore_eos=True, temperature=0.0)
sp      = SamplingParams(max_tokens=OUTTOK, min_tokens=OUTTOK, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp_warm, use_tqdm=False)  # warmup -> steady-state decode

res = {"model": MODEL, "pid": os.getpid(), "fp": _FP, "batch": B, "out": OUTTOK, "eager": EAGER}
if MODE == "bench":
    best = None
    for _ in range(2):
        t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
        gt = sum(len(x.outputs[0].token_ids) for x in o); tps = gt / dt
        best = tps if best is None or tps > best else best
    res["decode_tok_s"] = round(best, 1)
    print("BENCH", json.dumps(res))
else:
    o = llm.generate(prompts, sp, use_tqdm=False)
    res["tokens"] = sum(len(x.outputs[0].token_ids) for x in o)
    print("DECODE done", json.dumps(res))

if RESULT:
    with open(RESULT, "w") as f: json.dump(res, f)
_hb_stop.set(); time.sleep(0.05)
