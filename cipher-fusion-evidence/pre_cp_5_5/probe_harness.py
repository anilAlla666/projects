"""CP 5.5 pre-work plugin-routed Koopman feasibility probe harness.

Runs a single variant per process invocation. Forks for each variant outside
this script so vLLM EngineCore state cannot leak across measurements.

Variant selection: CIPHER_PROBE_HOOK env (none / python / ctypes) is read by
the cipher_vllm_plugin general_plugins entry point at worker subprocess init.

Workload: TinyLlama-1.1B-Chat-v1.0 FP16 N=1 decode, 128 generated tokens.
Reps per invocation: default 7. First rep is warmup (dropped). Reported tok/s
is mean over reps 2..N. Per-token latency from generated_tokens / elapsed.

Output: JSON to --out path with run_id, variant, reps, per-rep tok/s, mean,
stdev, crossings counter (from /tmp/cipher_probe_<worker_pid>.json if hook
variant), GPU clock samples, and reproducibility metadata.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time


def gpu_clocks():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=3).decode().strip()
        gr, mem, pwr, tmp = [x.strip() for x in out.split(",")]
        return {"clocks_gr_mhz": int(gr), "clocks_mem_mhz": int(mem),
                "power_w": float(pwr), "temp_c": int(tmp)}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("baseline", "python", "ctypes"))
    ap.add_argument("--model", default="/home/ubuntu/.cache/huggingface/hub/"
                    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16",
                    help="float16 or bfloat16; default float16 for fp16-only baseline")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable cudagraph + torch.compile to expose Python apply path")
    args = ap.parse_args()

    if os.path.isdir(args.model):
        snap_dir = args.model
        snaps = sorted(glob.glob(os.path.join(snap_dir, "*")))
        if snaps:
            args.model = snaps[0]

    print(f"[probe] variant={args.variant} model={args.model} "
          f"tokens={args.tokens} reps={args.reps} dtype={args.dtype} "
          f"pid={os.getpid()}", file=sys.stderr, flush=True)

    pre = {"start_time": time.time(), "gpu": gpu_clocks(), "pid": os.getpid()}

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=0.5,
              max_model_len=2048, enforce_eager=args.enforce_eager,
              disable_log_stats=True)

    # Install the probe hook via collective_rpc AFTER model load. The plugin's
    # register-time install at cipher_vllm_kv.py:_install_probe_hook is bypassed
    # in vLLM V1 — the worker apparently caches the apply reference (or compiles
    # past it) between general_plugins load and model forward. Verified
    # empirically: plugin-installed hook gives crossings=0; collective_rpc
    # install post-LLM() gives crossings = 88 layers × tokens, exact match for
    # TinyLlama-1.1B. See PRE_CP_5_5_PLUGIN_ROUTED_FEASIBILITY.md §3 for the
    # plumbing-vs-production-install separation.
    def _install_hook(self):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        import sys as _sys
        if args.variant == "baseline":
            _sys.modules["__main__"]._cipher_probe_counter = [0]
            return 0
        counter = [0]
        orig = UnquantizedLinearMethod.apply
        if args.variant == "ctypes":
            import ctypes
            rt = None
            for cand in ("libcipher_rt.so",
                         "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"):
                try:
                    rt = ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL); break
                except OSError:
                    continue
            assert rt is not None, "libcipher_rt.so not loadable in worker"
            noop = rt.cipher_rt_koopman_skip_dtype
            noop.restype = ctypes.c_uint64
            noop.argtypes = []
            def w(self_q, layer, x, bias=None):
                counter[0] += 1
                noop()
                return orig(self_q, layer, x, bias)
        else:  # python
            def w(self_q, layer, x, bias=None):
                counter[0] += 1
                return orig(self_q, layer, x, bias)
        UnquantizedLinearMethod.apply = w
        _sys.modules["__main__"]._cipher_probe_counter = counter
        return id(counter)

    def _read_crossings(self):
        import sys as _sys
        c = getattr(_sys.modules["__main__"], "_cipher_probe_counter", None)
        return c[0] if c else None

    def _reset_crossings(self):
        import sys as _sys
        c = getattr(_sys.modules["__main__"], "_cipher_probe_counter", None)
        if c:
            c[0] = 0
        return True

    llm.collective_rpc(_install_hook)

    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                        ignore_eos=True, min_tokens=args.tokens)
    prompt = "Once upon a time"

    per_rep = []
    for rep in range(args.reps):
        llm.collective_rpc(_reset_crossings)
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        rep_crossings = llm.collective_rpc(_read_crossings)[0]
        n_gen = len(outs[0].outputs[0].token_ids)
        tps = n_gen / elapsed if elapsed > 0 else 0.0
        per_rep.append({"rep": rep, "elapsed_s": elapsed,
                        "n_gen": n_gen, "tok_s": tps,
                        "crossings": rep_crossings,
                        "gpu_post": gpu_clocks()})
        print(f"[probe] rep={rep} elapsed={elapsed:.4f}s "
              f"n_gen={n_gen} tok_s={tps:.3f} crossings={rep_crossings}",
              file=sys.stderr, flush=True)

    measured = per_rep[1:] if len(per_rep) > 1 else per_rep
    tps_values = [r["tok_s"] for r in measured]
    elapsed_values = [r["elapsed_s"] for r in measured]
    mean_tps = statistics.mean(tps_values)
    stdev_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0
    mean_elapsed = statistics.mean(elapsed_values)

    # Sum per-rep crossings for the measured (post-warmup) reps; matches the
    # per-rep timing window exactly.
    crossings = sum((r.get("crossings") or 0) for r in measured)
    crossings_path = None

    import gc
    del llm
    gc.collect()

    post = {"end_time": time.time(), "gpu": gpu_clocks()}
    result = {
        "variant": args.variant,
        "model": args.model,
        "tokens": args.tokens,
        "reps_total": args.reps,
        "reps_measured": len(measured),
        "warmup_dropped": 1 if len(per_rep) > 1 else 0,
        "dtype": args.dtype,
        "per_rep": per_rep,
        "mean_tok_s": mean_tps,
        "stdev_tok_s": stdev_tps,
        "mean_elapsed_s": mean_elapsed,
        "crossings_total": crossings,
        "crossings_dump_path": crossings_path,
        "pre": pre,
        "post": post,
        "env_CIPHER_PROBE_HOOK": os.environ.get("CIPHER_PROBE_HOOK"),
        "env_CIPHER_KV_ALLOC": os.environ.get("CIPHER_KV_ALLOC"),
        "env_LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        "env_CUDA_INJECTION64_PATH": os.environ.get("CUDA_INJECTION64_PATH"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote {args.out}", file=sys.stderr, flush=True)
    print(f"[probe] mean_tok_s={mean_tps:.3f} stdev={stdev_tps:.3f} "
          f"crossings={crossings}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
