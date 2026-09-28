"""Test 2 driver — spawn N children, monitor, kill after deadline.

Distributes children across both H100s so model weights fit:
    Llama-3.1-8B fp16 ~ 16 GB; 4 instances per GPU ~ 64 GB / 80 GB.
Records aggregate tok/s, watts, and CUDA-error counts.
"""
import os, sys, json, time, signal, subprocess, threading
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import stress_common as sc

ROOT = Path(__file__).resolve().parent.parent
N_CHILDREN = int(os.environ.get("STRESS_T2_N", "8"))
DURATION_S = int(os.environ.get("STRESS_T2_DURATION", "300"))
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
SUFFIX     = os.environ.get("STRESS_SUFFIX",
                            "_baseline" if not USE_CIPHER else "")


def main():
    deadline = time.time() + DURATION_S
    print(f"[t2] spawning {N_CHILDREN} children, "
          f"deadline=+{DURATION_S}s, cipher={USE_CIPHER}", flush=True)

    children = []
    n_gpus = 2
    for cid in range(N_CHILDREN):
        gpu = cid % n_gpus  # round-robin across GPUs
        env = os.environ.copy()
        # Each child sees only its assigned GPU as cuda:0.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["STRESS_T2_DEADLINE"] = str(deadline)
        env["STRESS_T2_CHILD"] = str(cid)
        env["STRESS_T2_DEVICE"] = "cuda:0"
        env["STRESS_SUFFIX"] = SUFFIX
        # CIPHER env preserved from caller; LD_PRELOAD inherited.
        if USE_CIPHER:
            env["LD_PRELOAD"] = (f"{ROOT}/libcipher_hook.so "
                                 f"/usr/lib/x86_64-linux-gnu/libcuda.so")
            env.setdefault("CIPHER_FP8_COMPUTE", "on")
            env.setdefault("CIPHER_SUBSTITUTE_V2", "on")
            env.setdefault("CIPHER_FUSION_KERNELS", "on")
            # M1: enable weight-share so peer CIPHER children IPC-map the
            # same fp16 weight buffer instead of each loading its own
            # 16 GB copy. cipher_weight_share infrastructure ships in
            # libcipher_rt.so; env flag activates it.
            env.setdefault("CIPHER_WEIGHT_SHARE", "on")
        env["CIPHER"] = "1" if USE_CIPHER else "0"

        log_path = ROOT / "stress" / f"t2_child_{cid}{SUFFIX}.stdout"
        f = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "stress" / "t2_child.py")],
            env=env, stdout=f, stderr=subprocess.STDOUT,
        )
        children.append((cid, gpu, p, f))
        # Stagger spawns slightly so they don't all hammer model load at once.
        time.sleep(2.0)
        print(f"[t2] launched child {cid} on GPU {gpu} pid={p.pid}",
              flush=True)

    # Power/memory sampling thread.
    samples = []
    stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, n_gpus, 0.5), daemon=True)
    th.start()

    # Heartbeat loop until deadline, then kill leftovers.
    last_hb = 0
    while time.time() < deadline + 30:  # +30s grace
        now = time.time()
        if now - last_hb >= 30:
            last_hb = now
            alive = [c for c in children if c[2].poll() is None]
            print(f"[t2 hb {int(now-(deadline-DURATION_S)):4d}s] "
                  f"alive={len(alive)}/{len(children)} "
                  f"watts={samples[-1][1] if samples else 0:.0f} "
                  f"mem={samples[-1][2] if samples else 0}MiB",
                  flush=True)
        if all(c[2].poll() is not None for c in children):
            break
        time.sleep(1.0)

    # Force-kill any survivors.
    for cid, gpu, p, f in children:
        if p.poll() is None:
            print(f"[t2] killing child {cid} pid={p.pid}", flush=True)
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=5)
        f.close()

    stop.set(); th.join(timeout=2)

    # Aggregate.
    rows = []
    total_tokens = 0; total_calls = 0; cuda_errs = 0; ok_count = 0
    for cid, gpu, p, _ in children:
        path = ROOT / "stress" / f"t2_child_{cid}{SUFFIX}.json"
        if path.exists():
            d = json.load(open(path))
            rows.append(d)
            total_tokens += d.get("tokens", 0)
            total_calls  += d.get("calls", 0)
            cuda_errs    += d.get("cuda_errs", 0)
            ok_count     += int(d.get("ok", False))
        else:
            rows.append(dict(child_id=str(cid), missing_json=True,
                             returncode=p.returncode))

    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    max_mem = max(s[2] for s in samples) if samples else 0
    elapsed = DURATION_S
    agg_tps = total_tokens / elapsed
    tok_w = agg_tps / mean_w if mean_w > 0 else 0

    payload = dict(
        cipher=USE_CIPHER, n_children=N_CHILDREN,
        duration_s=elapsed,
        ok_count=ok_count, total_tokens=total_tokens,
        total_calls=total_calls, cuda_errs=cuda_errs,
        agg_tps=agg_tps, mean_w=mean_w, tok_w=tok_w,
        max_mem_mib=max_mem, watts_samples=len(samples),
        children=rows,
    )
    out_path = ROOT / "stress" / f"t2_concurrent{SUFFIX}.json"
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\n[t2] {ok_count}/{N_CHILDREN} ok, "
          f"agg_tps={agg_tps:.1f} W={mean_w:.0f} "
          f"tok/W={tok_w:.3f} cuda_errs={cuda_errs} "
          f"max_mem={max_mem}MiB",
          flush=True)
    print(f"[t2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
