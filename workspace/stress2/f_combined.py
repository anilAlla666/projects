"""F1, F3, F4, F5, F7 — adversarial tests."""
import os, sys, json, time, subprocess, gc, signal, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
ROOT = sc.ROOT


def f1_model_switch():
    """20 cycles 1B↔8B. No crash."""
    rt, _, _ = sc.init_cipher()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    P1B = "/home/ubuntu/models/Llama-3.2-1B"
    P8B = "/home/ubuntu/models/Llama-3.1-8B"
    crashes = 0
    for i in range(20):
        path = P1B if i % 2 == 0 else P8B
        try:
            tok = AutoTokenizer.from_pretrained(path)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
            model.requires_grad_(False); model.train(False)
            try: sc.patch_fusion(rt, model)
            except Exception: pass
            ids = tok("Hello world.", return_tensors="pt").input_ids.to("cuda:0")
            attn = torch.ones_like(ids)
            with torch.no_grad():
                _ = model.generate(ids, attention_mask=attn, max_new_tokens=8,
                                    do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            del model, tok, ids, attn
            gc.collect(); torch.cuda.empty_cache()
            print(f"  [F1] cycle {i:>2} {os.path.basename(path)} ok", flush=True)
        except Exception as e:
            crashes += 1
            print(f"  [F1] cycle {i}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break
    return dict(cycles=20, crashes=crashes)


def f3_proc_pressure():
    """Spawn 50 processes that each allocate a 100MB torch tensor and idle 10s."""
    children = []
    n = 50
    for i in range(n):
        cmd = [sys.executable, "-c",
               "import torch, time; "
               "x = torch.randn(25_000_000, device='cuda'); "
               "time.sleep(8); "
               "print('proc', x.sum().item())"]
        env = os.environ.copy()
        # Distribute across both GPUs.
        env["CUDA_VISIBLE_DEVICES"] = str(i % 2)
        p = subprocess.Popen(cmd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(p)
        if i % 10 == 9:
            time.sleep(0.5)
    # Wait up to 60 s — should finish in ~10 s.
    deadline = time.time() + 60
    pending = [p for p in children if p.poll() is None]
    while pending and time.time() < deadline:
        time.sleep(1.0)
        pending = [p for p in children if p.poll() is None]
    survivors = sum(1 for p in children if p.returncode == 0)
    hung = len(pending)
    for p in pending:
        try: p.kill()
        except Exception: pass
    return dict(spawned=n, ok=survivors, hung=hung)


def f4_kill_mid_inference():
    """Start a generate() in a subprocess; kill -9 mid-flight; check GPU still usable."""
    cmd = [sys.executable, "-c",
           "import os, sys, ctypes, torch; "
           "sys.path.insert(0, '/home/ubuntu/op31-prod-fix/stress'); "
           "import stress_common as sc; sc._setup_alloc_env(); "
           "rt,_,_ = sc.init_cipher(); "
           "model, tok, _ = sc.load_model('cuda:0', patch=True, rt=rt); "
           "ids = tok('Long prompt.', return_tensors='pt').input_ids.to('cuda:0'); "
           "attn = torch.ones_like(ids); "
           "_ = model.generate(ids, attention_mask=attn, max_new_tokens=10000, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if "LD_PRELOAD" not in env:
        env["LD_PRELOAD"] = (f"{ROOT}/libcipher_hook.so "
                             f"/usr/lib/x86_64-linux-gnu/libcuda.so")
    env.setdefault("CIPHER_FP8_COMPUTE", "on")
    env.setdefault("CIPHER_SUBSTITUTE_V2", "on")
    env.setdefault("CIPHER_FUSION_KERNELS", "on")
    p = subprocess.Popen(cmd, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Let it load + start generation, then kill -9.
    time.sleep(20.0)
    if p.poll() is None:
        os.kill(p.pid, signal.SIGKILL)
        p.wait(timeout=5)
        killed = True
    else:
        killed = False
    # Now try a fresh CUDA op.
    try:
        verify_cmd = [sys.executable, "-c",
                      "import torch; x = torch.randn(1024, device='cuda'); "
                      "print('OK', x.sum().item())"]
        r = subprocess.run(verify_cmd, env={"CUDA_VISIBLE_DEVICES": "0"},
                           capture_output=True, text=True, timeout=30)
        usable = "OK" in r.stdout
        verify_out = (r.stdout + r.stderr)[:200]
    except Exception as e:
        usable = False
        verify_out = f"{type(e).__name__}: {e}"
    return dict(killed=killed, gpu_usable_after_kill=usable,
                verify_output=verify_out)


def f5_clock_switch():
    """10 cycles 1200↔1980 MHz under load. Sudo required."""
    # Start a CUDA-busy process in background.
    busy_cmd = [sys.executable, "-c",
                "import torch, time; t0=time.time(); "
                "while time.time() - t0 < 30: "
                "  a=torch.randn(4096,4096,device='cuda',dtype=torch.float16); "
                "  b=torch.randn(4096,4096,device='cuda',dtype=torch.float16); "
                "  c=a@b; torch.cuda.synchronize()"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"
    p = subprocess.Popen(busy_cmd, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    crashes = 0
    for i in range(10):
        target = 1200 if i % 2 == 0 else 1980
        rc = subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0",
                             "-lgc", str(target)],
                            capture_output=True, text=True, timeout=10).returncode
        if rc != 0: crashes += 1
        time.sleep(0.5)
    # Restore lock to 1200.
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "1200"],
                   capture_output=True, text=True, timeout=10)
    p.wait(timeout=30)
    busy_ok = (p.returncode == 0)
    return dict(switch_failures=crashes, busy_proc_survived=busy_ok)


def f7_torch_compile():
    """torch.compile under CIPHER. Just check no crash."""
    rt, _, _ = sc.init_cipher()
    import torch
    try:
        @torch.compile
        def f(x):
            return torch.nn.functional.silu(x) * x.sum(dim=-1, keepdim=True)
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
        # Trigger compilation.
        y = f(x)
        torch.cuda.synchronize()
        # And one more invocation — exercise the compiled cache.
        y2 = f(x)
        torch.cuda.synchronize()
        return dict(crashed=False, out_finite=bool(torch.isfinite(y).all().item()),
                    out_finite2=bool(torch.isfinite(y2).all().item()))
    except Exception as e:
        return dict(crashed=True, error=f"{type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    which = sys.argv[1]
    fn = {"F1": f1_model_switch, "F3": f3_proc_pressure,
          "F4": f4_kill_mid_inference, "F5": f5_clock_switch,
          "F7": f7_torch_compile}[which]
    print(f"[{which}] starting", flush=True)
    t0 = time.perf_counter()
    out = fn()
    out["elapsed_s"] = time.perf_counter() - t0
    print(f"[{which}] {out}", flush=True)
    with open(os.path.join(os.path.dirname(__file__), f"{which.lower()}.json"),
              "w") as f:
        json.dump(out, f, indent=2)
