"""Step 18: online reinforcement learning for test writing on CodeContests cases (tools/cc_build.py), on a GPU host.

One iteration: serve the current adapter on every GPU, sample GROUP answers for each of PER_ITER training cases,
score every answer on its case's hidden pool (tools/cc_score.py, balanced accuracy), turn each group into advantages
(answer score minus the group's mean, scaled and clipped; groups whose answers all score the same are dropped), and
train one advantage-weighted pass from the current adapter (tools/train_judge.py --init-adapter). A dev set the
training never sees is scored with one answer per case before the first iteration and every EVAL_EVERY iterations:
that curve is the first, cheap test of whether anything is being learned.

    python3 tools/rl_cc.py --cases ~/cccases --work ~/s18 --iters 8        (on the GPU host, in ~/vbrepo2)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_score import parse, prompt, score  # noqa: E402

HOME = Path.home()
SERVE = HOME / "s13" / "serve_lora.sh"
BASE = "qwen38-27b"


def log(work: Path, **kw) -> None:
    line = json.dumps({"t": time.strftime("%H:%M:%S"), **kw})
    print(line, flush=True)
    with open(work / "rl.log", "a") as f:
        f.write(line + "\n")


def gpus() -> int:
    return int(subprocess.run("nvidia-smi --query-gpu=index --format=csv,noheader | wc -l", shell=True,
                              capture_output=True, text=True).stdout)


def stop_servers() -> None:
    subprocess.run("pkill -f '[v]llm serve'; sleep 20; pkill -f '[E]ngineCore'", shell=True)
    while True:
        used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                              capture_output=True, text=True).stdout.split()
        if max(int(x) for x in used) < 1000:
            return
        time.sleep(5)


def start_servers(work: Path, adapter: Path | None) -> list[str]:
    n, urls = gpus(), []
    for g in range(n):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(g), "PORT": str(8000 + g),
                            "LORA": f"--lora-modules policy={adapter}" if adapter else ""}
        subprocess.Popen(["bash", str(SERVE)], env=env, stdout=open(work / f"serve_{g}.log", "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        urls.append(f"http://127.0.0.1:{8000 + g}/v1")
    for g in range(n):
        while True:
            t = (work / f"serve_{g}.log").read_text(errors="replace")
            if "Application startup complete" in t:
                break
            if "initialization failed" in t or "Error" in t.splitlines()[-1:][0:1]:
                raise RuntimeError(f"server {g} failed")
            time.sleep(10)
    return urls


def sample(url: str, model: str, case: dict, n: int, temperature: float) -> list[dict]:
    import httpx2 as httpx
    body = {"model": model, "n": n, "temperature": temperature, "max_tokens": 12000,
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "messages": [{"role": "user", "content": prompt(case)}]}
    for attempt in range(5):
        try:
            r = httpx.post(url + "/chat/completions", json=body, timeout=3600)
            return [{"content": c["message"].get("content") or "",
                     "reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "",
                     "finish": c.get("finish_reason")} for c in r.json()["choices"]]
        except Exception:  # noqa: BLE001
            time.sleep(20)
    return []


def rollouts(urls: list[str], model: str, cases: list[dict], n: int, temperature: float) -> list[list[dict]]:
    with cf.ThreadPoolExecutor(16 * len(urls)) as ex:
        futs = [ex.submit(sample, urls[i % len(urls)], model, c, n, temperature) for i, c in enumerate(cases)]
        outs = [f.result() for f in futs]
    with cf.ThreadPoolExecutor(48) as ex:
        jobs = {ex.submit(score, c, parse(o["content"])): (i, j) for i, (c, os_) in enumerate(zip(cases, outs))
                for j, o in enumerate(os_)}
        for f in cf.as_completed(jobs):
            i, j = jobs[f]
            try:
                outs[i][j]["score"] = f.result()
            except Exception as e:  # noqa: BLE001
                outs[i][j]["score"] = {"ba": 0.5, "error": str(e)[:200], "parsed": False}
    return outs


def rename(src: Path, dst: Path) -> None:
    """Training saves `model.layers.*`; the served checkpoint names them `model.language_model.layers.*`."""
    from safetensors.torch import load_file, save_file
    dst.mkdir(parents=True, exist_ok=True)
    w = load_file(src / "adapter_model.safetensors")
    w2 = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v
          for k, v in w.items()}
    assert len(w2) == len(w) and all(".language_model.layers." in k for k in w2)
    save_file(w2, dst / "adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")


def dev_eval(work: Path, urls: list[str], model: str, dev: list[dict], it: int) -> float:
    outs = rollouts(urls, model, dev, 1, 0.7)
    s = [o[0]["score"] for o in outs if o]
    log(work, dev_iter=it, ba=round(statistics.mean(x["ba"] for x in s), 4),
        tpr=round(statistics.mean(x.get("tpr", 1) for x in s), 4), tnr=round(statistics.mean(x.get("tnr", 0) for x in s), 4),
        parsed=round(statistics.mean(1.0 if x.get("parsed") else 0.0 for x in s), 3), n=len(s))
    return statistics.mean(x["ba"] for x in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--per-iter", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--gate-iter", type=int, default=4, help="the cheap gate: dev must have risen by --gate-min here")
    ap.add_argument("--gate-min", type=float, default=0.03)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    files = sorted(a.cases.glob("*.json"))
    random.Random(0).shuffle(files)
    dev = [json.loads(f.read_text()) for f in files[:a.dev]]
    train = [json.loads(f.read_text()) for f in files[a.dev:]]
    log(a.work, start=True, dev=len(dev), train=len(train))
    adapter, served = None, None
    stop_servers()
    urls = start_servers(a.work, None)
    dev0 = dev_eval(a.work, urls, BASE, dev, 0)
    rng = random.Random(1)
    for it in range(1, a.iters + 1):
        batch = rng.sample(train, a.per_iter)
        outs = rollouts(urls, "policy" if served else BASE, batch, a.group, 1.0)
        rows, kept_groups, all_ba = [], 0, []
        for c, os_ in zip(batch, outs):
            bas = [o["score"]["ba"] for o in os_]
            all_ba += bas
            if len(set(bas)) < 2:
                continue
            kept_groups += 1
            m = statistics.mean(bas)
            for o, b in zip(os_, bas):
                adv = max(-2.0, min(2.0, (b - m) / a.scale))
                if adv == 0:
                    continue
                rows.append({"source": "own", "task": c["name"], "file": c["name"], "system": "", "tools": [],
                             "messages": [{"role": "user", "content": prompt(c)},
                                          {"role": "assistant", "content": o["content"], "reasoning": o["reasoning"]}],
                             "weight": round(adv, 4)})
        log(a.work, iter=it, sample_ba=round(statistics.mean(all_ba), 4), groups_with_signal=kept_groups,
            rows=len(rows), parsed=round(statistics.mean(1.0 if o["score"].get("parsed") else 0.0
                                                         for os_ in outs for o in os_), 3))
        data = a.work / f"it{it}.jsonl"
        data.write_text("".join(json.dumps(r) + "\n" for r in rows))
        stop_servers()
        out = a.work / f"ad{it}"
        cmd = [str(HOME / "tr/bin/torchrun"), "--standalone", f"--nproc_per_node={gpus()}", "tools/train_judge.py",
               "--data", str(data), "--out", str(out), "--epochs", "1", "--lr", str(a.lr), "--seg", "1",
               "--max-len", "32768", "--no-offload", "--neg-cap", "2.0", "--accum", "8"]
        if adapter:
            cmd += ["--init-adapter", str(adapter)]
        r = subprocess.run(cmd, cwd=HOME / "vbrepo2", stdout=open(a.work / f"train{it}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode or '"DONE"' not in (a.work / f"train{it}.log").read_text():
            log(a.work, iter=it, train_failed=True)
            return
        adapter = out
        served = a.work / f"ad{it}_hf"
        rename(out, served)
        urls = start_servers(a.work, served)
        if it % a.eval_every == 0 or it == a.iters or it == a.gate_iter:
            d = dev_eval(a.work, urls, "policy", dev, it)
            if it == a.gate_iter and d - dev0 < a.gate_min:
                log(a.work, gate_failed=True, dev0=round(dev0, 4), dev=round(d, 4))
                stop_servers()
                return
    stop_servers()
    log(a.work, done=True, adapter=str(served))


if __name__ == "__main__":
    main()
