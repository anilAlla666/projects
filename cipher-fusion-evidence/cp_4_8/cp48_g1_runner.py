"""CP 4.8 item 4 — G1 runner. 21-WL tensor_mfu_pct measurement.

For each gated WL: launch its driver under the PRODUCTION substrate
(libcipher_rt c2c5d313 via CUDA_INJECTION64_PATH), run a sustained window
(memo §5: 600 s), parse run_for_duration's end-summary from the driver log,
compute analytical tensor_mfu_pct (mfu_compute, embedding-corrected post
§3 spot-check), and grade vs the FROZEN cp48_ceilings.json pre-registration.

Pass per WL: measured tensor_mfu_pct >= 0.80 x ceiling. WL23 = stability-only.
Driver failure / anomaly -> recorded as status!="ok", surfaced, NOT averaged.

Usage: python3 cp48_g1_runner.py WL01 [WL06 ...]   (subset, e.g. validation)
       python3 cp48_g1_runner.py --all              (the 21 gated WLs)
       env WL_DURATION overrides the per-WL window (default 600).
"""
import os, sys, json, time, subprocess, re, glob

CP48 = "/home/ubuntu/cipher-fusion-evidence/cp_4_8"
DRIVERS = "/home/ubuntu/cipher_workloads/drivers"
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"          # c2c5d313 — production
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import mfu_compute as M

MODELS = {
    "tiny":    "/home/ubuntu/models/TinyLlama-1.1B",
    "mistral": "/home/ubuntu/models/Mistral-7B-v0.1",
    "minilm":  "/home/ubuntu/models/all-MiniLM-L6-v2",
    "sdxl":    "/home/ubuntu/models/stable-diffusion-xl-base-1.0",
    "clip":    "/home/ubuntu/models/clip-vit-large-patch14",
    "llava":   "/home/ubuntu/models/llava-1.5-7b-hf",
}

# Per-WL config. units: "tokens" = use the end-summary tokens= field;
# ("iters_x", K) = iters * K (K precomputed for WL06). mean_context feeds
# the attention term (decode WLs: small KV; prefill WLs: seq/2).
WL = {
    "WL01": dict(drv="wl01_decode_b1.py",      model="tiny",    units="tokens",       mctx=64),
    "WL02": dict(drv="wl02_decode_b8.py",      model="tiny",    units="tokens",       mctx=64),
    "WL03": dict(drv="wl03_prefill_b8.py",     model="mistral", units="tokens",       mctx=512),
    "WL04": dict(drv="wl04_vllm_serving.py",   model="tiny",    units="tokens",       mctx=64),
    "WL06": dict(drv="wl06_embeddings.py",     model="minilm",  units=("iters_x", 0), mctx=8),
    "WL07": dict(drv="wl07_lora_finetune.py",  model="tiny",    units="tokens",       mctx=128),
    "WL08": dict(drv="wl08_sdxl.py",           model="sdxl",    units=("iters_x", 1), mctx=0),
    "WL09": dict(drv="wl09_whisper.py",        model=None,      units=("iters_x", 1), mctx=0),
    "WL10": dict(drv="wl10_speculative.py",    model="tiny",    units="tokens",       mctx=64),
    "WL11": dict(drv="wl11_agentic.py",        model="tiny",    units="tokens",       mctx=128),
    "WL12": dict(drv="wl12_batch.py",          model="tiny",    units="tokens",       mctx=64),
    "WL13": dict(drv="wl13_long_context_32k.py", model="mistral", units="tokens",     mctx=16384),
    "WL14": dict(drv="wl14_torch_compile.py",  model="tiny",    units="tokens",       mctx=64),
    "WL16": dict(drv="wl16_prefix_caching.py", model="tiny",    units="tokens",       mctx=64),
    "WL17": dict(drv="wl17_training_full.py",  model="tiny",    units="tokens",       mctx=128),
    "WL19": dict(drv="wl19_clip.py",           model="clip",    units=("iters_x", 32), mctx=0),
    "WL20": dict(drv="wl20_llava.py",          model="llava",   units=("iters_x", 1), mctx=0),
    "WL21": dict(drv="wl21_code_generation.py", model="tiny",   units="tokens",       mctx=64),
    "WL22": dict(drv="wl22_rag.py",            model="tiny",    units="tokens",       mctx=64),
    "WL23": dict(drv="wl23_model_switch.py",   model="tiny",    units="tokens",       mctx=64),
}
ORDER = ["WL01", "WL06", "WL13",            # cheap-first harness validation
         "WL02", "WL03", "WL04", "WL07", "WL10", "WL11", "WL12", "WL14",
         "WL16", "WL17", "WL21", "WL22", "WL23", "WL19", "WL08", "WL09", "WL20"]

CEILINGS = {c["wl"]: c for c in
            json.load(open(f"{CP48}/cp48_ceilings.json"))["ceilings"]}
DURATION = int(os.environ.get("WL_DURATION", "600"))
END_RE = re.compile(r'tenant=\S+ end; iters=(\d+) \S+=(\d+) tokens=(\d+) '
                    r'elapsed=([\d.]+)s')


def wl06_tokens_per_batch():
    """WL06 reports a fixed batch-count, not tokens. Compute the exact token
    count of its 512 fixed sentences with the MiniLM tokenizer."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODELS["minilm"])
    sents = ["This is sentence number " + str(i) for i in range(512)]
    return sum(len(tk(s).input_ids) for s in sents)


def run_wl(wl):
    cfg = WL[wl]
    drv = f"{DRIVERS}/{cfg['drv']}"
    log = f"{CP48}/g1_logs/{wl}.log"
    os.makedirs(f"{CP48}/g1_logs", exist_ok=True)
    env = dict(os.environ)
    env["CUDA_INJECTION64_PATH"] = RT                  # production substrate
    env["CIPHER_TENANT_ID"] = f"cp48_g1_{wl.lower()}"
    env["WL_TENANT_ID"] = env["CIPHER_TENANT_ID"]
    env["WL_DURATION"] = str(DURATION)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if cfg["model"]:
        env["WL_MODEL"] = MODELS[cfg["model"]]
    t0 = time.time()
    print(f"[g1] {wl}: launch {cfg['drv']} window={DURATION}s", flush=True)
    with open(log, "w") as lf:
        try:
            rc = subprocess.run(["python3", drv], env=env, stdout=lf,
                                 stderr=subprocess.STDOUT,
                                 timeout=DURATION + 900).returncode
        except subprocess.TimeoutExpired:
            return dict(wl=wl, status="timeout", note=f"exceeded {DURATION+900}s")
    wall = time.time() - t0
    txt = open(log).read()
    m = END_RE.search(txt)
    if not m:
        tail = "\n".join(txt.splitlines()[-6:])
        return dict(wl=wl, status="no-end-summary", rc=rc,
                    note=f"run_for_duration end-line not found; log tail:\n{tail}")
    iters, units_lbl, tokens, elapsed = (int(m.group(1)), int(m.group(2)),
                                         int(m.group(3)), float(m.group(4)))
    # resolve analytical units
    um = WL[wl]["units"]
    if um == "tokens":
        units = tokens
    else:                                  # ("iters_x", K)
        units = iters * um[1]
    tmfu = M.compute_tensor_mfu(wl, units, elapsed, mean_context=WL[wl]["mctx"])
    cl = CEILINGS[wl]
    row = dict(wl=wl, status="ok", rc=rc, wall_s=round(wall, 1),
               iters=iters, units=units, units_label_total=units_lbl,
               tokens=tokens, elapsed_s=round(elapsed, 1),
               regime=cl["regime"], ceiling_pct=cl["ceiling_pct"],
               pass_pct=cl["pass_pct"], tensor_mfu_pct=None, verdict=None)
    if cl["ceiling_pct"] is None:                      # WL23 stability-only
        row["verdict"] = "stability-only (no tensor_mfu gate)"
        row["tensor_mfu_pct"] = round(tmfu, 4) if tmfu else None
    else:
        row["tensor_mfu_pct"] = round(tmfu, 4)
        row["verdict"] = "PASS" if tmfu >= cl["pass_pct"] else "FAIL"
        row["frac_of_ceiling"] = round(tmfu / cl["ceiling_pct"], 3)
    return row


def main():
    args = sys.argv[1:]
    wls = ORDER if (args == ["--all"] or not args) else args
    if "WL06" in wls:
        k = wl06_tokens_per_batch()
        WL["WL06"]["units"] = ("iters_x", k)
        print(f"[g1] WL06 token/batch resolved = {k}", flush=True)
    results = []
    for wl in wls:
        r = run_wl(wl)
        results.append(r)
        if r["status"] != "ok":
            print(f"[g1] {wl}: STATUS={r['status']} — surfacing, not averaging. "
                  f"{r.get('note','')}", flush=True)
        else:
            print(f"[g1] {wl}: tensor_mfu={r['tensor_mfu_pct']}%  "
                  f"ceiling={r['ceiling_pct']}%  verdict={r['verdict']}", flush=True)
        json.dump(r, open(f"{CP48}/g1_logs/{wl}_result.json", "w"), indent=2)
    out = dict(cp="4.8", gate="G1", substrate="libcipher_rt c2c5d313",
               window_s=DURATION, n_wl=len(results), results=results)
    json.dump(out, open(f"{CP48}/cp48_g1_result.json", "w"), indent=2)
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    print(f"\n[g1] DONE: {len(ok)}/{len(results)} ran clean; "
          f"{len(bad)} surfaced issues: {[r['wl'] for r in bad]}")


if __name__ == "__main__":
    main()
