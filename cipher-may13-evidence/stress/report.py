"""Compile pass/fail report from t1/t2/t3/t4 JSON outputs."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(name):
    p = ROOT / name
    if not p.exists(): return None
    return json.load(open(p))


def tok_w_drift(windows):
    if not windows or len(windows) < 2: return None
    ws = [w["mean_w"] for w in windows]
    return max(ws) - min(ws)


def main():
    print("="*70)
    print("CIPHER STRESS TEST REPORT — Llama-3.1-8B + FP8 + 1200 MHz")
    print("="*70)

    # ---- Test 1 ----
    t1c = load("t1_endurance.json")
    t1b = load("t1_endurance_baseline.json")
    print("\n[Test 1] 10K-token endurance")
    if t1c is None:
        print("  ERROR: t1_endurance.json missing")
    else:
        c1 = t1c["decoded"].get("tok_1", "")
        c5 = t1c["decoded"].get("tok_5000", "")
        c10 = t1c["decoded"].get("tok_10000", "")
        crash = t1c.get("crash")
        gen = t1c.get("generated", 0)
        pass_no_crash = (crash is None and gen == t1c["target_tokens"])
        coh1 = "OK" if c1 and "!!!" not in c1 else "GARBAGE"
        coh5 = "OK" if c5 and "!!!" not in c5 else ("GARBAGE" if c5 else "MISSING")
        coh10 = "OK" if c10 and "!!!" not in c10 else ("GARBAGE" if c10 else "MISSING")
        windows = t1c.get("windows_60s", [])
        w_drift = tok_w_drift(windows)
        max_mem = max((w.get("max_mem_mib", 0) for w in windows), default=0)
        pre_mem = t1c.get("pre_nvsmi_mib", 0)
        mem_growth = max_mem - pre_mem if max_mem else t1c.get("mem_growth_mib", 0)
        print(f"  CIPHER: gen={gen}/{t1c['target_tokens']} crash={crash}")
        print(f"          tps={t1c['tps']:.1f} tok/W={t1c['tok_w']:.3f} W={t1c['mean_w']:.0f}")
        print(f"          windows_drift_W={w_drift:.1f} max_mem={max_mem}MiB "
              f"growth_vs_pre={mem_growth}MiB" if w_drift is not None else
              f"          max_mem={max_mem}MiB growth_vs_pre={mem_growth}MiB")
        print(f"  coherence@1={coh1} @5000={coh5} @10000={coh10}")
        print(f"  tok_1   : {c1!r}")
        print(f"  tok_5000: {c5!r}")
        print(f"  tok_10000:{c10!r}")
        verdict = "PASS" if (pass_no_crash and coh1 == coh5 == coh10 == "OK"
                              and (mem_growth or 0) < 1024) else "FAIL"
        print(f"  --> {verdict}")
    if t1b is not None:
        print(f"  baseline: tps={t1b['tps']:.1f} tok/W={t1b['tok_w']:.3f} "
              f"tok_1={t1b['decoded'].get('tok_1','')!r}")

    # ---- Test 2 ----
    t2c = load("t2_concurrent.json")
    t2b = load("t2_concurrent_baseline.json")
    print("\n[Test 2] 8-client × 5 min sustained load")
    for label, t in [("CIPHER", t2c), ("baseline", t2b)]:
        if t is None:
            print(f"  {label}: missing")
            continue
        ok = t.get("ok_count", 0)
        n = t.get("n_children", 0)
        tps = t.get("agg_tps", 0)
        w = t.get("mean_w", 0)
        cuda_errs = t.get("cuda_errs", 0)
        first_text_ok = sum(1 for c in t.get("children", [])
                            if c.get("first_text") and "!!!" not in c["first_text"])
        verdict = ("PASS" if ok == n and cuda_errs == 0 and tps > 0
                   and first_text_ok == n else "FAIL")
        print(f"  {label}: ok={ok}/{n} cuda_errs={cuda_errs} agg_tps={tps:.1f} "
              f"W={w:.0f} tok/W={t.get('tok_w',0):.3f} coherent={first_text_ok}/{n}")
        for c in t.get("children", [])[:3]:
            print(f"    child {c.get('child_id')}: ok={c.get('ok')} "
                  f"calls={c.get('calls',0)} tokens={c.get('tokens',0)} "
                  f"first={c.get('first_text','')!r}")
        print(f"  --> {label} {verdict}")

    # ---- Test 3 ----
    t3c = load("t3_determinism.json")
    t3b = load("t3_determinism_baseline.json")
    print("\n[Test 3] Determinism @ T=0")
    for label, t in [("CIPHER", t3c), ("baseline", t3b)]:
        if t is None:
            print(f"  {label}: missing")
            continue
        intra = t.get("intra_match", 0)
        n = t.get("n_runs", 0)
        crashes = t.get("crashes", 0)
        unique = len(t.get("unique_hashes", []))
        verdict = "PASS" if intra == n and crashes == 0 else "FAIL"
        print(f"  {label}: intra_match={intra}/{n} unique_hashes={unique} "
              f"crashes={crashes} elapsed={t.get('elapsed_s', 0):.1f}s")
        print(f"  --> {label} intra-determinism {verdict}")
    if t3c is not None and t3b is not None:
        cref = t3c.get("ref_seq", [])
        bref = t3b.get("ref_seq", [])
        same = cref[:64] == bref[:64]
        if not same:
            first_div = None
            for i in range(min(len(cref), len(bref))):
                if cref[i] != bref[i]:
                    first_div = i; break
            print(f"  CIPHER vs baseline: divergent (first diff at token {first_div})")
            print(f"    cipher first 12: {cref[:12]}")
            print(f"    base   first 12: {bref[:12]}")
        else:
            print(f"  CIPHER vs baseline: identical")

    # ---- Test 4 ----
    t4c = load("t4_memleak.json")
    t4b = load("t4_memleak_baseline.json")
    print("\n[Test 4] 1000-call memory leak")
    for label, t in [("CIPHER", t4c), ("baseline", t4b)]:
        if t is None:
            print(f"  {label}: missing")
            continue
        completed = t.get("completed_calls", 0)
        target = t.get("target_calls", 1000)
        crashed = t.get("crashes", 0)
        nvsmi_growth = t.get("growth_nvsmi_mib", 0)
        # Leak rate (MiB / call), excluding warmup
        log = [e for e in t.get("log", []) if "nvsmi_mib" in e]
        if len(log) >= 2:
            d_mib = log[-1]["nvsmi_mib"] - log[0]["nvsmi_mib"]
            d_call = log[-1]["call"] - log[0]["call"]
            rate = d_mib / max(1, d_call)
        else:
            rate = float("nan")
        verdict = ("PASS" if completed == target and nvsmi_growth < 512
                   else "FAIL")
        print(f"  {label}: completed={completed}/{target} crashes={crashed}")
        print(f"          nvsmi {t.get('pre_nvsmi_mib')}MiB -> "
              f"{t.get('post_nvsmi_mib')}MiB  growth={nvsmi_growth}MiB  "
              f"rate~{rate:.2f} MiB/call")
        if t.get("last_err"):
            print(f"          last_err: {t['last_err']}")
        print(f"  --> {label} {verdict}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
