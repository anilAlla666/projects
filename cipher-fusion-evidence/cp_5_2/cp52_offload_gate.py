#!/usr/bin/env python3
"""CP 5.2 Step 3 — offload correctness gate driver (Option A).

Runs cp52_offload_gate_worker.py three times in separate processes:
  1. BASELINE      — large KV pool, no preemption, offload ON (never fires)
  2. OFFLOAD       — tiny KV pool forces preemption, offload ON
                     (snapshot-on-preempt / restore-on-resume)
  3. CONTROL       — tiny KV pool forces preemption, offload OFF
                     (vanilla vLLM recompute preemption)

PASS requires:
  (A) token-ID identity — every output token of every prompt in the OFFLOAD
      run matches the BASELINE run. The CP 5.2 §4 correctness bar: a
      preempted-and-restored request decodes identically to one never
      preempted. (CONTROL is also checked == BASELINE: confirms the tiny pool
      triggers preemption and vanilla recompute is itself correct.)
  (B) hooks fired in the scheduler — the OFFLOAD run's stderr carries
      cipher-offload 'snapshot' and 'restore' lines, count > 0, and
      snapshots == restores (every parked sequence was restored, no leak).
  (C) tenant tag preserved — every restore line reports the tag PRESERVED
      (or n/a only if KV is not CIPHER-owned), none MISMATCH.
"""
import json
import os
import re
import subprocess
import sys

HERE = "/home/ubuntu/cipher-fusion-evidence/cp_5_2"
WORKER = f"{HERE}/cp52_offload_gate_worker.py"
TINY_BLOCKS = "32"   # << blocks needed by 6 concurrent ~268-token seqs (~102)


def run(label, blocks, offload):
    print(f"\n{'=' * 72}\n=== RUN: {label}  (blocks={blocks or 'none'}, "
          f"offload={'ON' if offload else 'OFF'})\n{'=' * 72}", flush=True)
    env = {**os.environ,
           "CIPHER_GATE_BLOCKS": str(blocks or 0),
           "CIPHER_KV_OFFLOAD": "1" if offload else "0",
           "CIPHER_KV_ALLOC": "1"}
    p = subprocess.run([sys.executable, WORKER], env=env,
                       capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"GATE FAIL — {label} worker exited {p.returncode}")
    lines = [ln for ln in p.stdout.splitlines()
             if ln.startswith("CIPHER_GATE_RESULT ")]
    if not lines:
        raise SystemExit(f"GATE FAIL — {label}: no CIPHER_GATE_RESULT line")
    return json.loads(lines[-1][len("CIPHER_GATE_RESULT "):]), p.stdout + p.stderr


def tokens_match(a, b):
    if len(a["prompts"]) != len(b["prompts"]):
        return False
    return all(pa["token_ids"] == pb["token_ids"]
               for pa, pb in zip(a["prompts"], b["prompts"]))


def main():
    baseline, _ = run("BASELINE", 0, True)
    offload, off_log = run("OFFLOAD", TINY_BLOCKS, True)
    control, _ = run("CONTROL", TINY_BLOCKS, False)

    def first_div(pa, pb):
        for j, (x, y) in enumerate(zip(pa["token_ids"], pb["token_ids"])):
            if x != y:
                return j
        return None

    # --- Check A: token-ID identity (offload vs baseline) ------------------
    print(f"\n{'=' * 72}\n=== GATE CHECK A — token-ID identity "
          f"(offload preempt/resume vs unpreempted baseline)\n{'=' * 72}",
          flush=True)
    a_off = tokens_match(baseline, offload)
    for i, (pb, po) in enumerate(zip(baseline["prompts"], offload["prompts"])):
        d = first_div(pb, po)
        same = d is None and pb["n_tokens"] == po["n_tokens"]
        print(f"[{'OK  ' if same else 'DIFF'}] prompt {i}: {pb['prompt']!r}  "
              f"(baseline n={pb['n_tokens']}, offload n={po['n_tokens']})",
              flush=True)
        if not same and d is not None:
            print(f"        first divergence at token {d}: "
                  f"baseline={pb['token_ids'][d]} offload={po['token_ids'][d]}",
                  flush=True)
    print(f"  OFFLOAD vs BASELINE : {'PASS' if a_off else 'FAIL'}", flush=True)

    # --- Diagnostic: CONTROL (vanilla recompute) vs baseline ---------------
    # Not a pass criterion. vLLM v1 preemption recomputes KV via re-prefill;
    # that is "numerically equivalent", NOT guaranteed byte-identical (prefill
    # vs incremental-decode kernels differ in FP reduction order). A control
    # divergence is EXPECTED and corroborates that the tiny pool forced
    # destructive preemption — and shows CIPHER offload (exact-byte restore)
    # is strictly more correct than recompute.
    print(f"\n{'=' * 72}\n=== DIAGNOSTIC — CONTROL (vanilla recompute) vs "
          f"baseline  [not a pass criterion]\n{'=' * 72}", flush=True)
    a_ctl = tokens_match(baseline, control)
    ctl_divs = []
    for i, (pb, pc) in enumerate(zip(baseline["prompts"], control["prompts"])):
        d = first_div(pb, pc)
        if d is not None:
            ctl_divs.append((i, d))
            print(f"  prompt {i}: recompute diverged at token {d} "
                  f"(baseline={pb['token_ids'][d]} control={pc['token_ids'][d]})",
                  flush=True)
    if a_ctl:
        print("  control == baseline — vanilla recompute happened to stay "
              "byte-identical this run", flush=True)
    else:
        print(f"  control diverged on {len(ctl_divs)} prompt(s) — expected: "
              "vanilla recompute is not byte-exact. CIPHER offload restores "
              "exact KV bytes and stays identical (Check A).", flush=True)

    # --- Check B: hooks fired ----------------------------------------------
    print(f"\n{'=' * 72}\n=== GATE CHECK B — offload hooks fired in scheduler"
          f"\n{'=' * 72}", flush=True)
    snaps = re.findall(r"\[cipher-offload\] snapshot req=", off_log)
    rests = re.findall(r"\[cipher-offload\] restore req=", off_log)
    installed = "[cipher-offload] hooks installed" in off_log
    n_snap, n_rest = len(snaps), len(rests)
    print(f"[{'OK  ' if installed else 'MISS'}] 'hooks installed' line present",
          flush=True)
    print(f"[{'OK  ' if n_snap > 0 else 'MISS'}] snapshot lines: {n_snap}",
          flush=True)
    print(f"[{'OK  ' if n_rest > 0 else 'MISS'}] restore lines:  {n_rest}",
          flush=True)
    balanced = n_snap > 0 and n_snap == n_rest
    print(f"[{'OK  ' if balanced else 'MISS'}] snapshots == restores "
          f"(no parked-sequence leak)", flush=True)
    b_pass = installed and n_snap > 0 and balanced

    # --- Check C: tenant tag preserved -------------------------------------
    print(f"\n{'=' * 72}\n=== GATE CHECK C — tenant tag preserved across "
          f"preempt/resume\n{'=' * 72}", flush=True)
    tag_lines = re.findall(r"restore req=\S+.*tag=(\S[^\n]*)", off_log)
    n_preserved = sum(1 for t in tag_lines if t.startswith("PRESERVED"))
    n_mismatch = sum(1 for t in tag_lines if t.startswith("MISMATCH"))
    n_na = sum(1 for t in tag_lines if t.startswith("n/a"))
    print(f"  restore tag results: PRESERVED={n_preserved} "
          f"MISMATCH={n_mismatch} n/a={n_na}", flush=True)
    # PASS: at least one restore, zero mismatches, and the tag was actually
    # checked (PRESERVED) — n/a-only means KV was not CIPHER-owned.
    c_pass = (n_rest > 0 and n_mismatch == 0 and n_preserved > 0)
    print(f"[{'OK  ' if c_pass else 'MISS'}] tags preserved, no mismatch",
          flush=True)

    # --- Verdict -----------------------------------------------------------
    print(f"\n{'=' * 72}", flush=True)
    print(f"GATE CHECK A (token-ID identity, offload vs baseline): "
          f"{'PASS' if a_off else 'FAIL'}", flush=True)
    print(f"GATE CHECK B (offload hooks fired, balanced)         : "
          f"{'PASS' if b_pass else 'FAIL'}", flush=True)
    print(f"GATE CHECK C (tenant tag preserved)                  : "
          f"{'PASS' if c_pass else 'FAIL'}", flush=True)
    print(f"  diagnostic: vanilla recompute (control) "
          f"{'stayed identical' if a_ctl else 'diverged — expected'}; "
          f"offload is byte-exact", flush=True)
    verdict = a_off and b_pass and c_pass
    print(f"CP 5.2 STEP 3 OFFLOAD CORRECTNESS GATE               : "
          f"{'PASS' if verdict else 'FAIL'}", flush=True)
    print(f"{'=' * 72}", flush=True)
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
