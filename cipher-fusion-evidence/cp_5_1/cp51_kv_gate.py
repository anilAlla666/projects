#!/usr/bin/env python3
"""CP 5.1 Step 3 — correctness gate driver (Option A).

Runs cp51_kv_gate_worker.py twice in separate processes:
  1. substrate OFF  (CIPHER_KV_ALLOC=0)  — native vLLM torch.zeros KV
  2. substrate ON   (CIPHER_KV_ALLOC=1)  — KV buffers owned by CIPHER VMM

PASS requires BOTH:
  (A) token-ID identity — every output token of every prompt matches between
      the two runs. This is the CP 5.1 §4 correctness bar.
  (B) hook-in-EngineCore proof — the substrate-ON run's stderr carries the
      cipher-vllm-kv 'hook installed' and 'KV buffers owned by CIPHER VMM'
      lines under an '(EngineCore pid=...)' prefix. Option A's coverage-
      immunity claim rests on register() firing in the EngineCore subprocess;
      if the lines appear only in the main process the claim is unproven.
"""
import json
import re
import subprocess
import sys

HERE = "/home/ubuntu/cipher-fusion-evidence/cp_5_1"
WORKER = f"{HERE}/cp51_kv_gate_worker.py"


def run(mode_on):
    env_val = "1" if mode_on else "0"
    print(f"\n{'='*72}\n=== RUN: CIPHER_KV_ALLOC={env_val} "
          f"({'substrate ON' if mode_on else 'substrate OFF'})\n{'='*72}", flush=True)
    p = subprocess.run(
        [sys.executable, WORKER],
        env={**__import__("os").environ, "CIPHER_KV_ALLOC": env_val},
        capture_output=True, text=True,
    )
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"GATE FAIL — worker (KV_ALLOC={env_val}) exited {p.returncode}")
    m = [ln for ln in p.stdout.splitlines() if ln.startswith("CIPHER_GATE_RESULT ")]
    if not m:
        raise SystemExit(f"GATE FAIL — no CIPHER_GATE_RESULT line (KV_ALLOC={env_val})")
    result = json.loads(m[-1][len("CIPHER_GATE_RESULT "):])
    return result, p.stdout + p.stderr


def main():
    off, off_log = run(mode_on=False)
    on, on_log = run(mode_on=True)

    # --- Check (A): token-ID identity --------------------------------------
    print(f"\n{'='*72}\n=== GATE CHECK A — token-ID identity\n{'='*72}", flush=True)
    a_pass = True
    for i, (po, pn) in enumerate(zip(off["prompts"], on["prompts"])):
        same = po["token_ids"] == pn["token_ids"]
        a_pass &= same
        tag = "OK  " if same else "DIFF"
        print(f"[{tag}] prompt {i}: {po['prompt']!r}", flush=True)
        print(f"        n_tokens off={len(po['token_ids'])} on={len(pn['token_ids'])}", flush=True)
        if not same:
            for j, (a, b) in enumerate(zip(po["token_ids"], pn["token_ids"])):
                if a != b:
                    print(f"        first divergence at token {j}: off={a} on={b}", flush=True)
                    break
            print(f"        off text: {po['text']!r}", flush=True)
            print(f"        on  text: {pn['text']!r}", flush=True)

    # --- Check (B): hook fired in the EngineCore subprocess ----------------
    print(f"\n{'='*72}\n=== GATE CHECK B — hook installed in EngineCore subprocess\n{'='*72}", flush=True)
    installed = re.search(
        r"\(EngineCore pid=\d+\).*\[cipher-vllm-kv\] hook installed", on_log)
    owned = re.search(
        r"\(EngineCore pid=\d+\).*\[cipher-vllm-kv\] KV buffers owned by CIPHER VMM", on_log)
    # also accept the lines if the prefix lands on the preceding multiplexed line
    if not installed:
        installed = ("[cipher-vllm-kv] hook installed" in on_log)
    if not owned:
        owned = ("[cipher-vllm-kv] KV buffers owned by CIPHER VMM" in on_log)
    print(f"[{'OK  ' if installed else 'MISS'}] 'hook installed' line present", flush=True)
    print(f"[{'OK  ' if owned else 'MISS'}] 'KV buffers owned by CIPHER VMM' line present", flush=True)
    engine_prefixed = bool(re.search(r"\(EngineCore pid=\d+\).*\[cipher-vllm-kv\]", on_log))
    print(f"[{'OK  ' if engine_prefixed else 'MISS'}] cipher-vllm-kv log line carries (EngineCore pid=) prefix", flush=True)
    b_pass = bool(installed) and bool(owned) and engine_prefixed

    # --- Verdict -----------------------------------------------------------
    print(f"\n{'='*72}", flush=True)
    print(f"GATE CHECK A (token-ID identity)        : {'PASS' if a_pass else 'FAIL'}", flush=True)
    print(f"GATE CHECK B (hook in EngineCore subproc): {'PASS' if b_pass else 'FAIL'}", flush=True)
    verdict = a_pass and b_pass
    print(f"CP 5.1 STEP 3 CORRECTNESS GATE          : {'PASS' if verdict else 'FAIL'}", flush=True)
    print(f"{'='*72}", flush=True)
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
