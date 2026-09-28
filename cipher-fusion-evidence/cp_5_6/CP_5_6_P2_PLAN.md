# CP 5.6 Priority 2 — Teacher-Forced Correctness Gate + TPW Re-measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> All GPU work is sequential — the H100 is a shared resource; never run two
> measurement processes concurrently (contention corrupts tok/s and watts).

**Goal:** Re-measure the anchored 3.617× tok/W headline (CP 2.4 composed gate)
on the F1-fixed substrate `a7ac8e97`, behind a teacher-forced top-1
correctness gate, so the number is certified as efficiency on *verified*
output rather than unverified — probably degenerate — decode timing.

**Architecture:** Two decoupled measurements per (prompt, arm). (1) A
**free-running** greedy decode — production mode, power-windowed — yields
tok/s, watts, tok/W, and the decoded text. (2) A **teacher-forced** check —
an incremental KV-cache decode whose input at every step is the *gold* token,
never the model's own output — yields per-step argmax-vs-gold agreement over
128 positions. Teacher-forcing is cascade-free and uses the identical forward
path as production, so it is FP-tie-noise-free; it catches F1-style garbage at
~0% and accepts honest INT4 quant drift at ~95-100%. A pair passes at ≥99%
agreement; only passing pairs contribute their free-running TPW.

**Tech Stack:** Python 3.10, transformers 5.8.1, torch 2.11.0+cu130, HF
`AutoModelForCausalLM` (Mistral-7B-v0.1 FP16), CIPHER substrate
`libcipher_rt.so` md5 `a7ac8e97` (CP 5.6 P1), bash orchestration, `nvidia-smi`
power sampling, pytest.

**Anchors (read-only — DO NOT rebuild or rotate):**
- substrate: `libcipher_rt.so` `a7ac8e979c9c20c914e87d7c8509665a` (a7ac8e97)
- preserved fallback: `libcipher_rt.so.pre_cp56` `dc804eb3…` (dc804eb3)
- kmod `e2f50452`, libcipher_v2 `86618c30`, cipher_kv_bridge `8d6ffe3f`

---

## Amendments from advisor review (applied before execution)

The code actually built supersedes the code blocks below where they differ:

- **A1 — power: arm-level mean, no per-prompt epoch slicing.** 1 Hz
  `nvidia-smi` sampling gives only ~2-3 samples per ~2.4 s prompt window —
  too noisy for per-prompt power. Instead, per CP 2.4 `analyze_composed.py`:
  one **arm-level** mean power over the whole free-run window;
  per-prompt tok/W = per-prompt tok/s ÷ arm-level mean power. ~12-24 samples
  per arm is ample for a stable mean (VOLT-locked clock makes power near
  constant). `mean_power_window(rows,t0,t1)` → `mean_power(rows)`.
- **A2 — pre-flight cache check (Task 1 Step 2.5).** Before any 30-min GPU
  run, verify `teacher_forced()` reproduces `m.generate()` *exactly* on a
  `max_new=8` toy case. Catches a `cache_position`/`attention_mask` mismatch
  on a tiny case instead of after the full run.
- **A3 — explicit `attention_mask` in every TF forward.** `m.generate()`
  passes one internally; matching it removes a class of "vanilla TF = 99.7%
  not 100.0%" non-bugs.
- **A4 — verdict detects looping.** A passing-TF-but-looping all-on arm
  (`accept_rate==1.0` or `distinct_ratio<0.5`) does **not** return verdict A;
  it returns B with an explicit loop rationale — TF certifies substrate
  numerics, but a looping free-run means the tok/W numerator counts loop
  tokens (non-substrate cause: spec-decode / base-model).

---

## Background — why this plan exists

`F1_TPW_RECORDS_REVIEW.md` ruled the 3.617× headline **SUSPECT**: the CP 2.4
composed gate engaged the F1-degenerate Marlin full-GPU path (green ctx
present, primary-pinned GEMM), the all-on arm showed the `accept_rate=1.000`
loop signature, and the harness recorded **only timing** — zero output
correctness. CP 5.6 P1 (`f1_p1_verify.runlog`) shipped the cross-context
ordering fix (`a7ac8e97`): the full-GPU path now records a completion event
inside `PrimaryCtxGuard` and issues `cuStreamWaitEvent` on the caller stream.
P1's micro-repro went 99/100 degenerate → 0/100. **P2 is the end-to-end
re-measurement that closes CP 5.6.**

The naive gate — substrate free-run vs a single free-running gold sequence at
99% identity — is mechanically broken: FP-tie non-determinism + the
autoregressive cascade means a perfectly correct substrate can still diverge
and "fail". Teacher-forcing removes both confounds.

## Reference files (read before coding — do not modify)

- `/home/ubuntu/cipher_rt_phase4/spec_varied_driver.py` — CP 2.4 driver; the
  5-prompt set and arm/env convention are copied verbatim from it.
- `/home/ubuntu/cipher-fusion-evidence/cp_2_4/run_composed.sh` — CP 2.4
  orchestrator; arm env toggles + sentinel/watts pattern reused.
- `/home/ubuntu/cipher-fusion-evidence/cp_2_4/analyze_composed.py` — CP 2.4
  aggregation; CI/lift math reused.
- `/home/ubuntu/cipher-fusion-evidence/cp_2_4/composed_result.json` — the
  number under re-test: all-on/vanilla tok/W 3.6166× [3.591, 3.642].

## File Structure

| File | Responsibility |
|---|---|
| `/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py` | Per-process driver. `MODE=gold` captures the clean-FP16 gold sequences; `MODE=gate` runs one arm: free-run measurement + teacher-forced check. Lives beside `spec_varied_driver.py` so `cipher_spec_decode` / `tenant_register` imports resolve. |
| `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/run_cp56_p2.sh` | Orchestrator. Pass 1 = gold (no substrate). Pass 2 = 3 arms (vanilla/marlin/allon), each power-windowed via DECODE_START/END sentinel + continuous `nvidia-smi` sampler. |
| `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/analyze_cp56_p2.py` | Aggregation + report. Per (prompt,arm): tok/s, per-prompt mean power, tok/W, TF agreement, pass/fail @≥99%. Per arm: passing count, TPW mean+CI95 on passing pairs. Overall all-on/vanilla and marlin/vanilla TPW lift. Emits `cp56_p2_result.json` + verdict A/B/C. |
| `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/test_analyze_cp56_p2.py` | pytest for the pure aggregation logic — CI math, per-prompt power slicing by epoch window, ≥99% threshold. Runs with no GPU. |
| `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/` (dir) | All evidence: `gold.json`, `<arm>.json`, `<arm>.watts.csv`, `<arm>.log`, run logs, `cp56_p2_result.json`. |
| `/home/ubuntu/cipher-fusion-evidence/cp_5_6/CP_5_6_P2_REPORT.md` | Final report — verdict, per-arm table, A/B/C outcome, anchors. |

## Methodology decisions locked for this plan

1. **Substrate = `a7ac8e97`** (current `libcipher_rt.so`, CP 5.6 P1 fix). P2
   asks "does the P1 fix make the headline honest", so it measures the fixed
   build. `dc804eb3` is **not** in scope (out-of-scope per the advisor's
   "anchors as recorded").
2. **Teacher-forced = incremental KV-cache loop fed gold tokens**, *not* a
   single full no-cache forward. Only the incremental path is numerically
   identical to free-running greedy decode; that identity is what makes the
   vanilla sanity check land at exactly ~100% and what justifies the 99%
   threshold being "FP-tie-noise-free". A single full forward would inject
   cache-vs-no-cache FP drift and muddy the gate.
3. **Pair = one (prompt, arm).** 5 prompts × 3 arms = 15 pairs. One free-run
   per pair; CI is over the 5 prompts within an arm (matches the advisor's
   "pairs passing per arm" / "CI on passing pairs"). This differs from
   CP 2.4's 5 *repeated* pairs — the advisor's design re-bases the CI unit
   onto per-prompt, which is what the teacher-forced gate operates on.
4. **Power window excludes the teacher-forced section.** TPW is a
   free-running production measurement; the sentinel brackets only the
   free-run loop. Teacher-forcing runs after DECODE_END, unmeasured.
5. **Threshold ≥99%** per-step agreement over the 128 gold positions. Failing
   pairs are flagged with agreement rate + decoded text, never silently
   dropped. Decoded text + a `distinct_ratio` loop metric + `accept_rate` are
   recorded for **every** pair (passing included) so a "passing-but-looping"
   case is visible to human adjudication.

---

## Task 1: Build the teacher-forced + free-run driver

**Files:**
- Create: `/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py`

- [ ] **Step 1: Write the driver**

Create `/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py` with exactly:

```python
"""CP 5.6 Priority 2 — teacher-forced correctness gate + free-running TPW
re-measurement driver (finding F1 re-measurement).

MODE=gold : clean FP16, no substrate. Free-running greedy decode; dump the
            128-token gold sequence (token IDs + text) per prompt to GOLD_JSON.
MODE=gate : selected arm (env, same toggles as CP 2.4 run_composed.sh):
   (1) free-running greedy decode, timed + power-windowed via DECODE_START/END
       sentinel  -> production TPW measurement (tok/s, ids, text, accept_rate);
   (2) teacher-forced check -> incremental KV-cache decode FED THE GOLD tokens
       (cascade-free; identical forward path to production, so FP-tie-noise-
       free). Per-step argmax-vs-gold agreement over the 128 gold positions.
   Writes per-prompt JSON: free-run metrics + tf agreement -> OUT_JSON.

Arm selection (set by run_cp56_p2.sh): vanilla = no substrate; marlin/allon =
LD_PRELOAD libcipher_rt + CIPHER_* toggles. The driver itself is arm-agnostic.
"""
import os
import sys
import json
import time

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register

MODE = os.environ.get("MODE", "gate")              # gold | gate
MODEL = os.environ["WL_MODEL"]
TENANT = os.environ.get("WL_TENANT_ID", "cp56p2")
ARM = os.environ.get("ARM", "vanilla")
OUT_JSON = os.environ["OUT_JSON"]
GOLD_JSON = os.environ["GOLD_JSON"]
MAX_NEW = int(os.environ.get("WL_MAX_NEW", "128"))

# Verbatim from spec_varied_driver.py — the CP 2.4 varied 5-prompt set, open-
# ended so a fixed 128-token budget does not hit EOS early.
PROMPTS = [
    "The Pacific Ocean is the largest ocean on Earth, covering approximately",
    "Sarah opened the old letter with trembling hands. The handwriting was "
    "her grandmother's, and the date read",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot =",
    "If a train leaves Chicago at 3 PM traveling east at 60 mph and another "
    "leaves New York at 4 PM traveling west at 80 mph,",
    "User: What are the main differences between supervised and unsupervised "
    "learning?\nAssistant: The main differences are",
]

tenant = tenant_register.register(TENANT)
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda().eval()

# Operator-policy injection — install() no-ops when CIPHER_SPEC=0, so the same
# driver measures the vanilla / marlin baselines and the spec-on all-on arm.
import cipher_spec_decode
cipher_spec_decode.install()


def free_run(prompt):
    """One timed greedy 128-token generation — the production decode path."""
    enc = tok(prompt, return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    torch.cuda.synchronize()
    t0 = time.time()
    p0 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - p0
    t1 = time.time()
    seq = out[0].tolist()
    gen = seq[plen:]
    st = dict(cipher_spec_decode._last_stats)      # {} on non-spec arms
    prop = st.get("proposed")
    ar = (st.get("accepted") / prop) if prop else None
    distinct = (len(set(gen)) / len(gen)) if gen else 0.0
    return {"plen": plen, "full_ids": seq, "gen_ids": gen,
            "gen_tokens": len(gen), "wall_s": round(dt, 4),
            "tok_s": round(len(gen) / max(dt, 1e-6), 4),
            "t0_epoch": round(t0, 3), "t1_epoch": round(t1, 3),
            "accept_rate": ar, "distinct_ratio": round(distinct, 4),
            "text": tok.decode(gen)}


def teacher_forced(gold_full_ids, plen):
    """Incremental KV-cache decode FED THE GOLD prefix at every step.

    Step i consumes gold[plen+i-1] (never the model's own argmax) -> the
    chain cannot cascade. The KV-cache incremental path is numerically the
    SAME forward as free-running greedy decode, so for the vanilla arm this
    reproduces the gold run exactly (agreement == 100%); substrate arms
    diverge only by genuine substrate numerics. Returns per-step argmax-vs-
    gold agreement over the gold generation positions.
    """
    gold = torch.tensor([gold_full_ids], device="cuda")
    n_gen = gold.shape[-1] - plen
    past = None
    agree = 0
    mism = []
    with torch.no_grad():
        # prompt forward
        cur = gold[:, :plen]
        cache_pos = torch.arange(plen, device="cuda")
        out = m(input_ids=cur, past_key_values=past, use_cache=True,
                cache_position=cache_pos)
        past = out.past_key_values
        pred = int(out.logits[0, -1].argmax())
        for i in range(n_gen):
            g = int(gold[0, plen + i])
            if pred == g:
                agree += 1
            else:
                mism.append({"pos": i, "pred": pred, "gold": g})
            if i == n_gen - 1:
                break
            # teacher-force: next input is the GOLD token, not `pred`
            cur = gold[:, plen + i:plen + i + 1]
            cache_pos = torch.tensor([plen + i], device="cuda")
            out = m(input_ids=cur, past_key_values=past, use_cache=True,
                    cache_position=cache_pos)
            past = out.past_key_values
            pred = int(out.logits[0, -1].argmax())
    return {"n_steps": n_gen, "agree": agree,
            "agreement": round(agree / n_gen, 6) if n_gen else 0.0,
            "mismatches": mism[:25]}


# Warm — run the full prompt set once at the timed budget, untimed. Pays every
# one-time cost up front (Marlin lazy quant + NVRTC compile, cuDNN/Marlin JIT
# across the decode kv_len range, spec-arm draft load) so the timed free-run
# and the teacher-forced loop both observe a fully warm, Marlin-active substrate.
for _p in PROMPTS:
    _we = tok(_p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        m.generate(**_we, max_new_tokens=MAX_NEW, do_sample=False,
                   pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

if MODE == "gold":
    gold = {"model": MODEL, "max_new": MAX_NEW, "n_prompts": len(PROMPTS),
            "prompts": []}
    for i, p in enumerate(PROMPTS):
        fr = free_run(p)
        gold["prompts"].append(
            {"prompt": i, "plen": fr["plen"], "full_ids": fr["full_ids"],
             "gen_ids": fr["gen_ids"], "gen_tokens": fr["gen_tokens"],
             "distinct_ratio": fr["distinct_ratio"], "text": fr["text"]})
        print("GOLD prompt=%d gen_tokens=%d distinct_ratio=%.3f"
              % (i, fr["gen_tokens"], fr["distinct_ratio"]), file=sys.stderr)
    with open(GOLD_JSON, "w") as f:
        json.dump(gold, f, indent=2)
    print("wrote " + GOLD_JSON, file=sys.stderr)
    sys.exit(0)

# MODE == gate
with open(GOLD_JSON) as f:
    gold = json.load(f)
gold_by_prompt = {g["prompt"]: g for g in gold["prompts"]}

# Power-sentinel window — run_cp56_p2.sh keys its nvidia-smi sampler off
# DECODE_START / DECODE_END so watts.csv covers exactly the free-run loop.
# The teacher-forced loop runs AFTER DECODE_END and is intentionally not
# power-measured: TPW is a free-running production figure.
SENTINEL = "/tmp/cipher_cp56p2_%s.decode_window" % TENANT
with open(SENTINEL, "w") as _s:
    _s.write("DECODE_START %.3f\n" % time.time())

free = []
for i, p in enumerate(PROMPTS):
    fr = free_run(p)
    free.append(fr)
    print("RESULT prompt=%d tok_s=%.3f gen_tokens=%d wall=%.3fs "
          "accept_rate=%s distinct=%.3f"
          % (i, fr["tok_s"], fr["gen_tokens"], fr["wall_s"],
             ("%.3f" % fr["accept_rate"]) if fr["accept_rate"] is not None
             else "n/a", fr["distinct_ratio"]), file=sys.stderr)

with open(SENTINEL, "a") as _s:
    _s.write("DECODE_END %.3f\n" % time.time())

results = []
for i, p in enumerate(PROMPTS):
    g = gold_by_prompt[i]
    tf = teacher_forced(g["full_ids"], g["plen"])
    fr = free[i]
    row = {"prompt": i, "plen": fr["plen"],
           "tok_s": fr["tok_s"], "wall_s": fr["wall_s"],
           "gen_tokens": fr["gen_tokens"],
           "t0_epoch": fr["t0_epoch"], "t1_epoch": fr["t1_epoch"],
           "accept_rate": fr["accept_rate"],
           "distinct_ratio": fr["distinct_ratio"],
           "free_run_text": fr["text"], "free_run_gen_ids": fr["gen_ids"],
           "tf_agreement": tf["agreement"], "tf_agree": tf["agree"],
           "tf_n_steps": tf["n_steps"], "tf_mismatches": tf["mismatches"]}
    results.append(row)
    print("TFGATE prompt=%d tf_agreement=%.4f (%d/%d)"
          % (i, tf["agreement"], tf["agree"], tf["n_steps"]), file=sys.stderr)

out = {"cp": "5.6", "priority": "P2", "arm": ARM, "model": MODEL,
       "substrate_md5": os.environ.get("CIPHER_SUBSTRATE_MD5", "n/a"),
       "tenant": TENANT, "max_new": MAX_NEW, "n_prompts": len(PROMPTS),
       "prompts": results}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("wrote " + OUT_JSON, file=sys.stderr)
```

- [ ] **Step 2: Syntax-check the driver**

Run: `python3 -m py_compile /home/ubuntu/cipher_rt_phase4/tf_gate_driver.py && echo COMPILE_OK`
Expected: `COMPILE_OK`

- [ ] **Step 3: Snapshot checkpoint (the campaign tree is not a git repo)**

Run:
```bash
mkdir -p /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
cp /home/ubuntu/cipher_rt_phase4/tf_gate_driver.py \
   /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/tf_gate_driver.py.snapshot
```
Expected: snapshot copy exists (evidence preserved by snapshot + final tarball).

---

## Task 2: Build the orchestrator

**Files:**
- Create: `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/run_cp56_p2.sh`

- [ ] **Step 1: Write the orchestrator**

Create `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/run_cp56_p2.sh` with:

```bash
#!/usr/bin/env bash
# CP 5.6 P2 — teacher-forced correctness gate + TPW re-measurement.
# Pass 1: gold  — clean FP16, NO substrate, MODE=gold -> gold.json
# Pass 2: gate  — arms vanilla/marlin/allon, MODE=gate, each free-run power-
#                 windowed (DECODE_START/END sentinel + nvidia-smi sampler),
#                 then teacher-forced vs gold.
# Arm env toggles are identical to cp_2_4/run_composed.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py
TARGET=/home/ubuntu/models/Mistral-7B-v0.1
GOLD="${HERE}/gold.json"
VOLT_MHZ="${VOLT_MHZ:-1000}"
SUBSTRATE_MD5="$(md5sum "$BIN" | cut -c1-8)"
mkdir -p "$HERE"

echo "=== CP 5.6 P2 — TF gate + TPW re-measurement ==="
echo "substrate libcipher_rt.so md5=${SUBSTRATE_MD5}  (expect a7ac8e97)"
echo "target=$TARGET  VOLT_MHZ=$VOLT_MHZ"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset to nominal"
echo

# ---- Pass 1: gold (clean FP16, no substrate) ----
echo "--- pass 1: gold capture (clean FP16, no CIPHER) ---"
env MODE=gold ARM=gold WL_TENANT_ID=cp56p2_gold WL_MODEL="$TARGET" \
    OUT_JSON=/dev/null GOLD_JSON="$GOLD" CIPHER_SPEC=0 \
    python3 "$DRIVER" > "${HERE}/gold.log" 2>&1
if [ ! -s "$GOLD" ]; then
  echo "  !! gold capture failed — see ${HERE}/gold.log"; exit 1
fi
grep -h "^GOLD" "${HERE}/gold.log" | sed 's/^/  /'
echo

# ---- run_one <arm> ----
run_one() {
  local arm="$1"
  local tid="cp56p2_${arm}"
  local sentinel="/tmp/cipher_cp56p2_${tid}.decode_window"
  local json="${HERE}/${arm}.json"
  local log="${HERE}/${arm}.log"
  local csv="${HERE}/${arm}.watts.csv"
  rm -f "$sentinel" "$json"
  local common=( MODE=gate ARM="$arm" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
                 OUT_JSON="$json" GOLD_JSON="$GOLD" \
                 CIPHER_SUBSTRATE_MD5="$SUBSTRATE_MD5" )
  case "$arm" in
    vanilla) env "${common[@]}" CIPHER_SPEC=0 \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
    marlin)  env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
    allon)   env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=on \
               CIPHER_SPEC_DRAFT=ngram CIPHER_VOLT=on CIPHER_VOLT_MHZ="$VOLT_MHZ" \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
  esac
  local wl=$!
  local waited=0
  while ! grep -q DECODE_START "$sentinel" 2>/dev/null; do
    if ! kill -0 "$wl" 2>/dev/null; then
      wait "$wl" 2>/dev/null
      echo "  !! $arm exited before DECODE_START — see $log"; return 1
    fi
    (( waited >= 3000 )) && { kill "$wl" 2>/dev/null; echo "  !! $arm timeout"; return 1; }
    sleep 0.2; waited=$((waited+1))
  done
  # continuous power sampler — ABSOLUTE epoch ts so analyze can slice per-prompt
  ( echo "epoch_s,power_w,clock_sm_mhz"
    while ! grep -q DECODE_END "$sentinel" 2>/dev/null; do
      smp=$(nvidia-smi --query-gpu=power.draw,clocks.sm \
            --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
      echo "$(date +%s.%N),${smp:-0,0}"
      sleep 0.5
    done ) > "$csv" &
  local sampler=$!
  wait "$wl"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
  grep -hE "^(RESULT|TFGATE)" "$log" 2>/dev/null | sed 's/^/  /'
}

for arm in vanilla marlin allon; do
  echo "--- arm: $arm ---"
  run_one "$arm" || echo "  (arm $arm did not complete cleanly)"
  echo
done

echo "=== end pod state ==="
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i 0
echo "analyze: python3 ${HERE}/analyze_cp56_p2.py"
```

- [ ] **Step 2: Make executable and shell-lint**

Run: `chmod +x /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/run_cp56_p2.sh && bash -n /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/run_cp56_p2.sh && echo BASH_OK`
Expected: `BASH_OK`

---

## Task 3: Build the analyzer + its self-test

**Files:**
- Create: `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/analyze_cp56_p2.py`
- Test: `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/test_analyze_cp56_p2.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/test_analyze_cp56_p2.py`:

```python
"""Self-test for analyze_cp56_p2 pure logic — no GPU, no model.

analyze_cp56_p2.py guards its report driver under `if __name__ == "__main__"`,
so a plain import runs no analysis — only the pure helpers are exercised here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_cp56_p2 as an


def test_ci95_known():
    m, lo, hi = an.ci95([2.0, 2.0, 2.0, 2.0, 2.0])
    assert abs(m - 2.0) < 1e-9 and abs(hi - lo) < 1e-9


def test_ci95_spread():
    m, lo, hi = an.ci95([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-9 and lo < m < hi


def test_mean_power_window_slices_by_epoch():
    rows = [(100.0, 200.0), (101.0, 210.0), (102.0, 999.0)]
    # window [100.0, 101.5] keeps the first two samples only
    assert abs(an.mean_power_window(rows, 100.0, 101.5) - 205.0) < 1e-9


def test_mean_power_window_empty_returns_none():
    assert an.mean_power_window([(100.0, 200.0)], 500.0, 600.0) is None


def test_pass_threshold():
    assert an.passes(0.99) is True
    assert an.passes(0.991) is True
    assert an.passes(0.989) is False
    assert an.passes(0.0) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2 && python3 -m pytest test_analyze_cp56_p2.py -q`
Expected: FAIL — `analyze_cp56_p2.py` does not exist yet (collection error /
`ModuleNotFoundError`).

- [ ] **Step 3: Write the analyzer**

Create `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/analyze_cp56_p2.py`:

```python
"""CP 5.6 P2 — teacher-forced gate + TPW re-measurement analysis.

Reads gold.json + {vanilla,marlin,allon}.json + {arm}.watts.csv. Per
(prompt, arm): tok/s (free-run), per-prompt mean power (watts.csv rows whose
epoch ts falls in the prompt's [t0_epoch, t1_epoch] free-run window), tok/W,
and teacher-forced agreement. A pair PASSES at agreement >= 0.99. Per arm:
passing-pair count, and TPW = mean tok/W over PASSING pairs only, with 95% CI.
Overall lift: all-on/vanilla and marlin/vanilla TPW, matched per prompt on
prompts where both arms passed. Emits cp56_p2_result.json and an A/B/C verdict.
"""
import os
import csv
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["vanilla", "marlin", "allon"]
THRESHOLD = 0.99
HEADLINE = 3.6166                       # CP 2.4 composed all-on/vanilla tok/W
HEADLINE_CI = (3.5908, 3.6423)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    mu, s = mean(xs), std(xs)
    se = s / math.sqrt(len(xs)) if xs else 0.0
    return mu, mu - 1.96 * se, mu + 1.96 * se


def passes(agreement):
    return agreement >= THRESHOLD


def mean_power_window(rows, t0, t1):
    """rows: list of (epoch_s, power_w). Mean power over [t0, t1] inclusive."""
    pw = [p for (ts, p) in rows if t0 <= ts <= t1]
    return mean(pw) if pw else None


def load_watts(arm):
    path = os.path.join(HERE, "%s.watts.csv" % arm)
    rows = []
    try:
        for r in csv.DictReader(open(path)):
            try:
                rows.append((float(r["epoch_s"]), float(r["power_w"])))
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass
    return rows


def load_arm(arm):
    """-> dict prompt -> per-pair metrics, or None if the arm json is missing."""
    path = os.path.join(HERE, "%s.json" % arm)
    try:
        d = json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    watts = load_watts(arm)
    out = {}
    for r in d["prompts"]:
        mp = mean_power_window(watts, r["t0_epoch"], r["t1_epoch"])
        tw = (r["tok_s"] / mp) if mp else None
        out[r["prompt"]] = {
            "tok_s": r["tok_s"], "mean_power_w": mp, "tok_w": tw,
            "tf_agreement": r["tf_agreement"], "pass": passes(r["tf_agreement"]),
            "accept_rate": r.get("accept_rate"),
            "distinct_ratio": r.get("distinct_ratio"),
            "free_run_text": r.get("free_run_text", ""),
            "tf_mismatches": r.get("tf_mismatches", [])}
    return out


def main():
    res = {"cp": "5.6", "priority": "P2", "gate": "teacher-forced top-1",
           "threshold": THRESHOLD, "headline_under_test": HEADLINE,
           "substrate": "a7ac8e97", "arms": {}, "sanity": {}, "lift": {}}
    arms = {a: load_arm(a) for a in ARMS}

    print("prompt arm      tok/s    W      tok/W   tf_agree  pass")
    print("-" * 58)
    for p in range(5):
        for a in ARMS:
            d = arms.get(a)
            if not d or p not in d:
                print("%6d %-8s MISSING" % (p, a)); continue
            r = d[p]
            print("%6d %-8s %7.2f %6.1f  %s   %.4f   %s"
                  % (p, a, r["tok_s"],
                     r["mean_power_w"] if r["mean_power_w"] else float("nan"),
                     ("%.4f" % r["tok_w"]) if r["tok_w"] else " n/a  ",
                     r["tf_agreement"], "PASS" if r["pass"] else "FAIL"))
    print("-" * 58)

    # sanity — vanilla teacher-forced must be ~100%
    van = arms.get("vanilla") or {}
    van_agree = [van[p]["tf_agreement"] for p in van]
    sane = bool(van_agree) and min(van_agree) >= 0.99
    res["sanity"] = {"vanilla_tf_agreements": van_agree,
                     "vanilla_tf_min": min(van_agree) if van_agree else None,
                     "sane": sane}
    print("SANITY  vanilla TF min=%s  -> %s"
          % (("%.4f" % min(van_agree)) if van_agree else "n/a",
             "OK" if sane else "FAIL (gold capture / harness bug)"))

    # per-arm TPW on passing pairs
    for a in ARMS:
        d = arms.get(a) or {}
        passing = [p for p in d if d[p]["pass"] and d[p]["tok_w"]]
        tws = [d[p]["tok_w"] for p in passing]
        mu, lo, hi = ci95(tws) if len(tws) >= 2 else (mean(tws), None, None)
        res["arms"][a] = {
            "n_prompts": len(d), "n_passing": len(passing),
            "passing_prompts": sorted(passing),
            "tpw_mean": round(mu, 4) if tws else None,
            "tpw_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "failing": {str(p): {"tf_agreement": d[p]["tf_agreement"],
                                 "accept_rate": d[p]["accept_rate"],
                                 "distinct_ratio": d[p]["distinct_ratio"],
                                 "text_head": d[p]["free_run_text"][:240]}
                        for p in d if not d[p]["pass"]}}
        print("ARM %-8s passing=%d/%d  TPW=%s"
              % (a, len(passing), len(d),
                 ("%.4f %s" % (mu, res["arms"][a]["tpw_ci95"]))
                 if tws else "n/a"))

    # matched-prompt lift on prompts where BOTH arms passed
    for tag, num, den in [("allon/vanilla", "allon", "vanilla"),
                          ("marlin/vanilla", "marlin", "vanilla")]:
        dn, dd = arms.get(num) or {}, arms.get(den) or {}
        ratios = []
        for p in range(5):
            if (p in dn and p in dd and dn[p]["pass"] and dd[p]["pass"]
                    and dn[p]["tok_w"] and dd[p]["tok_w"]):
                ratios.append(dn[p]["tok_w"] / dd[p]["tok_w"])
        if ratios:
            mu, lo, hi = ci95(ratios) if len(ratios) >= 2 else (mean(ratios), None, None)
            res["lift"][tag] = {
                "n": len(ratios), "mean": round(mu, 4),
                "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None}
            print("LIFT %-15s %.4fx  n=%d  %s"
                  % (tag, mu, len(ratios),
                     res["lift"][tag]["ci95"] or ""))
        else:
            res["lift"][tag] = {"n": 0, "mean": None, "ci95": None}
            print("LIFT %-15s n/a (no prompt passed both arms)" % tag)

    # verdict A/B/C
    av = res["lift"].get("allon/vanilla", {})
    allon_pass = res["arms"].get("allon", {}).get("n_passing", 0)
    marlin_pass = res["arms"].get("marlin", {}).get("n_passing", 0)
    if not sane:
        verdict = ("INVALID", "vanilla teacher-forced sanity failed — "
                   "gold capture or harness bug; fix before interpreting")
    elif allon_pass == 0 or marlin_pass == 0:
        verdict = ("C", "a substrate arm passed zero pairs at >=99% TF — the "
                   "substrate still produces unverified/degenerate decode; "
                   "3.617x cannot be re-asserted; F1 not closed by P2")
    elif av.get("ci95") and av["ci95"][0] <= HEADLINE <= av["ci95"][1]:
        verdict = ("A", "TF gate passes and re-measured all-on/vanilla TPW CI "
                   "brackets 3.617x — headline re-asserted on verified output")
    else:
        verdict = ("B", "TF gate passes but re-measured TPW lift differs from "
                   "3.617x — report the corrected number; F1 closed, headline "
                   "restated")
    res["verdict"] = {"outcome": verdict[0], "rationale": verdict[1]}
    print("-" * 58)
    print("VERDICT %s — %s" % verdict)

    outp = os.path.join(HERE, "cp56_p2_result.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print("result -> " + outp)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the self-test to verify it passes**

Run: `cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2 && python3 -m pytest test_analyze_cp56_p2.py -q`
Expected: PASS — 5 passed.

---

## Task 4: Gold capture

**Files:** none created — produces `/home/ubuntu/cipher-fusion-evidence/cp_5_6/p2/gold.json`

- [ ] **Step 1: Capture gold (clean FP16, no substrate)**

Run:
```bash
mkdir -p /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
env MODE=gold ARM=gold WL_TENANT_ID=cp56p2_gold \
    WL_MODEL=/home/ubuntu/models/Mistral-7B-v0.1 \
    OUT_JSON=/dev/null GOLD_JSON=$PWD/gold.json CIPHER_SPEC=0 \
    python3 /home/ubuntu/cipher_rt_phase4/tf_gate_driver.py 2>&1 | tee gold.log
```
Expected: 5 `GOLD prompt=N gen_tokens=128 distinct_ratio=…` lines, `wrote …gold.json`.

- [ ] **Step 2: Verify gold sequences are coherent and non-looping**

Run:
```bash
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
python3 -c "
import json
g = json.load(open('gold.json'))
for p in g['prompts']:
    print('prompt', p['prompt'], 'gen_tokens', p['gen_tokens'],
          'distinct_ratio', round(p['distinct_ratio'],3))
    print('   ', repr(p['text'][:160]))
    assert p['gen_tokens'] >= 1
    assert p['distinct_ratio'] > 0.45, 'gold prompt %d looks looped' % p['prompt']
print('GOLD_OK')
"
```
Expected: `GOLD_OK`, each prompt's text reads as coherent English/code, no
prompt obviously degenerate. **If a gold prompt is itself looping** (distinct
ratio low / visibly repeating), STOP — clean FP16 Mistral should not loop;
investigate the model load before measuring substrate arms.

---

## Task 5: Sanity check — vanilla arm (HARD GATE)

**Files:** produces `vanilla.json`, `vanilla.watts.csv`, `vanilla.log`

This is the test-first gate of the whole measurement: the vanilla arm is
clean FP16 with no substrate, so its teacher-forced agreement against the
gold it would itself have produced must be ~100%. A low score means the gold
capture or the teacher-forced harness is broken — not the substrate.

- [ ] **Step 1: Run the vanilla arm only**

Run:
```bash
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
sentinel=/tmp/cipher_cp56p2_cp56p2_vanilla.decode_window; rm -f "$sentinel"
env MODE=gate ARM=vanilla WL_TENANT_ID=cp56p2_vanilla \
    WL_MODEL=/home/ubuntu/models/Mistral-7B-v0.1 \
    OUT_JSON=$PWD/vanilla.json GOLD_JSON=$PWD/gold.json \
    CIPHER_SPEC=0 CIPHER_SUBSTRATE_MD5=none \
    python3 /home/ubuntu/cipher_rt_phase4/tf_gate_driver.py 2>&1 | tee vanilla.log
```
Expected: 5 `RESULT …` lines then 5 `TFGATE prompt=N tf_agreement=…` lines,
`wrote …vanilla.json`. (No watts.csv here — this standalone run has no
sampler; the watts.csv for vanilla is produced when the full orchestrator
runs, or vanilla can be re-run under the Task 6 per-arm sampler form. tok/W
for vanilla still needs a watts.csv — see Step 4.)

- [ ] **Step 2: Assert the sanity gate**

Run:
```bash
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
python3 -c "
import json
d = json.load(open('vanilla.json'))
ag = [r['tf_agreement'] for r in d['prompts']]
print('vanilla TF agreements:', ag)
mn = min(ag)
print('min =', mn)
assert mn >= 0.99, 'SANITY FAIL: vanilla teacher-forced %.4f < 0.99 — gold/harness bug' % mn
print('SANITY_OK')
"
```
Expected: `SANITY_OK`, agreements at or extremely near 1.0000.

- [ ] **Step 3: STOP gate**

If sanity fails: do **not** proceed to Task 6. Invoke
`superpowers:systematic-debugging`, root-cause the gold capture / teacher-forced
loop (likely suspects: `cache_position` handling, prompt-length offset,
tokenizer pad token), fix `tf_gate_driver.py`, re-run Task 4 + Task 5. Only a
green `SANITY_OK` unlocks Task 6.

- [ ] **Step 4: Decide the run shape for Task 6**

Because vanilla also needs a power trace for its tok/W, the cleanest path is
to run **all three arms via the orchestrator** in Task 6 (it re-runs vanilla
under the sampler). The standalone vanilla run above exists only to gate
sanity early and cheaply. Confirm: Task 6 will use `run_cp56_p2.sh` (gold +
3 arms) so every arm gets a `watts.csv`.

---

## Task 6: Full measurement — all arms under the orchestrator

**Files:** produces `gold.*`, `vanilla.*`, `marlin.*`, `allon.*`,
`cp56_p2_result.json`

- [ ] **Step 1: Run the full orchestrator**

The orchestrator runs gold + all 3 arms, each free-run power-windowed, each
followed by teacher-forcing. Arms run strictly sequentially (uncontended H100).

Run:
```bash
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
bash run_cp56_p2.sh 2>&1 | tee run_cp56_p2.runlog
```
Expected: substrate md5 line shows `a7ac8e97`; pass-1 gold prints 5 `GOLD`
lines; each arm prints 5 `RESULT` + 5 `TFGATE` lines; `gold.json`,
`{vanilla,marlin,allon}.json`, `{vanilla,marlin,allon}.watts.csv` all written
non-empty.

- [ ] **Step 2: Sanity re-confirm on the orchestrator's vanilla run**

Run:
```bash
cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2
python3 -c "
import json
ag = [r['tf_agreement'] for r in json.load(open('vanilla.json'))['prompts']]
print('vanilla TF:', ag)
assert min(ag) >= 0.99, 'SANITY FAIL on orchestrator vanilla run'
print('SANITY_OK')
"
```
Expected: `SANITY_OK`.

- [ ] **Step 3: Analyze**

Run: `cd /home/ubuntu/cipher-fusion-evidence/cp_5_6/p2 && python3 analyze_cp56_p2.py 2>&1 | tee analyze.log`
Expected: the per-prompt table, `SANITY OK`, per-arm TPW lines, `LIFT` lines,
a `VERDICT A|B|C` line, `cp56_p2_result.json` written.

- [ ] **Step 4: Verification before completion**

Invoke `superpowers:verification-before-completion`. Confirm from the actual
files (not assumption): `cp56_p2_result.json` exists; `sanity.sane == true`;
every arm json has 5 prompts; watts.csv files non-empty and epoch-stamped;
the verdict letter matches the numbers. Re-read each arm's `failing` block
and decoded-text heads — a pair that *passes* the TF gate but whose free-run
text loops (low `distinct_ratio` / `accept_rate==1.0`) must be called out
explicitly in the report.

---

## Task 7: Report + evidence tarball

**Files:**
- Create: `/home/ubuntu/cipher-fusion-evidence/cp_5_6/CP_5_6_P2_REPORT.md`
- Create: `/home/ubuntu/cipher-fusion-evidence/cp_5_6_p2_evidence.tar.gz`

- [ ] **Step 1: Write the report**

Create `/home/ubuntu/cipher-fusion-evidence/cp_5_6/CP_5_6_P2_REPORT.md` covering:
1. **Scope** — CP 5.6 P2; re-measure 3.617× on `a7ac8e97` behind a
   teacher-forced gate; the F1 SUSPECT verdict this answers.
2. **Methodology** — teacher-forced incremental KV-cache decode fed gold;
   why cascade-free + FP-tie-noise-free; free-run TPW separate; 99% threshold;
   pair = (prompt, arm); n=5 prompts; substrate md5 (cite actual).
3. **Sanity** — vanilla TF per-prompt agreements (must be ~100%); state OK.
4. **Results table** — per (prompt, arm): tok/s, W, tok/W, TF agreement,
   PASS/FAIL. Per arm: passing count, TPW mean + CI95. all-on/vanilla and
   marlin/vanilla TPW lift + CI.
5. **Output-coherence note** — for every arm, decoded-text inspection;
   `accept_rate` and `distinct_ratio`; explicitly flag any passing-but-looping
   pair, and contrast with the CP 2.4 `accept_rate=1.000` loop signature.
6. **Verdict A / B / C** — quote `cp56_p2_result.json`'s verdict.
   - **A — headline holds:** all substrate arms pass the TF gate and the
     re-measured all-on/vanilla TPW CI brackets 3.617× → re-assert on verified
     output; CP 5.6 closes.
   - **B — headline corrected:** arms pass the TF gate but the re-measured
     lift differs materially from 3.617× → report the corrected number;
     CP 5.6 closes with the headline restated.
   - **C — headline fails:** a substrate arm fails the TF gate → the substrate
     still produces unverified/degenerate decode; 3.617× cannot be
     re-asserted; CP 5.6 does **not** close on P2 — escalate (P3 / further
     fix). **Note:** the A/B/C labels are reconstructed from the F1 review's
     verdict structure; the defining "prior prompt" is not in this session's
     context — confirm the mapping with the adjudicator.
7. **Anchors** — substrate `a7ac8e97` unchanged; no rebuild; kmod/v2/bridge
   unchanged. New artifacts: `tf_gate_driver.py`, `run_cp56_p2.sh`,
   `analyze_cp56_p2.py`, evidence under `cp_5_6/p2/`.
8. **CP 5.6 status** — closed (A/B) or open with next step (C); awaiting
   adjudication.

- [ ] **Step 2: Build the evidence tarball**

Run:
```bash
cd /home/ubuntu
tar czf cipher-fusion-evidence/cp_5_6_p2_evidence.tar.gz \
  -C cipher-fusion-evidence cp_5_6/p2 cp_5_6/CP_5_6_P2_PLAN.md \
  cp_5_6/CP_5_6_P2_REPORT.md
md5sum cipher-fusion-evidence/cp_5_6_p2_evidence.tar.gz \
  > cipher-fusion-evidence/cp_5_6_p2_evidence.tar.gz.md5
cat cipher-fusion-evidence/cp_5_6_p2_evidence.tar.gz.md5
```
Expected: tarball + md5 written.

- [ ] **Step 3: Final verification + WAIT for adjudication**

Confirm the tarball lists the report, result json, all arm jsons, watts csvs,
gold.json, the three scripts. Present the verdict and the per-arm TPW table.
Per campaign discipline, **STOP and WAIT for adjudication** — do not proceed
to any further CP.

---

## Self-Review (against the advisor's methodology spec)

**Spec coverage:**
- Teacher-forced top-1 gate → Task 1 `teacher_forced()`, Task 3 threshold. ✓
- Free-running TPW separate from correctness → Task 1 `free_run()` + sentinel
  window; teacher-forcing runs outside the window. ✓
- Gold = vanilla FP16 free-running greedy, 5 prompts × 128 tok → Task 4. ✓
- Sanity: vanilla teacher-forced ~100% before measuring substrate → Task 5
  HARD GATE with STOP. ✓
- Per (prompt, arm) pair; pass @≥99%; passing pairs contribute TPW; failing
  flagged with agreement + text → Task 3 analyzer. ✓
- Reporting: passing count per arm, TPW + CI95 on passing pairs, overall
  all-on/vanilla lift, A/B/C outcomes → Task 7. ✓
- Catches F1 garbage ~0%, accepts INT4 drift ~95-100% → inherent to
  teacher-forcing; threshold 0.99. ✓ (risk noted below)

**Open risks flagged for the advisor consult:**
- *INT4 drift vs the 99% line.* The advisor says honest INT4 drift is
  ~95-100%; the threshold is 99%. An honest marlin arm could land at, say,
  97% and "fail" the gate. The design handles this by flagging — never
  dropping — failing pairs with their agreement + text for human
  adjudication; a 97%-coherent-text fail is distinguishable from a
  ~0%-garbage fail. This is the advisor's spec, followed as written.
- *Passing-but-looping.* If a substrate arm passes teacher-forcing (Marlin
  numerically correct) yet its free-run text still loops, the loop is not a
  substrate-numerics bug — it is spec-decode or base-model behaviour. The
  report calls this out explicitly (Task 7 §5); the TPW still counts per the
  advisor's "passing pairs contribute their free-running TPW".
- *A/B/C provenance.* The outcome labels are reconstructed, not quoted from a
  defining source in context — confirm with the adjudicator.
- *CI unit.* Per-prompt (n=5), not CP 2.4's repeated-pairs (n=5 repeats).
  This follows the advisor's "pairs passing per arm" framing but is a
  genuine change from the CP 2.4 methodology — surfaced here so it is a
  conscious, reviewed choice.

**Placeholder scan:** none — all code is complete and literal.

**Type consistency:** `free_run()` keys (`t0_epoch`, `t1_epoch`, `tok_s`,
`gen_ids`, `text`, `accept_rate`, `distinct_ratio`) are consumed unchanged by
the gate writer and by `analyze_cp56_p2.load_arm`. `mean_power_window`,
`ci95`, `passes` signatures match `test_analyze_cp56_p2.py`. watts.csv header
`epoch_s,power_w,clock_sm_mhz` written by orchestrator, read by analyzer. ✓
