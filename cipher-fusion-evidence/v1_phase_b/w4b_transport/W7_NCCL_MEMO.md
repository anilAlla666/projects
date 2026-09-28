# W.7 — NCCL tuner plugin (M.1) — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO — STOP for Anil approval before any build
(design-memo → approve → build). **Anchors UNCHANGED** (verified on disk): cipher_rt_phase4
`8b5e928` / libcipher_rt.so md5 `9fe23143`, cipher_kmod `02fc2d1` (0.6.6), cipher_kv_bridge
`5a3db034`; cipher-fusion-evidence HEAD `0245c5e`, `w4b-close`→`dbb7dd9`.

## §0 — Provenance

W.4b CLOSED on branch A (density 3.239×). FWD-1 closed (the "52 ms forward" was a
DynamicCache prototype artifact; product vLLM is already static-KV; graph lever absent →
no new substrate lever; W.8 persistent-kernel DE-PRIORITIZED). Per W.4a §G + Memory #29
resequence (… → W.4 → **W.7** → W.8), W.7 NCCL is the next substep. W.7 = audit **SUBSTEP
M.1 — NCCL tuner** (`V1_CAPABILITY_AUDIT…:491`), capability #20.

## §1 — What W.7 targets + the regime / GPU-count constraint

**Substrate-line primitive (Memory #24):** the **NCCL tuner plugin** — the *sanctioned*
`NCCL_TUNER_PLUGIN` hook (NCCL itself `dlopen`s the path and resolves `ncclTunerPlugin_vN`).
This IS substrate-line (NCCL's own documented plugin ABI), **not** a framework patch. Any
path requiring patching NCCL/vLLM source → **HARD STOP, surface** (Memory #24).

**Most of W.7 already exists — re-port + validate, do NOT rebuild** (`cipher-may13-evidence`):
- `libcipher_nccl_tuner.so` (built) exports **`ncclTunerPlugin_v1` + `ncclTunerPlugin_v2`**
  (`nm -D` confirmed). Sources: `src/cipher_nccl_tuner.cpp` (+ `cipher_nccl.cpp`,
  `_v4`, `_neural`, `_bpf`), ABI header `include/cipher_nccl_tuner_abi.h` (mirrors NCCL
  2.21+ v2 / 2.17–2.20 v1).
- Behavior: `getCollInfo` returns an **(algorithm, protocol, nChannels)** tuple
  (RING/TREE/NVLS; nChannels seeded by the CfC `CipherNcclPolicy`). **Safety: if the RT
  bridge is unresolvable it returns `NCCL_ALGO_UNDEF`/`NCCL_PROTO_UNDEF` → NCCL uses its
  internal tuner = vanilla.** (`cipher_nccl_v4.cpp` is internal cipher logic — `*_decide`
  by bytes — not a plugin-ABI v4 entry.)
- **Net-new for W.7:** (a) confirm load on the installed **NCCL 2.26.2** (its lookup chain
  is `v3 → v2`; our plugin has v2, so it should bind via the v2 rung — must verify; if 2.26
  hard-requires v3, a thin v3-ABI shim is the only net-new code); (b) re-port the DSO into
  the v1 substrate-line + RT-bridge wiring; (c) the validation harness below.

**Regime / GPU-count constraint (load-bearing, surfaced honestly):** the *value* of an
NCCL tuner is choosing better collective algorithms for **cross-rank** AllReduce/AllGather
— a **multi-GPU** lever. **This pod is a single H100** (`nvidia-smi -L` → 1 GPU;
`torch.cuda.device_count()==1`). A 2-rank NCCL job runs on one GPU, but its AllReduce is a
**degenerate intra-GPU reduction with no NVLink/IB transfer to overlap or re-route** — so
the *overlap/p99-speedup* engagement number is **not meaningfully measurable here** and a
1-GPU number must NOT be presented as the multi-GPU lever (no faked multi-GPU result).
Therefore W.7 is scoped (per the W.4b "substrate+safety DELIVERED, regime
surface-with-options" pattern) to **tuner-plugin-correctness on 1 GPU**: load via the
sanctioned ABI, safe pass-through == vanilla, valid `getCollInfo` decisions, and
**bit-identical collective numerics** — with the overlap engagement gate **measured only
if a ≥2-GPU pod is provided** (else surfaced as regime-unreachable, not faked, not NOT-MET).

## §2 — Engagement definition (Memory #19 product-engagement-gate)

- **Capability:** #20 NCCL tuner plugin.
- **Workload class:** multi-GPU collective workloads — pre-training / multi-GPU serving
  with cross-rank AllReduce (audit §B.1). (Single-GPU decode, the W.4b regime, has **no
  tenant-level collectives** — W.7 does not serve it.)
- **Goal × workload:** advances **Goal 1 (density/throughput) at the multi-GPU collective
  workload class** via better-chosen collective algorithm/protocol/nChannels (and, with
  Layer-2 routing, AllReduce↔compute overlap). Does NOT advance single-GPU decode.
- **Engagement gate (the real lever):** *2-rank Mistral-7B pre-training step, AllReduce
  p99 ↓ ≥20% vs vanilla NCCL, correctness preserved* (audit `:497`). **Overlap-attributable
  = (vanilla AllReduce p99) ÷ (plugin AllReduce p99) on identical ≥2-GPU config.**
- **Engagement on THIS pod:** unmeasurable (1 GPU). The deliverable here is
  **plugin-correctness** (§4) — the substrate primitive proven load-able, safe, and
  numerically lossless, staged for a ≥2-GPU gate.

## §3 — Execution plan (7 atomic items; approve before item 1)

1. **ABI / load validation.** Re-port `libcipher_nccl_tuner.so` (+ RT-bridge DSO) into the
   v1 substrate-line. Confirm `NCCL_TUNER_PLUGIN=…/libcipher_nccl_tuner.so` binds on **NCCL
   2.26.2** (NCCL debug log shows the tuner plugin loaded via the `v2` rung). If 2.26 hard-
   requires `v3` → write a thin `ncclTunerPlugin_v3` ABI wrapper (the only net-new code;
   delegates to the existing v2 logic). No NCCL source patch (Memory #24).
2. **Safe pass-through / fallback.** With the RT bridge unresolvable, the plugin must return
   `UNDEF` → NCCL internal tuner. Verify a 2-rank (same-GPU) NCCL AllReduce job behaves
   identically *with the plugin in fallback* vs *no plugin* (no crash, same result, NCCL
   accepts).
3. **`getCollInfo` decision validity.** With the CfC policy active, sweep (collective,
   message-size, nRanks) and assert every returned (algo, proto, nChannels) tuple is valid
   and accepted by NCCL (no NCCL error / no fallback-to-internal due to bad tuple). Capture
   the decision table.
4. **Correctness (Memory #11) — HARD STOP gate.** On 2-rank (same-GPU) AllReduce + AllGather
   over fp16/bf16/fp32 buffers, the collective **output is bit-identical** plugin-active vs
   vanilla (algorithm/protocol selection does not change NCCL numerics). ANY divergence →
   HARD STOP, surface.
5. **Engagement (regime-honest).** If a **≥2-GPU pod is available**: run the 2-rank
   Mistral-7B pre-training step, report AllReduce p99 plugin vs vanilla (gate ≥20% ↓,
   correctness preserved). **Else (this 1-GPU pod):** SURFACE the overlap gate as
   measurement-unreachable; deliver items 1–4 as the substrate-primitive close (the W.4b
   surface-with-options disposition). Do **not** present a 1-GPU number as the lever.
6. **Write `W7_NCCL_FINDINGS.md`** — load/fallback/decision/correctness results + the
   engagement disposition (measured ≥2-GPU, or surfaced 1-GPU).
7. **Commit** (author `Anil`, no co-author; userspace plugin re-port → **anchors UNCHANGED**
   unless a v3 shim or RT-bridge edit touches a tracked substrate DSO — if it does, treat as
   a fresh substrate step: full backfill regression + 9-cell gate per Memory #16). Tag at
   close only if Anil directs. **STOP for Anil.**

## §4 — Decision gate

- **PASS (plugin-correctness, gateable on 1 GPU):** (a) loads via the sanctioned
  `NCCL_TUNER_PLUGIN` ABI on NCCL 2.26.2; (b) safe fallback == vanilla; (c) `getCollInfo`
  returns valid tuples NCCL accepts; (d) **collectives bit-identical** (Memory #11). This
  certifies the substrate primitive.
- **Engagement (the lever):** overlap-attributable AllReduce p99 ↓ **≥20%** vs vanilla on a
  **≥2-GPU** 2-rank Mistral step — **MEASURED only if ≥2 GPUs are provided**, else
  **SURFACED as regime-unreachable** (not faked, not NOT-MET; staged for a multi-GPU pod).
- **FAIL / HARD STOP:** any collective numeric divergence (Memory #11), or any need to patch
  NCCL/vLLM source (Memory #24).

## §5 — Discipline guarantees

- **Substrate-line (Memory #24):** only the sanctioned `NCCL_TUNER_PLUGIN` env hook; zero
  NCCL/vLLM source patching (HARD STOP if required).
- **Correctness (Memory #11):** bit-identical collective output every run; forced-divergence
  → HARD STOP.
- **Anchors (Memory #16):** userspace plugin re-port → UNCHANGED; any tracked-substrate edit
  (v3 shim landing in a versioned DSO / RT-bridge) → fresh substrate step + backfill gate.
- **Regime honesty:** the single-GPU pod cannot gate the multi-GPU overlap lever; that is
  surfaced, not faked — the deliverable here is the proven, lossless, sanctioned plugin
  primitive, staged for a ≥2-GPU engagement gate.

## §6 — One-line ask

Approve W.7 scoped as **tuner-plugin-correctness on this 1-GPU pod** (items 1–4 + 6–7, with
§5 engagement surfaced as ≥2-GPU-deferred) — or, if you can provision ≥2 GPUs and want the
full overlap gate now, say so and item 5 runs the real 2-rank lever. No code until approval.
