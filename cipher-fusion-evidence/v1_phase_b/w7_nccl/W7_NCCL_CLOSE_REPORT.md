# W.7 — NCCL tuner plugin (M.1) — CLOSE REPORT (plugin-correctness, 1-GPU scope)

**Date:** 2026-05-29. **Status:** primitive-correctness items (1–4, 6) VALIDATED;
engagement overlap gate (item 5) HARDWARE-DEFERRED to v1.x. **STOP at §4 verdict for
Anil.** **Anchors UNCHANGED** (cipher_rt_phase4 `8b5e928`/`9fe23143`, cipher_kmod
`02fc2d1`, cipher_kv_bridge `5a3db034`; cipher-fusion-evidence `w4b-close`→`dbb7dd9`).

## §1 — Primitive VALIDATED (items 1–4, 6)

Validated the **existing** `cipher-may13-evidence/libcipher_nccl_tuner.so` (re-port, **no
rebuild** — confirmed it loads & passes as-is). Harness: `w7_plugin_validate.c` (direct
ABI), `w7_stub_bridge.c` (active-path stub), `w7_nccl_bind.c` (real NCCL bind).

- **Item 1 — ABI bind under NCCL 2.26.2: PASS, no shim needed.** NCCL's lookup chain is
  **v4 → v3 → v2**; v4/v3 miss, then NCCL logs **"Using tuner plugin cipher-nccl-tuner-v2"**
  — the existing v2 export binds via the fallback rung. (`w7_nccl_bind` + `NCCL_DEBUG=INFO`;
  also `nm -D`: `ncclTunerPlugin_v1`+`v2` PRESENT.) **The only contingency net-new code (a
  thin v3 shim) is NOT required.**
- **Item 2 — loads + getCollInfo callable + safe fallback: PASS.** `init` rc=0
  (`ncclSuccess`); `getCollInfo` callable; with the RT-bridge symbol absent the plugin
  returns `NCCL_ALGO_UNDEF`/`NCCL_PROTO_UNDEF` (24/24 calls) → NCCL uses its internal tuner.
- **Item 4 — getCollInfo decision validity: PASS (defined=24, bad=0).** With the bridge
  resolvable and a fabric clearing the plugin's small-fabric guard (nRanks=16,nNodes=2),
  every returned `(algorithm, protocol, nChannels)` is a **legal NCCL tuple** across the
  message-size sweep {1 KiB … 1 GiB} × {AllReduce, Broadcast, AllGather, ReduceScatter}:
  algo ∈ {TREE(0),RING(1)}, proto ∈ {LL128(1),SIMPLE(2)}, nChannels ∈ {2,4,8}; NVLS→RING
  fallback when no NVSwitch. No out-of-range / no NCCL error.
- **Item 6 — fallback path clean: PASS.** Bridge unresolvable → all UNDEF → NCCL default,
  `init`/`destroy` clean, no crash. (Also: the plugin's own **small-fabric guard** —
  nRanks≤4 ∧ nNodes≤1 → passthrough — means on small fabrics it defers to NCCL regardless
  of the bridge; correct conservative behavior.)
- **Item 3 — pass-through correctness (Memory #11): PARTIAL on 1 GPU.** A valid **1-rank**
  NCCL `AllReduce` returns the identity result (2.0) **bit-identically** with the plugin
  loaded vs vanilla. A **real multi-rank** collective cannot run on this pod — NCCL
  **rejects 2 ranks on one GPU** ("Duplicate GPU detected … both on CUDA device 7000",
  `ncclInvalidUsage`); ≥2 physical devices (or MIG) are required. The plugin is by
  construction a **selection-hint emitter** (writes only algo/proto/nChannels out-params;
  it never touches the collective data path), so it **cannot alter collective numerics** —
  but the multi-rank bit-identical *measurement* is deferred with item 5 (below).

## §2 — Engagement gate (item 5): HARDWARE-DEFERRED to v1.x

The real lever — *2-rank Mistral-7B pre-training step, AllReduce p99 ↓ ≥20% vs vanilla,
correctness preserved* (audit `:497`) — is **NOT measurable on this single-H100 pod**: an
NCCL collective needs ≥2 ranks on ≥2 devices (NCCL rejects 2-on-1), and even then the
overlap lever requires real inter-GPU (NVLink/IB) transfer to hide behind compute. **No
1-GPU number is presented as the multi-GPU lever (none was faked).** Deferred to a ≥2-GPU
pod (Anil has no ≥2-GPU access this cycle). The same constraint defers the multi-rank
*pass-through bit-identical* measurement (item 3 tail).

## §3 — Anchors UNCHANGED

No substrate/kmod/bridge edit. The validation re-used the existing may13 plugin .so
verbatim (Memory #16: userspace validation only). The plugin's active path resolves
`cipher_nccl_record_decide` via `dlsym(RTLD_DEFAULT)`; the **V1-anchor** libcipher_rt.so
does NOT export it (the may13 RT does) — wiring the active bridge into the v1 substrate-line
is part of the broader re-port and would be a fresh substrate step (backfill + gate) **if**
pursued for a ≥2-GPU engagement run; it is NOT needed for the correctness validated here.

## §4 — What's net-new vs reported

- **Reported/pre-existing (re-used verbatim):** the tuner plugin itself
  (`libcipher_nccl_tuner.so`, `ncclTunerPlugin_v1/v2`, getCollInfo algo/proto/nChannels
  mapping, CfC-seeded nChannels, safe UNDEF fallback, small-fabric guard) — all from
  `cipher-may13-evidence`. **Nothing was rebuilt.**
- **Net-new this substep:** only the **validation harness** (`w7_plugin_validate.c`,
  `w7_stub_bridge.c`, `w7_nccl_bind.c`, `w7_nccl_integration.py`) — test code, not product.
- **Net-new NOT needed:** the contingency v3 ABI shim (NCCL 2.26.2 binds v2 directly).
- **Net-new deferred (for a ≥2-GPU engagement run, not done here):** wiring the active RT
  bridge (`cipher_nccl_record_decide`) into the v1 substrate-line + the Layer-2 overlap
  routing.

## §5 — Goal status (honest)

**NCCL tuner plugin correctness VALIDATED** (binds via the sanctioned `NCCL_TUNER_PLUGIN`
ABI under NCCL 2.26.2 with no shim; loads; emits valid getCollInfo decisions; safe
fallback; selection-hint-only so numerically lossless by construction; 1-rank pass-through
bit-identical). **The overlap engagement lever is UNPROVEN — hardware-deferred to v1.x, NOT
delivered.** Goal-1-at-multi-GPU-collective-workloads is therefore *substrate-primitive
ready*, not *engagement-proven*.

## §4 VERDICT (for Anil)

- **PASS (primitive, gateable on 1 GPU):** items 1, 2, 4, 6 PASS; item 3 PASS on the
  degenerate 1-rank path + lossless-by-construction. The plugin is load-able via the
  sanctioned ABI, valid, and safe.
- **Engagement (the lever):** overlap p99 ↓≥20% — **SURFACED as ≥2-GPU-deferred** (not
  faked, not NOT-MET). Multi-rank pass-through bit-identical deferred likewise.
- **No tag move, no v1.x build.** Awaiting Anil: accept the primitive-correctness close
  (W.7 substrate-primitive ready, engagement v1.x-deferred), then frontier → V.1 (CP 5.5
  soak) or as Anil directs.
