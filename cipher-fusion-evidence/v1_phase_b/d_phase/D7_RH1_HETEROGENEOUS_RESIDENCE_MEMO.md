# D.7 — R-H1 heterogeneous weight residence — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO — STOP for Anil approval before code
(design-memo → approve → build → close-gate → STOP). First D-phase substep; **unblocks the
V.1 Goal-1 soak.** **Anchors UNCHANGED at entry** (cipher_rt_phase4 `8b5e928`/`9fe23143`,
cipher_kmod `02fc2d1`/0.6.6, cipher_kv_bridge `5a3db034`).

## §0 — Provenance

V.1 item-1b found R-H1 (heterogeneous weight residence) net-new for the LOCKED Goal 1
(100 agents × ≥3–5 families, KL=0). D-phase plan (`D_PHASE_PLAN.md`) sequences D.7 first.

## §1 — What D.7 targets + current-vs-net-new

**Goal 1 residence math:** 100 agents each with its own full INT4 model copy = 100×~4 GB ≈
400 GB → **does NOT fit 80 GB.** It fits **only via weight-sharing**: agents of the *same
family* share one physical INT4 copy. **≥3–5 families × one shared copy each ≈ 12–20 GB**,
100 tenants bound to their family's copy. So R-H1 = **extend same-model weight-sharing to
≥3–5 DISTINCT families co-resident, with per-tenant routing to the right family kit.**

**Built (re-use verbatim, do NOT rebuild):**
- **Track-2 weight-sharing** (same-model N tenants → one read-only VMM arena copy; lossless
  KL=0; `cipher-track2-weight-sharing`, validated W.4b.6 formb6 at N=8 Mistral 13.49 GiB).
- **model_uuid routing primitives:** `apply_recipe` keys on `model_uuid_lo/hi` (G12);
  `CIPHER_REGISTER_MODEL` ioctl → model_uuid (G10); KV `cipher_rt_kv_dedup_model_hash` (G3).
- **Marlin** weight cache keyed by **`w_ptr`** (`cipher_rt_marlin_engine_observe_weight/
  lookup(w_ptr)`) — distinct families have distinct `w_ptr`s, so within-process keying
  separates them; WA arenas cap=100 (G2).

**Net-new for D.7:**
1. **Multi-family co-residence orchestration:** ≥3–5 family arenas (Mistral-7B, Qwen-7B,
   Llama-3-8B + an SLM) resident simultaneously (INT4), each shared across its tenants —
   the W.4b.6 single-family arena extended to N families. (cipher_kv_bridge VMM arenas,
   cuMemExport POSIX-FD; bridge UNCHANGED — orchestration is userspace + existing ioctls.)
2. **Per-tenant family routing:** each tenant's launches resolve to its family's Marlin
   INT4 kit + Koopman recipe via `model_uuid` (REGISTER_MODEL), with **no cross-family
   weight bleed** (the Memory-#11 catastrophe-guard analog: a tenant must never execute on
   another family's weights).
3. **JIT cubin cache scaling:** Marlin recompiles per (model, K, N) shape — at ≥3–5
   families budget JIT amortization across `model_uuid` where shapes repeat (e.g. 7B-class
   share K=N=4096 for o_proj). Verify the cubin cache doesn't thrash at 5 families.
4. **Heterogeneous-residence correctness gate** (below).

**Substrate-line (Mem #24):** CUDA-dispatch/symbol-intercept + existing `/dev/cipher`
ioctls; no vLLM/NCCL source patch. The CP 5.3 anchor (`cipher-marlin-primary-ctx-pin`,
Marlin is full-GPU-only) remains in force — D.7 does not change it.

## §2 — D.7 gate (the residence proof that unblocks V.1)

D.7 closes when, on this single H100:
- **≥3–5 distinct INT4 model families co-resident** in shared arenas (residence + HBM
  budget logged, fits 80 GB), each shared across ≥2 tenants;
- **per-tenant routing correct:** every tenant's decode runs on ITS family's kit —
  **each tenant output KL=0 vs that family standalone** (Memory #11); a forced
  mis-route (tenant→wrong family) is BLOCKED by the catastrophe guard → solo/abort, no
  corruption (the W.4b.3 first-coalesce-backstop analog, reused);
- **default-OFF:** the multi-family path ships gated; off → cuBLAS passthrough == vanilla.

(The full **100-agent** density + bursty soak is measured in **V.1**, not D.7. D.7 proves
the *residence + routing substrate* the V.1 Goal-1 soak runs on.)

## §3 — Execution plan (7 atomic items; approve before item 1)

1. **Multi-family arena residence:** load ≥3–5 families' INT4 weights into shared VMM
   arenas (extend formb6_weights pattern to N families); log HBM (target ≤ ~20–25 GiB for
   5×7B-INT4) + fits 80 GB with KV headroom.
2. **REGISTER_MODEL per family** → distinct model_uuid; bind each tenant to its family's
   arena + uuid.
3. **Per-tenant Marlin/Koopman routing:** dispatch resolves the tenant's family kit by
   model_uuid (Marlin `w_ptr` of that family's shared copy; recipe by uuid). Verify routing
   table at ≥3–5 families.
4. **Correctness gate (Mem #11):** each tenant KL=0 vs its family standalone; forced
   mis-route → catastrophe-guard BLOCK (no cross-family bleed). HARD STOP on any KL≠0.
5. **JIT cubin cache check:** confirm no thrash at 5 families (cubin reuse across uuids
   where (K,N) repeat; CUPTI launch-count parity).
6. **Backfill regression + 9-cell gate (Mem #16):** the standard cross-tree/kmod regression
   (kmod load/unload clean, /dev/cipher 0666, prior actuators intact, SC6 bit-identical) —
   D.7 is a fresh substrate step.
7. **Close report `D7_RH1_CLOSE.md`** + commit (author Anil; tag `d7-rh1-residence`; rotate
   anchors only if a substrate DSO changed). **STOP for Anil** before D.8.

## §4 — Decision gate

- **PASS:** ≥3–5 families co-resident on 80 GB, per-tenant routing KL=0, mis-route BLOCKED,
  cubin cache stable, 9-cell regression clean, default-OFF/passthrough preserved.
- **PARTIAL:** name the exact residence/routing limit (e.g. fits only K families < 3–5, or
  cubin thrash, or a routing gap) + measured-vs-target + eng-days. No faked residence.
- **HARD STOP (Mem #11):** any tenant KL≠0 vs standalone / any cross-family weight bleed.

## §5 — Discipline

Substrate-line only (Mem #24); fresh-substrate backfill + 9-cell gate + tag (Mem #16);
KL=0 per tenant (Mem #11); default-OFF, cuBLAS passthrough worst case; reuse Track-2 +
G3/G10/G12 + W.4b.3 backstop verbatim (no second copy). Commit as Anil, no co-author.

## §6 — One-line ask

Approve D.7 scoped as the ≥3–5-family residence + per-tenant-routing substrate (gate =
KL=0 routing + mis-route BLOCK, unblocking the V.1 Goal-1 soak) — or adjust the family set
/ residence budget / gate. No code until approval.
