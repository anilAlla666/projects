# W.6 sub-C — kmod GPU co-residence registry — CLOSE REPORT

**Date:** 2026-05-28
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `73fac17fc3fea30a0d8c0e5d2c633c9a` (CUDA-13, in-container build)
**Side anchor preserved:** `build_cuda13/libcipher_rt.so.w6sC` (identical bytes)
**kmod:** `cipher_kmod.ko` 0.6.6, srcversion `E27B31659B29969E8A75FE6`, ko md5 `9c0caef96e43798ef121996ad3adc971` (loaded == built)

## Verdict
**W.6 sub-C PASS.** The kmod GPU co-residence registry (NR 30 REGISTER / NR 31
QUERY) + libcipher_rt substrate registration deliver the authoritative
multi-tenant co-residence signal: each tenant learns how many distinct tenant
processes share this GPU right now and each peer's substrate model fingerprint
(W.6 sub-B). With W.6 sub-B's model identity, this completes W.4 POOL's
cross-tenant input: (co-resident pids + per-pid fingerprint) → legal
GEMM-coalescing groups. 9/9 close gate PASS, multi-tenant co_resident=2
verified cross-process (same-fp + distinct-fp), zero segfaults.

## A — kmod co-residence query + NR decision

**NR decision (adjudicated):** a NEW dedicated registry + two new NRs, NOT a
reuse of NR 29's read-path. Investigation (`cipher_stream_registry.c`) showed
NR 29's table is stream-keyed `(tgid, stream_handle) → tenant_id` with **no
per-tgid model_fingerprint and no liveness timestamp** — the two fields W.4
needs. Overloading it would duplicate the fingerprint across each tgid's stream
rows. A dedicated `tgid → (model_fingerprint, last_seen_ns)` registry is the
correct data model.

**File:** `cipher_kmod/cipher_coresidence_registry.c` (new, +Kbuild entry).
- `struct cipher_cohort_node { u32 tgid; u64 model_fingerprint; u64 last_seen_ns; }`, 256-bucket hashtable keyed on tgid, spinlock; model_registry two-phase alloc idiom.
- `cipher_cohort_register(tgid, fp)` — heartbeat-or-insert; `last_seen = ktime_get_ns()`.
- `cipher_cohort_snapshot(max, out, &n)` — prunes entries stale > `CIPHER_COHORT_STALE_NS` (30 s, monotonic) lazily during the walk, returns live snapshot + true live count.
- `cipher_dev_cohort_register` (NR 30) / `cipher_dev_cohort_query` (NR 31) — anti-spoof: key on `current->tgid`, ignore userspace tgid.

**ABI:** `cipher_ioctl.h` — `CIPHER_COHORT_REGISTER` `_IOW(C,30,…)`,
`CIPHER_COHORT_QUERY` `_IOWR(C,31,…)`; additive — NRs 1/27/28/29 byte-unchanged.
MODULE_VERSION 0.6.5 → 0.6.6.

**Heartbeat-or-insert design (advisor catch):** QUERY carries
`caller_fingerprint` and re-inserts the caller if it was pruned during an idle
gap, so a registered-then-idle process re-appears on its next poll. A one-shot
register + heartbeat-via-query alone would have pruned an idle-but-alive tenant
permanently. Verified by test_cohort T4/T5.

## B — substrate registration + authoritative multi_tenant signal

**File:** `cipher_rt_phase4/cipher_rt_coresidence.c` + `.h` (new). Lazy cached
`/dev/cipher` fd (classify_observer idiom); ABI mirrored locally (commit.c
precedent). `cipher_rt_coresidence_update(fp, peers, max, &n)` heartbeat-or-
inserts self via NR 31 (throttled ~1 ioctl/sec) and returns the live count
(incl. self); on any error returns the cached count (never breaks the launch
path).

**Classifier wiring:** `src/cipher_workload_detect.cpp` `classify_internal()` —
`model_fp = cipher_workload_model_fingerprint()` (W.6 sub-B); `co_resident =
cipher_rt_coresidence_update(model_fp, …)`; `multi_tenant = (co_resident >= 2)`.
This **replaces** the K.1-era `/proc`-maps host scan as the `multi_tenant`
gate with the authoritative kmod count. `co_resident_count` carried in the
profile (`include/cipher_workload_detect.h`, carved from `reserved[]`,
ABI-size-preserving). New CLASSIFY-CORES log: `co_resident=N multi=M
peers=[tgid:fp,…]` (W.4 consumes).

## C — kmod regression (Memory #16) + 9/9 close gate

- **ko load/unload cycle:** clean — `coresidence_registry ready … 256-bucket, 30 s liveness` on load; prior module `unloaded cleanly`; no oops. Loaded srcversion == built (`E27B31659B29969E8A75FE6`). `/dev/cipher` mode 0666 preserved.
- **kmod functional test** `test_cohort` (10/10 PASS): T1 single (co_resident=1, self fp readback), T2 same-model peer co-resident, T3 diff-model peer co-resident, T4 heartbeat-via-query keeps a query-only peer live, T5 stale-prune (killed peer aged out, heartbeating peer survived, count dropped to 2).
- **cohort churn soak** `cohort_churn` (90 s, pool 40): **PASS** — 124,600 register+query cycles, **errs=0**; peak 38,005 concurrent live registrants handled with no oops/BUG/leak (true count returned despite the 128-entry snapshot cap — defensive live-vs-filled counting); prune-under-load correct (decayed to co_resident=1 = heartbeating anchor only after the 30 s window). `dmesg` clean (only the init line).
- **strict Memory #16 30-min N=128 resolver-coherence soak** (`resolver_soak`, addendum 2026-05-28, kmod 0.6.6 loaded): **PASS** — **18,327,542,633 reads** over 1800 s by 128 writer threads (continuous NR-29 register churn → view-slot generation churn) + 8 reader threads (lock-free generation-pre/post-checked lookups). **INCOHERENT = 0 at every 60 s checkpoint** (t+60 … t+1800; HARD GATE per Memory #11), misses = 0. Pre-soak atomicity gate **128000/128000 coherent** (matches Memory #17 baseline). Reader lookup p50 88 ns / p99 262 ns / rate 10.18 M/s (descriptive — under sustained N=128 writer contention; uncontended baseline was p99 49 ns; no hard numeric gate vs the 95/35 ns observe-publish path since the original 4-subsystem harness was not preserved). `dmesg` clean (no oops/BUG/leak/call-trace). ko unload/reload cycle clean before AND after the soak (all registries — model/stream/coresidence — re-init; /dev/cipher 0666 restored).
- **9-cell close gate (real vLLM 0.21, build_cuda13 73fac17f bind-mounted over CDI path): 9/9 PASS, NO SEGFAULTS.** All prior actuators preserved — P1/P2 A4_BATCH_INFERENCE (obs 142k/143k), P3 A3 int4=1, T1 A3, C1 A4; marlin_engage/volt/koopman/attn counters intact. Every single-tenant cell reported `co_resident=1 multi=0` with the correct self fingerprint.

**Regression scope (Memory #16 strict — addendum 2026-05-28):** Anil
adjudicated the orthogonality argument as a HYPOTHESIS to be validated by the
soak, not a substitute for it. The strict 30-min N=128 soak above drives the
UNCHANGED W7-9 Step 5 resolver hot path (NR 29 register + view-slot generation
churn + lock-free mmap lookup) with kmod 0.6.6 loaded. Result: **the orthogonality
hypothesis is CONFIRMED empirically** — the new cohort hashtable / spinlock /
init-ordering does NOT perturb the resolver hot path (18.3 B coherent reads,
0 incoherent, no oops/leak, clean ko cycle). The original
`cipher_test_commit_n128` / `test_observe_publish` / `test_audit_chain`
harnesses were `/tmp` scratch (not preserved); the resolver path is the
clearest single-driver representative of the kmod hot path with a hard
coherence metric (`resolver_soak.c` in this dir reconstructs it from
`cipher_stream_resolver.c`). Deviation #2 (soak not run) is RESOLVED.

## D — engagement table

| Scenario | Cells | co_resident | multi | Fingerprints | W.4 verdict |
|----------|-------|-------------|-------|--------------|-------------|
| Single-tenant | P1/T1/C1 Llama-3-8B | 1 | 0 | `0xd40e…` (self) | n/a |
| Single-tenant | P2 Mistral-7B | 1 | 0 | `0x9fde…` | n/a |
| Single-tenant | P3 TinyLlama-AWQ | 1 | 0 | `0x458e…` (int4) | n/a |
| **M1 multi, same model** | 2× TinyLlama-1.1B | **2** | **1** | both `0xe45c…` | **COALESCE** |
| **M2 multi, distinct model** | TinyLlama-fp16 + TinyLlama-AWQ | **2** | **1** | `0xe45c…` + `0x458e…` | **SPLIT** |
| Negative (stale-prune) | test_cohort T5 | 2→ drop | — | killed peer pruned < 30 s | — |

Both M1 tenants observed the symmetric view `co_resident=2 multi=1
peers=[…,…]`; both M2 tenants likewise, with distinct fingerprints. The
co_resident signal is **model-agnostic**; heavy-model fingerprint discrimination
(Llama-3 / Mistral / TinyLlama-AWQ) is independently verified single-tenant
above + in W.6 sub-B. 2× 8B concurrent was **not** used for M1/M2 because vLLM's
`gpu_memory_utilization` profiling collides under co-residence (`Available KV
cache memory: -11.79 GiB` → engine dies before inference); light long-lived
tenants give a clean cross-process demonstration. Per-tenant GPU-memory
partitioning for dense 8B co-residence is W.4 POOL territory (see §E).

## E — engineering debt forecast (v1.x)

1. **Multi-GPU co-residence** (single-GPU-per-host assumption): the registry is
   kmod-global == host-global. Two processes on **different** GPUs of one host
   would appear co-resident — a false positive for W.4 coalescing (no shared
   HBM). This pod is a single H100 so the assumption holds; multi-GPU needs
   keying on `(gpu_uuid, tgid)`. **v1.x.**
2. **100-tenant scale:** the QUERY snapshot is `O(buckets)` with a 2 KiB
   `entries[128]` payload; at the Goal-1 100-agent target this is fine
   (`CIPHER_COHORT_MAX=128`), but the ~1/sec-per-tenant query rate × 100 tenants
   should be load-checked under V.1. The 30 s liveness window may need tuning if
   tenants idle between bursts longer than 30 s (they age out then re-insert on
   next query — correct but adds churn).
3. **Fingerprint collision at scale** (inherits W.6 sub-B debt
   [[w6-subb-model-fingerprint]]): `(max_k, max_m, dtype)` could collide for two
   models with identical FFN+vocab dims; harden with num_kv_heads / layer-count
   if observed.
4. **2× 8B concurrent OOM:** dense same-GPU 8B co-residence needs per-tenant
   memory partitioning (W.4 POOL) — out of scope for the co-residence *signal*.

## F — anchors + W.4 POOL readiness

| Component | Pre-W.6sC | Post-W.6sC |
|-----------|-----------|------------|
| `cipher_rt_phase4` HEAD | fa1a602 (w6-subB) | (commit) tag `w6-subC-coresidence` |
| `cipher_rt_phase4` libcipher_rt.so md5 | 4afb719c (W.6sB, host) → 73fac17f (CUDA-13 rebuild) | **73fac17fc3fea30a0d8c0e5d2c633c9a** |
| `cipher_kmod` HEAD | 8c643fc (W7-9 Step 5) | (commit) tag `w6-subC-kmod` |
| `cipher_kmod` version / srcversion | 0.6.5 | **0.6.6 / E27B31659B29969E8A75FE6** (ko md5 9c0caef9) |
| `cipher-fusion-evidence` HEAD | 28ac198 (w6-subB-close) | (commit) tag `w6-subC-close` |
| `cipher-platform` rev8 `/usr/lib/cipher/libcipher_rt.so` | 1f305ce6 (UNCHANGED) | 1f305ce6 (**UNCHANGED**) |

**Build/deploy methodology (recorded — cost time this substep):**
1. The canonical `build_cuda13/libcipher_rt.so` must be built **inside the
   `vllm/vllm-openai:v0.21.0` container (CUDA 13)** — a host build (nvcc 12.8)
   silently fails to load via CUDA injection in the container. Build cmd: container
   with `/home/ubuntu/cipher_rt_phase4` mounted, `apt-get install -y libssl-dev`
   (for `cipher_rt_audit.c` openssl/hmac.h on a clean build), CUDA-13 include +
   cupti/cusolver/libcrypto symlinks, `make CUDA_INCLUDE=… CUPTI_INCLUDE=… LDFLAGS=…`.
2. The cipher-platform rev8 **CDI hook overrides `-e CUDA_INJECTION64_PATH`** to
   `/usr/lib/cipher/libcipher_rt.so` (the .deb baseline). To gate-test a dev build
   WITHOUT touching the host system file (Memory: rev8 UNCHANGED), **bind-mount**
   it over the CDI path inside the ephemeral container:
   `-v build_cuda13/libcipher_rt.so:/usr/lib/cipher/libcipher_rt.so:ro`. The host
   file stayed `1f305ce6` throughout. Folding this `-v` into the canonical
   `run_close_gate.sh` is a recommended follow-up.

**W.4 POOL readiness:** input now COMPLETE — model identity from W.6 sub-B
(`cipher_workload_model_fingerprint`) + co-residence from W.6 sub-C
(`cipher_rt_coresidence_update` → NR 31 → per-pid `(tgid, fingerprint)` +
authoritative `multi_tenant_detected`). M1 demonstrates a single coalescable
group; M2 demonstrates two non-coalescable groups.
