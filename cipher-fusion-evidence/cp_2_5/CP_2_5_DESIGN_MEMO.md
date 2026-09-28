# CP 2.5 — Design Memo: drop LD_PRELOAD, ship `cipher-platform.deb`

**Date:** 2026-05-16. **Status:** DESIGN — awaiting approval. **No code,
no packaging, no rebuild until this memo is approved.**

CP 2.5 canonical scope: *"LD_PRELOAD dropped in favour of
CUDA_INJECTION64_PATH-only deployment; `cipher-platform.deb` produced for
clean install on neocloud nodes."* It closes Phase 2 (14/23 → 15/23) and is
the **deployment story** for the May 28 Ditlev demo — how CIPHER actually
lands on an operator's fleet.

Read §0 first. The audit found that "drop LD_PRELOAD" is **not** a packaging
change — it is a re-architecture of two interception substrates, with a
real engineering body and a schedule longer than the 3–5 days estimated.
§6 has the decisions that need your adjudication before the gate is fixed.

---

## 0. Audit-before-build findings

### 0.1 The injection lib has TWO independent interception mechanisms

`libcipher_rt.so` intercepts the workload through two unrelated mechanisms,
and only one of them is `CUDA_INJECTION64_PATH`-native:

**Mechanism A — `InitializeInjection` / `InitializeInjection2`.** The CUDA
driver, at `cuInit()`, reads `CUDA_INJECTION64_PATH`, `dlopen`s that `.so`,
and calls `InitializeInjection2()`. `cipher_inject.c` routes that to
`cipher_v2_init_body()` — the actuator-init chain: tenant register, ARB, SM
packer, partition router, CUPTI subscribe, **VOLT/DVFS**, matmul-substrate
init, **Marlin engine + actuator**, attn-substrate init, audit. **This entire
chain is `CUDA_INJECTION64_PATH`-native — it needs no `LD_PRELOAD`.**

**Mechanism B — link-order symbol interposition.** Two substrates work by
*exporting symbols that shadow* the real ones, which only happens if
`libcipher_rt` sits **ahead of** the target library in the dynamic linker's
global scope. That ordering is exactly what `LD_PRELOAD` provides:

- **cuBLAS shim** (`cipher_rt_cublas_shim.c`) — exports
  `cublasGemmEx@libcublas.so.13` (via `.symver` + `cublas_version.map`) plus a
  plain alias. PyTorch's `cublasGemmEx` calls resolve to the shim **only**
  because `LD_PRELOAD` puts us first.
- **attn substrate** (`cipher_rt_attn_dispatch.cpp`) — exports the plain
  mangled ATen SDPA `::call` symbols (`_ZN2at4_ops..._scaled_dot_product_*`).
  Same: interposes **only** under `LD_PRELOAD`.

A library `dlopen`'d late by the driver (at `cuInit`) is appended to the
**end** of the global scope — *behind* the already-loaded `libcublas` and
`libtorch_cpu`. It therefore **cannot shadow** their symbols. This is
standard ELF loader behaviour, not a tunable.

### 0.2 Empirical confirmation — the audit smoke

`audit/cp25_smoke.py` (20× fp16 matmul on CUDA), libcipher_rt `5e304549`,
`CIPHER_MARLIN=on`. Full result: `audit/SMOKE_RESULT.md`.

| arm | InitializeInjection chain | cuBLAS shim | attn trampolines |
|---|---|---|---|
| `CUDA_INJECTION64_PATH` only | runs **fully** (all actuators init) | **calls=0** — never fired | **tramp_calls=0** |
| `LD_PRELOAD` + `CUDA_INJECTION64_PATH` | runs fully | **calls=20** — every GEMM routed | active |

The finding is decisive: **`CUDA_INJECTION64_PATH`-only silently loses the
cuBLAS shim and the attn substrate.** Marlin routing depends on the cuBLAS
shim, so under `CUDA_INJECTION64_PATH`-only the **CP 2.4 composed 3.617×
result would not reproduce** — the GEMMs bypass the substrate entirely. CUPTI
(driver-level, Mechanism A) still works — it saw the kernel launches.

### 0.3 The packaging constraint — `libcipher_rt` drags `libtorch`

`libcipher_rt.so` links `-ltorch_cpu -lc10 -lcrypto` as `DT_NEEDED` (the attn
substrate needs ATen SDPA signatures). A `.deb` that installs this lib has a
**hard runtime dependency on a specific `libtorch`** (currently 2.11.0+cu130 —
the header pins `CIPHER_RT_ATTN_TORCH_VERIFIED`). "Clean install on a neocloud
node" must account for this: either the node's torch matches, or the `.deb`
ships/pins its own. §6 Decision 2.

### 0.4 Consequence — what CP 2.5 actually is

CP 2.5 is **not** "wrap the existing lib in a `.deb`." It is:

1. **Re-architect Mechanism B** — replace `LD_PRELOAD` link-order
   interposition with a mechanism that works from the late-loaded injection
   lib: **GOT/PLT patching** driven from `InitializeInjection2` (§2).
2. **Package** — `cipher-platform.deb` with kmod 0.4.8 + libcipher_v2 +
   libcipher_rt + install/uninstall scripting (§3).
3. **Gate** — prove the 3.617× composed result reproduces `LD_PRELOAD`-free,
   plus clean install/uninstall, plus no regressions (§4).

---

## 1. Scope

| Sub-task | What | Risk |
|---|---|---|
| (i) GOT/PLT-patch re-architecture | cuBLAS shim + attn substrate intercept via runtime GOT patching from `InitializeInjection2`, not `LD_PRELOAD` | **High** — new mechanism; RELRO, multi-module scan, ordering |
| (ii) `.deb` packaging | `cipher-platform.deb` — kmod, libs, `postinst`/`prerm`, env wiring | Medium — DKMS vs prebuilt kmod, libtorch dep |
| (iii) Gate | 3.617× reproduces `LD_PRELOAD`-free; install/uninstall clean; no regressions | Medium — measurement discipline |

In scope: the deployment-mode change and packaging. **Out of scope:** any
actuator behaviour change, ABI change, kmod source change. CP 2.5 is a
delivery-mechanism CP — the *what* CIPHER does is frozen; only *how it loads*
changes.

---

## 2. Design — GOT/PLT patching

### 2.1 Mechanism

When `InitializeInjection2()` runs (at `cuInit`, **before** the first
`cublasGemmEx` — confirmed by the smoke ordering: the init chain prints
before "matmul done"), a new `cipher_rt_got_patch_init()` step:

1. `dl_iterate_phdr()` — walk every loaded ELF object.
2. For each object, parse `.rela.plt` + `.rela.dyn`; find GOT slots whose
   relocation symbol is `cublasGemmEx`, `cublasGemmStridedBatchedEx`, or the
   three ATen SDPA `::call` mangled names.
3. Save the original pointer (for passthrough — replaces the current
   `dlvsym`/`RTLD_NEXT` resolution), `mprotect` the GOT page writable
   (handles RELRO), overwrite the slot with our trampoline address, restore
   page protection.

This is the well-trodden "PLT hook" / API-detour pattern (≈400–600 LOC,
self-contained, no new dependency). It patches **call sites directly**, so it
is **load-order-independent** — works precisely because it does not rely on
symbol-scope ordering.

### 2.2 What changes in the source

- **New TU** `cipher_rt_got_patch.c` — the `dl_iterate_phdr` GOT patcher.
- **cuBLAS shim** — the `.symver` export and `cublas_version.map` are
  **removed**; `cipher_rt_cublasGemmEx_impl` stays as the trampoline body,
  now installed by the GOT patcher instead of the linker. The "real"
  pointer comes from the saved GOT value, not `dlvsym`.
- **attn substrate** — same: the exported mangled ATen symbols are removed;
  the trampolines are installed by the GOT patcher.
- **`cipher_inject.c`** — `cipher_rt_got_patch_init()` added to the chain,
  ordered **after** matmul/attn substrate init (the trampolines must be
  registered before the GOT is pointed at them).
- **Makefile** — `--version-script` dropped; the attn TU no longer needs
  `-fvisibility=default` on the intercept symbols.

`libcipher_rt.so` is rebuilt — new campaign anchor md5 recorded; `5e304549`
saved as the rollback point. **No kmod change, no ABI change, no
libcipher_v2 change.**

### 2.3 Why not the alternatives (see §6 Decision 1)

- **`/etc/ld.so.preload`** — keeps the `LD_PRELOAD` *mechanism* node-wide,
  zero re-architecture. Rejected as the primary: (a) it is still
  `LD_PRELOAD`-family — not "`CUDA_INJECTION64_PATH` only"; (b) `libcipher_rt`
  drags `libtorch` (§0.3), so **every process on the node** — `sshd`, `cron`,
  apt — would load ~hundreds of MB of `libtorch`; fragile and invasive.
  Kept as a documented fallback only.
- **Driver-export-table / `cuLaunchKernel`-level routing** — intercept GEMMs
  at the kernel-launch granularity via the CUPTI path. Rejected: cuBLAS GEMMs
  would have to be recognised by cubin/kernel name and substituted — far
  harder and more fragile than the clean `cublasGemmEx` API intercept; a
  Phase-5-scale redesign, not CP 2.5.

### 2.4 Viability confirmed — GOT relocations exist for both substrates

GOT patching only works if the call sites route through the GOT. Verified
pre-memo with `objdump -R` (`audit/GOT_RELOC_CHECK.txt`):

- `libtorch_cuda.so` has `R_X86_64_JUMP_SLOT` relocations for
  `cublasGemmEx@libcublas.so.13` and `cublasGemmStridedBatchedEx@…` — the
  cuBLAS shim's targets. **Catchable.**
- `libtorch_cpu.so` has `R_X86_64_JUMP_SLOT` relocations for all three ATen
  SDPA `::call` symbols the attn substrate hooks (flash / efficient / cudnn).
  **Catchable** — confirms the attn migration is structurally viable on this
  stack, not assumed (the symbols are both GOT-relocated *and* exported — the
  interposable pattern LD_PRELOAD relied on; GOT patching reaches the same
  slots directly).

So the GOT patcher has a real target set for both substrates. This is the
audit's positive result — the §0.2 smoke proved the *negative* (injection-only
loses both), this proves GOT patching can *recover* both.

---

## 3. `cipher-platform.deb` design

```
cipher-platform_0.5.0_amd64.deb
├── /opt/cipher/lib/libcipher_rt.so          (GOT-patch build, new anchor)
├── /opt/cipher/lib/libcipher_v2.so          (86618c30, frozen)
├── /opt/cipher/kmod/cipher_kmod.ko          (0.4.8 e2f50452)  — see Decision 3
├── /opt/cipher/bin/cipher-run               (env-wiring launcher wrapper)
├── /etc/profile.d/cipher-platform.sh        (exports CUDA_INJECTION64_PATH)
├── /usr/lib/modules-load.d/cipher.conf      (load kmod at boot)
└── DEBIAN/{control,postinst,prerm,postrm}
```

- **`postinst`** — `depmod`, `modprobe cipher_kmod` (or DKMS build per
  Decision 3), verify `/dev/cipher` + `/dev/cipher_kvdedup` at 0666 (the
  devnode callback codified in 0.4.8), `ldconfig`.
- **`prerm`/`postrm`** — `rmmod cipher_kmod`, remove `/opt/cipher`, undo
  `ld.so` / profile wiring. Uninstall must leave the node byte-clean.
- **Deployment model** — the operator runs their workload normally; CIPHER
  activates via the single env var `CUDA_INJECTION64_PATH=/opt/cipher/lib/
  libcipher_rt.so`, set fleet-wide by `/etc/profile.d/cipher-platform.sh`.
  **No `LD_PRELOAD`, no per-job command-line change** — the Ditlev story.
- **`control`** — `Depends:` encodes the libtorch constraint (Decision 2).

Install + uninstall are themselves gate criteria (§4 c).

---

## 4. Workload + gate

Gate workload: the **CP 2.4 composed gate, re-run `LD_PRELOAD`-free** —
Mistral-7B B=1, n=5 matched pairs, three arms (vanilla / Marlin / all-on),
activated **only** by `CUDA_INJECTION64_PATH` pointing at the installed
`/opt/cipher/lib/libcipher_rt.so`.

**Gate criteria:**

| # | Criterion | PASS condition |
|---|---|---|
| (a) | Interception parity | cuBLAS shim + attn trampolines fire under `CUDA_INJECTION64_PATH`-only — `MATMUL calls > 0`, GEMMs routed (the §0.2 smoke goes from calls=0 → calls>0) |
| (b) | 3.617× reproduces | composed tok/W lift, `LD_PRELOAD`-free, n=5 — within CI of the CP 2.4 result [3.591, 3.642]; **honest-gap stated if not** |
| (c) | `.deb` install/uninstall | clean install on a from-scratch node state; `/dev/cipher` 0666; uninstall leaves node byte-clean |
| (d) | No regressions | CP 3.3 gate 4/4; KV-dedup 5/5; `.symver`→GOT migration changes no actuator output |
| (e) | Anchors held | kmod 0.4.8 `e2f50452` + libcipher_v2 `86618c30` + ABI + taint unchanged; libcipher_rt rebuilt → fresh md5 recorded |

n=5 matched pairs, mean ± 95 % CI — campaign measurement discipline.

---

## 5. Honest fallback / risk

1. **GOT patching is the schedule risk.** RELRO (`-z relro -z now`) makes the
   GOT read-only after load — the patcher must `mprotect` it; if a target
   object is built `-z now` with full RELRO the page must be re-protected
   after. Multi-module scan must catch every caller of `cublasGemmEx`
   (PyTorch's CUDA backend; possibly more). If patching proves unreliable on
   this stack, the documented fallback is `/etc/ld.so.preload` (§2.3) — the
   memo commits, **now**, that a fallback to `/etc/ld.so.preload` would be
   **reported as a partial close** ("LD_PRELOAD mechanism retained node-wide,
   removed from the per-job command line"), not relabelled as success.
2. **If 3.617× does not reproduce** `LD_PRELOAD`-free: the gap is measurable
   and attributable — GOT patch missed a caller (→ `MATMUL calls` low), or a
   trampoline-vs-`.symver` passthrough-path difference. Report the v2-deploy
   number with CI and the attributed cause; do not reframe.
3. **libtorch dependency** (§0.3) may force the `.deb` to be larger or
   torch-version-pinned than "clean install" implies — surfaced honestly in
   the report.

---

## 6. Open decisions — your adjudication

**Decision 1 — interception re-architecture.**
- (A) **GOT/PLT patching** from `InitializeInjection2` — true
  `CUDA_INJECTION64_PATH`-only; the recommended approach.
- (B) `/etc/ld.so.preload` — no re-architecture, but still `LD_PRELOAD`-family
  and drags `libtorch` node-wide. **Coupling:** Decision 2(iii) sever-libtorch
  substantially cleans this option — without the `libtorch` drag, a
  system-wide preload of the small stand-alone `.so` is unobjectionable.
  **1(B) + 2(iii) together** is a credible fast path against the May 28 date,
  at the cost only of the literal "injection-only" wording.
- (C) Reframe CP 2.5 scope to packaging-only and keep per-job `LD_PRELOAD`.
- **Recommendation: (A).** It is the only option that delivers the literal
  "`CUDA_INJECTION64_PATH` only" Ditlev story. (B) — best paired with 2(iii) —
  is the honest fallback if (A) proves unreliable on this stack.

**Decision 2 — libtorch dependency in the `.deb`.**
- (i) `Depends: libtorch2.11` (or equivalent) — assumes the node has matching
  torch; smallest `.deb`.
- (ii) Bundle a pinned `libtorch_cpu`/`c10` in `/opt/cipher/lib` — self-
  contained, larger `.deb`, version-frozen.
- (iii) Sever the dependency — the attn substrate is **PASSTHROUGH-only today**
  (no live actuator until T4.6.3); the ATen-header link could be made
  build-time-only / weak so the runtime `.so` does not hard-need `libtorch`.
  Cleanest, slightly more work.
- **Recommendation: (iii) if cheap, else (ii).** A platform `.deb` should not
  hard-couple to the tenant's torch build.
- **Addendum (post-build-empirics, 2026-05-16).** D2(iii)'s interpretation is
  refined by the build empirics: the bare-drop (sever *all* torch libs) was
  tested and **failed** — the attn TU's inline ATen header code pulls 8
  out-of-line `c10::` symbols, one of them a *data* symbol
  (`c10::UndefinedTensorImpl::_singleton`) that cannot be lazy-deferred, and
  libc10 sits in an RTLD_LOCAL scope invisible to a separately dlopen'd
  libcipher_rt. Decision: **sever `libtorch_cpu`** (the heavy, 451 MB,
  version-fragile component — *that* is D2(iii)'s real intent) and **retain
  `libc10`** (1.5 MB, unversioned soname, ABI-stable core) via the linker
  (`-lc10` + `-rpath`). `cipher-platform.deb` bundles `libc10.so`. The `.deb`
  stays self-contained and version-independent for tenant workloads — it does
  not couple to the tenant's heavy torch build, only carries the small stable
  c10 core. This is D2 = (iii) on the component that mattered, with (ii)'s
  bundling applied to the 1.5 MB remainder.

**Decision 3 — kmod in the `.deb`: prebuilt `.ko` vs DKMS.**
- (i) Ship the prebuilt `cipher_kmod.ko` (0.4.8) — works only on the matching
  kernel `6.8.0-1046-nvidia`.
- (ii) DKMS — rebuilds on the node's kernel; portable across the fleet,
  needs kernel headers present.
- **Recommendation: (ii) DKMS** for a real fleet; (i) acceptable if the demo
  fleet is kernel-homogeneous. Your call on the neocloud node reality.

**Decision 4 — attn substrate migration.** The attn substrate is
PASSTHROUGH-only (no live actuator until T4.6.3). Migrate it to GOT patching
in this STEP, or defer? The viability question is **settled** — §2.4's
`objdump -R` confirms `libtorch_cpu.so` carries GOT relocations for all three
hooked SDPA `::call` symbols, so migrating attn is the *same patcher* pointed
at three more slots, not new mechanism. **Recommendation: migrate now** —
near-zero marginal cost, and leaving a known-broken substrate in a shipped
`.deb` is worse. (Defer remains defensible only if the build STEP wants to
minimise gate surface — but it would ship a `.deb` with a dead substrate.)

---

## 7. Calendar — honest scope

You estimated 3–5 days. The audit revises this:

| Sub-task | Estimate | Risk |
|---|---|---|
| GOT/PLT-patch re-architecture + per-substrate verification | ~3–5 days | High — new mechanism, RELRO, multi-module |
| `.deb` packaging + install/uninstall on a clean node | ~2 days | Medium |
| Gate — 3.617× `LD_PRELOAD`-free, n=5 + regressions | ~1–2 days | Medium |

**Realistic envelope: ~1.5 weeks, not 3–5 days.** The packaging is 3–5 days;
the GOT-patch re-architecture is the body of the work and the schedule risk.
Still **one atomic STEP, one report at the end.** Stated now so the May 28
demo planning is not surprised — if CP 2.5 is demo-critical, Decision 1(B)
(`/etc/ld.so.preload`) is the fast path that makes the date at the cost of
the literal "injection-only" claim.

---

## 8. Discipline — anchors held

**CP 2.5 is pure userspace + packaging.** The GOT patcher and the
substrate changes live in `libcipher_rt.so`; the `.deb` wraps existing
artifacts.

- **No kmod source change. No ABI change.** kmod 0.4.8 `e2f50452` — shipped
  as-is in the `.deb`. ABI ioctl nrs untouched.
- Anchors `55ab8c0c` (kmod fallback) / `86618c30` (libcipher_v2) — frozen.
- `libcipher_rt.so` rebuilt — `5e304549` saved as the rollback point; the new
  GOT-patch build records a fresh md5 as the campaign anchor.
- Taint expected stable at 12288.

---

## 9. New artifacts CP 2.5 will produce (on approval)

- `cipher_rt_got_patch.c` — the GOT/PLT patcher (new TU)
- cuBLAS shim + attn substrate edited to GOT-install (no `.symver`)
- `libcipher_rt.so` rebuilt (new anchor md5 recorded)
- `cipher-platform.deb` + `DEBIAN/` scripting + `cipher-run` launcher
- evidence under `cipher-fusion-evidence/cp_2_5/`: audit smoke + GOT-reloc
  check (done — `audit/`), interception-parity logs, the `LD_PRELOAD`-free
  composed re-run, install/uninstall transcripts, `CP_2_5_REPORT.md`

---

**No code, no packaging, no rebuild until this memo is approved.** Please
adjudicate §6 decisions 1–4 — Decision 1 defines whether CP 2.5 delivers the
literal "injection-only" claim or the `/etc/ld.so.preload` fallback, and
that choice drives the calendar against the May 28 demo.
