# CP 2.5 — LD_PRELOAD-free deployment + `cipher-platform.deb` — REPORT

**Date:** 2026-05-16. **Status:** STEP closed — all five gate criteria
**PASS**. CP 2.5 closes as a full close (Phase 2: 14/23 → **15/23**).

CP 2.5 canonical scope: retire the per-job `LD_PRELOAD` so the CIPHER runtime
deploys via `CUDA_INJECTION64_PATH` alone, and ship the platform as a
self-contained `cipher-platform.deb`. One atomic STEP over multiple sessions;
this report lands at STEP close. Build order (approved memo §6): GOT/PLT
patcher → cuBLAS shim re-target → attn substrate re-target → libtorch sever →
`cipher_inject` wiring → composed-gate reproduction → `.deb` → install
transcripts.

Memo §6 decisions, as adjudicated: **D1 = (A)** GOT/PLT patching · **D2 =
(iii)** sever libtorch (refined post-build — see §3) · **D3 = (ii)** DKMS ·
**D4** migrate the attn substrate now.

---

## 1. The mechanism — GOT/PLT patching replaces link-order interposition

**The problem (audit, pre-build).** Two of the v2 stack's interception
mechanisms were `LD_PRELOAD`-dependent: the cuBLAS GEMM shim (a `.symver`
alias on `cublasGemmEx`) and the ATen SDPA substrate (three trampolines on
the `_scaled_dot_product_*_attention::call` symbols). Both rely on link-order
symbol interposition — a library `dlopen`'d *late* by the CUDA driver via
`CUDA_INJECTION64_PATH` cannot provide it. The audit smoke
(`audit/SMOKE_RESULT.md`) was decisive: under `CUDA_INJECTION64_PATH`-only the
cuBLAS shim fired **0** times vs **20** under `LD_PRELOAD`; the attn
trampolines, **0**. The `InitializeInjection` actuator-init chain (tenant,
ARB, SMP, PR, CUPTI, MARLIN, GREEN, VOLT) is already injection-native — only
those two subsystems needed re-architecture.

**The fix.** `cipher_rt_got_patch.{c,h}` — a runtime GOT/PLT patcher driven
from `InitializeInjection2`. `dl_iterate_phdr` walks every loaded ELF object;
`.rela.plt` (JUMP_SLOT) and `.rela.dyn` (GLOB_DAT) are parsed; registered
symbols are matched; `write_got_slot` handles full RELRO (reads
`/proc/self/maps`, `mprotect` RW → write → restore). Idempotent. The cuBLAS
shim and all three SDPA trampolines were re-targeted onto it; the `.symver`
alias and `cublas_version.map` version script were retired.

**Verified.** Standalone test (`got_patch_test/`, built full RELRO `-z now
-z relro`) — baseline-miss, patch-hit, idempotent, durable. In a live
PyTorch process under `CUDA_INJECTION64_PATH`-only: `GOT: patch applied —
8 slot(s) across 70 module(s)`, all interception restored (gate (a)).

---

## 2. Gate

All five criteria are measured under the rebuilt `libcipher_rt` (`c2c5d313`),
`CUDA_INJECTION64_PATH`-only end to end — no `LD_PRELOAD` anywhere in the
deploy path.

| # | Criterion | Result | Verdict |
|---|---|---|---|
| (a) | Interception parity under injection-only | cuBLAS shim `MATMUL calls=20` (audit baseline: 0); SDPA `tramp_calls=20 cudnn=20` (`cp25_sdpa_smoke.py`); GOT 8 slots / 70 modules | **PASS** |
| (b) | 3.617× reproduces `LD_PRELOAD`-free | composed (allon/vanilla) tok/W **3.602×**, 95 % CI **[3.529, 3.676]**, n=5, Mistral-7B B=1 — mean inside the CP 2.4 CI [3.591, 3.642], CIs overlap heavily | **PASS** |
| (c) | `.deb` install / uninstall | `dpkg -i` → smoke → `dpkg -r` → `dpkg -i` all rc=0; `/dev/cipher` 0666; uninstall leaves the node byte-clean | **PASS** |
| (d) | No regressions | CP 3.3 gate 4/4; T4.6.4 KV-dedup 5/5; alloc unit test ALL PASS; Fix A Marlin smoke clean; `/dev` nodes 0666 | **PASS** |
| (e) | Anchors held | kmod source unchanged (srcversion identical); libcipher_v2 `86618c30`; ABI + taint 12288 unchanged; libcipher_rt rebuilt → fresh anchor `c2c5d313` | **PASS** |

### (b) — the composed result, `LD_PRELOAD`-free

`run_composed_injection_only.sh` (derived from `cp_2_4/run_composed.sh`; the
**only** change is that the marlin/allon arms drop `LD_PRELOAD` and keep
`CUDA_INJECTION64_PATH` — the CP 2.4 script is preserved untouched as the
baseline-comparison artifact). n=5 matched pairs × 3 arms (vanilla / marlin /
all-on), Mistral-7B B=1, VOLT_MHZ=1000.

| Lift (tok/W) | CP 2.5 injection-only | CP 2.4 LD_PRELOAD baseline |
|---|---|---|
| marlin / vanilla | 1.296× [1.288, 1.304] | 1.307× |
| allon / marlin | 2.780× [2.720, 2.840] | 2.768× |
| **allon / vanilla (composed)** | **3.602× [3.529, 3.676]** | **3.617× [3.591, 3.642]** |

tok/s composition is clean and matches CP 2.4 within noise: marlin/vanilla
1.079×, allon/marlin 1.653× (independent Mistral n-gram spec arm: 1.639×),
allon/vanilla 1.784×. The 3.617× **reproduces** `LD_PRELOAD`-free —
3.602× is statistically indistinguishable from the v2-deploy number. Per
memo §5.2, no honest-gap statement is needed.

### (d) — regression checks (`gate_d/`)

All run under libcipher_rt `c2c5d313`; CP 3.3 and the Fix A smoke under
`CUDA_INJECTION64_PATH`-only.

- **CP 3.3 gate 4/4** — `flop_gate`, injection-only. The GOT path is
  *exercised*, not bypassed: the log carries `GOT: patch applied` and
  `CUBLAS-SHIM: real cublasGemmEx resolved`. (a) ring=256 PASS · (b) 3/3
  tenants MFU>0 PASS · (c1) r_util **0.9809** (baseline 0.9811) PASS · (c2)
  ratio **1.170** (baseline 1.169) PASS.
- **T4.6.4 KV-dedup 5/5** — (a) 16 runs, **15,500 pages cross-process
  verified, 0 fail**; (b) tenant teardown / release fop PASS; (c) slab
  regression = alloc unit test (below); (d) 4-tenant churn PASS; (e) module
  unload safety — `rmmod` refused with a tenant fd open, succeeded at
  refcount 0, re-`insmod` clean.
- **alloc unit test** — `=== ALL PASS ===`, rc=0. The binary emits 15 `ok:`
  assertions; the T4.6.4 report labelled this "14/14" — a count-label drift
  in the harness, **not a regression** (all assertions pass either way).
- **`/dev/cipher` + `/dev/cipher_kvdedup`** — both `crw-rw-rw-` (0666),
  re-verified fresh after the (e) re-`insmod`.
- **Fix A Marlin smoke** — M=32 GEMM, `CIPHER_MARLIN=on`, injection-only:
  MARLIN engine init OK, actuator ENABLED, NVRTC cubin 10/10, **MATMUL
  calls=20 handled=17 passthrough=3** — Marlin executed in the device-0
  primary context (Fix A), **clean exit, no hang** (the prior bug deadlocked
  here).

### (c) — `cipher-platform.deb`

`cipher-platform_1.0_amd64.deb` (md5 `bb924248`, 502 KB):

- `/usr/lib/cipher/` — `libcipher_rt.so`, `libcipher_v2.so`, `libc10.so`.
- `/usr/bin/cipher-run` — launcher; sets `CUDA_INJECTION64_PATH` and prepends
  `LD_LIBRARY_PATH=/usr/lib/cipher` so libcipher_rt's `DT_NEEDED libc10.so`
  resolves to the bundled copy on a node without PyTorch.
- `/usr/src/cipher-kmod-0.4.8/` — DKMS source package (D3); `postinst` runs
  `dkms add/build/install` + `modprobe`, `prerm` runs `modprobe -r` +
  `dkms remove`. Both idempotent.

Install cycle on the live pod (`install_cycle.log`): `dpkg -i` (DKMS builds
against `6.8.0-1046-nvidia`) → installed-mode smoke `cipher-run python3
cp25_smoke.py` (GOT 8 slots, `MATMUL calls=20`, rc=0) → `dpkg -r` (module
unloaded, DKMS deregistered, `/dev/cipher` gone, `/usr/lib/cipher` empty —
byte-clean) → `dpkg -i` again (idempotent, smoke rc=0). DKMS emits a benign
Secure-Boot signing warning — the node has no Secure Boot; the module
installs and loads regardless.

**Torch-free resolution — measured, not asserted.** The "self-contained, no
tenant-side PyTorch" claim was verified directly: under `cipher-run`'s
`LD_LIBRARY_PATH=/usr/lib/cipher`, `ldd /usr/lib/cipher/libcipher_rt.so`
resolves `libc10.so => /usr/lib/cipher/libc10.so`, and in a live process
that never imports torch, `/proc/self/maps` shows `libc10` mapped from
`/usr/lib/cipher/libc10.so` — the **bundled** copy, not torch's. The
build-tree `libcipher_rt` carries `DT_RUNPATH` into the torch tree; because
`LD_LIBRARY_PATH` is searched before `DT_RUNPATH`, `cipher-run` deterministically
binds the bundled core on a node with no PyTorch installed.

---

## 3. Audit-revealed refinement of D2(iii) — the bare-drop finding

This section records a mid-build empirical correction, because the *way* it
was handled is the point.

**The pre-memo assumption.** D2(iii) — "sever the libtorch dependency" — was
written on the assumption (memo §0.3, Makefile comment) that the attn
translation unit referenced *only inline* ATen symbols, so dropping all torch
libraries from the `libcipher_rt` link would leave no undefined symbols.

**The assumption was wrong.** The bare-drop build (`2f845393`) carried **8
undefined `c10::` symbols** — the attn TU's inline ATen header code calls
out-of-line c10 *core* functions (`UndefinedTensorImpl::_singleton`,
`throwNullDataPtrError`, `materialize_cow_storage`, `torchCheckFail`, …).
Zero `at::` / libtorch_cpu symbols — all 8 were `c10::`.

**It was tested, not assumed.** Three options were characterised against
*measured* constraints, not guesses:
- The driver tolerates an injection-lib load failure (bogus path and the
  bare-drop lib both gave `cuInit rc=0`, graceful skip) — so a broken
  injection lib fails *silently*.
- **The bare-drop `libcipher_rt` fails to load even inside a PyTorch
  process** — `undefined symbol: c10::UndefinedTensorImpl::_singleton`. It is
  a *data* symbol (resolved eagerly, not lazy-deferrable), and libc10 sits in
  an RTLD_LOCAL scope that a separately `dlopen`'d injection library cannot
  see. Option (C) bare-drop was therefore **dead** — it breaks the gate, not
  merely non-PyTorch processes.
- Option (B) weak stubs — rejected: the RTLD_LOCAL finding makes runtime
  interposition fragile and silent-failing.

**The chosen option (A), on the measured boundary.** Link `-lc10` (+ rpath);
bundle `libc10.so` in the `.deb`. libc10 is **1.5 MB**, soname `libc10.so`
(unversioned), an ABI-stable core. The heavy, version-fragile **`libtorch_cpu`
(451 MB) is severed** — which is D2(iii)'s *real* intent: do not couple the
platform `.deb` to the tenant's heavy torch build. D2(iii) thus holds, on the
component that actually mattered; (ii)'s bundling is applied only to the
1.5 MB stable remainder. The memo §6 D2 addendum records this refinement.

**The pattern.** The assumption was an audit guess; it was *characterised
empirically before any claim was made* — the bare-drop was built, its failure
mode was reproduced and root-caused (data symbol, RTLD_LOCAL scope), all three
options were documented with evidence, and the decision followed the measured
constraint. The broken bare-drop artifact is preserved
(`libcipher_rt.so.cp2_5_baredrop_BROKEN`, `2f845393`) as evidence. This is the
engineering-marvel epistemic posture: **characterize the boundary
empirically; do not assume it.**

---

## 4. Anchors

| Component | Anchor | Status |
|---|---|---|
| kmod | source unchanged — srcversion `E427CAFA4E94D548233DC7A` | reference build `e2f50452`; CP 2.5 touched no kmod code |
| libcipher_v2 | `86618c30` (`libcipher_v2.so.v0.2.0`) | unchanged; bundled in the `.deb` |
| libcipher_rt | `5e304549` → **`c2c5d313`** | **new CP 2.5 anchor**, supersedes `5e304549` |
| ABI / taint | `/dev/cipher` ioctl ABI unchanged; taint 12288 | unchanged |

**kmod anchor nuance (gate (e)).** The DKMS-rebuilt, node-local
`cipher_kmod.ko` carries md5 `6654d9e5`, not the reference `e2f50452`. The
**srcversion is identical** — `E427CAFA4E94D548233DC7A` on both. The bytes
differ only because DKMS strips debug info (2.14 MB → 123 KB) and
Secure-Boot-signs the result. Same source → same module; the `e2f50452`
anchor stands as the canonical reference build, and the `.deb` rebuilds that
exact source per node (D3 = DKMS).

**libcipher_v2 housekeeping note.** The `.deb` bundles the **anchored**
`libcipher_v2.so.v0.2.0` (`86618c30`). A later, un-anchored
`libcipher_v2.so` (`cc0479b8`) also exists on disk; it is **Phase 2
housekeeping — not blocking CP 2.5** and is flagged for later reconciliation.

---

## 5. Artifacts

| Artifact | md5 / status |
|---|---|
| `cipher_rt_phase4/libcipher_rt.so` | **`c2c5d313e2c24b687ba344cd9fe14a2f`** — CP 2.5 anchor |
| `cipher_rt_phase4/cipher_rt_got_patch.{c,h}` | GOT/PLT patcher (new) |
| `cipher-platform_1.0_amd64.deb` | `bb92424803a04bf989e92d48d2f76336` |
| `libcipher_rt.so.cp2_5_baredrop_BROKEN` | `2f845393…` — dead bare-drop, preserved as evidence |
| `libcipher_rt.so.pre_cp2_5` | `5e304549…` — rollback point |
| `composed_injection_only_result.json` | composed 3.602× tok/W [3.529, 3.676] |
| `audit/SMOKE_RESULT.md`, `audit/GOT_RELOC_CHECK.txt` | pre-build audit |
| `cp25_sdpa_smoke.py`, `cp25_loadtest_nontorch.py` | load + interception smokes |
| `gate_d/` | CP 3.3 gate, KV-dedup, alloc, Fix A smoke, module-unload logs |
| `install_cycle.log` | full `dpkg` lifecycle transcript |
| `got_patch_test/` | standalone RELRO-aware GOT patcher test |

---

## 6. Close

CP 2.5 retired the per-job `LD_PRELOAD`: the CIPHER runtime now deploys via
`CUDA_INJECTION64_PATH` alone, with GOT/PLT patching restoring the cuBLAS and
SDPA interception that link-order interposition used to provide. The 3.617×
composed tok/W result **reproduces** `LD_PRELOAD`-free at 3.602×
[3.529, 3.676] — within the CP 2.4 CI. The platform ships as a self-contained
`cipher-platform.deb` (DKMS kmod, bundled runtime + c10 core, `cipher-run`
launcher) with a clean install / uninstall / reinstall cycle on the live pod.
All five gate criteria pass; no regressions; anchors held.

The one mid-build correction — the D2(iii) bare-drop assumption — was caught
by building and testing the bare-drop rather than trusting the audit guess,
and resolved on the measured boundary (sever the 451 MB version-fragile
`libtorch_cpu`, retain the 1.5 MB ABI-stable `libc10`). §3 records it in full.

**Recommendation:** close CP 2.5 (Phase 2: 14/23 → **15/23**). Awaiting user
adjudication.
