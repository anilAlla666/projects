# CP 2.5 — build progress (atomic STEP across sessions)

CP 2.5 — drop LD_PRELOAD for CUDA_INJECTION64_PATH-only deployment; ship
`cipher-platform.deb`. Phase 2 close (14/23 → 15/23). One atomic STEP; the
single `CP_2_5_REPORT.md` lands when all gate criteria are measured. This
file is the durable cross-session checkpoint — not the report.

---

## Session 2026-05-16 — STARTED

### Done
- **Audit-before-build.** Examined the v2 stack's interception architecture
  (`cipher_inject.c`, `cipher_rt_cublas_shim.c`, `cipher_rt_attn_dispatch.cpp`,
  Makefile, `cublas_version.map`). Two independent mechanisms: (A) the
  `InitializeInjection2` actuator-init chain — CUDA_INJECTION64_PATH-native;
  (B) link-order symbol interposition (cuBLAS `.symver` shim + ATen SDPA
  trampolines) — LD_PRELOAD-dependent.
- **Audit smoke** — `audit/cp25_smoke.py`, `audit/SMOKE_RESULT.md`.
  CUDA_INJECTION64_PATH-only: cuBLAS shim `calls=0` (never fires), attn
  `tramp_calls=0`; the InitializeInjection chain runs fully. With LD_PRELOAD:
  shim `calls=20` (every GEMM routed). **Decisive: injection-only silently
  loses the cuBLAS shim + attn substrate** — the CP 2.4 3.617× would not
  reproduce.
- **GOT-reloc viability check** — `audit/GOT_RELOC_CHECK.txt` (`objdump -R`):
  `libtorch_cuda.so` has `R_X86_64_JUMP_SLOT` relocs for `cublasGemmEx` +
  `cublasGemmStridedBatchedEx`; `libtorch_cpu.so` has them for all 3 hooked
  SDPA `::call` symbols. **GOT patching is viable for BOTH substrates** — the
  positive result (smoke proved the negative; this proves recovery).
- **`CP_2_5_DESIGN_MEMO.md` written** — audit findings, the GOT/PLT-patching
  re-architecture design, `cipher-platform.deb` structure, gate criteria
  (a)–(e), honest calendar (~1.5 weeks, not 3–5 days — the GOT-patch
  re-architecture is the body of the work), 4 open decisions.

### Awaiting
**User adjudication of design memo §6 decisions 1–4** before any code:
1. Interception re-architecture: (A) GOT/PLT patching [recommended] /
   (B) `/etc/ld.so.preload` fallback / (C) packaging-only reframe.
2. libtorch dependency in the `.deb`.
3. kmod packaging: prebuilt `.ko` vs DKMS.
4. attn substrate — migrate now [recommended] or defer.

No code, no packaging, no rebuild until the memo is approved.

### Anchors — held
kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`, libcipher_rt `5e304549`
(rollback point; GOT-patch build will record a fresh anchor), taint 12288.
No ABI change in scope.

---

## Session 2026-05-16 (cont.) — BUILD STARTED (memo approved)

User adjudicated memo §6: D1=(A) GOT/PLT patching · D2=(iii) sever libtorch ·
D3=(ii) DKMS · D4=migrate attn now. Calendar ~1.5 weeks acknowledged; CP 2.5
may land just after the May 28 demo (demo material = CP 2.4 + T4.6.4 +
CP 3.3/3.4, already done). Build STEP: 9 items, GOT patcher first.

### Build item 1 — GOT patcher — DONE + VERIFIED
- **`cipher_rt_phase4/cipher_rt_got_patch.{c,h}`** written. `dl_iterate_phdr`
  walks every loaded ELF object; parses `.rela.plt` (JUMP_SLOT) + `.rela.dyn`
  (GLOB_DAT); matches registered symbols; `write_got_slot` handles RELRO
  (reads `/proc/self/maps`, mprotect RW → write → restore original prot).
  Idempotent (slot==trampoline → skip); `cipher_rt_got_register` /
  `_apply` / `_init` API. Real-fn resolution deliberately left to the
  substrates (dlopen+dlsym) — saved-slot value is a lazy-PLT hazard.
- **Standalone test** `cp_2_5/got_patch_test/` — `got_patch_test.c`, no CUDA,
  built **full RELRO** (`-z now -z relro`, confirmed `BIND_NOW`+`GNU_RELRO`).
  **PASS**: baseline-miss, patch-hit, idempotent (re-apply 0 slots), durable.
  Clean build, no warnings. Evidence: `got_patch_test/NOTE.md`, `test_run.log`.
  md5: got_patch.h `ff87a54a`, got_patch.c `0c4251b7`.

### Build items 2–5 — DONE (with one open fork — see below)
- **Item 2 — cuBLAS shim** re-targeted: `.symver` + `cublasGemmEx` alias
  removed; `cipher_rt_cublas_shim_register_got()` registers
  `cublasGemmEx → cipher_rt_cublasGemmEx_impl`. Real-fn resolution unchanged
  (dlopen libcublas + dlvsym).
- **Item 3 — attn substrate** re-targeted: `cipher_rt_attn_register_got()`
  registers the 3 SDPA `::call` trampolines; trampolines made HIDDEN
  (pragma); `resolve_lazy` switched `dlsym(RTLD_NEXT)` →
  `dlopen("libtorch_cpu.so",RTLD_NOLOAD)+dlsym` (verified: soname clean,
  resolves).
- **Item 5 — `cipher_inject.c`** wired: registrations + `cipher_rt_got_patch_init()`
  appended to the init chain. Makefile: `cipher_rt_got_patch.o` added,
  `--version-script` dropped, `$(TORCH_LIBS)` dropped.
- **Rebuild — SUCCEEDED.** libcipher_rt `5e304549` → **`2f845393`** (rollback
  saved: `/home/ubuntu/libcipher_rt.so.pre_cp2_5` = `5e304549`). Verified:
  no libtorch/libc10 in DT_NEEDED; `cublasGemmEx` + ATen SDPA NOT exported;
  `InitializeInjection` + GOT-patcher symbols exported.

### OPEN FORK — item 4 (sever libtorch): bare-drop FAILED, needs adjudication

Dropping all torch libs left **8 undefined `c10::` symbols** (the attn TU's
*inline* ATen header code calls out-of-line c10 core fns:
`UndefinedTensorImpl::_singleton` [**data**], `throwNullDataPtrError`,
`materialize_cow_storage`, `c10::detail::torchCheckFail`, …). The pre-memo
assumption ("attn TU references only inline symbols") was wrong — all 8 are
`c10::`, **zero `at::` / libtorch_cpu** symbols.

**Empirically tested (not assumed):**
- Driver tolerates injection-lib load failure — bogus path + bare-drop lib
  both give `cuInit rc=0` (graceful skip).
- **Bare-drop libcipher_rt (`2f845393`) FAILS TO LOAD even in a PyTorch
  process** — `undefined symbol: c10::UndefinedTensorImpl::_singleton`. It is
  a *data* symbol (resolved eagerly, can't be lazy-deferred), and libc10 sits
  in an RTLD_LOCAL scope (torch loads it there) invisible to a separately
  dlopen'd libcipher_rt. → injection silently does not happen. **Option (C)
  bare-drop is DEAD** — it breaks the gate, not just non-PyTorch processes.

**Resolution — recommend (A):** link `-lc10` (+ `-rpath`); the `.deb`
bundles libc10.so. libc10 = **1.5 MB**, soname `libc10.so` (unversioned),
ABI-stable core. The heavy version-fragile **libtorch_cpu (451 MB) IS
severed** — D2(iii)'s real intent (don't couple to the tenant's heavy torch
build) is met; only the small stable core is retained + bundled.
(B) weak stubs — rejected: the RTLD_LOCAL finding makes runtime
interposition fragile, silent-failure mode.

**State:** active `libcipher_rt.so` restored to the working rollback
`5e304549`; broken bare-drop kept as `libcipher_rt.so.cp2_5_baredrop_BROKEN`
(`2f845393`). All source edits (items 2,3,5) are correct and in place —
only the Makefile `TORCH_LIBS` line needs the `-lc10` fix, then rebuild.
**AWAITING user adjudication of (A).**

### OPEN FORK — RESOLVED: user adjudicated (A), 2026-05-16

User approved (A): link `-lc10`, bundle `libc10.so` in `cipher-platform.deb`.
D2(iii) interpretation refined post-build-empirics — memo §6 D2 addendum
written: sever `libtorch_cpu` (451 MB, version-fragile), retain `libc10`
(1.5 MB, unversioned soname, ABI-stable core) via the linker.

### Item 4 (A) fix + rebuild — DONE + VERIFIED

- **Makefile fix applied.** `C10_LIBS := -Wl,-rpath,$(TORCH_LIB) -L$(TORCH_LIB)
  -lc10`; `$(TARGET)` link line `$(OBJS) $(LIBS) $(C10_LIBS) -lcrypto`. TORCH
  comment block updated (libc10 IS linked, libtorch_cpu is not).
- **Clean rebuild SUCCEEDED.** `make clean && make`, no errors (only
  pre-existing benign warnings: unused params in torch headers, unused
  `env_on`). New **libcipher_rt anchor `c2c5d313`** (`5e304549` → `c2c5d313`;
  bare-drop `2f845393` is the dead intermediate). md5 file:
  `cipher_rt_phase4/libcipher_rt.so.cp2_5.md5`.
- **Verify 1 — anchor.** `c2c5d313e2c24b687ba344cd9fe14a2f`.
- **Verify 2 — DT_NEEDED.** `libc10.so` present; **no `libtorch_cpu`**
  (`readelf -d` grep torch_cpu → none). NEEDED set: libcupti, libcuda,
  **libc10.so**, libcrypto, libstdc++, libgcc_s, libc, ld-linux.
- **Verify 3 — 8 c10:: symbols resolve.** All 8 undefined `c10::` symbols
  (incl. the data symbol `UndefinedTensorImpl::_singleton`) confirmed
  `--defined-only` in `libc10.so`; `ldd` resolves `libc10.so` via rpath;
  standalone `ctypes.CDLL` dlopen of libcipher_rt succeeds — no undefined
  symbol at load.
- **Verify 4 — loads in both process types** (`cp_2_5/cp25_loadtest_nontorch.py`
  + `audit/cp25_smoke.py`, no LD_PRELOAD, CUDA_INJECTION64_PATH only):
  - **non-PyTorch** CUDA process: full InitializeInjection chain runs,
    `cuInit rc=0 device_count=1`, GOT patch 0 slots/20 modules (correct — no
    cublas/torch GOT slots present), libcipher_rt loaded (vs bare-drop which
    failed even in PyTorch).
  - **PyTorch** process: full chain runs, **`GOT: patch applied — 8 slot(s)
    across 70 module(s)`**, `CUBLAS-SHIM: real cublasGemmEx resolved`,
    `MATMUL: exit totals — calls=20 ... passthrough=20` — **all 20 GEMMs
    routed through the shim under CUDA_INJECTION64_PATH-only** (vs audit
    ARM 1 `calls=0`). The GOT re-architecture works; gate (a) mechanism
    confirmed.
- **Verify 5 — broken bare-drop preserved** as
  `cipher_rt_phase4/libcipher_rt.so.cp2_5_baredrop_BROKEN` (`2f845393`).

### SDPA substrate — GOT interception verified under injection-only

`cp_2_5/cp25_sdpa_smoke.py` (20× `F.scaled_dot_product_attention`,
CUDA_INJECTION64_PATH only): **`[cipher-attn] tramp_calls=20 ... cudnn=20`** —
all 20 SDPA calls routed through the GOT-patched trampolines (cuDNN path, as
expected on H100/cu13). `handled=0 passthrough=20` — correct, attn is
PASSTHROUGH-only until T4.6.3. Closes Test C's matmul-only gap.

**Target count clarified:** `4 target(s) registered` = 1 cuBLAS
(`cublasGemmEx`) + 3 SDPA. `cublasGemmStridedBatchedEx` has a GOT reloc
(audit `GOT_RELOC_CHECK.txt`) but is **not registered** — that is parity with
the CP 2.4 LD_PRELOAD baseline (the `.symver` shim only ever interposed
`cublasGemmEx`; StridedBatched was never hooked). Not a regression.

### Item 6 — composed gate, injection-only — LAUNCHED (background)

`cp_2_5/run_composed_injection_only.sh` (derived from `cp_2_4/run_composed.sh`,
**only change**: marlin/allon arms drop `LD_PRELOAD`, keep
`CUDA_INJECTION64_PATH`). CP 2.4 script left untouched as the
baseline-comparison artifact. n=5 pairs × 3 arms, Mistral-7B B=1,
VOLT_MHZ=1000. Analysis: `analyze_composed_injection_only.py` →
`composed_injection_only_result.json`. Pass: composed (allon/vanilla)
tok/W ≥ 3.5×. Runlog: `composed_injection_only.runlog`.

### Item 8 test-env decision — user adjudicated 2026-05-16

**Live pod, dpkg cycle.** `dpkg -i` / `dpkg -r` / `dpkg -i` again on this pod —
install, idempotency, clean uninstall. postinst/prerm (item 7) designed for
this: DKMS build against the running kernel `6.8.0-1046-nvidia`.

### Item 7 — cipher-platform.deb — BUILT

`cp_2_5/cipher-platform_1.0_amd64.deb` (md5 `bb924248`, 502 KB). Built with
`fakeroot dpkg-deb --build`. Layout:
- `/usr/lib/cipher/` — `libcipher_rt.so` (`c2c5d313`), `libcipher_v2.so`
  (`86618c30`), `libc10.so` (`b814a98f`, bundled — D2(iii) addendum).
- `/usr/bin/cipher-run` — launcher; sets `CUDA_INJECTION64_PATH` +
  prepends `LD_LIBRARY_PATH=/usr/lib/cipher` so libcipher_rt's DT_NEEDED
  `libc10.so` resolves to the bundled copy on a torch-free node.
- `/usr/src/cipher-kmod-0.4.8/` — DKMS source pkg (kmod .c/.h + Kbuild +
  Makefile + `dkms.conf`); kbuild-generated `.mod.c` and the standalone
  `probe_microbench.c` excluded.
- `DEBIAN/postinst` — `dkms add/build/install` + `modprobe` + /dev/cipher
  fallback mknod; `DEBIAN/prerm` — `modprobe -r` + `dkms remove` + node
  cleanup. Both idempotent.
- `control` — `Depends: dkms, libc6, libstdc++6, libgcc-s1`;
  `Recommends: linux-headers-generic`.

**Note — libcipher_v2 source file:** the anchored build is
`libcipher_v2/libcipher_v2.so.v0.2.0` (`86618c30`); the plain
`libcipher_v2/libcipher_v2.so` on disk is a later un-anchored build
(`cc0479b8`). The .deb bundles the **anchored 86618c30** per the held
anchor. (libcipher_rt has no DT_NEEDED on libcipher_v2; v2 is bundled per
the item-7 spec.)

### Item 6 — composed gate, injection-only — DONE — **PASS**

n=5, Mistral-7B B=1, libcipher_rt `c2c5d313`, CUDA_INJECTION64_PATH-only
(no LD_PRELOAD). `composed_injection_only_result.json`:
- **composed (allon/vanilla) tok/W = 3.602× — 95% CI [3.529, 3.676]**
- CP 2.4 baseline: 3.617×, CI [3.591, 3.642]. CP 2.5 mean 3.602 is **inside**
  the CP 2.4 CI; CIs overlap heavily. **Gate (b) PASS** — 3.617× reproduces
  LD_PRELOAD-free, indistinguishable from the v2-deploy number.
- tok/s composition clean: marlin/vanilla 1.079×, allon/marlin 1.653×
  (CP 2.4 1.647×; independent Mistral n-gram spec 1.639×), allon/vanilla
  1.784× (CP 2.4 1.795×). All within noise.
- Gate (a) parity already shown (MATMUL calls=20, SDPA tramp_calls=20).

### Next — sequencing (user-confirmed 2026-05-16)

**item 6 → gate (d) regression checks → item 8 dpkg cycle → item 9 report.**

**gate (d) — regression checks — DONE — ALL PASS** (`cp_2_5/gate_d/`):
- [x] **CP 3.3 gate 4/4** — flop_gate under `c2c5d313` injection-only;
  GOT engaged (`GOT: patch applied`, `CUBLAS-SHIM resolved` in log — not
  vacuous). a/b/c1/c2 all PASS; r_util 0.9809 (baseline 0.9811), ratio
  1.170 (1.169). `gate_d/cp33/`.
- [x] **T4.6.4 KV-dedup 5/5** — `kvdedup_xproc`: (a) 16 runs, 15500 pages
  cross-process verified, 0 fail; (b) teardown PASS; (d) 4-tenant churn
  PASS. (c) = alloc_unit_test. (e) = module unload. `gate_d/kvdedup_xproc.log`.
- [x] **alloc_unit_test** — `=== ALL PASS ===` rc=0. Binary emits 15 `ok:`
  assertions; the T4.6.4 report labelled it "14/14" — count-label drift in
  the harness, **not a regression**. `gate_d/alloc_unit_test.log`.
- [x] **/dev/cipher + /dev/cipher_kvdedup 0666** — both `crw-rw-rw-`
  (511,0 / 510,0), confirmed fresh after the (e) re-insmod.
- [x] **Fix A Marlin smoke** (injection-only) — M=32 GEMM, `CIPHER_MARLIN=on`:
  MARLIN engine init OK, actuator ENABLED, NVRTC cubin 10/10, **MATMUL
  calls=20 handled=17 passthrough=3** — Marlin executed in the primary
  context, **clean exit, no hang** (the prior bug hung here).
  `gate_d/marlin_fixA_smoke.log`.
- [x] **(e) module unload safety** — rmmod refused with a tenant fd open;
  succeeded at refcount 0; re-insmod (anchor `e2f50452`) clean; both /dev
  nodes back at 0666. `gate_d/module_unload_e.log`.

### Item 8 — dpkg install cycle — DONE — **PASS** (`cp_2_5/install_cycle.log`)

`run_install_cycle.sh` on the live pod — `dpkg -i` → smoke → `dpkg -r` →
`dpkg -i` → smoke. All rc=0:
- **install**: DKMS add/build/install of cipher-kmod 0.4.8 against
  `6.8.0-1046-nvidia`; files in /usr/lib/cipher; cipher-run present.
- **smoke (installed mode)**: `cipher-run python3 cp25_smoke.py` — GOT patch
  8 slots, CUBLAS-SHIM resolved, MATMUL calls=20. rc=0.
- **uninstall**: module unloaded, DKMS deregistered, /dev/cipher absent,
  /usr/lib/cipher empty, cipher-run gone — node byte-clean.
- **reinstall**: idempotent — rebuilt, /dev/cipher back `crw-rw-rw-`,
  smoke rc=0.
- Benign: DKMS emits a Secure-Boot signing warning (`/sys/firmware/efi/
  efivars not found`) — node has no Secure Boot; module installs and loads
  regardless.

### kmod anchor — gate (e) nuance (verified)

The DKMS-rebuilt node-local `cipher_kmod.ko` md5 is `6654d9e5`, **not** the
reference anchor `e2f50452`. **srcversion is identical** —
`E427CAFA4E94D548233DC7A` on both. The .ko bytes differ only because DKMS
**strips debug info** (2.14 MB → 123 KB) and **Secure-Boot-signs** the
result. CP 2.5 touched **zero kmod code**; same source → same srcversion →
same module. The `e2f50452` anchor stands as the canonical reference build;
the .deb rebuilds that exact source per-node (D3 = DKMS).

### Item 9 — `CP_2_5_REPORT.md` — DONE

`CP_2_5_REPORT.md` written. All five gate criteria PASS: (a) parity,
(b) 3.602× composed, (c) install/uninstall, (d) regressions, (e) anchors.
Includes the dedicated §3 "Audit-revealed refinement of D2(iii)" section and
the libcipher_v2 housekeeping note.

## CP 2.5 — STEP CLOSED (2026-05-16)

All 9 build items done; 5/5 gate criteria PASS. CP 2.5 closes as a full
close — Phase 2: 14/23 → **15/23**. Awaiting user adjudication.

Anchors at close: kmod 0.4.8 (srcversion `E427CAFA…`, reference build
`e2f50452`), libcipher_v2 `86618c30`, **libcipher_rt `c2c5d313`** (new CP 2.5
anchor), taint 12288, ABI unchanged.

### Report requirements (user-confirmed)
- Dedicated section: **"Audit-revealed refinement of D2(iii)"** — the
  pre-memo assumption that the attn TU referenced only inline ATen symbols
  was wrong; the bare-drop was empirically tested; three options
  characterized with evidence; link-`libc10` chosen on measured constraints.
  Frame as the engineering-marvel pattern: characterize boundaries
  empirically before claiming.
- libcipher_v2 note: the .deb bundles the **anchored `86618c30`**
  (`libcipher_v2.so.v0.2.0`); the later un-anchored on-disk
  `libcipher_v2.so` (`cc0479b8`) is flagged as **Phase 2 housekeeping, not
  blocking CP 2.5**.

### Anchors — held (user-confirmed)
kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`, **libcipher_rt `c2c5d313`**
(new CP 2.5 anchor, supersedes `5e304549`). No ABI change.
