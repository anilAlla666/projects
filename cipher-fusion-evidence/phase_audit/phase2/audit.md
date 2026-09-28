# CIPHER Phase 2 Evidence Audit — CP 2.1 through CP 2.5

**Audit date:** 2026-05-15
**Auditor scope:** read-only. Commands used: `ls`, `stat`, `md5sum`, `wc`, `grep`, `nm`, `readelf`, `tar -t`, `diff`, file reads. No code written, no workloads run.
**Pod:** Lambda H100 80GB SXM5, driver 580.105.08, kernel 6.8.0-1046-nvidia.

**Canonical Phase 2 plan** (verbatim from the canonical Phase 0–9 plan, recovered from
`/home/ubuntu/.claude/paste-cache/1e68679e0eca813b.txt` lines 61–69):

```
Phase 2 (CP 2.1 - CP 2.5):
  CP 2.1 libcipher_hook audit, port/keep/delete inventory
  CP 2.2 libcipher.so v2 skeleton via CUDA_INJECTION64_PATH
  CP 2.3 Tenant identity propagation, end-to-end accounting
  CP 2.4 Marlin + DVFS + speculative ported, 2.96x tok/W
  CP 2.5 LD_PRELOAD path dropped, cipher-platform.deb
```

---

## 1. Status table

| CP | One-line scope | STATUS |
|----|----------------|--------|
| CP 2.1 | libcipher_hook audit, port/keep/delete symbol inventory | **NOT DONE** |
| CP 2.2 | libcipher.so v2 skeleton loaded via CUDA_INJECTION64_PATH | **SHIPPED** |
| CP 2.3 | Tenant identity propagation, end-to-end accounting | **PARTIAL** |
| CP 2.4 | Marlin + DVFS + speculative ported, 2.96× tok/W reproduces through new dispatch | **NOT DONE** |
| CP 2.5 | LD_PRELOAD path dropped, cipher-platform.deb produced | **NOT DONE** |

**Headline:** 1 SHIPPED, 1 PARTIAL, 3 NOT DONE. Phase 2 delivered the *plumbing* (a working
CUDA-injection skeleton + kernel-rooted tenant identity) but did **not** deliver the
*payload* (perf actuators ported to the new dispatch) and did **not** package or cut over
(no `.deb`, LD_PRELOAD stack still present and still the only place perf was ever measured).

---

## CP 2.1 — libcipher_hook audit, port/keep/delete inventory — **NOT DONE**

### Gate intent
Produce an audit of the existing `libcipher_hook.so` LD_PRELOAD shim layer and an explicit
**port / keep / delete inventory**: for each intercepted symbol or shim (cuLaunchKernel,
cuLaunchKernelEx, cublasGemmEx, dispatch guard, etc.) decide whether it ports to the v2
injection model, stays as a dev-only shim, or is dropped.

### Evidence search
- `grep -rin "port.*keep|keep.*delete|libcipher_hook.*audit|symbol.*inventory"` across
  `/home/ubuntu/cipher-phase2-evidence/`, `/home/ubuntu/PHASE_1*.md`, `/home/ubuntu/PHASE_2*.md`
  → **zero hits.**
- `/home/ubuntu/cipher-phase2-evidence/` directory listing: no file named or scoped as a
  hook audit or inventory. Contents are the Phase 1.5 kmod build tree + the v2 lib + smoke logs.
- `libcipher_hook.so` itself exists only on the *old* op31-prod stack at
  `/home/ubuntu/cipher-may13-evidence/libcipher_hook.so` (md5 not load-bearing here; it is the
  audit *subject*, not an audit *artifact*). It was never inventoried in a Phase 2 document.

### What does exist (and why it is not the inventory)
`PHASE1_INSIGHT_REPORT.md` §3 and §6 (`/home/ubuntu/cipher-phase2-evidence/PHASE1_INSIGHT_REPORT.md`,
17821 bytes, mtime 2026-05-13 07:03) argue *strategically* that LD_PRELOAD should be replaced
by `CUDA_INJECTION64_PATH` and states "the existing `libcipher_hook.so` LD_PRELOAD path remains
for local development and CI but stops being the production deployment shape." That is a
**direction-setting paragraph**, not a symbol-by-symbol port/keep/delete inventory. No charitable
mapping: the gate explicitly requires an *inventory*, and no inventory artifact exists on disk.

### Verdict
**NOT DONE.** No evidence the libcipher_hook audit ran as a scoped deliverable.
What closes it: a document enumerating every `libcipher_hook.so` interception with a
port/keep/delete decision and rationale per entry.

---

## CP 2.2 — libcipher.so v2 skeleton via CUDA_INJECTION64_PATH — **SHIPPED**

### Gate intent
A v2 injection library that the CUDA driver loads via the documented `CUDA_INJECTION64_PATH`
environment variable at `cuInit()`, exposing the `InitializeInjection` entrypoint(s).

### Evidence on disk
- **Artifact:** `/home/ubuntu/libcipher_v2/libcipher_v2.so` — md5 `cc0479b836e560619e2b286ca1caecb7`,
  16968 bytes (current build), mtime 2026-05-13 10:05.
- **Evidence-bundle copy:** `/home/ubuntu/cipher-phase2-evidence/libcipher_v2/libcipher_v2.so` —
  md5 `86618c30896470b642fcc6985d8dc632`, 16496 bytes, mtime 2026-05-13 07:03. This is the
  Phase-2-era v0.2.0 build (identical md5 to `/home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0`).
  The two differ because the live `/home/ubuntu/libcipher_v2/` copy was rebuilt with the CUPTI
  path added (Phase 3 Task 5). The Phase-2 evidence bundle preserves the as-of-Phase-2 binary.
- **Source:** `cipher_inject.c` (1399 B), `cipher_tenant.c` (1636 B), `cipher_v2_internal.h`,
  `Makefile` — all present in `/home/ubuntu/cipher-phase2-evidence/libcipher_v2/`.
- **Exported symbols** (`nm -D /home/ubuntu/libcipher_v2/libcipher_v2.so`):
  `InitializeInjection` (T) and `InitializeInjection2` (T) — both driver-injection entrypoints.
- **Linking footprint** (`readelf -d`): `NEEDED libc.so.6`, `NEEDED libcupti.so.12`. (The
  Phase-2-era v0.2.0 build was pure-libc, 16 KB, per `PHASE_2_NOTES.md`; the cupti NEEDED
  appears in the post-Phase-2 rebuild.)
- **Load proof:** `/home/ubuntu/cipher-phase2-evidence/smoke_with_v2_inject.log` line 2–3:
  `[cipher_v2:dbg] init body running` / `[cipher_v2] tenant 'smoke-A' registered`. The
  injection library was demonstrably dlopen'd and its init body executed inside a real CUDA
  process. `PHASE_2_NOTES.md` integration ledger step 4 ("inject lib, no env var") and step 5
  both PASS.

### Verdict
**SHIPPED.** A v2 injection skeleton exists, exports the correct entrypoints, is loaded via
the documented mechanism, and is proven to run inside a CUDA process. The skeleton is genuinely
a *skeleton* — see CP 2.4 for what it does NOT contain.

---

## CP 2.3 — Tenant identity propagation, end-to-end accounting — **PARTIAL**

### Gate intent
Tenant identity must propagate end-to-end: from `CUDA_INJECTION64_PATH` userspace, through
`/dev/cipher`, into the kernel module's per-PID accounting, and surface in stats — *and*
"end-to-end accounting" implies per-tenant attribution of actual GPU work.

### Evidence — identity propagation (met)
- `cipher_tenant.c` (`/home/ubuntu/libcipher_v2/cipher_tenant.c`): reads `CIPHER_TENANT_ID`,
  opens `/dev/cipher`, issues `CIPHER_REGISTER_TENANT` (ioctl nr 1) with `gettid()`→pid /
  `getpid()`→tgid, closes. The gettid/getpid mapping is correct and documented (Finding 8).
- `PHASE_2_NOTES.md` integration ledger (`/home/ubuntu/cipher-phase2-evidence/PHASE_2_NOTES.md`,
  10444 B): **11 of 11 steps PASS**, including step 5 (env var set → `[cipher_v2] tenant
  'test-tenant-A' registered`; TENANT column populated in `/proc/cipher/stats`) and step 6
  (two concurrent tenants, both visible per-PID and per-TGID, no cross-tenant pollution).
- Kernel artifact: `cipher_kmod.ko` 0.2.0 in `/home/ubuntu/cipher-phase2-evidence/` (772456 B),
  with `cipher_dev.c` implementing `CIPHER_REGISTER_TENANT` and an anti-spoof check
  (validate-against-current per Finding 9).
- Smoke log confirms registration fires in a live workload.

### Evidence — end-to-end *accounting* (NOT met)
- The kmod 0.2.0 ABI has **only nr 1 live**; nrs 2/3/4 (`SNAPSHOT`, `RESET`, `GET_VERSION`)
  return `-ENOSYS` (`PHASE_2_NOTES.md` "Artifacts shipped"). There is **no per-tenant work
  accounting ioctl in Phase 2.**
- The launch-counting / accounting path (`cipher_cupti.c`, `CIPHER_SUBMIT_LAUNCH_STATS`,
  ioctl nr 7) is explicitly tagged in `cipher_v2_internal.h` as **"Phase 3 Task 5 (ioctl nr 7)"**
  and is not part of the Phase 2 kmod 0.2.0. The CUPTI source was added to the live
  `libcipher_v2/` tree *after* Phase 2 (it is why the live `.so` md5 differs from the
  Phase-2 evidence-bundle `.so`).
- The v2 smoke log teardown line `[CIPHER F1] Intercepts: 0 | Substitutions: 0` shows the
  injection library captured no per-tenant GPU-work attribution during the run.
- "Accounting" in Phase 2 is limited to a static TENANT *label* on per-PID/per-TGID rows of
  `/proc/cipher/stats`. Per-tenant launch/throughput/MFU attribution is deferred to Phase 3.

### Verdict
**PARTIAL.** Tenant *identity propagation* is fully shipped and proven (11/11, concurrent
multi-tenant, anti-spoofed). The *end-to-end accounting* half of the gate is not met: Phase 2
stamps a label but does not attribute GPU work to it; that is Phase 3 Task 5 work.
What closes it: per-tenant launch/work accounting wired and surfaced (the nr-7 CUPTI path),
which is post-Phase-2.

---

## CP 2.4 — Marlin + DVFS + speculative ported, 2.96× tok/W reproduces through new dispatch — **NOT DONE**

### Gate intent (binding, no charitable reading)
The three perf actuators — Marlin INT4 GEMM, DVFS adaptive clock, speculative decode — must
be **ported to the v2 `CUDA_INJECTION64_PATH` dispatch**, and the **2.96× tok/W** figure must
**reproduce through that new dispatch**.

### The 2.96× figure — it exists, but on the WRONG stack
The 2.96× tok/W number is real and is on disk:
- `/home/ubuntu/cipher-may13-evidence/SCORECARD.md` line 16:
  `tok/W (B=1) | 0.099 | 0.476 (1.62×) | 0.833 (2.84×) | 0.868 (2.96×)` and line 21:
  "vs system torch 2.7 baseline (0.293 tok/W) the full stack is 2.96× tok/W."
- `/home/ubuntu/cipher-may13-evidence/INVESTOR_REPORT.md` lines 11, 28:
  "2.96× tok/W vs FP16 baseline (Marlin + spec + DVFS, B=1)"; "0.868 tok/W ... +196% (2.96×)".

**But the SCORECARD header (lines 1–9) explicitly attributes this to the OLD stack:**
> "Libs: `libcipher_hook.so` + `libcipher_rt.so` rebuilt 2026-05-02 ... `CIPHER_DVFS` adaptive
> clock loop in the thermal-feedback sampler. Date: 2026-05-02"

So 2.96× was measured on the **LD_PRELOAD `libcipher_hook.so` + `libcipher_rt.so` stack**,
on **Llama-3.1-8B, B=1, 200-token decode**, dated **2026-05-02** — i.e. *before* Phase 2's
injection work and on the very stack Phase 2 was supposed to replace. It is NOT a v2-dispatch
result. (Search confirms: `grep -rn "2.96"` finds the figure only in those two op31-era
reports plus unrelated NVIDIA kernel-source files; no Phase 2 document reproduces it.)

### The v2 dispatch contains zero perf actuators
- `nm -D /home/ubuntu/libcipher_v2/libcipher_v2.so` exports only `InitializeInjection` /
  `InitializeInjection2`. No Marlin, DVFS, or speculative symbols.
- Source of the v2 lib is exhausted by three files: `cipher_inject.c` (entrypoints),
  `cipher_tenant.c` (ioctl nr 1 registration), `cipher_cupti.c` (Phase-3 launch counting).
  **No actuator code of any kind.** `cipher_v2_internal.h` describes the lib's entire job as
  (1) tenant stamp, (2) CUPTI launch count.
- The v2 smoke log (`smoke_with_v2_inject.log`) records `tps = 71.31` on **TinyLlama** with
  Marlin — but that Marlin is the *old* `libcipher_rt.so` LD_PRELOAD path (line 1:
  `loading rt from /workspace/libcipher_rt.so`; line 34: `[CIPHER F1] cuLaunchKernel hook
  installed`). The v2 injection library is loaded *alongside* the old LD_PRELOAD stack purely
  to register a tenant — it does not dispatch Marlin itself.
- The v2 smoke log reports **no tok/W at all** — only tps. There is no Phase 2 tok/W
  measurement, let alone a 2.96× one, through the new dispatch.
- `diff` of `smoke_with_v2_inject.log` vs `smoke_without_v2_inject.log`: the only delta is
  `tps 71.31` vs `71.29` — i.e. the v2 lib is a performance no-op (it only adds a tenant
  stamp). This is consistent with the gate NOT being attempted: nothing perf-bearing moved
  into the v2 path.

### Why this is not SUPERSEDED
A SUPERSEDED verdict would need substitute work that ports the actuators to the injection
model and reproduces (or beats) 2.96× through it. No such work exists. The post-Phase-2
`.symver` matmul-routing substrate and VOLT/Marlin actuators live in
`/home/ubuntu/cipher_rt_phase4/` and are framed as **Phase 4 / T4.x.x** work on a *different*
library (`libcipher_rt.so` rebuilt with `.symver` cuBLAS interposition), not on the
`CUDA_INJECTION64_PATH` v2 dispatch. They are parallel later-phase work, not a Phase 2
substitute, and they do not reproduce the 2.96× B=1 Llama-8B figure (per project memory,
T4.5 Marlin regressed on TinyLlama B=1). No honest equivalence argument can be made.

### Verdict
**NOT DONE.** The 2.96× tok/W figure is genuine but stranded on the pre-Phase-2 LD_PRELOAD
stack (op31-prod, 2026-05-02, Llama-3.1-8B). The v2 `CUDA_INJECTION64_PATH` dispatch is a
tenant-registration skeleton with **no Marlin, no DVFS, no speculative code** and **no tok/W
measurement of any kind**. The porting of perf actuators to the new dispatch was not
attempted. This is the central drift of Phase 2.

---

## CP 2.5 — LD_PRELOAD path dropped, cipher-platform.deb produced — **NOT DONE**

### Gate intent
The LD_PRELOAD deployment path must be *dropped*, and a packaged `cipher-platform.deb` must
be produced.

### Evidence — cipher-platform.deb (does not exist)
- `find / -name "*.deb" 2>/dev/null` → only `/var/cache/apt/archives/libssl3_*.deb` and
  `libssl-dev_*.deb` (stock Ubuntu system packages). **No `cipher-platform.deb`, no CIPHER
  `.deb` of any name, anywhere on the pod.**
- `grep -rln "cipher-platform"` → hits only inside chat-session cache files
  (`.claude/paste-cache/*`, `.claude/projects/*.jsonl`) — i.e. the *plan text itself* — never
  in any build script, Makefile, or artifact. No packaging was ever scripted.

### Evidence — LD_PRELOAD path dropped (not dropped)
- `libcipher_hook.so` (138576 B, mtime 2026-05-13 11:01) and `libcipher_rt.so` (785456 B)
  remain present and live at `/home/ubuntu/cipher-may13-evidence/`, with `.preroadmap`
  backups preserved.
- The Phase 2 v2 smoke run *itself depends on* the LD_PRELOAD stack: `smoke_with_v2_inject.log`
  line 1 is `loading rt from /workspace/libcipher_rt.so` and line 34 installs the
  `libcipher_hook.so` cuLaunchKernel hook. Far from being dropped, LD_PRELOAD is still the
  actuation path; v2 injection rides alongside it.
- `PHASE1_INSIGHT_REPORT.md` §6 explicitly states the LD_PRELOAD path "remains for local
  development and CI" — i.e. a deliberate decision NOT to drop it in Phase 2.

### Verdict
**NOT DONE.** No `cipher-platform.deb` exists on the pod. The LD_PRELOAD stack was not
dropped — it is preserved, live, and is still the only path through which perf was ever
measured. Neither half of the gate was met.
What closes it: a Debian package build (control file + `dpkg-deb`/`debuild` producing
`cipher-platform*.deb`) and a deployment cutover that retires the LD_PRELOAD libs from the
production path.

---

## 2. Phase 2 honest accounting — drift summary

Phase 2's canonical intent was a **cutover**: move CIPHER from the LD_PRELOAD shim model to
the NVIDIA-documented `CUDA_INJECTION64_PATH` injection model, carry the perf actuators across,
prove the 2.96× tok/W result still holds through the new path, package it as a `.deb`, and
retire LD_PRELOAD.

What actually shipped is the **first half of the plumbing only**:
- CP 2.2 (injection skeleton) — genuinely SHIPPED.
- CP 2.3 (tenant identity) — identity propagation SHIPPED, accounting deferred → PARTIAL.

What did **not** happen:
- CP 2.1 — the hook audit / inventory was never produced as an artifact.
- CP 2.4 — **no perf actuator was ported to the injection dispatch**; the 2.96× figure is
  real but lives entirely on the old LD_PRELOAD stack (2026-05-02, Llama-3.1-8B), and the v2
  path has no Marlin/DVFS/speculative code and no tok/W measurement.
- CP 2.5 — no `.deb` was ever built; LD_PRELOAD was explicitly *kept*, not dropped.

The perf-actuator work did not stop — it **drifted into Phase 3/4** under different framing:
the `.symver` cuBLAS-interposition substrate, VOLT DVFS, and Marlin actuator now live in
`/home/ubuntu/cipher_rt_phase4/` as T4.x.x deliverables on a *rebuilt `libcipher_rt.so`*, not
on the v2 injection library. So CIPHER kept doing perf work, but never on the dispatch path
Phase 2 was chartered to build, and never re-validated the 2.96× headline through it.

**Bottom line for Phase 2:** the injection/identity substrate is real and SHIPPED-grade; the
performance cutover that was the *point* of Phase 2 was not done, and the marquee 2.96× tok/W
number remains a pre-Phase-2 LD_PRELOAD result that has never been reproduced through the new
dispatch.
