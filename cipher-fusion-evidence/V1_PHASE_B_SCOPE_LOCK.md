# V1 Phase B scope-lock: cipher-platform.deb packaging path for plugin bundling

**Date:** 2026-05-26
**Pre-state anchor:** `v1-substrate-driver-worker-init` plus B.0 plus B.0.5
- `cipher_rt_phase4` `8613812e` tag `v1-substrate-driver-worker-init`
- `libcipher_rt.so` md5 `1d91e7da`
- `cipher_kmod` `8c643fc` tag `week-13-14-complete`
- `cipher_vllm_plugin/cipher_vllm_kv.py` md5 `b89a9b6e` (file on disk; reinstalled in vllm_env post-Phase-A)
- `cipher-fusion-evidence` `c52d2ab` (post-B.0.5 archaeology commit)

**Authority:**
- Anil 2026-05-26 V1 substrate work sequence Phase B + Anil 2026-05-26 B.0.5 surface adjudication selecting Option (ii) packaging path
- Anil 2026-05-26 Phase B B.0 + B.0.5 measurement directive
- `v1-goal5-contract-lock` memory (2026-05-26 Goal 5 contract preserved at apt-install + env-var deployment tier)

**Type:** packaging-and-deployment work; bundle libcipher_rt.so + cipher_vllm_plugin into a single OS-package-level dependency (cipher-platform.deb) so the customer-visible deployment surface is OS-package install + env var, the plugin is substrate-internal implementation detail. Tag `v1-substrate-platform-package` reserved for B.5 close.

---

## 1. Why this campaign

Per V1 substrate work sequence locked 2026-05-26, Phase B's goal was to move cipher_vllm_plugin functionality into libcipher_rt.so so customers get plugin behavior without pip-installing cipher-vllm-kv. B.0 measurement and B.0.5 archaeology established:

- Track 2 SC6 76% N=4 Mistral-7B weight savings reproduces at substrate-only (plugin uninstalled). The 76% is the cipher_kv_bridge weight-arena chain (Track 2), NOT the CP 5.1 KV-cache hook chain. B.0 evidence: `v1_phase_b/results/b0_substrate_only_baseline.json`.

- Week 5 cross-tenant KV-dedup measurements (45 GiB N=4 TinyLlama HBM saved per `week5-complete` Step 3.B, 83% hit rate Mistral-7B N=4, KL=0 bit-identity across 6 pairs) ARE plugin-load-bearing. `cipher_vllm_kvdedup.py:7-9` docstring is the authoritative chain: "walks the CIPHER-VMM-backed KV pages allocated by CP 5.1's `cipher_vllm_kv` ... calls `cipher_rt_kv_dedup_alias` on each 2 MiB page." B.0.5 evidence: `v1_phase_b/results/b0_5_plugin_hook_archaeology.md`.

- The original V1 substrate work sequence proposed cudaMalloc/cuMemAlloc GOT-patch intercept; B.0.5 §8 found this structurally challenged because PyTorch's caching allocator pools memory between `torch.zeros` and `cudaMalloc`, so LD_PRELOAD interception cannot reliably identify which `cudaMalloc` corresponds to vLLM KV-cache allocations.

Per Anil 2026-05-26 surface adjudication picking Option (ii): the plugin is the right shape for the vLLM-Python-class-method problem; the question is the deployment surface, not the implementation language. Phase B ships cipher-platform.deb that bundles libcipher_rt.so + cipher_vllm_plugin so customer-visible deployment is OS-package install plus env var, plugin is substrate-internal not customer-facing. Goal 5 contract preserved at apt-install + env-var tier (same complexity tier as NVIDIA CUDA Toolkit / NCCL / cuDNN install).

## 2. The strong prior

Three measurements ground Phase B's gate language:

1. **A.2 hard gate (Phase A precedent)**: `cipher_rt_cublas_shim_calls` in worker reaches approximately 11000 over TinyLlama-1.1B FP16 vLLM V1 N=1 128-token decode. Evidence at `v1_phase_a/results/a2_cuda_injection_PASS.json` (shim_calls=19090 PASS under CUDA_INJECTION64_PATH, plugin uninstalled). This is the cuBLAS shim layer; Phase B inherits this gate.

2. **Week 5 KV-dedup gate** (Phase B-specific): per `week5-complete` memory headlines:
   - N=4 TinyLlama HBM saved approximately 45802 MiB (~45 GiB) at Step 3.B
   - N=2 TinyLlama HBM saved approximately 42306 MiB at Step 2 exit gate
   - N=4 Mistral-7B KL = 0.000000e+00 across all 6 pairs at Step 3.C (bit-identity)
   - Hit rate 83.20% at Mistral-7B N=4 (above 60% gate)
   - False-positive rate 0.015% absolute at N=2 different-prompt
   - Substrate: 67 unique physicals back 22968 virtuals at N=4 TinyLlama (4x sharing)

3. **Track 2 SC6 gate** (B.0 verified at substrate-only): Mistral-7B N=4 76.0% savings; TinyLlama N=4 59.6% savings. Both 7/7 integration checks PASS. Evidence at `v1_phase_b/results/b0_substrate_only_baseline.json`.

The prior says Branch A is reachable: each measurement worked in its respective baseline (Week 5 with pip-installed plugin; B.0 at substrate-only for Track 2 SC6; Phase A A.2 at substrate-only for cuBLAS shim). Phase B's job is to deliver all three under one customer deployment surface (apt install + env var), with the plugin moved from customer pip-install to substrate-internal package contents.

Risk concentrates on (a) Python entry-point registration via .deb post-install hooks behaving correctly (vs the pip install -e behavior we test against today), (b) packaging metadata correctness, (c) interaction with existing CUDA/torch/vLLM installations on customer systems.

## 3. Three locked decisions

1. **Goal 5 contract preserved at OS-package + env-var tier** per Anil 2026-05-26 Option (ii) pick. Customer deployment surface:
   - Step 1: install cipher-platform package (`apt install cipher-platform` for Debian/Ubuntu target; `dnf install cipher-platform` if RPM target added)
   - Step 2: set `CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so` (vLLM V1 path) OR `LD_PRELOAD=/usr/lib/cipher/libcipher_rt.so` (non-vLLM-V1 path)
   - Step 3: run existing workload unchanged
   
   No pip install of cipher-vllm-kv. No `--quantization` CLI flag. No `quant_config` Python wrapping. No model code change. No Python import hook required from customer side. Plugin (.py files) live inside the .deb at standard system path; vLLM's `general_plugins` entry point discovers them via standard Python entry-points mechanism after .deb post-install registers the entry point.

2. **v1 target: Ubuntu 22.04 .deb only** (per `cipher-platform.deb` precedent at `cipher-cp25-closed` memory). Reasoning: Ubuntu 22.04 ships Python 3.10 which matches the existing `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` ABI. Ubuntu 24.04 ships Python 3.12 which would require a rebuilt cipher_kv_bridge .so (separate v1.x scope). RPM target (RHEL/Rocky) also v1.x follow-on. Avoid multi-target scope bloat at v1; Ubuntu 22.04 is the dominant ML serving distro and is the build/test environment on this pod (`cipher-pod-environment` Lambda H100 with Ubuntu).

   **Python ABI lock**: `.deb` is tagged for `cpython-310` only. Customer running Ubuntu 24.04 (Python 3.12) gets a clean install-time error (declared Depends on `python3 (>= 3.10), python3 (<< 3.11)`) rather than silent runtime import failure. v1.x roadmap: add `cipher_kv_bridge.cpython-312-...so` via parallel build, ship multi-ABI .deb.

3. **Substrate code UNCHANGED through Phase B**. cipher_rt_phase4 stays at `8613812e`; libcipher_rt.so md5 `1d91e7da` is what the .deb ships. cipher_kmod stays at `8c643fc`. cipher_vllm_plugin .py files at md5 `b89a9b6e` are what the .deb ships. Phase B is packaging infrastructure work, not substrate engineering. Anchors do NOT rotate through Phase B.

4. **CP 5.2 plugin bundled in .deb without separate gate.** `cipher_kv_offload.py` (Week 5 CP 5.2 KV offload) ships in the .deb alongside CP 5.1 + Week 5 KV-dedup. Per `cipher-cp52-closed` memory, CP 5.2 is correctness-equivalent (no measurement uplift cited at close). No Phase B Gate D for KV-offload; bundling-for-completeness only.

5. **Phase B is a cipher-platform package UPDATE not a from-scratch build** (empirical finding 2026-05-26 anchor verification): `cipher-platform 1.0` is already installed on this pod (`dpkg -s cipher-platform` confirms). It ships `/usr/lib/cipher/libcipher_rt.so + libc10.so + libcipher_v2.so + /usr/src/cipher-kmod-0.4.8/` via DKMS source. Packaging scaffolding exists at `/tmp/cipher-deb-build/cipher-platform/{DEBIAN,usr}/`; the .deb artifact at `cipher-fusion-evidence/cp_2_5/cipher-platform_1.0_amd64.deb`. Phase B's actual work is to UPDATE this existing package: bump version 1.0 to 2.0 (additive: adds Python plugin files and entry-point registration); refresh libcipher_rt.so to current build (md5 `1d91e7da`); refresh DKMS source to current kmod week-13-14-complete (0.6.5); add plugin Python files to `/usr/lib/python3/dist-packages/`; extend post-install hook to register entry points. The cipher-platform-1.0 infrastructure is the starting point; Phase B does NOT design packaging from zero.

## 4. Five-substep sequence with file:line citations

### B.1 package structure design (1 ED, design memo only)

**Calendar:** ~1 ED (design only, no build).

**Deliverable:** `cipher-fusion-evidence/v1_phase_b/PACKAGE_DESIGN.md` (decision doc, no code).

**Content per Anil 2026-05-26 spec:**
- Package format target: `.deb` for v1 (Ubuntu 22.04 only per §3 decision 2 Python ABI lock). `.rpm` and Ubuntu 24.04 follow-on tracked in deferred-scope.
- Package layout: where each component lives
  - `/usr/lib/cipher/libcipher_rt.so` (substrate)
  - `/usr/lib/cipher/libc10.so` (substrate dependency per CP 2.5 `cipher-platform.deb` precedent)
  - `/usr/lib/python3/dist-packages/cipher_vllm_kv.py` (CP 5.1 plugin)
  - `/usr/lib/python3/dist-packages/cipher_vllm_kvdedup.py` (Week 5 KV-dedup plugin)
  - `/usr/lib/python3/dist-packages/cipher_kv_offload.py` (CP 5.2 KV offload plugin)
  - `/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` (Python C extension; Python 3.10 ABI per §3 lock)
  - `/usr/lib/python3/dist-packages/cipher_model_fingerprint.py` (Track 2 SC4/SC6 fingerprint helper; needed for Gate C SC6 test)
  - Standard system entry-point registration via `dist-info/entry_points.txt`
- Post-install hook: register entry points via standard Python tooling at .deb install time
- Package metadata: name `cipher-platform`, version `0.1.0` (V1 release), maintainer + license
- Dependencies declared: CUDA 13.0+, Python 3.10 (locked range `>= 3.10, << 3.11`), vLLM 0.21.0+ (Recommends not Depends since substrate works without vLLM too), DKMS (kmod via cipher-platform's existing DKMS-bundled-source mechanism per Q5 finding)
- Uninstall semantics: `apt remove cipher-platform` cleanly removes all files, unregisters entry point, leaves vLLM in vanilla state

**R-B.1 entry-point registration proof-of-concept** (advisor catch 2026-05-26: push entry-point investigation from B.3 reactive to B.1 proactive):
- Build a tiny POC `.deb` that bundles ONE Python file with ONE entry-point declared via standard `dist-info/entry_points.txt`
- Install on clean Ubuntu 22.04 env: `apt install ./poc.deb`
- Verify: `python3 -c "import importlib.metadata as m; print([e.name for e in m.entry_points(group='vllm.general_plugins') if 'poc' in e.name])"` returns the registered entry point
- If POC PASS: Q6 option (a) standard dist-info confirmed; B.2 proceeds with that approach
- If POC FAIL: surface to Anil BEFORE B.2 with mechanism finding; pick Q6 (b) explicit-write or (c) hybrid pip-inside-deb
- POC artifact lives at `v1_phase_b/packaging/poc/` for evidence trail; ~0.5 ED added to B.1 (total B.1 now ~1.5 ED)

**Verification gate at B.1 close (paperwork plus POC, no production build):**
- `PACKAGE_DESIGN.md` lands on disk
- R-B.1 POC PASS (entry-point registration mechanism validated end-to-end on this pod)
- Surface to Anil for review before B.2 entry per scope-lock discipline (same as Phase A R-A.1 caveat: surface-immediately, do not silently extend)
- Anil pick on any §10 open items in this scope-lock before B.2

### B.2 packaging implementation (3-5 ED)

**Calendar:** 3-5 ED (build infrastructure, no substrate code change).

**Touches:**
- New directory `cipher-fusion-evidence/v1_phase_b/packaging/debian/` with:
  - `control` (package metadata, dependencies, description)
  - `rules` (build steps; symlinks libcipher_rt.so from cipher_rt_phase4 build dir, copies .py from cipher_vllm_plugin)
  - `postinst` (post-install hook: register Python entry points)
  - `prerm` (pre-remove hook: unregister entry points cleanly)
  - `changelog`, `copyright`, `compat`
- Build script `cipher-fusion-evidence/v1_phase_b/packaging/build_deb.sh` that runs `dpkg-deb` (or `debuild`) and produces `cipher-platform_0.1.0_amd64.deb` artifact
- libcipher_rt.so and .py files copied or symlinked from existing repos (NOT modified)

**ABI impact:** NONE. No substrate code change. cipher_rt_phase4 + cipher_kmod + cipher_vllm_plugin all UNCHANGED.

**Estimate detail (revised down per §3 decision 5 finding):**
- Starting point `/tmp/cipher-deb-build/cipher-platform/{DEBIAN,usr}/` scaffolding exists. Version bump 1.0 to 2.0 + libcipher_rt.so refresh + DKMS source refresh: ~1 ED
- Add Python plugin files (cipher_vllm_kv.py + cipher_vllm_kvdedup.py + cipher_kv_offload.py + cipher_kv_bridge.so + cipher_model_fingerprint.py) under `/usr/lib/python3/dist-packages/` per §4 B.1 layout: ~0.5 ED
- Extend post-install hook to register Python entry points (mechanism validated in B.1 R-B.1 POC): ~0.5 ED
- build_deb.sh smoke build on this pod: ~0.5 ED
- Total B.2: ~2.5 ED (revised down from 3-5 ED original)

**Close gate for B.2:**
- `cipher-platform_0.1.0_amd64.deb` artifact builds clean
- `dpkg -c cipher-platform_0.1.0_amd64.deb` lists expected file layout per B.1 design
- `dpkg-deb -I cipher-platform_0.1.0_amd64.deb` shows correct metadata

### B.3 no-pip verification (1 ED)

**Calendar:** 1 ED.

**Methodology:** clean-env three-gate measurement post-`apt install`.

**Pre-conditions at B.3 start:**
- `pip uninstall -y cipher-vllm-kv` (verify via `pip show cipher-vllm-kv` returns non-zero AND `importlib.metadata.entry_points(group='vllm.general_plugins')` returns empty list of cipher entries)
- `apt install ./cipher-fusion-evidence/v1_phase_b/packaging/cipher-platform_0.1.0_amd64.deb` (B.2 artifact)
- Verify: `dpkg -L cipher-platform` lists expected files; `importlib.metadata.entry_points(...)` now includes `cipher_vllm_kv` + `cipher_vllm_kvdedup` registered via .deb post-install hook (NOT via pip)

**Three measurement gates (HARD per Anil 2026-05-26 spec):**

| Gate | Methodology | Threshold |
|---|---|---|
| Gate A: cuBLAS shim layer (Phase A A.2 precedent) | re-run `v1_phase_a/verify_no_plugin.py --cuda-injection-path` per Phase A precedent | worker `cipher_rt_cublas_shim_calls >= 11000` |
| Gate B: Week 5 KV-dedup | re-run Week 5 Step 3.B harness (TinyLlama N=4) + Step 3.C harness (Mistral-7B N=4 KL gate) | TinyLlama N=4 HBM saved >= 45000 MiB (1.7% margin under 45802 baseline; tight per advisor catch 2026-05-26 because `cipher_rt_kv_dedup_alias` is deterministic cuMemUnmap+cuMemMap accounting, not a noisy benchmark); Mistral-7B N=4 KL=0 exact bit-identity + hit rate >= 80% |
| Gate C: Track 2 SC6 | re-run `phase_c/sc6_run.py Mistral-7B shared` + `... independent` + `sc6_aggregate.py` | Mistral-7B N=4 savings >= 75% (matches B.0 substrate-only baseline within 1% tolerance) |

All three gates must PASS for Branch A.

**Output evidence:**
- `v1_phase_b/results/b3_gate_a_cublas_shim.json`
- `v1_phase_b/results/b3_gate_b_kvdedup.json`
- `v1_phase_b/results/b3_gate_c_sc6.json`
- Aggregated `b3_three_gate_summary.md`

### B.4 uninstall test (0.5 ED)

**Calendar:** 0.5 ED.

**Methodology:**
- `apt remove cipher-platform` (also test `apt purge cipher-platform`)
- Verify: `dpkg -L cipher-platform` errors (package gone); `importlib.metadata.entry_points(...)` no longer lists cipher entries; libcipher_rt.so removed from `/usr/lib/cipher/`
- Smoke test: run vanilla vLLM TinyLlama N=1 decode WITHOUT cipher-platform installed; expect normal vLLM behavior (no broken state)
- Re-install `apt install ./cipher-platform_0.1.0_amd64.deb`; re-run Gate A from B.3; expect PASS again (verifies install + remove + reinstall round-trip is clean)

**Output evidence:** `v1_phase_b/results/b4_uninstall_test.md` (PASS/FAIL log).

### B.5 close-out (0.5 ED)

**Calendar:** 0.5 ED.

**Touches:**
- `cipher-fusion-evidence/V1_PHASE_B_COMPLETE.md` (new): close-out doc with Branch verdict, anchors, three-gate evidence chain, ledger row, residue
- `cipher-fusion-evidence/V1_GOAL5_DEPLOYMENT_LEDGER.md` update per ledger-gate discipline (Phase B row added in same commit as close-out)
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/v1-phase-b-platform-package.md` (new memory file)
- `MEMORY.md` index update (one line)
- Tag `v1-substrate-platform-package` lands at this commit IF substrate code rotated (per §3 locked decision 3, substrate is UNCHANGED through Phase B, so tag lives on cipher-fusion-evidence commit, not cipher_rt_phase4)

**Verification:** scope-lock branch framing language (§8) maps to actual B.3 + B.4 outcomes; ledger row reflects current state; honest residue declared.

---

## 5. Total estimate

B.1 includes +0.5 ED for R-B.1 entry-point registration POC (advisor catch 2026-05-26: push reactive→proactive). B.2 revised down to 2.5 ED per §3 decision 5 finding (cipher-platform 1.0 scaffolding exists; Phase B is UPDATE not from-scratch build).

| Branch | B.1 | B.2 | B.3 | B.4 | B.5 | Total |
|---|---|---|---|---|---|---|
| **Branch A** (.deb builds clean, all 3 gates PASS, uninstall clean) | 1.5 | 2.5 | 1 | 0.5 | 0.5 | **6 ED** approximately 1 cal-week |
| **Branch B** (.deb builds, 1 of 3 gates fails, surface mechanism finding) | 1.5 | 2.5 | 1 | 0.5 | 0.5 (surface) | **6 ED** approximately 1 cal-week before surface |
| **Branch C** (.deb builds + 3 gates PASS, uninstall leaves broken state) | 1.5 | 2.5 | 1 | 1.5 (debug + surface) | 0.5 | **7 ED** approximately 1.5 cal-weeks |
| **Branch D** (R-B.1 POC fails at B.1 OR .deb mechanism fundamentally incompatible with vLLM plugin discovery at B.3) | 1.5 (POC fail surface) | 0-2 (debug if reached) | 0 | 0 | 0.5 (surface) | **2-4 ED** approximately 0.5-1 cal-week before surface |

Range **2-7 ED** matches Anil's 1-2 cal-week Phase B budget (revised down from 6.5-8 ED original estimate due to packaging-scaffolding-exists finding). Branch D early-surface possible at B.1 POC if entry-point mechanism is structurally incompatible (advisor catch 2026-05-26 pushes this risk to B.1 not B.3).

---

## 6. Risks

| ID | Risk | Likelihood | Impact | Mitigation | Detection (gate) |
|---|---|---|---|---|---|
| **R-B.1** | Python entry-point registration via .deb post-install hook does not match `pip install -e` behavior; vLLM's `general_plugins` discovery does not see cipher entries | MEDIUM | BLOCKER for Branch A | Investigate dist-info entry_points.txt path + Python's importlib.metadata search; test on clean Ubuntu container before B.3 | B.2 close: `importlib.metadata.entry_points(group='vllm.general_plugins')` includes cipher entries post-`apt install` |
| **R-B.2** | Installed Python files (cipher_vllm_kv.py etc.) at `/usr/lib/python3/dist-packages/` conflict with pip-installed cipher-vllm-kv on customer systems that already have the pip package | LOW | MAJOR | Document upgrade path: customer should `pip uninstall cipher-vllm-kv` before `apt install cipher-platform`; .deb post-install hook checks and warns if pip version present | B.3 includes a pre-pip-uninstall check in the test sequence |
| **R-B.3** | libcipher_rt.so at `/usr/lib/cipher/` is not on the dynamic linker's default search path; LD_PRELOAD-style use requires absolute path; CUDA_INJECTION64_PATH always requires absolute path | LOW | MINOR | Document customer env var as `CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so`; .deb post-install can optionally add to `/etc/ld.so.conf.d/cipher.conf` if SONAME-only LD_PRELOAD is desired | B.3 test uses absolute path; B.4 uninstall removes ld.so.conf entry if added |
| **R-B.4** | cipher_kmod still requires separate install (DKMS or pre-built .ko); cipher-platform.deb does NOT bundle kmod because kmod is kernel-version-specific | MEDIUM | MEDIUM | Document cipher_kmod as separate package (cipher-dkms or cipher-modules); cipher-platform.deb declares Depends on cipher-modules; existing kmod install per `cipher-pod-environment` is already met on this pod for testing | B.2 declares dependency; B.3 verifies kmod loaded before each gate |
| **R-B.5** | Per Anil R-A.1-equivalent caveat: any unexpected mechanism finding during B.2 or B.3 (analogous to Phase A's cuGetProcAddress finding) | MEDIUM | MAJOR | Surface IMMEDIATELY per Anil 2026-05-26 caveat; do not push through silently; HARD STOP at B.3 gate failure per Branch D framing | B.3 gate failure triggers HARD STOP and surface |
| **R-B.6** | vLLM upgrade (e.g., 0.21.0 -> 0.22.x) breaks plugin discovery because vLLM's `load_general_plugins` API changes | LOW | MINOR | Phase B targets vLLM 0.21.0 per package metadata Depends; vLLM API stability tracked separately as a CIPHER deployment-environment concern | B.3 + B.4 use vLLM 0.21.0 on this pod; cross-version compat is v1.x scope |
| **R-B.7** | cipher_kv_bridge.so (Python C extension) requires specific Python ABI; .deb installation must match the system Python ABI exactly | LOW | MAJOR (if violated) | .deb metadata declares Python version compatibility explicitly; per Anil v1 target Ubuntu 22.04 ships Python 3.10 which matches our current build | B.2 close: verify .deb installs cleanly on a fresh Ubuntu 22.04 (target distro) |

---

## 7. Honest residue declared up-front

1. **Anil's V1 substrate work sequence spec's CP 5.1 attribution.** Per B.0.5 archaeology §3 and §10 item 1: the spec attributed Track 2 SC6's 76% to CP 5.1; empirically the 76% is Track 2's cipher_kv_bridge weight-arena chain. Phase B scope-lock language reflects the actual mechanism (CP 5.1 = vLLM KV-cache hook for Week 5 KV-dedup; Track 2 SC6 = transformers weight-arena chain via cipher_kv_bridge.so independent of plugin). The spec text addendum is informational; does not change Phase B work scope under Option (ii).

2. **Anil's V1 substrate work sequence spec's cudaMalloc/cuMemAlloc intercept approach.** Per B.0.5 §8: that approach is structurally challenged. Option (ii) packaging path sidesteps it; Phase B does NOT pursue the cudaMalloc-intercept engineering. Recorded so the spec text addendum is informational and the intercept path stays parked.

3. **Plugin file md5 b89a9b6e is what the .deb ships.** No plugin code change in Phase B. If a future Phase needs plugin source change (e.g., new tenant-tagging logic), that ships in a new .deb version, not as Phase B substep.

4. **Track 3 v1 SC1-SC6 test sequence still not located** (carries from B.0 §9 and B.0.5 §10 item 3). Add to Phase B B.1 design memo entry checklist as a sub-investigation. If Track 3 v1 SC1-SC6 doesn't have a runnable harness today, surface as Phase B follow-on or defer to CP 5.5 Phase E entry. Does NOT block Phase B per Anil 2026-05-26 spec.

5. **RPM (RHEL/Rocky/Fedora) target NOT in v1 Phase B scope.** Per §3 locked decision 2: v1 ships .deb only. RPM tracked as v1.x follow-on. If a customer requires RPM at v1, surface to Anil for v1 scope amendment.

6. **cipher_kmod packaging is SEPARATE from cipher-platform.deb.** Per R-B.4: kmod is kernel-version-specific and ships via DKMS or pre-built .ko. cipher-platform.deb declares dependency on cipher-modules. The kmod packaging itself is NOT Phase B scope (kmod is unchanged through Phase A and Phase B; it ships under its existing cipher_kmod build-and-distribute mechanism per `cipher-pod-environment` precedent).

7. **Branch labels (B.1 Branch A/B/C/D) in this scope-lock are Phase B internal outcomes**, not to be confused with Anil's prior "5 product goals Branch A required in v1" adjudication. Same naming-collision discipline as Phase A scope-lock §7.

---

## 8. Pre-commit framing per Phase B branch

Per `WEEK_14_FOLLOWUP_OPTION_2_SCOPE_LOCK.md` §3 decision 3 template, applied to Phase B per Anil 2026-05-26 V1 Phase A scope-lock §8 precedent.

**Phase B Branch A outcome statement template:**
> V1 Phase B closed via packaging-path Option (ii) at cipher-fusion-evidence [new commit] tag `v1-substrate-platform-package`. cipher-platform.deb (v0.1.0, amd64, Ubuntu 22.04 target) bundles libcipher_rt.so + cipher_vllm_kv.py + cipher_vllm_kvdedup.py + cipher_kv_offload.py + cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so. Post-install hook registers Python entry points via standard `importlib.metadata` mechanism. All three measurement gates PASS under `apt install cipher-platform` + `CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so` (NO pip install): Gate A cuBLAS shim worker `shim_calls=[X]` (>= 11000); Gate B Week 5 KV-dedup TinyLlama N=4 HBM saved `[Y] MiB` (>= 40000) + Mistral-7B N=4 KL=0 + hit rate `[Z]%` (>= 80%); Gate C Track 2 SC6 Mistral-7B N=4 savings `[W]%` (>= 75%). Uninstall test PASS: `apt remove cipher-platform` cleanly removes all files, unregisters entry points, vLLM vanilla path works. Goal 5 contract preserved at OS-package install + env-var deployment tier per Anil 2026-05-26 Option (ii) framing; same complexity tier as NVIDIA CUDA Toolkit install. Substrate anchors UNCHANGED through Phase B (cipher_rt_phase4 `8613812e`, libcipher_rt.so md5 `1d91e7da`, cipher_kmod `8c643fc`).

**Phase B Branch B outcome statement template:**
> V1 Phase B partially closed: cipher-platform.deb builds clean but 1 of 3 measurement gates fails. Gate [A/B/C] measured `[number]` against threshold `[number]`; mechanism diagnosis at `V1_PHASE_B_COMPLETE.md` §[N]. Surface to Anil for adjudication: (a) accept partial Phase B close with documented gate-specific limitation, (b) extend Phase B for additional substep to address the gate failure, (c) HARD STOP and re-architect packaging path. Substrate anchors UNCHANGED; .deb artifact preserved for inspection.

**Phase B Branch C outcome statement template:**
> V1 Phase B closed via packaging-path Option (ii) BUT uninstall test surfaced cleanup-state issue: `apt remove cipher-platform` leaves [specific issue]. Substrate file cleanup OK but [entry-point registration / config file / etc.] persists. Surface to Anil for adjudication: ship Phase B with known-issue documented OR add cleanup hook fix as B.4.1 sub-substep. Three measurement gates PASS pre-uninstall; uninstall-correctness is the only residue.

**Phase B Branch D outcome statement (HARD STOP):**
> V1 Phase B HARD STOP at B.2 or B.3: .deb mechanism fundamentally incompatible with vLLM's plugin discovery. Specific mechanism: [entry-point registration via .deb post-install does NOT match pip install -e behavior / importlib.metadata search path differs / ABI mismatch / other]. Surface to Anil for alternative packaging mechanism choice: (a) switch to Python wheel (.whl) distributed via .deb (hybrid; .deb installs the wheel via internal pip), (b) ship as Snap package (auto-handles entry points via Snap interfaces), (c) revert to Option (iv) two-tier Goal 5 contract from B.0.5 §9, (d) revert to Option (v) reverse 2026-05-26 contract lock for vLLM workloads. No silent fallback; Anil adjudication required.

---

## 9. No HARD STOP at scope-lock landing

Scope-lock ready for Anil review. No prerequisite blocks Phase B entry. The pre-state anchors at §0 are verified on disk. B.0 + B.0.5 evidence committed to cipher-fusion-evidence at `0783f1d9` + `c52d2ab` respectively. Next prompt on Anil approval is B.1 design memo.

V1 substrate work sequence: **Phase A (CLOSED) → Phase B (this scope-lock) → Phase C BF16 → Phase D FP8 → Phase E CP 5.5**. Each phase has its own scope-lock document per Anil 2026-05-26 sequence spec.

---

## 10. Adjudication record (open items for Anil pre-commit review)

**Q1 (CLOSED 2026-05-26):** Option (ii) packaging path selected per Anil 2026-05-26 surface adjudication. Goal 5 contract preserved at OS-package + env-var deployment tier.

**Q2 (CLOSED 2026-05-26 per Anil):** Ubuntu 22.04 .deb only for v1. Python 3.10 ABI lock per §3 decision 2 confirmed. Ubuntu 24.04 (Python 3.12) plus RPM are v1.x follow-on scope. Rationale (Anil 2026-05-26): cipher_kv_bridge.cpython-310 ABI matches existing build; multi-distro at v1 introduces scope bloat without clear customer-side benefit (Ubuntu 22.04 is the dominant ML serving distro through 2026-2027). Declared `Depends: python3 (>= 3.10, << 3.11)` for clean install-time error on incompatible distros.

**Q3 (CLOSED 2026-05-26 by empirical finding):** package name `cipher-platform` continues per CP 2.5 close precedent. The package already exists on this pod at v1.0; Phase B ships v2.0 (additive major bump for Python plugin bundling). No new name needed.

**Q4 (CLOSED 2026-05-26 per Anil):** standard `/usr/lib/python3/dist-packages/` path. Rationale (Anil 2026-05-26): CIPHER-namespaced alternative requires customer-side PYTHONPATH machinery which violates the locked Goal 5 contract (no Python path manipulation in customer step). Standard system path keeps `importlib.metadata` discovery automatic; plugin module names are already CIPHER-prefixed (`cipher_vllm_kv`, `cipher_vllm_kvdedup`, `cipher_kv_offload`); no realistic namespace collision risk.

**Q5 (CLOSED 2026-05-26 by empirical finding):** cipher-platform 1.0 already bundles cipher_kmod-0.4.8 via DKMS source at `/usr/src/cipher-kmod-0.4.8/` (verified `dpkg -L cipher-platform`). Phase B uses the existing DKMS-bundled-in-cipher-platform approach (Q5 option c-equivalent under existing infrastructure, but cleaner than building separate cipher-modules.deb). Phase B's update bumps the DKMS source from cipher-kmod-0.4.8 to cipher-kmod-0.6.5 (week-13-14-complete equivalent; verify exact version mapping in B.1 design memo). No separate cipher-modules.deb needed; Phase A's "cipher_kmod UNCHANGED" stays true because Phase B refreshes the DKMS source to ship the existing week-13-14-complete kmod, not modify it.

**Q6 (CLOSED 2026-05-26 per Anil):** option (a) standard `dist-info/entry_points.txt`. Rationale (Anil 2026-05-26): cheapest validation, lowest risk. `pip install -e cipher-vllm-kv` already produces dist-info today; Phase B packages that dist-info into the .deb at the standard system path. R-B.1 POC at B.1 validates end-to-end on Ubuntu 22.04 before B.2 commits production scaffolding; POC failure surfaces with mechanism evidence + scope-lock amendment before any wasted B.2 work. Falls back to (b) post-install programmatic write or (c) hybrid pip-inside-deb ONLY on POC failure with explicit Anil re-adjudication.

---

## 11. Framing for B.5 close-out (V1_PHASE_B_COMPLETE.md)

Per Anil 2026-05-26 directive: the B.5 close-out doc frames the deliverable as **cipher-platform v2.0** (additive major bump from existing v1.0), NOT first-time packaging.

The empirical finding: `cipher-platform 1.0` is already installed on this pod (verified via `dpkg -s cipher-platform` 2026-05-26 returning `Package: cipher-platform / Version: 1.0 / Status: install ok installed / Maintainer: CIPHER Platform <anil.0666369@gmail.com>`). It ships:
- `/usr/lib/cipher/libcipher_rt.so` (substrate, older version)
- `/usr/lib/cipher/libc10.so` (substrate dependency)
- `/usr/lib/cipher/libcipher_v2.so` (Phase 3 substrate)
- `/usr/src/cipher-kmod-0.4.8/` (DKMS source tree, 17 files including `Kbuild`, `Makefile`, `dkms.conf`, kmod source .c files)
- `/usr/bin/cipher-run` (launcher script)

Phase B builds on this shipping infrastructure additively:
- libcipher_rt.so refreshes from older version to current `1d91e7da` (Phase A close build)
- cipher-kmod DKMS source refreshes from 0.4.8 to current week-13-14-complete (0.6.5)
- NEW: Python plugin files added under `/usr/lib/python3/dist-packages/`
- NEW: post-install hook extended to register Python entry points
- Version bumps 1.0 to 2.0 (major bump per semantic-versioning convention for new feature addition)

**Framing for the V1_PHASE_B_COMPLETE.md narrative (diligence story):** A reviewer reading the close-out doc should hear "mature shipping package, additive v1.0 to v2.0 update adding plugin bundle and entry-point registration" rather than "first packaging attempt." The empirical evidence of cipher-platform 1.0 on disk supports this framing; B.5 close-out doc cites the verbatim `dpkg -s cipher-platform` output as evidence of pre-existing shipping infrastructure. This matters for the diligence story: CIPHER's packaging is a mature engineering artifact with version-1 history, not a green-field deployment exercise.

V1_PHASE_B_COMPLETE.md template language carry-into:
- §1 outcome statement: "cipher-platform v2.0 ships at this commit, additive update to the existing v1.0 package (cipher-cp25-closed precedent 2026-05-16)"
- §2 evidence chain: cite `dpkg -s cipher-platform` v1.0 baseline; cite `dpkg -L cipher-platform` post-install verification at v2.0
- §6 ledger row: Phase B close row notes "v2.0 additive update; customer install path unchanged from v1.0 deployment pattern; plugin bundling is the new feature"

## 12. Related memory

- [[v1-phase-a-driver-worker-init]]: Phase A close baseline; Branch D (a) CUDA_INJECTION64_PATH deployment path that Phase B inherits and extends
- [[v1-goal5-contract-lock]]: 2026-05-26 Goal 5 contract; Phase B preserves at OS-package + env-var tier per Option (ii)
- [[cipher-cp25-closed]]: prior cipher-platform.deb packaging precedent (libcipher_rt.so + libc10.so + kmod packaging at CP 2.5)
- [[cipher-cp51-closed]]: CP 5.1 plugin hook origin; substrate-deliverable not measurement-uplift; Phase B bundles this plugin in .deb
- [[cipher-cp52-closed]]: CP 5.2 KV offload plugin; bundled in .deb alongside CP 5.1
- [[week5-complete]]: Week 5 KV-dedup measurement chain (plugin-dependent); Phase B Gate B preserves these numbers
- [[week6-kvdedup-autotrigger]]: cipher_vllm_kvdedup.py auto-trigger; bundled in .deb
- [[cipher-track2-weight-sharing]]: Track 2 SC6 76% weight-sharing (substrate-only; B.0 verified); Phase B Gate C re-verifies under .deb deployment
- [[cipher-evidence-commit-discipline]]: followed at B.0, B.0.5, this scope-lock, B.5 close-out
- [[cipher-fresh-session-anchor-verify]]: anchor verification done at scope-lock entry
- [[cipher-proceed-not-ask]]: "Proceed to Phase B" + B.0 + B.0.5 + Option (ii) surface adjudication each executed without re-litigation per detailed scope
- [[cipher-pod-environment]]: pod is Ubuntu (apt/dpkg available); B.2 builds + tests on this pod natively for v1
