# V1 Phase B scope-lock addendum: B.1' Option (vi) container path

**Date:** 2026-05-26
**Amendment to:** `V1_PHASE_B_SCOPE_LOCK.md` (cipher-fusion-evidence `e33d62d`)
**Trigger:** Anil 2026-05-26 adjudication after B.1 R-B.1 POC surfaced Branch C virtualenv sys.path scope finding (`v1_phase_b/packaging/poc/results/poc_install_verify.json`)
**Type:** addendum per cipher addendum convention (W14 Step 2 G addendum precedent `6be1d4f`; W14 Step 3 S3.C KL addendum precedent `ef8b830`). Original scope-lock `V1_PHASE_B_SCOPE_LOCK.md` body preserved verbatim; addendum supersedes specific sections per §3 below.

---

## 1. Pre-state anchors (UNCHANGED through B.1 plus B.1')

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-platform .deb at v1.0 (CP 2.5 close artifact; reframed by this addendum)
- cipher-fusion-evidence `159386f` (B.1 deliverables commit; this addendum lands at next commit)

---

## 2. Reframe summary

The B.1 R-B.1 POC validated that the standard `dist-info/entry_points.txt` mechanism (Q6 (a)) works for system Python but does NOT auto-discover entry points in virtualenv-isolated Python because virtualenv `sys.path` does not include `/usr/lib/python3/dist-packages/`. Production ML deployments commonly use virtualenvs (including this pod's `vllm_env`).

Anil 2026-05-26 adjudication picked Option (C) container path over the five POC-result options surfaced in `poc_install_verify.json`. The container path sidesteps the .deb-vs-virtualenv discovery problem by construction: the container ships its own Python with the plugin pip-installed at build time inside the container; container Python IS the deployment Python; entry-point discovery happens automatically.

The .deb path work is NOT discarded. PACKAGE_DESIGN.md content is reframed from "v1 deployment unit" to "v1 bare-metal deployment role for cipher_kmod prerequisite installer" per §6 below.

---

## 3. Sections of V1_PHASE_B_SCOPE_LOCK.md superseded by this addendum

| Original section | Original content | Superseded by addendum section |
|---|---|---|
| §1 Why this campaign | Option (ii) packaging path framing | §4 Why container path (this addendum) |
| §3 decision 1 | Customer deployment recipe via apt install plus env var | §5 decision 1 (container deployment recipe) |
| §3 decision 2 | Ubuntu 22.04 .deb only with Python 3.10 ABI lock | §5 decision 2 (container base image plus host-prereq .deb) |
| §3 decision 4 | CP 5.2 plugin bundled in .deb without separate gate | §5 decision 4 (CP 5.2 plugin pip-installed in container) |
| §3 decision 5 | Phase B is cipher-platform package UPDATE | §5 decision 5 (Phase B is cipher-platform-vllm container BUILD plus thin cipher-platform .deb v2.0 refresh for kmod prereq) |
| §4 substep B.1 through B.5 | .deb packaging substeps | §7 substep B.1' through B.5' (this addendum, container substeps) |
| §5 total estimate | 6 ED Branch A | §8 total estimate (this addendum, 5.5-7 ED Branch A) |
| §6 risks R-B.1 through R-B.7 | .deb risks | §9 risks R-B.1' through R-B.8' (this addendum, container-specific) |
| §8 Branch A/B/C/D framing | .deb verdicts | §10 Branch A/B/C/D framing (this addendum, container verdicts) |
| §10 Q1 through Q6 | .deb adjudication record | §11 Q1-Q6 unchanged; Q7 added for cipher_kmod host prereq |
| §11 framing for B.5 close-out | cipher-platform v2.0 .deb framing | §12 framing for B.5' close-out (this addendum, container v2.0 framing) |

Sections of V1_PHASE_B_SCOPE_LOCK.md NOT superseded (still authoritative):
- §2 The strong prior (three measurement gates per Phase A precedent)
- §3 decision 3 (substrate code UNCHANGED through Phase B; substrate anchors do not rotate)
- §7 honest residue items 1-7 (carry verbatim)
- §12 related memory (carry verbatim, plus this addendum adds references)

---

## 4. Why container path (supersedes §1)

The cipher_vllm_plugin chain is load-bearing for Week 5 KV-dedup measurements per B.0.5 archaeology. The plugin needs to be discoverable by vLLM's Python at engine init. Two paths surfaced:

(ii) .deb packaging: install plugin at OS-system-Python path. POC surfaced the virtualenv-isolation wrinkle: production ML deployments routinely use virtualenvs and `/usr/lib/python3/dist-packages/` is not in virtualenv sys.path.

(vi) container path: ship the deployment unit as an OCI container that bundles its OWN Python with plugin pre-installed. Container Python IS the deployment Python. Customer runs the container with one env var; plugin discovery happens automatically inside the container.

Container path is the cleaner mechanism. Container deployment is also the norm for ML serving (vLLM/TGI/SGLang all ship container-first; NVIDIA Triton, Inference Server, NIM all ship containers). Goal 5 contract preserved verbatim: customer runs ONE container with ONE env var. No pip install (plugin is inside the container, not in customer's Python), no Python path manipulation (container Python sees its own packages), no CLI flag, no application code change.

---

## 5. Locked decisions (supersedes §3 decisions 1, 2, 4, 5; decision 3 preserved)

1. **Goal 5 contract preserved at container-run + env-var deployment tier.** Customer deployment surface:
   - Step 1 (one-time per host): host installs cipher-platform .deb v2.0 (refresh of CP 2.5 v1.0 with kmod source bumped to week-13-14-complete; libcipher_rt.so NO LONGER bundled in .deb because that lives in container)
   - Step 2 (per deployment): `docker run --gpus all -e CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so cipher-platform-vllm:2.0 [vllm-serve-args]`
   
   Step 1 parallels NVIDIA driver install (one-time per host, standard ops pattern). Step 2 is the customer-deployment-unit run. No pip install of cipher_vllm_kv. No `--quantization` CLI flag. No Python path manipulation. No model code change.

2. **v1 base image: `nvidia/cuda:12.x-runtime-ubuntu22.04`** as the safer default. Alternative: `vllm/vllm-openai:0.21.0` if licensing review confirms re-distribution is permitted; if not, use the NVIDIA CUDA base and pip install vLLM into the container at build time. Ubuntu 22.04 base preserves the Python 3.10 ABI match with `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` per scope-lock §3 decision 2 Python ABI lock (carries through to container path).

3. (preserved from original §3 decision 3) **Substrate code UNCHANGED through Phase B.** cipher_rt_phase4 `8613812e`; libcipher_rt.so md5 `1d91e7da`; cipher_kmod `8c643fc`; cipher_vllm_plugin md5 `b89a9b6e`. Phase B is packaging-and-deployment infrastructure work, not substrate engineering.

4. **CP 5.2 plugin pip-installed in container at build time** (container build runs `pip install -e /path/to/cipher_vllm_plugin` inside the container's Python). Same plugin chain (CP 5.1 cipher_vllm_kv + Week 5 KV-dedup cipher_vllm_kvdedup + CP 5.2 cipher_kv_offload) bundled. No separate Gate D for KV-offload (correctness-equivalent per `cipher-cp52-closed`).

5. **Phase B ships TWO artifacts:**
   - **cipher-platform-vllm:2.0 container** (v1 deployment unit; Phase B B.2' through B.5' builds this)
   - **cipher-platform .deb v2.0** (v1 host prerequisite installer for cipher_kmod; thin refresh of CP 2.5 v1.0 to bump DKMS kmod source from 0.4.8 to week-13-14-complete; libcipher_rt.so REMOVED from .deb because it lives in container)
   
   Both artifacts ship as part of v1 release. The .deb is one-time-per-host (parallel to NVIDIA driver install). The container is per-deployment.

---

## 6. PACKAGE_DESIGN.md disposition (preserved as v1.x bare-metal scope reframe)

`v1_phase_b/PACKAGE_DESIGN.md` is preserved on disk. Its content is reframed:

- Sections 1, 2 (cipher-platform v1.0 baseline + v2.0 metadata): still authoritative for the v1 cipher-platform .deb v2.0 thin-refresh role (kmod source bump only; libcipher_rt.so + plugin sections superseded by container)
- Sections 3 (customer deployment recipe): SUPERSEDED by §5 decision 1 of this addendum (container plus host-prereq)
- Section 5 (R-B.1 POC plan): COMPLETED at B.1; reframed as evidence trail (POC outcome informed the Option (vi) pick)
- Sections 6, 7, 8, 9: retain as informational v1.x bare-metal-deployment reference

The PACKAGE_DESIGN.md file is annotated at the top with a header noting "v1 deployment role: thin cipher-platform .deb v2.0 refresh for kmod prereq only; v1.x bare-metal deployment scope deferred per 2026-05-26 Option (vi) container pick." Original content preserved for audit trail.

---

## 7. Five-substep sequence container path (supersedes §4 B.1 through B.5)

### B.1' (this addendum, complete at this commit, ~0.5 ED actual)

This addendum lands the scope amendment. PACKAGE_DESIGN.md header annotation. Devang Nebius update message drafted to `v1_phase_b/DEVANG_NEBIUS_UPDATE.md` per Anil 2026-05-26 parallel action item.

**Verification gate at B.1' close (this addendum commit):**
- This addendum file lands on disk
- PACKAGE_DESIGN.md carries reframe header
- DEVANG_NEBIUS_UPDATE.md drafted (action item for user to send externally; not gating B.2' start)
- Surface to Anil for Q7 (cipher_kmod host prereq) adjudication plus explicit "proceed to B.2'" signal before container build starts

### B.2' container build pipeline (3-4 ED)

**Touches:**
- New `v1_phase_b/container/Dockerfile` (base image, COPY substrate libs, pip install plugin inside container Python, ENV defaults)
- New `v1_phase_b/container/entrypoint.sh` (thin wrapper that exec's vLLM serve with substrate env vars set)
- New `v1_phase_b/container/build_container.sh` (docker build + image tag)
- New `v1_phase_b/container/.dockerignore` (excludes cipher-fusion-evidence and other large dirs from build context)

**Dockerfile responsibilities:**
- FROM `nvidia/cuda:12.x-runtime-ubuntu22.04` (or vllm-base if licensing OK)
- apt install python3, pip install vllm==0.21.0 + dependencies
- COPY substrate libraries from build context: libcipher_rt.so, libc10.so, libcipher_v2.so to `/usr/lib/cipher/`
- COPY cipher_kv_bridge.so to container Python site-packages
- COPY cipher_vllm_plugin source tree, run `pip install -e /opt/cipher_vllm_plugin/` inside container (entry-point registration happens via pip install; container Python sees its own site-packages)
- ENV CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so (default; customer can override)
- ENTRYPOINT [entrypoint.sh]

**Estimate detail:**
- Dockerfile authoring + entrypoint.sh: ~1.5 ED
- build_container.sh + smoke build on this pod: ~1 ED
- Image size optimization (multi-stage build if needed): ~0.5-1 ED
- Total B.2': ~3-3.5 ED

**Close gate for B.2':**
- `cipher-platform-vllm:2.0` image builds clean
- Image size <5 GB (CUDA runtime + vLLM base already ~3 GB; substrate plus plugin negligible)
- `docker run --rm cipher-platform-vllm:2.0 --help` succeeds (smoke; entrypoint chain works)

### B.3' container verification (1-1.5 ED)

**Methodology:** all three gates run INSIDE one `docker run` invocation per gate type (container is the deployment unit; gates run inside the unit; gates run with host-side cipher_kmod loaded per C.1 prerequisite).

| Gate | Methodology | Threshold |
|---|---|---|
| Gate A: cuBLAS shim | inside container, run `verify_no_plugin.py --cuda-injection-path` analog (modified to run plugin-installed-in-container; semantically the same gate as Phase A A.2) | worker `cipher_rt_cublas_shim_calls >= 11000` |
| Gate B: Week 5 KV-dedup | inside container, run TinyLlama N=4 + Mistral-7B N=4 Week 5 Step 3 harnesses | TinyLlama N=4 HBM saved >= 45000 MiB; Mistral-7B N=4 KL=0 + hit >= 80% |
| Gate C: Track 2 SC6 | inside container, run sc6_run.py both phases plus sc6_aggregate.py | Mistral-7B N=4 savings >= 75% |

All three gates inside ONE container run = deployment-unit verification.

**Output evidence:** `v1_phase_b/container/results/b3_three_gate_in_container.json`.

### B.4' uninstall test (0.5 ED)

`docker rm cipher-platform-vllm:2.0` cleanup test. Verify no host-side state leaked (container should leave nothing on host). Re-run smoke after removal to verify clean state.

### B.5' close-out (0.5 ED)

V1_PHASE_B_COMPLETE.md draft framing cipher-platform-vllm:2.0 as v1 deployment unit. cipher-platform .deb v2.0 as v1 host-prereq. Ledger row added per ledger-gate discipline.

---

## 8. Total estimate (supersedes §5)

| Branch | B.1' | B.2' | B.3' | B.4' | B.5' | Total |
|---|---|---|---|---|---|---|
| **Branch A** (container builds clean, all 3 gates PASS inside container, uninstall clean) | 0.5 | 3-3.5 | 1-1.5 | 0.5 | 0.5 | **5.5-6.5 ED** approximately 1-1.5 cal-weeks |
| **Branch B** (container builds, 1 of 3 gates fails inside container) | 0.5 | 3-3.5 | 1-1.5 | 0.5 | 0.5 (surface) | **5.5-6.5 ED** approximately 1-1.5 cal-weeks before surface |
| **Branch C** (gates PASS in container but uninstall leaves host-side leak) | 0.5 | 3-3.5 | 1-1.5 | 1-1.5 (debug) | 0.5 | **6.5-7.5 ED** approximately 1.5 cal-weeks |
| **Branch D** (container mechanism fundamentally fails; e.g., libcipher_rt.so does not LD_PRELOAD inside container due to CUDA driver mount issue) | 0.5 | 1-2 (debug + surface) | 0 | 0 | 0.5 (surface) | **2-3 ED** approximately 0.5 cal-week before surface |

Range **2-7.5 ED** matches Anil 1-2 cal-week Phase B budget.

---

## 9. Risks (supersedes §6 R-B.1 through R-B.7; container-specific R-B.1' through R-B.8')

| ID | Risk | Likelihood | Impact | Mitigation | Detection |
|---|---|---|---|---|---|
| **R-B.1'** | nvidia-container-toolkit must be installed on host to make `--gpus all` work; this is a separate prereq from cipher-platform .deb | MEDIUM | MAJOR (BLOCKER without it) | Document as customer prereq parallel to NVIDIA driver install; nvidia-container-toolkit is standard for any GPU container workload (vLLM/TGI/SGLang all require it) | B.2' build script checks `docker run --gpus all` smoke first |
| **R-B.2'** | Container's libcipher_rt.so cannot find host CUDA driver libraries at runtime | LOW | BLOCKER | nvidia-container-toolkit volume-mounts host CUDA driver libs into container; standard CUDA container deployment pattern; verified via smoke at B.2' close | B.2' smoke includes minimal CUDA init test |
| **R-B.3'** | Container size exceeds reasonable v1 distribution limit (Docker Hub free tier 10 GB; private registries no hard limit) | LOW | MINOR | Multi-stage build to strip build-time deps; CUDA runtime base (3 GB) + vLLM (1.5 GB) + substrate libs (10 MB) + plugin Python (5 MB) = ~4.5 GB realistic; under 5 GB target | B.2' close: `docker images cipher-platform-vllm:2.0` size check |
| **R-B.4'** | cipher_kmod host-prereq mismatch with container's substrate ABI | MEDIUM | MAJOR | cipher-platform .deb v2.0 ships kmod source matching week-13-14-complete; container substrate libs (libcipher_rt.so md5 1d91e7da) is from that exact tag; ABI consistency by construction; cipher-platform .deb release-cadence couples to container release-cadence | B.3' Gate A inside container exercises the kmod ABI; mismatch surfaces as ioctl failures |
| **R-B.5'** | Q7 cipher_kmod host-prereq adjudication unresolved | MEDIUM | BLOCKER | Surface to Anil in §11 Q7 before B.2' starts; Anil 2026-05-26 recommended C.1 (host-package prereq parallel to NVIDIA driver pattern); confirm at addendum review | this addendum surfaces Q7 |
| **R-B.6'** | container registry choice (GHCR vs Docker Hub vs NVIDIA NGC) not yet locked | LOW | MINOR | v1 ship via local-build + tarball distribute (`docker save` + customer `docker load`) acceptable until registry choice closes; v1.x adds registry push | B.2' build script supports both `docker save` tarball and registry push paths |
| **R-B.7'** | vLLM image base licensing (vllm/vllm-openai:0.21.0) may not permit re-distribution | LOW | MINOR | Default to NVIDIA CUDA base; pip install vLLM at container build time (avoids re-distributing vLLM's image); v1.x switches to vllm-base if licensing review clears | B.1' addendum surfaces; B.2' Dockerfile uses NVIDIA CUDA base by default |
| **R-B.8'** | R-A.1-equivalent caveat: unexpected mechanism finding during B.2' or B.3' inside container (analogous to Phase A cuGetProcAddress) | MEDIUM | MAJOR | SURFACE IMMEDIATELY per scope-lock §7 R-A.1-equivalent caveat; do NOT push through silently; HARD STOP at B.3' gate failure per Branch D framing | B.3' gate failure triggers HARD STOP and surface |

---

## 10. Branch framing pre-commit (supersedes §8; container-specific verdicts)

**Phase B Branch A outcome statement template:**
> V1 Phase B closed via Option (vi) container path at cipher-fusion-evidence [new commit] tag `v1-substrate-platform-package` (alias `v1-cipher-platform-vllm-container`). cipher-platform-vllm:2.0 OCI container (built on `nvidia/cuda:12.x-runtime-ubuntu22.04` base; substrate libs + pip-installed cipher_vllm_plugin chain) ships as v1 deployment unit. cipher-platform .deb v2.0 ships as v1 host-prerequisite installer for cipher_kmod (refreshed DKMS source from 0.4.8 to week-13-14-complete; libcipher_rt.so REMOVED from .deb because it lives in container). All three measurement gates PASS inside one `docker run --gpus all -e CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so cipher-platform-vllm:2.0` invocation: Gate A cuBLAS shim_calls=[X] (>= 11000); Gate B Week 5 KV-dedup TinyLlama N=4 [Y] MiB (>= 45000) + Mistral-7B KL=0 + hit [Z]% (>= 80%); Gate C Track 2 SC6 Mistral-7B [W]% (>= 75%). docker rm cleanup PASS. Goal 5 contract preserved verbatim: customer runs ONE container with ONE env var; host-prereq cipher-platform .deb is one-time-per-host parallel to NVIDIA driver install. Substrate anchors UNCHANGED through Phase B (cipher_rt_phase4 `8613812e`, libcipher_rt.so md5 `1d91e7da`, cipher_kmod `8c643fc`).

**Phase B Branch B outcome statement template:**
> V1 Phase B partially closed: cipher-platform-vllm:2.0 container builds clean but 1 of 3 measurement gates fails inside container. Gate [A/B/C] measured [number] against threshold [number]; mechanism diagnosis at `V1_PHASE_B_COMPLETE.md` §[N]. Surface to Anil for adjudication: (a) accept partial Phase B close with documented gate-specific limitation, (b) extend Phase B for additional substep to address the gate failure, (c) HARD STOP and re-architect container approach. Substrate anchors UNCHANGED.

**Phase B Branch C outcome statement template:**
> V1 Phase B closed via Option (vi) container path BUT uninstall test surfaced host-side state leak: `docker rm` removes container but [specific host-side state] persists. Substrate file cleanup OK but [config / cache / log / etc.] persists. Surface to Anil for adjudication: ship Phase B with known-issue documented OR add cleanup hook fix as B.4'.1 sub-substep. Three measurement gates PASS pre-uninstall.

**Phase B Branch D outcome statement (HARD STOP):**
> V1 Phase B HARD STOP at B.2' or B.3': container mechanism fundamentally fails. Specific mechanism: [libcipher_rt.so does not LD_PRELOAD inside container due to CUDA driver mount issue / cipher_kmod ioctl from container fails / other]. Surface to Anil for alternative deployment mechanism choice: (a) revert to Option (ii) .deb path with virtualenv-aware setup script, (b) switch to a different container runtime (Singularity for HPC; Snap; Flatpak), (c) split substrate into in-container + on-host components more cleanly, (d) re-architect substrate to not require shared CUDA driver state with host (heavy). No silent fallback; Anil adjudication required.

---

## 11. Adjudication record (supersedes §10; Q1-Q6 carry; Q7 added)

**Q1-Q6**: closed per V1_PHASE_B_SCOPE_LOCK.md §10 + Q1 re-pick. Q1 reframes from Option (ii) to Option (vi) container path per Anil 2026-05-26.

**Q7 (OPEN, surface for Anil pick):** cipher_kmod host-prereq deployment mechanism. Three options per Anil 2026-05-26 surface:
- (C.1) cipher-platform .deb v2.0 host-side install + container (TWO customer steps; step a is one-time-per-host parallels NVIDIA driver). NOT a Goal 5 violation because parallels existing NVIDIA driver tooling.
- (C.2) Container ships kmod source; runtime DKMS build inside container. Requires privileged container OR host kernel-headers volume-mount. Brittle.
- (C.3) Container assumes cipher_kmod is on host without specifying mechanism. Vague; not deployable.

Anil 2026-05-26 recommendation: **(C.1)**. Surface for confirmation. If Anil picks differently, §5 decision 5 (cipher-platform .deb v2.0 role) and §6 PACKAGE_DESIGN.md disposition revise.

---

## 12. Framing for B.5' close-out (supersedes §11)

Per Anil 2026-05-26 directive folded into container reframe: the B.5' close-out doc frames the deliverable as **cipher-platform-vllm:2.0** (the OCI container) plus **cipher-platform .deb v2.0** (the host prerequisite installer). Both ship as v1 release artifacts.

The empirical baseline cited: dpkg -s cipher-platform v1.0 output (CP 2.5 close 2026-05-16) shows existing shipping infrastructure that v2.0 .deb refreshes for the kmod prereq role. The container is NEW v1 work but builds on mature substrate (Phase A close `v1-substrate-driver-worker-init` substrate libs) and mature plugin chain (`cipher_vllm_plugin` md5 `b89a9b6e`).

**Framing for V1_PHASE_B_COMPLETE.md narrative (diligence story):** A reviewer reading the close-out doc hears "mature deployment platform: containerized deployment unit (cipher-platform-vllm:2.0) plus host-prerequisite installer (cipher-platform .deb v2.0; refresh of CP 2.5 v1.0); deployment pattern parallels NVIDIA's own driver-plus-container model that the ML serving industry runs on (NVIDIA NGC, nvidia-container-toolkit, NIM)." Diligence-story strength: CIPHER's deployment model aligns with the de facto industry standard for GPU-accelerated ML containers, not a bespoke distribution mechanism.

---

## 13. Devang Nebius parallel action item (informational; not gating B.2')

Per Anil 2026-05-26 directive: send Devang one-paragraph technical update + sharpened question to inform the container-path scope before B.2' build starts. Draft message at `v1_phase_b/DEVANG_NEBIUS_UPDATE.md`. Action item for user-side external send; not gating B.2' start. Response informs the C.1 vs alternative framing in Q7 above; does not block container build (B.2' can proceed with C.1 default per Anil recommendation, adjust if Devang surfaces different Nebius pattern).

---

## 14. Related memory

Carry from V1_PHASE_B_SCOPE_LOCK.md §12 + additions:
- [[v1-phase-a-driver-worker-init]]: Phase A close + CUDA_INJECTION64_PATH deployment path that container path makes the documented v1 customer-facing path
- [[v1-goal5-contract-lock]]: 2026-05-26 Goal 5 contract; container path preserves at container-run + env-var tier
- [[cipher-cp25-closed]]: v1.0 cipher-platform.deb precedent; v2.0 .deb thin refresh for kmod prereq role
- [[cipher-cp51-closed]]: CP 5.1 plugin hook origin; bundled in container at build time
- [[cipher-cp52-closed]]: CP 5.2 KV offload plugin; bundled in container
- [[week5-complete]]: Week 5 KV-dedup plugin chain; bundled in container
- [[g6-audit-chain]]: kmod week-13-14-complete (0.6.5); .deb v2.0 refreshes DKMS source
- [[cipher-pod-environment]]: Ubuntu pod environment; B.2' container build on this pod natively
- [[cipher-evidence-commit-discipline]]: followed at this B.1' addendum commit
- [[cipher-proceed-not-ask]]: Anil "Option (C) container selected" plus B.0.5 plus B.1 POC surface adjudication chain
