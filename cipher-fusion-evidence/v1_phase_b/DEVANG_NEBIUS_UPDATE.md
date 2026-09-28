# Devang Nebius update + Phase B deployment design for Nebius

**Date drafted:** 2026-05-26 (refined per Anil 2026-05-26 customer-model correction; supersedes 2026-05-26 B.0.7 mechanism-investigation draft)
**Action item:** Anil to send externally
**Purpose:** parallel action item per V1_PHASE_B_SCOPE_LOCK_ADDENDUM_B1_DOUBLE_PRIME.md (B.1''). Devang response window 1-3 business days; overlaps Phase B B.2''-B.5'' execution. Not gating B.1'' commit or B.2'' substrate engineering start.

---

## Draft message (refined per corrected customer model: neocloud ops team is CIPHER's customer)

**Subject:** CIPHER Phase A close + Phase B deployment design for Nebius

> Devang : Phase A of CIPHER v1 closed last week. Substrate ships driver-level worker-init via NVIDIA's CUDA_INJECTION64_PATH; A.2 verification gate PASS at 19090 cuBLAS shim calls per H100 workload. All five v1 substrate goals are now Branch-A-feasible.
>
> Phase B scoping the deployment artifact for neocloud ops teams. Leaning: cipher-platform v2.0 .deb that Nebius (and other neocloud operators) install on each H100 node via standard apt. Once installed, vanilla `docker run --gpus all vllm/vllm-openai:0.21.0` invocations from your customers transparently pick up CIPHER substrate via nvidia-container-toolkit's CDI mechanism (same path NVIDIA uses for its own driver library injection). Customers see faster inference; they don't see CIPHER. cipher-platform ships with operator-facing CLI (status / verify / refresh / version / logs) and runbook for standard ops scenarios (post-driver-update refresh, debugging, uninstall).
>
> Two questions for Nebius's GPU node ops:
>
> (1) Does your team install host-side .deb packages with DKMS-based kernel modules today (NVIDIA driver, proprietary observability, scheduling)? Looking to confirm cipher-platform's deployment model matches your existing ops patterns.
>
> (2) Are you on nvidia-container-toolkit 1.17+ with CDI mode enabled? cipher-platform's CDI extension mechanism requires this; older NVIDIA stacks would need a different deployment path.
>
> Happy to share the Phase A close evidence pack + B.0.7 mechanism investigation summary if useful.
>
> : Anil

---

## Context for the question (not part of the send)

CIPHER v1 ships ONE deployment artifact under M1 CDI mechanism + P1 in-place patch sub-path (per Anil 2026-05-26 Q8/Q9/Q10/Q11 close + customer-model correction):

**cipher-platform .deb v2.0** : one-time-per-host installer by neocloud ops team. Bundles:
- Substrate libraries: `libcipher_rt.so` + `libc10.so` + `libcipher_v2.so` at `/usr/lib/cipher/`
- Plugin chain: `cipher_vllm_kv.py` + `cipher_vllm_kvdedup.py` + `cipher_kv_offload.py` + `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` + `cipher_model_fingerprint.py` + `cipher_vllm_kv-0.2.0.dist-info/` at `/usr/lib/python3/dist-packages/`
- DKMS source: `/usr/src/cipher-kmod-0.6.5/` (refreshed from CP 2.5 era 0.4.8)
- CDI patch script: `/usr/lib/cipher/cipher_cdi_patch.sh` (postinst patches /var/run/cdi/nvidia.yaml in-place to add CIPHER mounts + CUDA_INJECTION64_PATH env to existing nvidia.com/gpu=all entry's containerEdits)
- Operator CLI: `/usr/bin/cipher-platform` with subcommands status / verify / refresh / version / logs
- Operator runbook: `/usr/lib/cipher/runbook.md`
- Health check: `/usr/lib/cipher/health-check.sh`
- Optional systemd path-watcher unit: `cipher-platform-watch.service` (auto re-patch on /var/run/cdi/nvidia.yaml change; convenience layer)
- postinst hook: DKMS register/build/install + cipher_cdi_patch.sh first-apply + banner
- prerm hook: DKMS unregister + revert CDI patch + remove operator-facing files

**Neocloud ops customer step:** standard apt install + standard verify + standard documented post-driver-update refresh (`sudo cipher-platform refresh`). Normal B2B infrastructure-operator experience.

**End-user customer step (neocloud's customer): NONE.** `docker run --gpus all vllm/vllm-openai:0.21.0 serve --model meta-llama/Llama-3-8B` runs UNCHANGED. CIPHER substrate fires transparently via host's nvidia-container-toolkit reading the patched CDI spec. Goal 5 contract verbatim preserved for the end-user (the ML team renting GPU compute), which is the contract's scope.

## Customer-model correction folded in

CIPHER sells to neoclouds (Nebius, Lambda, CoreWeave, RunPod). Neocloud ops teams are CIPHER's customer: they install, configure, monitor, and maintain CIPHER as part of their normal infrastructure operations; they have a contract with CIPHER; they get docs, support, CLI, runbooks. The neocloud's customer (ML team renting GPU compute) is the END-USER; that team does NOT install, configure, or know about CIPHER; Goal 5 contract verbatim applies to the END-USER.

Prior B.0.5 → B.0.7 → B.0.7++ investigation was framed as "make CIPHER invisible to whoever installs it." That was wrong: the neocloud ops team is a normal infrastructure operator and gets a normal infrastructure-operator deployment experience. M1 CDI mechanism + P1 in-place patch + operator CLI all CORRECT under the corrected model; what changes is the customer-facing surface (which is now explicitly operator-facing, not stealth).

## Why this send is parallel to B.1'' commit (informs B.2'' substrate engineering, does not gate B.1'' commit)

P1 in-place patch sub-path locked per Anil 2026-05-26 customer-model-corrected adjudication. Devang answer about Nebius's existing host-package + DKMS + nvidia-container-toolkit version patterns informs:
- B.2'' Dockerfile / build-script assumptions about target host environment
- B.3'' verification matrix (do we add a Nebius-specific gate?)
- Runbook content (Nebius-specific post-driver-upgrade procedure)

Send timing: parallel to B.1'' commit (now). Devang response window typically 1-3 business days for Nebius technical surface. B.2'' substrate engineering window is ~3 ED; Devang signal lands at-or-before B.3'' verification begins.

---

## Related memory

- [[v1-phase-a-driver-worker-init]]: Phase A close + CUDA_INJECTION64_PATH that M1 CDI sets transparently
- [[v1-goal5-contract-lock]]: Goal 5 contract verbatim that M1 mechanism preserves for the end-user
- [[cipher-fusion-campaign]]: Devang is a stakeholder per campaign discipline
- [[cipher-cp25-closed]]: CP 2.5 cipher-platform v1.0 .deb precedent
