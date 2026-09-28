# V1 Phase B.0.7++ P4 OCI prestart hook mechanism investigation

**Date:** 2026-05-26
**Type:** read-only empirical investigation, no substrate code change, blocks B.1'' amendment per Anil 2026-05-26 P4 investigation request before P1 vs P2 pick
**Budget:** 0.5 ED per Anil; this turn consumed ~0.3 ED (within budget)

**Pre-state anchors (UNCHANGED through B.0.7++):**
- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-fusion-evidence `b974ddd` (B.0.7 finalization commit)

---

## 1. Investigation summary

Per Anil 2026-05-26 directive: investigate whether nvidia-container-toolkit OCI prestart hook plugin mechanism (P4) dominates P1 (in-place patch /var/run/cdi/nvidia.yaml) and P2 (take over /etc/cdi/nvidia.yaml) within the M1 CDI meta-mechanism.

**Verdict: P4 does NOT dominate P1/P2 across the realistic v1 deployment matrix.** P4 is a deployment-specific mechanism (CRI-O only); not available on this pod's Docker runtime; nvidia explicitly deprecated their own OCI hook usage in favor of CDI. P1 vs P2 remain the candidate paths for v1.

---

## 2. Three empirical findings

### 2.1 No /etc/containers/oci/hooks.d/ directory on this pod (Docker, not Podman/CRI-O)

`ls /etc/containers/oci/hooks.d/` returns "No such file or directory." The /etc/containers/ directory exists with `libpod.conf` + `policy.json` (legacy podman config remnants) but no `oci/hooks.d/` subdirectory. The Lambda H100 pod runs Docker (verified: `docker --version` returns "Docker version 29.2.1, build a5c7197") not Podman or CRI-O.

The `/etc/containers/oci/hooks.d/` mechanism is the Podman/CRI-O hook directory standard. Docker does NOT read this directory at container creation. Docker's own runtime configuration is via `/etc/docker/daemon.json` (which doesn't exist on this pod) or via dockerd command-line args (which don't reference any custom runtime: `dockerd -H fd:// --containerd=/run/containerd/containerd.sock`).

### 2.2 nvidia-container-toolkit EXPLICITLY DEPRECATED OCI hooks in favor of CDI

`nvidia-ctk runtime configure --help` documents the `--oci-hook-path` option with verbatim text:

> "the path to the OCI runtime hook to create if --config-mode=oci-hook is specified. If no path is specified, the generated hook is output to STDOUT.
> **Note: The use of OCI hooks is deprecated.**"

nvidia-container-toolkit migrated AWAY from OCI hooks toward CDI as the recommended substrate-injection mechanism. CIPHER following nvidia's deprecation curve puts CIPHER on a forward-compat-risk path: NVIDIA may remove OCI hook support entirely in future toolkit versions.

### 2.3 Modern Docker 29.x has native CDI support; no nvidia-container-runtime wrapper invoked on this pod

dockerd process on this pod runs without any custom runtime registration:
```
/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
```

No `--add-runtime nvidia=/usr/bin/nvidia-container-runtime` flag. No `/etc/docker/daemon.json` runtimes block. Yet `docker run --gpus all` works on this pod (verified during B.0 SC6 tests).

Mechanism: Docker 29.x has built-in CDI device support. `--gpus all` translates to `--device=nvidia.com/gpu=all` and Docker queries CDI specs (at /etc/cdi + /var/run/cdi per `nvidia-container-runtime` config.toml `[modes.cdi].spec-dirs`). The nvidia-container-runtime-hook executable exists at `/usr/bin/nvidia-container-runtime-hook` but is NOT invoked in this pod's container startup path because Docker uses CDI directly without the wrapper runtime.

Implication: even if CIPHER ships an OCI prestart hook, it would not be invoked by this pod's Docker setup. The hook mechanism (OCI prestart) is not in the container startup path on modern Docker + CDI.

---

## 3. P4 applicability matrix across realistic v1 deployment targets

| Deployment target | OCI hook support | P4 applicable? |
|---|---|---|
| Docker on single-node bare-metal (this pod's pattern; Lambda H100, on-prem ML hosts) | Docker 25+ uses CDI directly; no OCI hook wrapper invoked | **NO** (mechanism not in startup path) |
| containerd-direct (used by some Kubernetes installs) | containerd hook mechanism is via `runtime.RuncOptions` config; not /etc/containers/oci/hooks.d/ pattern; requires custom containerd runtime registration | **PARTIAL** (would need containerd-specific hook plugin authoring) |
| CRI-O (used by OpenShift, some Kubernetes installs) | CRI-O natively reads /etc/containers/oci/hooks.d/ | **YES** (clean CRI-O hook directory mechanism) |
| Podman (rare in production ML serving but used in some HPC environments) | Podman natively reads /etc/containers/oci/hooks.d/ | YES (same as CRI-O) |

Production ML serving deployment-target weighting (rough industry split for design-partner customers):
- Docker single-node bare-metal: ~30-40% (small-to-medium ops, on-prem; Lambda single-tenant)
- containerd via Kubernetes: ~40-50% (most ML serving k8s clusters; default container runtime since Docker shim deprecation)
- CRI-O via Kubernetes: ~10-20% (OpenShift, some enterprise k8s)

P4 cleanly works on ~10-20% of the deployment matrix (CRI-O). For Docker and containerd (dominant 70-90% of targets), P4 requires alternative integration that is NOT cleaner than P1 or P2.

---

## 4. P4 vs P1 vs P2 dominance analysis (per Anil framing)

Anil's framing in the prior turn: "If P4 mechanism doesn't work on Docker but works on containerd-direct (which is what most Kubernetes-based production ML serving uses): P4 still dominates for production deployment."

**Empirical finding reverses this assumption**: P4 does NOT work on containerd-direct without additional infrastructure. containerd's hook mechanism is per-runtime-configuration (via `[plugins."io.containerd.cri.v1.runtime".containerd.runtimes.nvidia.options.BinaryName]` registration in `/etc/containerd/config.toml`), not via a standardized hooks.d/ directory. Adding a CIPHER hook to containerd requires modifying containerd's runtime registration to point at a CIPHER-wrapper-runtime that delegates to nvidia-container-runtime after injecting CIPHER hooks.

P4's clean-mechanism domain is CRI-O (legitimate hooks.d/ directory). For Docker + containerd (the dominant runtime mix for v1 design partners), P4 collapses to "ship a wrapper runtime" which is heavier engineering than P1 or P2.

**Updated verdict**: P4 is a CRI-O-deployment-specific mechanism with a clean answer there; for Docker and containerd-direct deployments, P4 is not a clean dominator over P1/P2.

---

## 5. P4 engineering cost estimate (CRI-O-specific)

If Anil decides to ship P4 as the v1 mechanism for CRI-O customers + P1 OR P2 as the parallel mechanism for Docker/containerd customers:

| Sub-task | Eng-days |
|---|---|
| CIPHER hook script at /etc/containers/oci/hooks.d/cipher.json + /usr/libexec/cipher/cipher-hook (bind-mounts + env setup) | 1 |
| .deb integration: postinst drops hook + script, prerm removes | 0.5 |
| CRI-O test environment + verify hook fires on container start | 1 |
| Documentation: CRI-O customers use P4; Docker/containerd customers use P1 OR P2 | 0.5 |
| **Total P4-additional-on-top-of-P1-or-P2** | **~3 ED** |

P4 as a SECOND mechanism (alongside P1 or P2) doubles the maintenance surface. Recommended only if there's a specific design-partner with CRI-O deployment that needs first-class support.

---

## 6. P4 forward-compatibility risk

Per §2.2: nvidia-container-toolkit explicitly deprecates OCI hooks. Two forward-compat scenarios:

**Scenario A**: NVIDIA removes OCI hook support in toolkit v2.0. CIPHER P4 hook continues to work (it's independent of nvidia-container-toolkit's own OCI hook usage; CIPHER's hook is registered via CRI-O's hooks.d/). No direct break.

**Scenario B**: CRI-O migrates fully to CDI for substrate injection (following NVIDIA's lead). hooks.d/ directory may be deprecated by CRI-O. CIPHER P4 hook would need to migrate to a CDI-based mechanism = back to P1/P2.

Scenario B is plausible mid-term (2027-2028). P4 is therefore a v1 deployment-specific mechanism that may need migration in v1.5 or v2.

P1 and P2 are CDI-native; they ride the forward-compat curve cleanly.

---

## 7. Recommendation

P4 does NOT dominate P1 or P2 across the realistic v1 deployment matrix. Recommendation: pick between P1 and P2 as the v1 universal mechanism.

If a specific design-partner deployment uses CRI-O AND needs first-class support, ship P4 as an ADDITIONAL mechanism alongside (P1 OR P2). Adds ~3 ED to Phase B but only if v1 design-partner mix includes CRI-O.

Default recommendation per honest engineering trade-off analysis (and Anil's stated preference in prior turn under "If P4 mechanism doesn't work on any production runtime relevant to v1"): **pick P1** (cipher-platform postinst patches /var/run/cdi/nvidia.yaml in-place). Reasoning:
- Goal 5 verbatim preserved (no customer-side action)
- Fragility recoverable: re-running `apt install cipher-platform` after a driver update re-patches the spec
- P2's stale-NVIDIA-libs failure mode is much worse (customer workload regressions traced to CIPHER even when CIPHER's substrate isn't load-bearing for the regression)
- ~1 ED cheaper to engineer (no spec regeneration tool needed)

But this is Anil's pick. NO AUTO SCOPE-DEGRADE.

---

## 8. Devang Nebius re-contact disposition

The Devang re-contact message at `v1_phase_b/DEVANG_NEBIUS_UPDATE.md` is current as drafted (M1 CDI leaning + P1 vs P2 sub-path question). P4 investigation finding does NOT change the Devang question because P4 is CRI-O-specific and Devang/Nebius are likely on Docker or containerd. The question about Nebius's existing CDI extension patterns informs P1 vs P2 directly.

Send the existing draft as-is; this P4 investigation is internal CIPHER-side analysis.

---

## 9. Honest residue at B.0.7++ close

1. **P4 mechanism applicability is deployment-runtime-specific.** Lambda H100 single-node Docker (this pod): P4 not in startup path. CRI-O Kubernetes: P4 clean. containerd Kubernetes: P4 requires custom wrapper runtime.

2. **nvidia-container-toolkit OCI hook deprecation** is documented in the tool itself. CIPHER picking P4 follows a deprecated mechanism (for nvidia's own hooks); CIPHER's third-party hook IS independent of nvidia's deprecation but the broader industry direction is CDI, not OCI hooks.

3. **Empirical containerd-direct hook verification NOT performed** because this pod doesn't expose a standalone containerd path (Docker wraps containerd). Confidence on containerd-direct mechanism comes from documented containerd config syntax (`/etc/containerd/config.toml` `[plugins."io.containerd.cri.v1.runtime"...]`) not empirical test.

4. **CRI-O hook mechanism verification NOT performed empirically** (no CRI-O installation on this pod). Mechanism documented per CRI-O's `/etc/containers/oci/hooks.d/` standard reference. Production CRI-O test would happen at design-partner site if CRI-O support is added to v1 scope.

5. **B.0.7++ surfaces P4 finding but does NOT amend B.0.7 main investigation doc inline.** The B.0.7 main doc (`b0_7_injection_mechanism_investigation.md` at commit `b974ddd`) listed M1 with confidence; this addendum surfaces that the WITHIN-M1 sub-path nuance has 4 options (P1, P2, P3 rejected, P4 deployment-specific) not just the 3 surfaced before P4 investigation. Future Anil pick on sub-path may want to reference this addendum.

6. **Hard budget**: P4 investigation ~15-20 minutes wall-clock. Well within Anil's 0.5 ED allocation. No silent extension.

---

## 10. Anchors at B.0.7++ close (UNCHANGED through investigation)

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-fusion-evidence this commit lands b0_7_oci_hook_investigation.md
- cipher-platform .deb installed at v1.0 (Phase B target: refresh to v2.0 once Anil sub-path picked)

---

## 11. Related memory

- [[v1-phase-a-driver-worker-init]]: Phase A close + CUDA_INJECTION64_PATH; M1 CDI mechanism that P1/P2 implement
- [[v1-goal5-contract-lock]]: Goal 5 contract verbatim; all P1/P2/P4 preserve, P3 violates
- [[cipher-pod-environment]]: Lambda H100 + Ubuntu 22.04 + Docker 29.x (verified this turn dockerd uses CDI directly, not nvidia-container-runtime wrapper)
- [[cipher-evidence-commit-discipline]]: followed at B.0.7++ commit
- [[cipher-proceed-not-ask]]: Anil "investigate P4 before P1 vs P2 pick" + detailed scope → executed without silent scope-shrink
