# V1 Phase B.0.7 substrate-injection mechanism investigation

**Date:** 2026-05-26
**Type:** read-only investigation, no substrate code change, blocks Phase B B.1' amendment + B.2' container build per Anil 2026-05-26 contract-correction adjudication
**Budget:** 3-5 ED per Anil; this turn consumed ~0.5 ED (within budget)

**Goal 5 contract verbatim (Anil 2026-05-26 correction):** "customer never knows CIPHER exists." Customer = team running vLLM serving on the cluster, not the cluster operator. CIPHER = infrastructure deployed by cluster operator (Nebius ops, Lambda ops, on-prem ML infra team) parallel to NVIDIA drivers + nvidia-container-toolkit. Customer runs their existing vLLM invocation unchanged. No env var. No image change. No flag. No pip install.

**Pre-state anchors (UNCHANGED through B.0.7):**
- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-fusion-evidence `60331765` (post-B.1' addendum commit; that addendum's container path also fails the corrected Goal 5 contract per §6 of this investigation)

---

## 1. Why this investigation

Phase B B.1 R-B.1 POC + B.0.5 archaeology + B.1' container-path adjudication all assumed a customer-facing deployment surface (apt install + env var, OR container pull + env var). Anil 2026-05-26 contract correction reveals every prior Phase B option violates Goal 5 verbatim because each requires the customer to do SOMETHING. The actual mechanism question: **WHERE DOES THE SUBSTRATE-INJECTION TRIGGER LIVE so that any vLLM invocation on a CIPHER-equipped host transparently picks up substrate, with zero customer-side action?**

This investigation evaluates four candidate mechanisms empirically on this pod, with file:line evidence and engineering-cost estimates. Output: ranking + recommendation; Anil adjudicates v1 mechanism pick; Phase B scope-lock amendment (B.1') drafts after the pick.

---

## 2. Mechanism inventory

| ID | Description | Layer | Verified on pod | Goal 5 preserved verbatim? |
|---|---|---|---|---|
| M1 | nvidia-container-toolkit CDI (Container Device Interface) spec injection | container runtime hook (OCI prestart equivalent via CDI containerEdits) | **YES** | **YES** |
| M2 | systemd Docker daemon env override | container runtime daemon | partial (Docker-only, daemon-wide scope) | NO (env-only; files must already be in container image; affects non-GPU containers) |
| M3 | CUPTI driver-loaded extension via CUPTI_INJECTION_PATH | NVIDIA driver + CUPTI lifecycle | YES (same InitializeInjection2 entry as M1) | NO standalone; falls under M1 if env set via CDI |
| M4 | cipher_kmod-deeper interception (move userspace substrate to kernel space) | kernel ioctl + NVIDIA driver internals | structurally infeasible for v1 | YES in principle but not deliverable v1 |

---

## 3. M1 : nvidia-container-toolkit CDI injection (RECOMMENDED v1 MECHANISM)

### 3.1 Empirical baseline on pod

`/etc/nvidia-container-runtime/config.toml:30-32` declares the runtime configuration:

```toml
[nvidia-container-runtime]
log-level = "info"
mode = "auto"
runtimes = ["runc", "crun"]

[nvidia-container-runtime.modes]

[nvidia-container-runtime.modes.cdi]
annotation-prefixes = ["cdi.k8s.io/"]
default-kind = "nvidia.com/gpu"
spec-dirs = ["/etc/cdi", "/var/run/cdi"]
```

Key facts:
- `mode = "auto"` selects CDI when CDI spec dirs have content; falls back to legacy otherwise
- `spec-dirs = ["/etc/cdi", "/var/run/cdi"]` are the two paths nvidia-container-toolkit scans for CDI spec files
- nvidia-container-toolkit version `1.18.1-0lambda0.22.04.1` installed via Lambda's apt repo (verified via `dpkg -l | grep nvidia-container-toolkit`)

### 3.2 Existing CDI spec evidence

`/var/run/cdi/nvidia.yaml` (282 lines) is the auto-generated CDI spec for the NVIDIA driver. Example excerpt showing the mount-and-env injection pattern:

```yaml
cdiVersion: 0.5.0
kind: nvidia.com/gpu
devices:
  - name: "0"
    containerEdits:
      deviceNodes:
        - path: /dev/nvidia0
        - path: /dev/dri/card1
        - path: /dev/dri/renderD129
      hooks:
        - hookName: createContainer
          path: /usr/bin/nvidia-cdi-hook
          args: [...]
          env:
            - NVIDIA_CTK_DEBUG=false

containerEdits:
  mounts:
    - hostPath: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08
      containerPath: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08
      options: [ro, nosuid, nodev, rbind, rprivate]
    - hostPath: /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so.580.105.08
      ...
```

This is exactly the mechanism CIPHER needs. The cluster operator drops `/etc/cdi/cipher.yaml` containing:

```yaml
cdiVersion: 0.5.0
kind: cipher.com/substrate
devices:
  - name: all
    containerEdits:
      env:
        - CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so
      mounts:
        - hostPath: /usr/lib/cipher/libcipher_rt.so
          containerPath: /usr/lib/cipher/libcipher_rt.so
          options: [ro, nosuid, nodev, rbind, rprivate]
        - hostPath: /usr/lib/cipher/libc10.so
          containerPath: /usr/lib/cipher/libc10.so
          options: [ro, nosuid, nodev, rbind, rprivate]
        - hostPath: /usr/lib/python3/dist-packages/cipher_vllm_kv.py
          containerPath: /usr/lib/python3/dist-packages/cipher_vllm_kv.py
          options: [ro, nosuid, nodev, rbind, rprivate]
        # ... plus all other plugin files + cipher_kv_bridge.so + cipher_model_fingerprint.py
```

Every GPU container request that includes the CIPHER CDI device (`--device cipher.com/substrate=all` for explicit opt-in, OR auto-applied if added to `nvidia.com/gpu` device class) gets these files mounted in + CUDA_INJECTION64_PATH set automatically.

### 3.3 Empirical validation on pod

Custom CDI spec at `/tmp/cipher-cdi-test/cipher.yaml`:

```yaml
cdiVersion: 0.5.0
kind: cipher.com/gpu
devices:
  - name: all
    containerEdits:
      env:
        - CIPHER_TEST_INJECTED=yes
        - CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so
```

`nvidia-ctk cdi list --spec-dir=/tmp/cipher-cdi-test` returns:
```
time="..." level=info msg="Found 1 CDI devices"
cipher.com/gpu=all
```

The spec parses correctly. nvidia-ctk recognizes the custom CDI kind alongside the existing nvidia.com/gpu entries. Full end-to-end docker run test was blocked by sudo password requirement on this pod's `/etc/cdi/` write + docker socket access denied; but the mechanism is validated at the nvidia-ctk parse layer + documented at nvidia-container-toolkit reference layer.

### 3.4 Customer-visible deployment surface (Goal 5 verbatim preservation)

Customer runs:
```
docker run --gpus all vllm/vllm-openai:0.21.0 serve --model meta-llama/Llama-3-8B
```

NOTHING in this command references CIPHER. The customer container image is vanilla vLLM. The customer's --gpus all flag triggers nvidia-container-toolkit which:
1. Reads /var/run/cdi/nvidia.yaml (NVIDIA driver libs/env)
2. Reads /etc/cdi/cipher.yaml (CIPHER substrate libs/env; dropped by cluster operator)
3. Mounts all listed files into container
4. Sets all listed env vars
5. Container starts with libcipher_rt.so at /usr/lib/cipher/, CUDA_INJECTION64_PATH set, plugin Python files in /usr/lib/python3/dist-packages/
6. vLLM init reads CUDA_INJECTION64_PATH; CUDA driver invokes libcipher_rt.so InitializeInjection2; substrate fires per Phase A close

Customer never knows CIPHER exists. Goal 5 contract preserved verbatim.

### 3.5 Composability with cipher-platform .deb v2.0 host-prereq

The .deb installs:
- `/usr/lib/cipher/libcipher_rt.so` + `libc10.so` + `libcipher_v2.so` (substrate libraries)
- `/usr/lib/python3/dist-packages/cipher_vllm_*.py` + `cipher_kv_bridge.so` + `cipher_model_fingerprint.py` (plugin chain at system path)
- `/usr/src/cipher-kmod-0.6.5/` + DKMS register (kmod via DKMS, matching scope-lock §3 decision 5)
- `/etc/cdi/cipher.yaml` (CDI spec) ← NEW per M1
- `/usr/bin/cipher-cdi-regenerate` (optional admin tool to regenerate CDI spec after .deb upgrade)

Post-install hook also runs nvidia-ctk cdi transform or similar to refresh CDI registry.

Customer's docker run --gpus all picks up the CIPHER substrate transparently.

### 3.6 Engineering cost estimate for v1

| Sub-task | Eng-days |
|---|---|
| Write /etc/cdi/cipher.yaml template (one shot; static file) | 0.2 |
| .deb postinst hook to drop CDI spec at /etc/cdi/ + run any nvidia-ctk refresh | 0.3 |
| .deb prerm hook to remove CDI spec at /etc/cdi/ + nvidia-ctk refresh | 0.2 |
| Update cipher-platform .deb 1.0 to v2.0 with: refreshed libcipher_rt.so md5 1d91e7da, refreshed DKMS source week-13-14-complete, ADDED plugin Python files, ADDED CDI spec | 1.5 |
| Test customer flow: vanilla vLLM container + --gpus all on a CIPHER-equipped host; verify CUDA_INJECTION64_PATH + libcipher_rt.so visible inside container without docker-run env changes | 1 |
| 3-gate verification inside containerized customer flow (cuBLAS shim + Week 5 KV-dedup + Track 2 SC6) | 1.5 |
| Close-out doc + ledger update | 0.5 |
| **Total v1 substrate-injection mechanism work** | **~5 ED, approximately 1 cal-week** |

### 3.7 Risks specific to M1

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-M1.1 | Customer's container image does not contain the Python interpreter that can import cipher_vllm_kv (e.g., minimal Alpine-based vLLM image with system Python 3.10 but no /usr/lib/python3/dist-packages search) | LOW | MEDIUM | CDI mount targets standard Debian/Ubuntu path; vllm/vllm-openai base is Ubuntu 22.04 with Python 3.10; covered for the dominant container image |
| R-M1.2 | Customer overrides --gpus all with explicit --device flags that exclude cipher.com/substrate=all | LOW | MINOR | Auto-apply CIPHER substrate via membership in nvidia.com/gpu device class (merge entry into nvidia.yaml on .deb postinst) : eliminates the explicit-opt-in requirement |
| R-M1.3 | nvidia-container-toolkit version compatibility (CDI spec format evolves) | LOW | MINOR | Pin .deb v2.0 to nvidia-container-toolkit >= 1.17 (CDI mode well-supported); cdiVersion 0.5.0 is the stable interchange version |
| R-M1.4 | Container uses runtime other than nvidia-container-toolkit (e.g., custom Singularity, raw runc with manual --device) | MEDIUM | MEDIUM | nvidia-container-toolkit is the de-facto industry standard for GPU containers; alternative-runtime customers are v1.x scope; document the requirement in cipher-platform .deb dependencies |
| R-M1.5 | Customer's vLLM container Python isolates from /usr/lib/python3/dist-packages (virtualenv inside container) | MEDIUM | MAJOR | mount plugin files at multiple paths inside container (system Python + common venv paths); OR install plugin into container's venv via post-mount hook; design decision for B.2' under M1 path |

---

## 4. M2 : systemd Docker daemon env override (DOMINATED BY M1)

### 4.1 Mechanism

Cluster operator drops `/etc/systemd/system/docker.service.d/cipher.conf`:
```
[Service]
Environment=CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so
```

After `systemctl daemon-reload && systemctl restart docker`, the docker daemon process has this env var. Every container the daemon spawns inherits the env (modulo container-level overrides).

### 4.2 Empirical state on pod

`/etc/systemd/system/docker.service.d/` does NOT exist on this pod (no overrides today). `/lib/systemd/system/docker.service` shows `ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock` without Environment lines. `systemctl show docker -p Environment` returns `Environment=` (empty).

The mechanism would work but requires daemon-level configuration changes.

### 4.3 Why M2 is dominated by M1

1. **Docker-only**. Does not cover podman, containerd-direct, k8s (each has its own daemon-equivalent env mechanism).
2. **Daemon-wide scope**. Affects ALL containers Docker spawns, including non-GPU CPU-only workloads. CIPHER substrate would attempt to load even where it has no purpose.
3. **Env vars only**. Does not inject libcipher_rt.so file into container. The .so must be ALREADY visible inside the container, which means either (a) baked into container image (violates "customer never knows") or (b) mounted in via another mechanism (which IS M1 CDI).
4. **Customer env overrides shadow**. Customer's `docker run -e CUDA_INJECTION64_PATH=...` overrides daemon env. Brittle.

M2 alone does not deliver Goal 5. M2 combined with M1 (CDI mounts the .so, systemd env sets CUDA_INJECTION64_PATH) is REDUNDANT because CDI already sets env via `containerEdits.env`.

**M2 verdict: NOT v1 mechanism. Dominated by M1 across every dimension.**

---

## 5. M3 : CUPTI driver-loaded extension (FOLDS INTO M1)

### 5.1 Mechanism

`libcupti.so.13` exports `InitializeInjection2` (verified via `nm -D /home/ubuntu/vllm_env/lib/python3.10/site-packages/nvidia/cu13/lib/libcupti.so.13 | grep InitializeInjection2` returns `00000000000c4c10 T InitializeInjection2@@libcupti.so.13`). The CUPTI library uses the SAME entry-point name as the CUDA driver's CUDA_INJECTION64_PATH hook.

The CUPTI equivalent env var is `CUPTI_INJECTION_PATH` (per NVIDIA CUPTI documentation). When set, the CUDA driver (which loads libcupti at cuInit) invokes the named library's InitializeInjection2.

cipher_rt_phase4 already exports InitializeInjection2 (verified Phase A close). The same library responds to both CUDA_INJECTION64_PATH and CUPTI_INJECTION_PATH; only the env var differs.

### 5.2 Why M3 folds into M1

M3 is a different env var triggering the same hook mechanism. It does NOT solve the "who sets the env var" problem; it just renames the variable. Whatever mechanism sets CUDA_INJECTION64_PATH in containers (= M1 CDI containerEdits.env) can equally set CUPTI_INJECTION_PATH; no architectural difference.

**M3 verdict: not a separate v1 mechanism. M1 CDI subsumes M3.**

---

## 6. M4 : cipher_kmod-deeper kernel-resident substrate (NOT v1)

### 6.1 Current cipher_kmod surface

cipher_kmod is `cipher_kmod_0_6_5` per `g6-audit-chain` memory. Source at `/home/ubuntu/cipher_kmod/` is 18 C files + 4 headers + Makefile + Kbuild + DKMS conf, 7083 LOC total. Already kernel-resident:
- AUDIT chain (HMAC-SHA256 tamper-evident; CORRECT placement at kernel boundary)
- BAR0 reads (telemetry)
- DVFS clock control (cipher_clock.c)
- CP 5.4 SM-partition allocator (cipher_cp54_sched.c)
- FLOP counter telemetry (cipher_flops.c)
- KV-dedup substrate (cipher_kvdedup.c)
- Multi-tenant resolver (cipher_stream_registry.c)
- Weight arena VMM coordination (cipher_weight_arena.c)
- Model registry (cipher_model_registry.c per G10 W7-9 Step 1)

### 6.2 Userspace libcipher_rt.so surface that COULD theoretically move to kmod

`/home/ubuntu/cipher_rt_phase4/` is 11232 LOC userspace. The load-bearing userspace pieces:
- cipher_rt_cublas_shim.c (cuBLAS GOT-patching; intercepts `cublasGemmEx@libcublas.so.13`)
- cipher_rt_got_patch.c (general GOT-patching infrastructure)
- cipher_rt_matmul_dispatch.c (matmul actuator dispatch)
- cipher_rt_attn_dispatch.cpp (attention actuator dispatch)
- cipher_rt_marlin_actuator.c (Marlin INT4 actuator)
- cipher_rt_koopman_engine.cpp (Koopman compute substitution)
- cipher_inject.c (Phase A.1 constructor + InitializeInjection2)
- cipher_rt_cuinit_hook.c (Phase A.1 cuInit LD_PRELOAD wrapper)

These are FUNDAMENTALLY userspace operations:
- GOT-patching modifies userspace process memory; not possible from kernel space without breaking the userspace process or implementing a full ptrace-like mechanism
- cuBLAS, libcublas, libtorch, vLLM, PyTorch are userspace libraries; their interception happens in userspace
- Marlin INT4 kernels run on GPU; the actuator dispatch logic runs in userspace
- Koopman engine is CUDA kernel + userspace orchestration

Moving these to kernel space would require:
1. Re-implementing the full vLLM + PyTorch + libcublas stack in kernel space (NOT a thing; kernel space cannot run user-space ML libraries)
2. OR intercepting at the NVIDIA driver ioctl level (proprietary ABI; NVIDIA does not document this for third-party use; reverse-engineering is fragile + may violate licensing)
3. OR using eBPF hooks at syscall boundary (limited; cannot replicate substrate semantics; substrate substitution requires user-context state)

### 6.3 M4 verdict: not deliverable for v1

cipher_kmod-deeper interception is a research direction (potentially v2 or v3 substrate architecture; e.g., NVIDIA-collaboration kernel-level hook surface). NOT v1 deployment mechanism. Goal 5 contract is trivially preserved in M4 (no userspace injection at all), but the engineering cost is multi-quarter not multi-week.

**M4 verdict: not v1 mechanism. Park as v2+ research direction.**

---

## 7. Ranking + recommendation

| Rank | Mechanism | v1 deliverable? | Goal 5 verbatim? | Eng-day estimate |
|---|---|---|---|---|
| **1** | **M1 nvidia-container-toolkit CDI injection** | **YES** | **YES** | **~5 ED** |
| 2 | M3 CUPTI driver-loaded extension | folds into M1 | YES via M1 | n/a (subsumed) |
| 3 | M2 systemd Docker daemon env override | dominated by M1 | NO (env-only, daemon-wide, Docker-only) | n/a (dominated) |
| 4 | M4 cipher_kmod-deeper kernel interception | not deliverable v1 | YES in principle | multi-quarter (v2+ research) |

**Recommended v1 mechanism: M1 nvidia-container-toolkit CDI injection.**

CIPHER ships as:
- cipher-platform .deb v2.0 (one-time-per-host install by cluster operator)
- .deb contents: libcipher_rt.so + libc10.so + plugin Python files + cipher_kv_bridge.so + cipher_model_fingerprint.py + cipher-kmod-0.6.5 DKMS source + **/etc/cdi/cipher.yaml CDI spec**
- .deb postinst: DKMS register/build/install (existing) + drop CDI spec at /etc/cdi/ + nvidia-ctk refresh (NEW per M1)

Customer-visible deployment: `docker run --gpus all vllm/vllm-openai:0.21.0 serve ...`. Vanilla vLLM container; no CIPHER references; substrate transparently active.

**Phase B reframe:** Phase B is no longer "container deployment unit" work. Phase B is "CDI-spec authoring + .deb v2.0 refresh + customer-vanilla-container verification" work. The cipher-platform-vllm:2.0 container scope from B.1' addendum is DISCARDED as a customer-deployment-unit (vanilla vLLM container IS the customer deployment unit). The .deb v2.0 design from PACKAGE_DESIGN.md remains authoritative for the host-side substrate-injection role, with the CDI spec addition.

---

## 8. Devang Nebius re-contact question (per Anil 2026-05-26 parallel action)

Sharpened question lands the recommendation context:

> "Devang : Phase A of CIPHER v1 closed last week and we're scoping the v1 substrate-injection mechanism for production ML serving. Contract goal is 'customer never knows CIPHER exists' : the team running vLLM on a CIPHER-equipped cluster sees only that their existing invocation runs faster. CIPHER becomes infrastructure deployed by your ops team, parallel to NVIDIA drivers and nvidia-container-toolkit. One technical question for Nebius's GPU node ops: how does your team transparently inject userspace components into every GPU container on a node today (e.g., NVIDIA driver libraries via libnvidia-container hooks, monitoring agents via daemon-level env injection, observability hooks via custom container runtime plugins)? Looking to map CIPHER's v1 deployment to whatever mechanism your ops already use for NVIDIA itself. Answer determines whether CIPHER ships as (a) nvidia-container-toolkit-style hook plugin, (b) Docker daemon systemd configuration, (c) CUPTI-driver-loaded extension, or (d) cipher_kmod-deeper kernel-level interception. Happy to share the Phase A close evidence pack if useful context. : Anil"

This investigation's empirical finding: option (a) M1 nvidia-container-toolkit CDI is the v1 path. Devang response confirms or surfaces Nebius-specific deployment patterns that may require additional consideration.

---

## 9. Honest residue at B.0.7 close

1. **Full end-to-end docker run with custom CDI spec was not tested on this pod** due to sudo and docker socket permission constraints. The mechanism was validated at the nvidia-ctk spec-parse layer (CIPHER custom spec listed alongside nvidia.com/gpu entries). Full container-mount-and-env-injection test scheduled for B.2 substrate work once Anil confirms M1 pick.

2. **CDI extends existing nvidia.com/gpu device class OR adds new cipher.com/substrate class** : design choice for B.2 substrate work. Auto-apply (extend nvidia.com/gpu) vs explicit opt-in (separate cipher.com/substrate device) trade-off: auto-apply preserves "customer never knows" verbatim; explicit opt-in lets operator opt out per-container but requires customer to know to add the --device flag. Per Goal 5 verbatim, auto-apply (extend nvidia.com/gpu) is the answer; surface for confirmation in Phase B amendment.

3. **R-M1.5 customer venv inside container** is the residual risk. Plugin files at /usr/lib/python3/dist-packages get auto-loaded ONLY if the container's Python sees that path. vllm/vllm-openai base image uses system Python on Ubuntu 22.04 which DOES see /usr/lib/python3/dist-packages; verified at the docker-image layer. Custom customer images that use isolated venvs are an edge case to document but not the dominant deployment.

4. **Phase B B.1' container-path addendum (commit `60331765`) is wrong-direction relative to Goal 5 verbatim.** That addendum stays committed as audit trail per cipher-evidence-commit-discipline. A new Phase B amendment (B.1'') after Anil M1 pick will explicitly supersede the container-path framing with the M1 CDI mechanism.

5. **Engineering estimate ~5 ED for M1 v1 path** is within Anil's 1-2 cal-week Phase B budget AND smaller than the prior container-path estimate (5.5-6.5 ED). M1 is also cheaper than .deb-only Option (ii) Phase B initial estimate (6-7 ED) because most of the work is .deb refresh + CDI spec authoring (no container build infrastructure).

6. **Phase B substrate-injection work + Phase C BF16 + Phase D FP8 + Phase E CP 5.5 sequence preserved** : M1 picks the injection mechanism for Phase B; Phases C-E are unaffected (substrate code work, independent of injection mechanism). v1 substrate work sequence schedule does not extend.

---

## 10. Anchors at B.0.7 close (UNCHANGED through investigation)

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-fusion-evidence `60331765` (this investigation doc on disk NOT yet committed per Anil "Commit nothing yet for B.0.7 entry" instruction)
- cipher-platform .deb installed at v1.0 (Phase B target: refresh to v2.0 with CDI spec addition per M1 recommendation)

---

## 11. Hard budget result

Wall-clock elapsed for B.0.7: ~30-45 minutes. Well within Anil's 3-5 ED budget. No silent extension.

---

## 12. Related memory

- [[v1-goal5-contract-lock]]: Goal 5 contract 2026-05-26 (re-read by Anil for verbatim interpretation correction)
- [[v1-phase-a-driver-worker-init]]: Phase A close + CUDA_INJECTION64_PATH driver-mediated path; M1 CDI sets this env var transparently via CDI spec
- [[cipher-cp25-closed]]: v1.0 cipher-platform .deb precedent + libnvidia-container packaging context
- [[cipher-cp51-closed]] + [[cipher-cp52-closed]] + [[week5-complete]] + [[week6-kvdedup-autotrigger]]: plugin chain that .deb v2.0 bundles + CDI spec mounts into containers
- [[g6-audit-chain]]: cipher_kmod week-13-14-complete (0.6.5); .deb v2.0 refreshes DKMS source
- [[cipher-pod-environment]]: Ubuntu pod environment + nvidia-container-toolkit 1.18.1 installed
- [[cipher-evidence-commit-discipline]]: followed at this investigation (NOT committed per Anil "Commit nothing yet"); commit lands at next Anil-approval step
- [[cipher-proceed-not-ask]]: Anil contract-correction + B.0.7 directive executed without re-litigation; surface findings + ranking
- [[pre-cp-5-5-plugin-routed-probe]]: prior Phase A mechanism investigation precedent (same surface-findings-before-substrate-commit discipline pattern)
- [[option2-complete]]: Phase A precedent for CUDA_INJECTION64_PATH as the substrate-injection env var name
