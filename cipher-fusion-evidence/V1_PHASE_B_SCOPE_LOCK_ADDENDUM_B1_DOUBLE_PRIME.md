# V1 Phase B scope-lock addendum: B.1'' M1 CDI + P1 in-place patch + neocloud-ops customer model

**Date:** 2026-05-26
**Amendment to:** `V1_PHASE_B_SCOPE_LOCK.md` (cipher-fusion-evidence `e33d62d`) + `V1_PHASE_B_SCOPE_LOCK_ADDENDUM_B1_PRIME.md` (cipher-fusion-evidence `60331765`)
**Trigger:** Anil 2026-05-26 customer-model correction after B.0.7++ P4 OCI hook investigation closed (cipher-fusion-evidence `4486fde`)
**Type:** addendum per cipher addendum convention. Original scope-lock + B.1' addendum preserved verbatim as wrong-direction audit trail; B.1'' supersedes both per §3 below.

---

## 1. Corrected customer model (NEW; supersedes implicit customer model carried through B.0.5 → B.0.7 → B.0.7++)

CIPHER's go-to-market is **B2B infrastructure sold to neoclouds**. The customer model has TWO distinct actors with distinct contract scopes:

| Actor | Role | Sees CIPHER? | Contract with CIPHER? |
|---|---|---|---|
| **Neocloud ops team** (Nebius, Lambda, CoreWeave, RunPod, on-prem ML infra ops) | CIPHER's CUSTOMER | YES (they bought it) | YES (B2B infrastructure contract; docs, CLI, runbook, support) |
| **Neocloud's customer** (ML team renting GPU compute; runs vLLM workloads, training, inference) | END-USER of CIPHER's substrate | NO (transparent) | NO (their contract is with the neocloud) |

**Goal 5 contract verbatim applies to the END-USER only:** "customer never knows CIPHER exists" = the ML team running `docker run --gpus all vllm/vllm-openai:0.21.0` invocations on neocloud GPU compute sees only faster inference; their invocation is unchanged; no flags, no env vars, no pip installs, no Python code changes, no plugin install awareness.

**The neocloud ops team is a normal infrastructure operator.** They install cipher-platform .deb via standard apt, run standard verify, follow a documented post-driver-update refresh procedure, monitor via standard journald, contact CIPHER support per standard B2B vendor channels. The neocloud ops team KNOWS CIPHER exists (they BOUGHT it). Operational requirements (postinst banner, CLI tool, runbook, periodic refresh after upstream NVIDIA driver upgrade) are NORMAL B2B infrastructure-operator UX, not Goal 5 violations.

**What this corrects:** B.0.5 → B.0.7 → B.0.7++ investigation was framed as "make CIPHER invisible to whoever installs it." That framing collapsed the two actors and over-constrained the deployment design. Under the corrected model:
- M1 nvidia-container-toolkit CDI mechanism: STILL CORRECT (this is how the substrate gets into end-user GPU containers without the end-user knowing)
- P1 in-place patch of /var/run/cdi/nvidia.yaml: STILL CORRECT mechanism layer
- Goal 5 verbatim for the end-user: PRESERVED
- Operational fragility under driver updates: NO LONGER a contract violation (handled by the neocloud ops team's standard ops process via documented `sudo cipher-platform refresh`)
- systemd path-watcher auto re-patch: DEMOTED from requirement to CONVENIENCE (ship if budget allows; .deb without it is shippable)
- .deb postinst may print operator-facing banners and ship operator-facing tooling

---

## 2. Pre-state anchors (UNCHANGED through B.1'')

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
- cipher-platform .deb at v1.0 (CP 2.5 close artifact; refreshed to v2.0 by this addendum's substep map)
- cipher-fusion-evidence `4486fde` (B.0.7++ P4 investigation commit; this addendum lands at next commit)

---

## 3. Sections of prior scope-lock + B.1' addendum superseded by this addendum

| Source doc | Section | Superseded by B.1'' section |
|---|---|---|
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §1 Why this campaign (Option (ii) packaging path framing) | §4 Why M1 CDI + P1 + operator-facing tooling |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §3 decision 1 (customer deployment recipe via apt + env var; original collapsed customer model) | §5 decision 1 (corrected customer model, neocloud ops as install actor, end-user is invisible) |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §3 decision 5 (Phase B is cipher-platform package UPDATE) | §5 decision 5 (Phase B is cipher-platform .deb v2.0 refresh with CDI patch + operator CLI + runbook) |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §4 substeps B.1 through B.5 (.deb packaging substeps) | §7 substeps B.1'' through B.5'' |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §5 total estimate (6 ED Branch A) | §8 total estimate (5-6 ED Branch A) |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §6 risks R-B.1 through R-B.7 (.deb risks) | §9 risks R-B.1'' through R-B.8'' (CDI-specific + operator-CLI-specific) |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §8 Branch A/B/C/D framing (.deb verdicts) | §10 Branch framing (CDI-patch verdicts) |
| `V1_PHASE_B_SCOPE_LOCK.md` (original) | §11 framing for B.5 close-out | §12 framing for B.5'' close-out (B2B infrastructure to neoclouds) |
| `V1_PHASE_B_SCOPE_LOCK_ADDENDUM_B1_PRIME.md` (B.1') | ENTIRE container-path scope | SUPERSEDED IN FULL by this B.1'' (B.1' kept as wrong-direction audit trail per cipher addendum convention; container path violated original Goal 5 framing AND collapsed customer model) |

Sections of original scope-lock NOT superseded (still authoritative):
- §2 The strong prior (three measurement gates per Phase A precedent)
- §3 decision 3 (substrate code UNCHANGED through Phase B; substrate anchors do not rotate)
- §7 honest residue items 1-7 (carry verbatim)
- §12 related memory (carry verbatim, plus this addendum adds references)

---

## 4. Why M1 CDI + P1 in-place patch + operator-facing tooling (supersedes original §1 and B.1' §4)

**M1 mechanism:** nvidia-container-toolkit CDI spec injection. The same mechanism NVIDIA itself uses to bind-mount libnvidia-ml.so + libcuda.so into every `docker run --gpus all` container carries CIPHER's libcipher_rt.so + plugin chain transparently. End-user docker run is UNCHANGED.

**P1 sub-path:** cipher-platform .deb v2.0 postinst patches `/var/run/cdi/nvidia.yaml` in-place to add CIPHER mounts + CUDA_INJECTION64_PATH env to the existing `nvidia.com/gpu=all` device's `containerEdits`. P1 was the recommended sub-path at B.0.7++ close (cipher-fusion-evidence `4486fde`) per:
- P4 OCI hook mechanism doesn't apply on modern Docker (NVIDIA explicitly deprecated; not in startup path on this pod's containerd-direct runtime)
- P2 (take-over /etc/cdi/nvidia.yaml) has worse failure mode (CIPHER owns NVIDIA's spec; stale-NVIDIA-libs regressions trace to CIPHER)
- P3 (separate cipher.com/gpu kind) violates Goal 5 verbatim (requires customer --device flag)
- P1's main fragility (driver-update wipes patch) is RECOVERABLE via documented `sudo cipher-platform refresh` (normal B2B ops UX per §1 corrected customer model)

**Operator-facing tooling (NEW per corrected customer model):** the neocloud ops team is a normal infrastructure operator. cipher-platform .deb v2.0 ships:
- `/usr/bin/cipher-platform` CLI (status / verify / refresh / version / logs)
- `/usr/lib/cipher/runbook.md` operator runbook
- `/usr/lib/cipher/health-check.sh` standalone health check
- Optional `cipher-platform-watch.service` systemd path-watcher (auto re-patch convenience)
- Structured journald logging via `cipher-platform.service`

This tooling was previously misframed as "leaking CIPHER's existence." Under the corrected customer model, it is normal infrastructure-operator UX and improves the product's surface area as a B2B infrastructure sale.

Goal 5 contract verbatim PRESERVED for the end-user: `docker run --gpus all vllm/vllm-openai:0.21.0 ...` is UNCHANGED. The substrate fires transparently via host's nvidia-container-toolkit reading the patched CDI spec.

---

## 5. Locked decisions (supersedes original §3 decisions 1, 5; decision 3 preserved)

1. **Goal 5 contract preserved verbatim for the END-USER.** End-user (ML team renting GPU compute) deployment surface: `docker run --gpus all vllm/vllm-openai:0.21.0 serve --model meta-llama/Llama-3-8B` runs UNCHANGED. No flags, no env vars, no pip installs, no Python code changes, no plugin install awareness.

2. **Neocloud ops team is the install actor.** Standard B2B infrastructure-operator UX:
   - One-time-per-host: `apt install cipher-platform` (installs substrate libs + plugin chain + DKMS kmod source + CDI patch script + operator CLI + runbook + optional systemd watcher)
   - Per-driver-upgrade: documented `sudo cipher-platform refresh` step (or auto-handled if optional systemd watcher shipped)
   - Per-cluster-verification: `cipher-platform verify` runs minimal health check
   - Standard journald logging via `cipher-platform.service`

3. (preserved from original §3 decision 3) **Substrate code UNCHANGED through Phase B.** cipher_rt_phase4 `8613812e`; libcipher_rt.so md5 `1d91e7da`; cipher_kmod `8c643fc`; cipher_vllm_plugin md5 `b89a9b6e`. Phase B is packaging-and-deployment infrastructure work, not substrate engineering.

4. **Ubuntu 22.04 .deb with Python 3.10 ABI lock.** Preserves Python 3.10 ABI match with `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so`. v1 ships Ubuntu 22.04 only; v1.x adds 24.04 if neocloud design partners need.

5. **Phase B ships ONE artifact: cipher-platform .deb v2.0** (refresh of CP 2.5 v1.0). Bundles:
   - Substrate libraries (libcipher_rt.so + libc10.so + libcipher_v2.so) at `/usr/lib/cipher/`
   - Plugin chain (cipher_vllm_kv + cipher_vllm_kvdedup + cipher_kv_offload + cipher_kv_bridge.so + cipher_model_fingerprint + dist-info entry points) at `/usr/lib/python3/dist-packages/`
   - DKMS source `/usr/src/cipher-kmod-0.6.5/` (refreshed from CP 2.5 era 0.4.8 to week-13-14-complete)
   - CDI patch script `/usr/lib/cipher/cipher_cdi_patch.sh` (P1 in-place patch of /var/run/cdi/nvidia.yaml)
   - Operator CLI `/usr/bin/cipher-platform`
   - Operator runbook `/usr/lib/cipher/runbook.md`
   - Standalone health check `/usr/lib/cipher/health-check.sh`
   - Optional systemd path-watcher unit `cipher-platform-watch.service` (ship if B.2'' budget allows; .deb without it is shippable)
   - postinst: DKMS register/build/install + cipher_cdi_patch.sh first-apply + operator banner
   - prerm: DKMS unregister + revert CDI patch + remove operator-facing files

---

## 6. PACKAGE_DESIGN.md disposition (preserved; updated annotation)

`v1_phase_b/PACKAGE_DESIGN.md` is preserved on disk. Header annotation already noting prior B.1' container-path reframe is updated to note B.1'' supersedes BOTH and the .deb v2.0 role is the v1 deployment artifact under M1 CDI + P1 + operator-facing tooling. Sections 1-2 (cipher-platform v1.0 baseline + v2.0 metadata) become authoritative again for the v2.0 .deb design under B.1''. Section 3 (customer deployment recipe) is SUPERSEDED by §5 decision 1+2 of this addendum (corrected customer model). Sections 6, 7, 8, 9 (informational reference) retained.

The B.1' container-path addendum is preserved as wrong-direction audit trail (cipher addendum convention).

---

## 7. Five-substep sequence M1 CDI + P1 + operator CLI (supersedes original §4 and B.1' §7)

### B.1'' (this addendum, complete at this commit, ~0.5 ED actual)

This addendum lands the scope amendment. Devang Nebius update message refreshed at `v1_phase_b/DEVANG_NEBIUS_UPDATE.md` per Anil 2026-05-26 corrected-customer-model directive.

**Verification gate at B.1'' close (this addendum commit):**
- This addendum file lands on disk
- DEVANG_NEBIUS_UPDATE.md refreshed (action item for user to send externally; not gating B.2'' start)
- Surface to Anil for explicit "proceed to B.2''" signal before substrate engineering starts

### B.2'' .deb v2.0 build pipeline (~2-3 ED + ~0.5-0.75 ED operator tooling)

**Touches:**
- New `v1_phase_b/packaging/debian/control` (cipher-platform v2.0 metadata; depends nvidia-container-toolkit >= 1.17, dkms, python3, python3-pip)
- New `v1_phase_b/packaging/debian/postinst` (DKMS register/build/install + cipher_cdi_patch.sh first-apply + banner)
- New `v1_phase_b/packaging/debian/prerm` (DKMS unregister + revert CDI patch + remove operator-facing files)
- New `v1_phase_b/packaging/cipher_cdi_patch.sh` (P1 in-place patch script; idempotent; reads /var/run/cdi/nvidia.yaml, ensures CIPHER entries present in nvidia.com/gpu=all containerEdits, writes atomic in-place)
- New `v1_phase_b/packaging/cipher_platform_cli.py` (`/usr/bin/cipher-platform`; subcommands status / verify / refresh / version / logs)
- New `v1_phase_b/packaging/runbook.md` (operator runbook)
- New `v1_phase_b/packaging/health-check.sh` (standalone health check)
- Optional new `v1_phase_b/packaging/cipher-platform-watch.service` + `cipher-platform-watch.path` (systemd path-watcher; ship if budget allows)
- New `v1_phase_b/packaging/build_deb.sh` (debuild wrapper)

**postinst responsibilities:**
- DKMS register/build/install for cipher-kmod-0.6.5 source
- Run cipher_cdi_patch.sh first-apply on /var/run/cdi/nvidia.yaml
- Enable + start cipher-platform-watch.service IF the optional systemd watcher shipped in this build
- Print operator banner: "cipher-platform v2.0 installed. Run `sudo cipher-platform verify` to test substrate health. After future NVIDIA driver upgrades, run `sudo cipher-platform refresh` to re-apply CDI patch."

**cipher-platform CLI subcommands:**
- `cipher-platform status`: print CDI patch applied (yes/no + summary of injected entries), substrate libcipher_rt.so version + md5, kmod loaded (yes/no) + ABI version, plugin dist-info entry-points registered (yes/no)
- `cipher-platform verify`: run minimal health check (substrate library loadable, kmod /dev/cipher accessible, CDI patch present in /var/run/cdi/nvidia.yaml, plugin entry-points discoverable). Exit 0 PASS, exit 1 FAIL.
- `cipher-platform refresh`: re-apply CDI patch (operator-invokable; what gets called by systemd watcher if shipped)
- `cipher-platform version`: print .deb version + substrate libcipher_rt.so md5 + kmod version + plugin version
- `cipher-platform logs`: tail recent CIPHER-related entries from systemd journal (`journalctl -u cipher-platform.service --since '1 hour ago'`)

**runbook.md sections:**
1. Post-install verification steps (`apt install` → `cipher-platform verify`)
2. Post-driver-update refresh procedure (`apt upgrade nvidia-driver-*` → `sudo cipher-platform refresh` → `cipher-platform verify`)
3. Common debugging scenarios (CIPHER not firing in container, kmod loading issues, CDI patch missing, journald log inspection)
4. Uninstall + clean-room recovery (`apt remove cipher-platform` → verify /var/run/cdi/nvidia.yaml CIPHER entries removed → verify cipher_kmod unloaded)
5. Contact info for CIPHER support

**Estimate detail:**
- debian/control + postinst + prerm: ~0.5 ED
- cipher_cdi_patch.sh (idempotent P1 patch): ~1 ED
- cipher_platform_cli.py: ~0.5 ED
- runbook.md + health-check.sh: ~0.25 ED
- build_deb.sh + first build + smoke install on this pod: ~0.5-0.75 ED
- Optional systemd watcher: +0.25 ED if shipped
- Total B.2'': ~2.75-3.25 ED (+0.25 if systemd watcher)

**Close gate for B.2'':**
- `cipher-platform_2.0.0_amd64.deb` builds clean
- `dpkg -i` smoke install on this pod succeeds (postinst runs, banner prints, CDI patch applies, kmod builds via DKMS)
- `cipher-platform status` reports all green
- `cipher-platform verify` exits 0

### B.3'' three-gate verification inside vanilla vLLM container + CLI smoke (1-1.5 ED)

**Methodology:** Run gates inside a VANILLA vLLM container (`docker run --gpus all vllm/vllm-openai:0.21.0 ...`) on this pod with cipher-platform .deb v2.0 installed on the host. CDI mechanism transparently injects substrate; gates measure substrate firing.

| Gate | Methodology | Threshold |
|---|---|---|
| Gate A: cuBLAS shim | vanilla vLLM container with arbitrary serve; check `/dev/cipher` ioctl `cipher_rt_cublas_shim_calls >= 11000` from host-side reader | shim_calls >= 11000 |
| Gate B: Week 5 KV-dedup | vanilla vLLM container with TinyLlama N=4 + Mistral-7B N=4 Week 5 Step 3 harnesses (harnesses ship as part of .deb evidence or run from host orchestrator) | TinyLlama N=4 HBM saved >= 45000 MiB; Mistral-7B N=4 KL=0 + hit >= 80% |
| Gate C: Track 2 SC6 | vanilla vLLM container running sc6_run.py both phases + sc6_aggregate.py | Mistral-7B N=4 savings >= 75% |
| Gate E (NEW per corrected customer model): cipher-platform CLI smoke | `cipher-platform status` + `cipher-platform verify` + `cipher-platform version` all succeed; `cipher-platform refresh` re-applies cleanly (idempotent) | all subcommands exit 0; verify reports all-green |

All gates run with .deb v2.0 installed on host and substrate firing transparently in vanilla vLLM container = deployment-unit verification under corrected customer model.

**Output evidence:** `v1_phase_b/results/b3_three_gate_under_deb.json` + `v1_phase_b/results/b3_cli_smoke.json`.

### B.4'' uninstall test (0.5 ED)

`apt remove cipher-platform` cleanup test. Verify CDI patch reverted in /var/run/cdi/nvidia.yaml, cipher_kmod unloaded + DKMS source removed, /usr/lib/cipher/ removed, /usr/bin/cipher-platform removed, plugin dist-info entry-points removed. Re-run vanilla vLLM container smoke after removal: substrate should NOT fire (no /dev/cipher access; libcipher_rt.so not present); confirms clean uninstall.

**Output evidence:** `v1_phase_b/results/b4_uninstall.json`.

### B.5'' close-out (~0.5 ED)

V1_PHASE_B_COMPLETE.md draft frames cipher-platform .deb v2.0 as v1 deployment artifact + B2B infrastructure sold to neoclouds. Ledger row added per ledger-gate discipline naming the neocloud as the install actor. Tag candidate `v1-substrate-cdi-platform` on cipher-fusion-evidence at close-out commit.

---

## 8. Total estimate (supersedes original §5 and B.1' §8)

| Branch | B.1'' | B.2'' | B.3'' | B.4'' | B.5'' | Total |
|---|---|---|---|---|---|---|
| **Branch A** (.deb builds clean, all gates PASS, CLI smoke PASS, uninstall clean) | 0.5 | 2.75-3.25 | 1-1.5 | 0.5 | 0.5 | **5.25-6.25 ED** approximately 1-1.5 cal-weeks |
| **Branch B** (.deb builds, 1 of 4 gates fails) | 0.5 | 2.75-3.25 | 1-1.5 | 0.5 | 0.5 (surface) | **5.25-6.25 ED** before surface |
| **Branch C** (gates PASS but uninstall leaves leak) | 0.5 | 2.75-3.25 | 1-1.5 | 1-1.5 (debug) | 0.5 | **6.25-7.25 ED** approximately 1.5 cal-weeks |
| **Branch D** (CDI mechanism fundamentally fails; e.g., libcipher_rt.so does not load in container via CDI mount, or kmod ioctl from container fails) | 0.5 | 1-2 (debug + surface) | 0 | 0 | 0.5 (surface) | **2-3 ED** before surface |

Range **2-7.25 ED** matches Anil 1-2 cal-week Phase B budget. +1 ED versus stealth-deployment misframing budget; the +1 ED buys the operator-facing CLI + runbook surface that improves the B2B infrastructure product.

---

## 9. Risks (supersedes original §6 R-B.1 through R-B.7 and B.1' §9 R-B.1' through R-B.8')

| ID | Risk | Likelihood | Impact | Mitigation | Detection |
|---|---|---|---|---|---|
| **R-B.1''** | nvidia-container-toolkit < 1.17 (no CDI support) on target neocloud host | MEDIUM | BLOCKER | cipher-platform v2.0 .deb depends nvidia-container-toolkit >= 1.17; dpkg refuses install on older versions with clear error; runbook §1 documents prerequisite | postinst checks toolkit version; Devang Q2 confirms Nebius posture |
| **R-B.2''** | NVIDIA driver upgrade wipes CIPHER's in-place CDI patch (P1 mechanism's known fragility) | HIGH | MAJOR if undetected | Documented `sudo cipher-platform refresh` in runbook §2; optional systemd path-watcher (cipher-platform-watch.service) auto-recovers if shipped; under corrected customer model, this is NORMAL B2B ops UX not a contract violation | `cipher-platform verify` detects missing patch; systemd watcher detects file change if shipped |
| **R-B.3''** | postinst's cipher_cdi_patch.sh races with concurrent nvidia-ctk cdi generate | LOW | MINOR | Atomic write via temp-file + rename; documented runbook step "do not run apt install during driver upgrade" | postinst smoke + B.3'' Gate A would surface mismatch |
| **R-B.4''** | cipher-platform CLI tool surface creep beyond v1 budget | LOW | MINOR | Lock v1 CLI to 5 subcommands (status / verify / refresh / version / logs); additional subcommands (metrics endpoint, etc.) deferred to v1.x | scope-lock in §5 decision 5 |
| **R-B.5''** | DKMS kmod build fails on neocloud host kernel (kernel-headers mismatch) | MEDIUM | MAJOR | cipher-platform .deb depends `linux-headers-generic` (Ubuntu 22.04 standard); runbook §3 debugging section covers DKMS troubleshooting; CP 2.5 v1.0 precedent worked on Lambda H100 pod with kernel 6.8.0-1046-nvidia | postinst DKMS build failure surfaces with explicit error |
| **R-B.6''** | systemd path-watcher (if shipped) over-triggers and thrashes nvidia.yaml | LOW | MINOR | Debounce in watcher (5s settling window); cipher-platform refresh idempotent (no-op if already patched); ship watcher as OPTIONAL convenience, not requirement | watcher logs to journald; B.2'' smoke includes settle-test |
| **R-B.7''** | end-user vLLM container has custom /etc/cdi/ override that takes precedence over /var/run/cdi/nvidia.yaml | LOW | MAJOR | Runbook §3 covers this debugging case; cipher-platform verify reports source of effective CDI spec; v1.x adds /etc/cdi/ patch path if neocloud design partners need | verify subcommand checks effective spec source |
| **R-B.8''** | R-A.1-equivalent caveat: unexpected mechanism finding during B.2'' or B.3'' (analogous to Phase A cuGetProcAddress or B.0.7 CDI collision) | MEDIUM | MAJOR | SURFACE IMMEDIATELY per scope-lock §7 R-A.1-equivalent caveat; do NOT push through silently; HARD STOP at B.3'' gate failure per Branch D framing | B.3'' gate failure triggers HARD STOP and surface |

---

## 10. Branch framing pre-commit (supersedes original §8 and B.1' §10)

**Phase B Branch A outcome statement template:**
> V1 Phase B closed via M1 CDI + P1 in-place patch + operator-facing CLI at cipher-fusion-evidence [new commit] tag `v1-substrate-cdi-platform`. cipher-platform .deb v2.0 ships as v1 deployment artifact for neocloud ops teams; install via standard apt; one-time-per-host parallel to NVIDIA driver install + nvidia-container-toolkit install. Neocloud's customers (end-users running vLLM workloads) see UNCHANGED `docker run --gpus all vllm/vllm-openai:0.21.0` invocations transparently picking up CIPHER substrate via host's patched CDI spec. All measurement gates PASS inside vanilla vLLM container with .deb v2.0 installed on host: Gate A cuBLAS shim_calls=[X] (>= 11000); Gate B Week 5 KV-dedup TinyLlama N=4 [Y] MiB (>= 45000) + Mistral-7B KL=0 + hit [Z]% (>= 80%); Gate C Track 2 SC6 Mistral-7B [W]% (>= 75%); Gate E cipher-platform CLI smoke all PASS. apt remove cleanup PASS. Goal 5 contract verbatim preserved for the end-user (the ML team renting GPU compute); neocloud ops team gets normal B2B infrastructure-operator deployment experience with CLI + runbook + standard journald monitoring. Substrate anchors UNCHANGED through Phase B (cipher_rt_phase4 `8613812e`, libcipher_rt.so md5 `1d91e7da`, cipher_kmod `8c643fc`).

**Phase B Branch B outcome statement template:**
> V1 Phase B partially closed: cipher-platform .deb v2.0 builds clean but 1 of 4 measurement gates fails inside vanilla vLLM container. Gate [A/B/C/E] measured [number] against threshold [number]; mechanism diagnosis at `V1_PHASE_B_COMPLETE.md` §[N]. Surface to Anil for adjudication: (a) accept partial Phase B close with documented gate-specific limitation, (b) extend Phase B for additional substep to address the gate failure, (c) HARD STOP and re-architect mechanism approach. Substrate anchors UNCHANGED.

**Phase B Branch C outcome statement template:**
> V1 Phase B closed BUT uninstall test surfaced host-side state leak: `apt remove cipher-platform` removes package but [specific host-side state] persists. Substrate file cleanup OK but [config / cache / log / CDI patch fragment / etc.] persists. Surface to Anil for adjudication: ship Phase B with known-issue documented OR add cleanup hook fix as B.4''.1 sub-substep. Three measurement gates + CLI smoke PASS pre-uninstall.

**Phase B Branch D outcome statement (HARD STOP):**
> V1 Phase B HARD STOP at B.2'' or B.3'': CDI mechanism fundamentally fails. Specific mechanism: [libcipher_rt.so does not load in vanilla vLLM container via CDI mount / cipher_kmod ioctl from container fails / patched CDI entry not honored by docker --gpus all / other]. Surface to Anil for alternative deployment mechanism choice: (a) revert to P2 take-over /etc/cdi/nvidia.yaml, (b) investigate CRI-O-specific P4 OCI hook path for narrow neocloud subset, (c) re-architect substrate to load via different entry point that does not require CDI injection, (d) other. No silent fallback; Anil adjudication required.

---

## 11. Adjudication record (supersedes original §10 and B.1' §11)

**Q1 (Phase B mechanism family)** : closed at M1 nvidia-container-toolkit CDI per Anil 2026-05-26 (B.0.7 mechanism investigation pick over LD_PRELOAD-only Q6 (a) path).

**Q8 (Phase B v1 mechanism = M1 CDI)** : closed per Anil 2026-05-26 (B.0.7 investigation).

**Q9 (within-M1 sub-path = P1 in-place patch)** : closed per Anil 2026-05-26 customer-model-corrected adjudication. P1 selected over P2 (worse failure mode), P3 (Goal 5 violation), P4 (not applicable to modern Docker; CRI-O-only).

**Q10 (CDI persistence verification)** : closed per B.0.7+ empirical test (`v1_phase_b/results/b0_7_cdi_persistence_verification.json`); Q9 (a) sibling-yaml-same-kind FAILS at CDI collision-detection; P1/P2/P3/P4 alternatives surfaced and adjudicated.

**Q11 (auto re-patch mechanism)** : closed at OPTIONAL per Anil 2026-05-26 customer-model correction. Operational fragility under driver updates is NOT a contract violation under corrected customer model. systemd path-watcher (cipher-platform-watch.service) ships if B.2'' budget allows; .deb without it is shippable; documented `sudo cipher-platform refresh` step in runbook §2 is the baseline mechanism.

**Q12 (operator-facing tooling scope)** : NEW per Anil 2026-05-26 customer-model correction. Closed at:
- /usr/bin/cipher-platform CLI with 5 subcommands (status / verify / refresh / version / logs)
- /usr/lib/cipher/runbook.md operator runbook (5 sections)
- /usr/lib/cipher/health-check.sh standalone health check
- Optional cipher-platform-watch.service systemd path-watcher
- Standard journald logging via cipher-platform.service
- Prometheus metrics endpoint DEFERRED to v1.x (if design partners request)

**Q13 (customer model)** : NEW per Anil 2026-05-26 customer-model correction. Closed at two-actor model:
- Neocloud ops team = CIPHER's customer (B2B infrastructure contract; docs, CLI, runbook, support; install actor)
- Neocloud's customer (ML team renting GPU compute) = END-USER (Goal 5 contract verbatim applies; transparent substrate firing)

---

## 12. Framing for B.5'' close-out (supersedes original §11)

Per Anil 2026-05-26 customer-model correction: the B.5'' close-out doc frames the deliverable as **cipher-platform .deb v2.0** (B2B infrastructure sold to neoclouds; install actor is the neocloud ops team; deployment pattern parallels NVIDIA driver + nvidia-container-toolkit install model). The end-user (the ML team renting neocloud GPU compute) sees UNCHANGED vLLM container invocations transparently pick up substrate via host CDI injection.

**Framing for V1_PHASE_B_COMPLETE.md narrative (diligence story):** A reviewer reading the close-out doc hears "B2B infrastructure deployment: cipher-platform .deb v2.0 installs on neocloud H100 hosts via standard apt; operator-facing CLI + runbook + journald monitoring for normal B2B infrastructure-operator UX; substrate fires transparently inside vanilla vLLM containers via nvidia-container-toolkit's CDI mechanism; deployment pattern parallels NVIDIA's own driver-plus-toolkit model that the ML serving industry runs on (NVIDIA NGC, nvidia-container-toolkit, NIM); neoclouds (Nebius, Lambda, CoreWeave, RunPod) are the design-partner cohort." Diligence-story strength: CIPHER's go-to-market matches the natural neocloud-substrate sale per Memory #10; deployment model aligns with the de facto industry standard for GPU-accelerated ML hosts.

**Ledger row (NEW per Q13 close, names neocloud as install actor):**

| Artifact | Install actor | Trigger | Cadence | Notes |
|---|---|---|---|---|
| cipher-platform .deb v2.0 | Neocloud ops team (Nebius, Lambda, CoreWeave, RunPod, on-prem ML infra ops) | `apt install cipher-platform` | One-time-per-host on neocloud H100 hosts; refresh on NVIDIA driver upgrade via `sudo cipher-platform refresh` (or auto via optional cipher-platform-watch.service) | B2B infrastructure contract with CIPHER; gets docs, CLI, runbook, support |
| Substrate firing in end-user container | Neocloud's customer (ML team renting GPU compute) | UNCHANGED `docker run --gpus all vllm/vllm-openai:0.21.0 ...` | Per end-user deployment | Goal 5 contract verbatim PRESERVED; end-user sees only faster inference |

---

## 13. Devang Nebius parallel action item (informational; not gating B.2'')

Per Anil 2026-05-26 directive: refreshed Devang one-paragraph technical update + sharpened question at `v1_phase_b/DEVANG_NEBIUS_UPDATE.md` (verbatim Anil 2026-05-26 draft). Action item for user-side external send; not gating B.2'' start. Response informs:
- B.2'' Dockerfile / build-script assumptions about target host environment
- B.3'' verification matrix (Nebius-specific gate?)
- Runbook content (Nebius-specific post-driver-upgrade procedure)

Devang response window 1-3 business days; lands at-or-before B.3'' verification begins. B.2'' substrate engineering can proceed on Anil "proceed to B.2''" signal without waiting.

---

## 14. Related memory

Carry from original §12 + B.1' §14 + additions:
- [[v1-phase-a-driver-worker-init]]: Phase A close + CUDA_INJECTION64_PATH that M1 CDI sets transparently
- [[v1-goal5-contract-lock]]: 2026-05-26 Goal 5 contract verbatim for the end-user; corrected customer model in §1 of this addendum
- [[cipher-cp25-closed]]: CP 2.5 cipher-platform .deb v1.0 precedent; v2.0 refresh under M1 CDI + P1 + operator CLI
- [[cipher-cp51-closed]]: CP 5.1 plugin hook; bundled in .deb plugin chain
- [[cipher-cp52-closed]]: CP 5.2 KV offload plugin; bundled in .deb plugin chain
- [[week5-complete]]: Week 5 KV-dedup plugin chain; bundled in .deb plugin chain
- [[g6-audit-chain]]: kmod week-13-14-complete (0.6.5); .deb v2.0 DKMS source
- [[neocloud-substrate-audit]]: 2026-05-21 read-only audit at week-5-complete anchors; B1+B2 substrate-ready for the neocloud customer cohort that B.1'' explicitly names
- [[cipher-pod-environment]]: Ubuntu pod environment; B.2'' .deb build + B.3'' container verification on this pod natively
- [[cipher-evidence-commit-discipline]]: followed at this B.1'' addendum commit
- [[cipher-proceed-not-ask]]: Anil 2026-05-26 customer-model-corrected proceed signal chain
- [[cipher-fusion-campaign]]: campaign discipline for go-to-market framing alignment

---

## 15. Honest residue

1. P1 fragility under driver updates is real and documented; under corrected customer model it is NOT a contract violation but it IS an operational responsibility for the neocloud ops team. Runbook §2 documents the procedure.
2. Optional systemd path-watcher reduces operational burden but adds ~0.25 ED. Ship decision deferred to B.2'' execution; Anil can revise during B.2'' if cycles tight.
3. B.1' container-path addendum stays in audit trail; do not delete. Wrong-direction lesson is part of the campaign record per cipher addendum convention.
4. P2 (take-over /etc/cdi/nvidia.yaml) remains the fallback if Branch D HARD STOP fires. ~3-4 ED additional eng-time.
5. Devang response may surface Nebius-specific CDI extension patterns (e.g., they may already in-place-patch /var/run/cdi/nvidia.yaml for their own observability agent; or they may take-over /etc/cdi/nvidia.yaml). If so, B.1'' may need a minor revision at B.3'' time; surface immediately per R-A.1-equivalent caveat.
6. v1 release ships Ubuntu 22.04 .deb only; Ubuntu 24.04 + other distros deferred to v1.x per Q4 close.
7. Prometheus metrics endpoint deferred to v1.x per Q12 close; design-partner request gating.
8. End-user vLLM container with custom /etc/cdi/ override that takes precedence over /var/run/cdi/nvidia.yaml is a known unhandled case; runbook §3 covers debugging; v1.x may add /etc/cdi/ patch path if neocloud design partners need.

9. **B.6''.9.3 (Path β.3, 2026-05-26): NR 27 CIPHER_REGISTER_MODEL kmod/mmap interaction bug.** F-VLLM-PLUGIN-CRASH bisect (B.6''.9.1) root-caused vLLM v0.21.0 EngineCore segfault to the cipher_vllm_kv plugin's NR 27 ioctl downstream code path; ASan rebuild (B.6''.9.3 Step 3c-3) ran clean, narrowing the bug to a kmod/mmap-region interaction outside userspace ASan coverage. Audit (B.6''.9.3) confirms NR 27 model_uuid has ZERO v1 consumers: SC6 weight-sharing (Track 2 76.0% baseline) uses NRs 21-24 ARENA path only; substrate libcipher_rt and cipher_kv_bridge have no NR 27 references; cipher_model_uuid() Python accessor has no readers anywhere; CIPHER_KVDEDUP defaults off in v1. rev6 gates NR 27 off via CDI env `CIPHER_REGISTER_MODEL=0` (zero v1 customer-observable impact; framework-agnostic; Memory #9 driver-API moat unchanged; Memory #20 ABI-additive discipline preserved — no kmod modification). v1.x picks up KASAN root-cause diagnostic + targeted kmod fix per Memory #20 regression protocol, then re-enables. Path Z subprocess-wrapper artifact preserved at `/tmp/cipher_vllm_kv_pathz.py` + `/tmp/cipher_vllm_kv_register_subproc.py` as v1.x alternative-design reference; build_asan + build_py312_asan preserved as KASAN follow-up audit trail.

10. **B.6''.9.3 cross-model KV-dedup deferred to v1.x.** The cross-tenant cross-model KV-dedup differentiation via `model_uuid` (`cipher_rt_kv_dedup_put` at cipher_rt_kv_alloc.c:574-581) requires (a) NR 27 re-enable per item 9 above and (b) `CIPHER_KVDEDUP=1` default flip. Both deferred to v1.x. Same-model KV-dedup (Week 5 plugin path, model-uuid-agnostic) is unaffected and remains live; Mistral-7B N=4 76.0% Track 2 baseline (which uses ARENA NRs 21-24, not NR 27) is unaffected.

15. **B.6''.9.8.5 + .8.5b infrastructure ships forward-compatible (env-gated OFF).** Audit at B.6''.9.8.5b.2 (`v1_phase_b/results/b6_9_8_5b_2_close.json`) confirms ALL current v1 actuators operate at NON-CUfunction layers: Marlin via cuBLAS shim (`cipher_rt_marlin_actuator.c:cipher_rt_matmul_register_actuator`), Koopman via cuBLAS shim (`cipher_rt_koopman_engine.cpp` same pattern), VOLT via NVML init-time clock lock (`cipher_rt_volt.h:"No per-launch classifier in v1"`). Zero v1 actuators wire `cipher_rt_graph_substitute_decide`. The .8.5 shims (`cuGraphAddKernelNode + _v2 + cuGraphExecKernelNodeSetParams + _v2 + cudaGraphAddKernelNode`) + .8.5b cuLaunchKernel-capture-detect ship as v1.x forward-compatible scaffolding; default-OFF env gate (`CIPHER_GRAPH_CAPTURE_DETECT=1`) prevents accidental activation; production rev7 .deb is SEGV-safe in default config. Current v1 actuator engagement under graph capture proceeds via existing cuBLAS GOT path (Marlin/Koopman dispatch substitute kernels via `cuLaunchKernel` during capture; PyTorch's normal capture mechanism records substitute kernels as graph nodes; replay runs Marlin/Koopman natively). Goals 2 + 4 actuator engagement does NOT regress by skipping .8.5b activation.

16. **v1.x per-CUfunction actuator architecture: Option C chosen.** If v1.x adds per-CUfunction actuators (e.g., FlashAttention substitution at FlashAttn kernel handles, direct-CUDA custom-kernel substitution), the chosen architecture is **Option C — registration-time substitution at `cuModuleGetFunction` / `cuLibraryGetKernel`** per B.6''.9.8.5b.2 audit recommendation. Mechanism: extend existing cuModuleGetFunction shim with weak `cipher_rt_kernel_substitute_decide(CUmodule mod, const char* name)` hook; when non-NULL substitute returned, swap CUfunction handle. ALL subsequent uses (cuLaunchKernel, cuGraphAddKernelNode, graph capture, replay) use the substituted handle automatically. Pros: zero per-launch overhead, zero SEGV class (no `cuStreamGetCaptureInfo` per launch), framework-agnostic (Memory #9 strengthened), composes with graph capture natively, ABI-additive (Memory #20). Cons: rules out per-launch input-dependent decisions (Marlin/Koopman OOD gates use cuBLAS layer for this anyway). Estimated ~1.5 ED at v1.x time (0.25 rollback .8.5b cuStreamGetCaptureInfo + 0.5 cuModuleGetFunction hook + 0.25 cuLibraryGetKernel extension + 0.5 build+test+commit). v1.x kicks off when the first per-CUfunction actuator (FlashAttention or other) is scoped in.

---

## 16. Pre-substrate-engineering checkpoint

Per scope-lock §4 R-A.1-equivalent caveat carry: SURFACE BEFORE B.2'' substrate engineering. NO AUTO SCOPE-DEGRADE.

This B.1'' commits with corrected customer model + M1 CDI + P1 + operator-facing CLI scope locked. B.2'' substrate engineering is BLOCKED on explicit Anil "proceed to B.2''" signal. No tool calls touch substrate or build the .deb at this commit.
