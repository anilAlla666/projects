# V1 Phase B.1 PACKAGE_DESIGN.md (cipher-platform v2.0)

> **REFRAMED 2026-05-26 per `V1_PHASE_B_SCOPE_LOCK_ADDENDUM_B1_PRIME.md` Option (vi) container path adjudication.** This document's content is preserved on disk for audit trail. Its v1 deployment role is reframed:
> - Sections 1, 2 (cipher-platform v1.0 baseline + v2.0 .deb metadata): still authoritative for the v1 cipher-platform .deb v2.0 thin-refresh role (DKMS kmod source bump from 0.4.8 to week-13-14-complete; libcipher_rt.so REMOVED from .deb because it lives in container per addendum §5 decision 5)
> - Section 3 (customer deployment recipe): SUPERSEDED by addendum §5 decision 1 (one container run + one env var; host-prereq .deb is one-time-per-host)
> - Section 5 (R-B.1 POC plan): COMPLETED at B.1; outcome informed the Option (vi) container pick
> - Sections 6, 7, 8, 9: retain as informational v1.x bare-metal-deployment reference
> 
> v1 deployment unit is now cipher-platform-vllm:2.0 container per addendum. cipher-platform .deb v2.0 is the v1 host-prerequisite installer for cipher_kmod. Both ship as v1 release artifacts.

---


**Date:** 2026-05-26
**Scope:** Phase B.1 deliverable per V1_PHASE_B_SCOPE_LOCK.md §4 B.1. Decision doc; no production code. R-B.1 entry-point registration POC at `v1_phase_b/packaging/poc/` validates the load-bearing mechanism end-to-end on this pod before B.2 commits production scaffolding.
**Pre-state anchors verified on disk at B.1 entry:** cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`; libcipher_rt.so md5 `1d91e7da`; cipher_kmod `8c643fc` tag `week-13-14-complete`; cipher-fusion-evidence `e33d62d` (Phase B scope-lock); cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`; cipher-platform .deb installed at v1.0.

---

## 1. Pre-existing cipher-platform v1.0 infrastructure (empirical baseline)

`dpkg -s cipher-platform` returns:

```
Package: cipher-platform
Status: install ok installed
Priority: optional
Section: utils
Maintainer: CIPHER Platform <anil.0666369@gmail.com>
Architecture: amd64
Version: 1.0
Depends: dkms, libc6, libstdc++6, libgcc-s1
Recommends: linux-headers-generic
Description: CIPHER GPU efficiency platform - runtime, substrate and kmod
 Self-contained deployment of the CIPHER platform for NVIDIA H100 nodes.
 Bundles the CUDA injection runtime (libcipher_rt), the Phase 3 substrate
 (libcipher_v2) and the pinned ABI-stable c10 core (libc10), plus the
 cipher_kmod kernel module shipped as a DKMS source package that rebuilds
 against the running kernel. Workloads launch under CUDA_INJECTION64_PATH
 via the cipher-run wrapper - no LD_PRELOAD and no dependency on a
 tenant-side PyTorch install.
```

`dpkg -L cipher-platform` shows files installed at:
- `/usr/bin/cipher-run` (launcher script)
- `/usr/lib/cipher/libcipher_rt.so` (substrate)
- `/usr/lib/cipher/libc10.so` (substrate dependency)
- `/usr/lib/cipher/libcipher_v2.so` (Phase 3 substrate)
- `/usr/src/cipher-kmod-0.4.8/` (DKMS source tree, 17 files)

Build scaffolding at `/tmp/cipher-deb-build/cipher-platform/`:
- `DEBIAN/control` (package metadata)
- `DEBIAN/postinst` (DKMS register/build/install + modprobe + /dev/cipher mknod fallback)
- `DEBIAN/prerm` (modprobe -r + DKMS remove)
- `usr/lib/cipher/{libcipher_rt.so,libc10.so,libcipher_v2.so}`
- `usr/bin/cipher-run`
- `usr/src/cipher-kmod-0.4.8/{Kbuild,Makefile,dkms.conf,*.c,*.h}` (DKMS source bundle)

The .deb artifact lives at `/tmp/cipher-deb-build/cipher-platform_1.0_amd64.deb` (502 KB; built 2026-05-16 per `cipher-cp25-closed` memory) and is also archived at `cipher-fusion-evidence/cp_2_5/cipher-platform_1.0_amd64.deb` for reference.

---

## 2. cipher-platform v2.0 design (additive update)

v2.0 is an ADDITIVE major bump from v1.0. The customer-visible deployment surface stays the same per Goal 5 contract: one package install + one env var. The version bump reflects ADDED functionality (Python plugin bundle + entry-point registration), not breaking change. All v1.0 files remain at their existing paths; v2.0 adds new files only.

### 2.1 Package metadata (proposed DEBIAN/control for v2.0)

```
Package: cipher-platform
Version: 2.0
Section: utils
Priority: optional
Architecture: amd64
Depends: dkms, libc6, libstdc++6, libgcc-s1, python3 (>= 3.10), python3 (<< 3.11)
Recommends: linux-headers-generic, python3-vllm (>= 0.21.0)
Maintainer: CIPHER Platform <anil.0666369@gmail.com>
Description: CIPHER GPU efficiency platform - runtime, substrate, kmod, and vLLM plugin bundle
 Self-contained deployment of the CIPHER platform for NVIDIA H100 nodes.
 v2.0 ADDITIVE update over v1.0: bundles the cipher_vllm_plugin Python
 entry-point so vLLM workloads get CIPHER's KV-cache buffer ownership,
 cross-tenant KV-dedup, and CP 5.2 KV offload without a separate pip
 install. Customer deployment unchanged: one apt install + one env var
 (CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so for vLLM V1,
 or LD_PRELOAD=/usr/lib/cipher/libcipher_rt.so for other CUDA workloads).
```

Reference: `/tmp/cipher-deb-build/cipher-platform/DEBIAN/control` (v1.0 baseline; v2.0 changes are Version + Depends.python3 + Description text + Recommends.python3-vllm).

### 2.2 Package layout (v2.0 = v1.0 file set plus new Python plugin files)

Carry-over from v1.0 (unchanged paths, refreshed contents):
- `/usr/lib/cipher/libcipher_rt.so` (refreshed to md5 `1d91e7da` per Phase A close `v1-substrate-driver-worker-init`)
- `/usr/lib/cipher/libc10.so` (unchanged from v1.0)
- `/usr/lib/cipher/libcipher_v2.so` (unchanged from v1.0; legacy)
- `/usr/src/cipher-kmod-0.6.5/` (refreshed DKMS source from cipher_kmod week-13-14-complete; v1.0 shipped 0.4.8)
- `/usr/bin/cipher-run` (unchanged from v1.0; launcher convenience)

NEW in v2.0 (per V1_PHASE_B_SCOPE_LOCK.md §4 B.1 layout):
- `/usr/lib/python3/dist-packages/cipher_vllm_kv.py` (CP 5.1 plugin from `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py` md5 `b89a9b6e`)
- `/usr/lib/python3/dist-packages/cipher_vllm_kvdedup.py` (Week 5 KV-dedup plugin from `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kvdedup.py`)
- `/usr/lib/python3/dist-packages/cipher_kv_offload.py` (CP 5.2 KV offload plugin from `/home/ubuntu/cipher_vllm_plugin/cipher_kv_offload.py`)
- `/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` (Python C extension from `/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so`)
- `/usr/lib/python3/dist-packages/cipher_model_fingerprint.py` (Track 2 SC4/SC6 helper from `/home/ubuntu/cipher-fusion-evidence/phase_c/cipher_model_fingerprint.py`)
- `/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/` (standard dist-info dir per Q6 (a))
  - `METADATA` (PEP 621 / 643 metadata)
  - `entry_points.txt` (the load-bearing file; Python `importlib.metadata` reads from here)
  - `RECORD` (file inventory)
  - `WHEEL` (build metadata)
  - `INSTALLER` (containing "deb")

### 2.3 entry_points.txt contents

Standard PEP 621 entry-point format mirroring what `pip install -e cipher-vllm-kv` produces today (per `cipher_vllm_plugin/setup.py` entry_points block):

```
[vllm.general_plugins]
cipher_vllm_kv = cipher_vllm_kv:register
cipher_vllm_kvdedup = cipher_vllm_kvdedup:register
```

Reference: `/home/ubuntu/cipher_vllm_plugin/setup.py:30-35` (current entry_points dict). Order matters per `WEEK_5_STEP_1_DESIGN_MEMO.md R-W5.3` (kvdedup MUST register after cipher_vllm_kv); Python 3.7+ dict insertion order preserves this in the rendered text file.

### 2.4 DEBIAN/postinst extension (additive over v1.0 postinst)

v1.0 postinst already handles DKMS register/build/install + modprobe + /dev/cipher mknod fallback per `/tmp/cipher-deb-build/cipher-platform/DEBIAN/postinst`. v2.0 extends this with NO additional shell code: the dist-info directory at `/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/` is already discovered by `importlib.metadata.entry_points()` at the standard system path. No post-install Python entry-point registration script needed.

The R-B.1 POC validates this end-to-end on Ubuntu 22.04 before B.2 commits to it.

If R-B.1 POC PASS: v2.0 postinst is identical to v1.0 postinst (DKMS-only logic).
If R-B.1 POC FAIL: scope-lock §10 Q6 falls back to (b) post-install programmatic write or (c) hybrid pip-inside-deb; v2.0 postinst extends with the chosen fallback logic; scope-lock amendment surface to Anil before B.2.

### 2.5 DEBIAN/prerm extension (additive over v1.0 prerm)

v1.0 prerm already handles modprobe -r + DKMS remove per `/tmp/cipher-deb-build/cipher-platform/DEBIAN/prerm`. v2.0 prerm needs NO additional shell code: `apt remove cipher-platform` removes all files including the dist-info directory; Python's `importlib.metadata.entry_points()` re-scans on each call and naturally returns empty for removed entries.

R-B.1 POC validates this end-to-end at step 6 (post-uninstall entry_points returns empty).

If POC step 6 PASS: v2.0 prerm is identical to v1.0 prerm.
If POC step 6 FAIL: prerm extends with explicit Python entry-point unregistration (programmatic write to remove the dist-info dir contents); scope-lock amendment surface.

---

## 3. Customer deployment recipe (cipher-platform v2.0)

Per V1_PHASE_B_SCOPE_LOCK.md §3 decision 1 + Goal 5 contract lock 2026-05-26:

```bash
# Step 1: install cipher-platform v2.0
sudo apt install ./cipher-platform_2.0_amd64.deb

# Step 2 (one of two):
# (a) vLLM V1 deployment
export CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so
# OR
# (b) non-vLLM-V1 CUDA workload deployment
export LD_PRELOAD=/usr/lib/cipher/libcipher_rt.so

# Step 3: run existing workload unchanged
python my_existing_vllm_serve_script.py  # or whatever the customer runs today
```

NO `pip install cipher-vllm-kv` (plugin lives inside the .deb).
NO `--quantization cipher_koopman` CLI flag.
NO `quant_config` Python wrapping.
NO model code change.
NO `PYTHONPATH` manipulation (Q4 closed: standard system path is in sys.path by default).
NO Python import hook from customer side.

vLLM detects the bundled plugin via standard `importlib.metadata.entry_points(group='vllm.general_plugins')` at engine init; the plugin auto-registers its hooks; cuBLAS shim + KV-cache buffer ownership + KV-dedup all fire automatically.

---

## 4. Q-closures from scope-lock §10 (Anil 2026-05-26)

| Q | Decision | Source |
|---|---|---|
| Q1 | Option (ii) packaging path | Anil 2026-05-26 B.0.5 surface adjudication |
| Q2 | Ubuntu 22.04 .deb only; Python 3.10 ABI lock | Anil 2026-05-26 |
| Q3 | Name `cipher-platform`, version 2.0 (additive bump from v1.0) | CLOSED by empirical finding 2026-05-26 |
| Q4 | `/usr/lib/python3/dist-packages/` standard path | Anil 2026-05-26 |
| Q5 | DKMS bundled in cipher-platform (existing pattern, refreshed source) | CLOSED by empirical finding 2026-05-26 |
| Q6 | Standard `dist-info/entry_points.txt` mechanism (R-B.1 POC validates) | Anil 2026-05-26 |

All six Q items now CLOSED. No open adjudication items remain for B.1 entry.

---

## 5. R-B.1 POC plan (deliverable at v1_phase_b/packaging/poc/)

Per V1_PHASE_B_SCOPE_LOCK.md §4 B.1 R-B.1 sub-element + Anil 2026-05-26 B.1 spec.

**Build pattern** (mirroring cipher-platform 1.0 raw dpkg-deb approach):

```
v1_phase_b/packaging/poc/
├── build_poc.sh              # builds the POC .deb via dpkg-deb --build
├── cipher_rb1_poc.py         # the POC Python module (stub register() function)
├── cipher-rb1-poc/           # the staging dir for dpkg-deb --build
│   ├── DEBIAN/
│   │   └── control           # minimal package metadata
│   └── usr/lib/python3/dist-packages/
│       ├── cipher_rb1_poc.py
│       └── cipher_rb1_poc-0.0.1.dist-info/
│           ├── METADATA
│           ├── entry_points.txt
│           ├── RECORD
│           ├── WHEEL
│           └── INSTALLER
├── cipher-rb1-poc_0.0.1_amd64.deb   # build artifact for evidence
└── results/
    └── poc_install_verify.json      # 7-step test sequence output
```

**POC entry_points.txt contents:**

```
[vllm.general_plugins]
cipher_rb1_poc = cipher_rb1_poc:register
```

**POC cipher_rb1_poc.py contents:**

```python
"""POC stub for R-B.1 entry-point registration validation.

Phase B.1 deliverable per V1_PHASE_B_SCOPE_LOCK.md §4 B.1. The register()
function does no real CIPHER work; the POC measures whether
importlib.metadata.entry_points(group='vllm.general_plugins') discovers
the entry after apt install (and returns empty after apt remove).
"""
import sys


def register():
    print("[cipher-rb1-poc] POC entry point fired", file=sys.stderr)
```

**POC test sequence (7 steps per scope-lock):**

| Step | Command | Expected |
|---|---|---|
| 1 | `dpkg -L cipher-rb1-poc` | exits non-zero (pre-install) |
| 2 | `sudo apt install ./cipher-rb1-poc_0.0.1_amd64.deb` | exits 0 |
| 3 | `dpkg -L cipher-rb1-poc` | lists dist-info files |
| 4 | `python3 -c "import importlib.metadata as m; print([(e.name, e.value) for e in m.entry_points(group='vllm.general_plugins') if 'rb1_poc' in e.name])"` | `[('cipher_rb1_poc', 'cipher_rb1_poc:register')]` |
| 5 | `sudo apt remove -y cipher-rb1-poc` | exits 0 |
| 6 | step 4 command repeated | `[]` (empty list) |
| 7 | `dpkg -L cipher-rb1-poc` | exits non-zero (post-uninstall) |

All 7 PASS = R-B.1 POC PASS = Branch A holds = B.2 proceeds with Q6 (a) standard dist-info confirmed.

ANY failure at steps 1-7 = surface to Anil with mechanism evidence BEFORE committing B.1 deliverables. Per R-A.1-equivalent caveat: do NOT debug independently.

---

## 6. B.2 implementation plan summary (carries from this design memo into B.2 scope)

Once B.1 POC validates the mechanism and Anil approves Branch A:

| B.2 sub-task | ED |
|---|---|
| Copy `/tmp/cipher-deb-build/cipher-platform/` to `v1_phase_b/packaging/cipher-platform-v2/`; bump version 1.0 to 2.0 in DEBIAN/control | 0.2 |
| Refresh `usr/lib/cipher/libcipher_rt.so` to md5 `1d91e7da` (Phase A close build) | 0.1 |
| Refresh `usr/src/cipher-kmod-0.4.8/` to `usr/src/cipher-kmod-0.6.5/` (week-13-14-complete source); update dkms.conf VER | 0.3 |
| Copy plugin Python files from `/home/ubuntu/cipher_vllm_plugin/` and `cipher_kv_bridge.so` and `cipher_model_fingerprint.py` per §2.2 layout | 0.3 |
| Generate `cipher_vllm_kv-0.2.0.dist-info/` directory (re-use `pip install -e` artifacts as template; verify against POC structure) | 0.5 |
| Extend DEBIAN/control Depends with python3 version range; update Description | 0.1 |
| build_deb.sh that invokes `dpkg-deb --build` | 0.2 |
| Smoke build + dpkg-deb -c verification | 0.3 |
| Buffer for issues | 0.5 |
| **Total B.2** | **2.5 ED** |

B.2 estimate carries from scope-lock §5; this design memo confirms the breakdown.

---

## 7. Honest residue at B.1 design close

1. **R-B.1 POC is the load-bearing validation.** This memo's confidence on the standard dist-info mechanism is based on documented PEP 621/660 behavior; the POC empirically verifies on Ubuntu 22.04 + this pod's specific Python 3.10 install. If the POC reveals a wrinkle (file ownership, ABI mismatch, sys.path scope issue), surface immediately before B.2.

2. **POC validates discovery, not full register() lifecycle.** The 7-step sequence verifies `importlib.metadata.entry_points()` returns the registered entry; it does NOT verify that vLLM's `load_general_plugins()` actually invokes `register()`. Full register() lifecycle is validated at B.3 Gate A under actual vLLM-on-CIPHER execution. This is the right scope for B.1 (mechanism validation) vs B.3 (production validation).

3. **cipher-kmod 0.6.5 DKMS source bundle path** assumes a clean copy of week-13-14-complete source. If kmod source on disk has uncommitted modifications relative to the tag, B.2 must use a clean git checkout to bundle. Document at B.2 entry.

4. **cipher_kv_bridge.so ABI tag (`cpython-310-x86_64-linux-gnu`)** locks the .deb to Python 3.10 only. Ubuntu 22.04 ships Python 3.10 (verified); the existing pod is Ubuntu so this constraint is met. Customer running Ubuntu 24.04 (Python 3.12) gets a clean Depends-mismatch error per §2.1 Depends declaration. v1.x roadmap adds multi-ABI .so build.

5. **plugin file md5 b89a9b6e is what the .deb ships.** No plugin code change in Phase B per scope-lock §3 decision 3. Future plugin source changes ship in cipher-platform v2.x or v3.0, not as Phase B substep.

6. **The R-B.1 POC artifact (.deb) lives in cipher-fusion-evidence committed alongside this design memo.** ~few KB binary file; small enough that committing it for evidence is reasonable. Per scope-lock §11 framing for B.5: the .deb artifact is reproducible evidence of the mechanism validation.

---

## 8. Next steps after this memo lands

1. Build R-B.1 POC at `v1_phase_b/packaging/poc/` per §5.
2. Run 7-step verification sequence per §5.
3. Document results in `v1_phase_b/packaging/poc/results/poc_install_verify.json`.
4. Surface verdict (Branch A / B / C / D per scope-lock §8) to Anil with PACKAGE_DESIGN.md + POC artifacts + verification JSON.
5. Hold for explicit Anil "proceed to B.2" signal before any B.2 production scaffolding work.

---

## 9. Related memory

- `v1-phase-a-driver-worker-init`: Phase A close + Branch D (a) CUDA_INJECTION64_PATH deployment that v2.0 .deb makes the documented customer-facing path
- `v1-goal5-contract-lock`: 2026-05-26 Goal 5 contract; v2.0 .deb preserves at OS-package + env-var tier
- `cipher-cp25-closed`: v1.0 cipher-platform.deb precedent (libcipher_rt.so + libc10.so + kmod packaging at CP 2.5 close)
- `cipher-cp51-closed`: CP 5.1 plugin hook origin; v2.0 bundles cipher_vllm_kv.py
- `cipher-cp52-closed`: CP 5.2 KV offload plugin; v2.0 bundles cipher_kv_offload.py
- `week5-complete`: Week 5 KV-dedup plugin chain; v2.0 bundles cipher_vllm_kvdedup.py
- `g6-audit-chain`: kmod week-13-14-complete (0.6.5); v2.0 refreshes DKMS source from v1.0's 0.4.8
- `cipher-pod-environment`: Ubuntu pod environment; v2.0 target is Ubuntu 22.04 per Q2 lock
- `cipher-evidence-commit-discipline`: followed at this B.1 deliverable commit
- `cipher-proceed-not-ask`: Anil "Proceed to B.1" + detailed scope = execute and document, surface POC findings
