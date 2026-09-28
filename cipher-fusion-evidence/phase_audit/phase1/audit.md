# CIPHER Phase 1 — Evidence Audit (CP 1.1 – CP 1.5)

**Audit date:** 2026-05-15
**Auditor scope:** Phase 1, checkpoints CP 1.1–CP 1.5 of the canonical Phase 0–9 plan.
**Method:** read-only. `ls -l`, `stat`, `md5sum`, `wc -l`, `grep`, `diff`, `strings`. No code written, no workloads run.
**Pod:** Lambda H100 80GB SXM5 · kernel 6.8.0-1046-nvidia · NVIDIA driver 580.105.08.

---

## 0. Framing finding (read this first)

**The canonical "Phase 0–9 plan with checkpoints CP 1.1–CP 1.5" does not exist as a discrete artifact on disk.** A `grep` across all `/home/ubuntu/*.md` and the `cipher-may13-evidence/` tree for `"CP 1."`, `"canonical"`, `"phase 0-9"`, `"checkpoint 1."`, `"first load gate"`, `"observation only"` returned only Phase 4 documents and incidental hits — never a Phase 1 checkpoint spec.

What the team actually executed was two real milestones, both tarballed 2026-05-13:

- **Phase 1** — `cipher-phase1-evidence.tar.gz` (662 KB, 05:27) — organized around *building the kmod and producing an architectural insight*.
- **Phase 1.5** — `cipher-phase1.5-evidence.tar.gz` (764 KB, 06:37) — organized around *decode coverage + empirical overhead measurement + lifecycle hardening*.

This audit therefore maps the canonical CP 1.1–1.5 *onto* evidence that was generated under a different organizing principle (insight-first, not checkpoint-first). Where that mapping is clean it is stated SHIPPED; where the canonical CP names an artifact the team never produced (a standalone hook-strategy report; a DKMS package), it is stated PARTIAL with the gap named. No charitable interpretation is applied: a paragraph inside a different document is not the same artifact as the document the CP names, and that distinction is preserved below.

---

## 1. CP-by-CP status table

| CP | One-line scope | STATUS |
|----|----------------|--------|
| CP 1.1 | NVIDIA open kernel module audit + hook-strategy written report | **PARTIAL** |
| CP 1.2 | Hook strategy decision documented | **SHIPPED** |
| CP 1.3 | cipher_kmod skeleton, DKMS compile, observation only | **PARTIAL** |
| CP 1.4 | First-load gate (kmod loads on the pod kernel) | **SHIPPED** |
| CP 1.5 | First architectural insight report | **SHIPPED** |

---

## 2. Per-CP evidence treatment

### CP 1.1 — NVIDIA open kernel module audit + hook-strategy written report — **PARTIAL**

**Canonical intent.** Two halves: (a) audit the NVIDIA open-source GPU kernel module tree to understand the driver's structure and pick an attachment point; (b) write that audit and the resulting hook strategy up as a report.

**What is on disk.**
- The NVIDIA open kernel module reference tree exists at `/home/ubuntu/ext/open-gpu-kernel-modules/`. **But its dating disqualifies it as Phase 1 evidence:** the directory mtime is `May 14 14:08` and `git log -1` reports commit `51edebee` dated `2026-04-28`, tagging **driver 595.71.05** (`version.mk`: `NVIDIA_VERSION = 595.71.05`). Phase 1 was tarballed `May 13 05:27` — the open-tree clone post-dates Phase 1 by ~33 hours, and it is the *wrong driver version* for this pod (pod runs 580.105.08). It was cloned for later Phase 4 driver-fusion work, not for a Phase 1 audit.
- The hook-strategy *reasoning* does exist, but embedded inside `PHASE1_INSIGHT_REPORT.md` §1 (lines 24–32): the decision to register a kprobe/kretprobe on the exported symbol `nvidia_unlocked_ioctl`, the rationale that the module holds no link-time reference to `nvidia.ko` (`depends:` empty, symbol resolved via kallsyms at `register_kprobe()` time), and the design intent to "sit beside nvidia.ko, not inside it."
- A `grep -ril "hook strategy"` across all Phase 1 / Phase 1.5 evidence dirs and `/home/ubuntu/*.md` returned **zero standalone hook-strategy documents** — only Phase 4 files.

**Gaps and how to close each.**
1. *No NVIDIA-open-tree audit was performed before the hook decision was made.* The hook-point choice (`nvidia_unlocked_ioctl`) was made from prior knowledge of the RM ABI, not from an audit artifact. **Closes by:** producing a dated written audit of the open-gpu-kernel-modules tree *at the pod's driver version (580.105.08)* — symbol map, ioctl dispatch path, candidate attach points — and acknowledging it as retrospective.
2. *No discrete hook-strategy report.* The reasoning is a subsection of the insight report. **Closes by:** extracting §1 into a standalone `PHASE1_HOOK_STRATEGY.md`, or formally accepting the insight-report subsection as the strategy record (a documentation decision, not a build gap).

**Why not SUPERSEDED.** The substitute artifact (the open-gpu tree) post-dates the decision it would have informed and is a different driver branch. Equivalence to "audit-then-decide" cannot be honestly argued for an audit that happened after the decision. PARTIAL is the honest call.

---

### CP 1.2 — Hook strategy decision documented — **SHIPPED**

**Canonical intent.** The chosen hook strategy is written down and justified.

**Evidence.**
- Path: `/home/ubuntu/cipher-phase1-evidence/PHASE1_INSIGHT_REPORT.md` §1 "What we built" (lines 15–70) and §3 "What this enables that LD_PRELOAD cannot" (lines 170–231).
- md5: `93b0575a4f8d7f1fad9a4fedb89c3236` · 351 lines · mtime `2026-05-13 05:27`.
- The strategy as documented: kprobe + kretprobe on `nvidia_unlocked_ioctl`; out-of-tree module loaded *alongside* `nvidia.ko`; no link-time dependency (`depends:` empty — confirmed independently by `strings cipher_kmod.ko | grep depends=` → `depends=`); RCU per-PID hashtable in the hot path; `/proc/cipher/stats` as the read-out surface; `/dev/cipher` reserved as the future control surface.
- The decision is justified against the alternative (LD_PRELOAD / CUDA API interception) in §3 with four named structural advantages of the kernel-boundary position.
- Corroborating source artifacts (the strategy as built): `cipher_probe.c` (215 lines, md5 `4dd4dc143675453387974a8d8ce7a48b`) contains the two `struct kprobe` instances; `Kbuild` (md5 `99dcce8dd8a4009cc7f90d20a0d36536`) is the out-of-tree obj-m link list.

**Reproducibility.** `strings /home/ubuntu/cipher-phase1-evidence/cipher_kmod.ko | grep -E "depends=|vermagic="` reproduces the empty-depends / pod-kernel-vermagic claim that the strategy rests on.

**Honest note.** SHIPPED, with one caveat: the decision record is a *subsection of the insight report*, not a standalone decision document. The content fully satisfies the CP's intent (the strategy is documented and justified); only the packaging differs from a literal reading of "decision documented."

---

### CP 1.3 — cipher_kmod skeleton, DKMS compile, observation only — **PARTIAL**

**Canonical intent.** Three parts: (a) a kernel-module skeleton exists; (b) it compiles, packaged via **DKMS**; (c) the module is observation-only (no behavior change to the driver).

**What is on disk — what passes.**
- *Skeleton exists and compiled.* `/home/ubuntu/cipher-phase1-evidence/` holds the full source set plus all build outputs:

  | file | lines | md5 |
  |------|------:|-----|
  | `cipher_main.c` | 73 | `b48d3036efe4e4f50abd82087b0626b4` |
  | `cipher_probe.c` | 215 | `4dd4dc143675453387974a8d8ce7a48b` |
  | `cipher_ioctl_decode.c` | 61 | (present) |
  | `cipher_proc.c` | 200 | (present) |
  | `cipher_dev.c` | 105 | `1d3e091487d64c5dcdd368978c36e6fd` |
  | `cipher_internal.h` | 70 | (present) |
  | `Kbuild` | 13 | `99dcce8dd8a4009cc7f90d20a0d36536` |
  | `Makefile` | 16 | `c5e52fe7571c6cafcfbb8b1a478603f4` |

  Source total ≈ 724 lines C/H (matches the report's "roughly 700 lines"). Compiled object files for every TU are present (`cipher_main.o`, `cipher_probe.o`, `cipher_ioctl_decode.o`, `cipher_proc.o`, `cipher_dev.o`, `cipher_kmod.o`, `cipher_kmod.mod.o`) plus the linked `cipher_kmod.ko` (718 384 bytes, md5 `42211f22b4219bd35494de6dba98141d`, version `0.1.0`). A compiled `.ko` with all `.o` artifacts present is proof the skeleton compiled clean.
- *Observation-only — substantially true.* The kprobe handler only reads `regs->si` and bumps `atomic64_t` counters; it does not modify ioctl flow. Confirmed behaviorally under CP 1.4 (byte-identical smoke output with vs without the module).

**Gaps and how to close each.**
1. *No DKMS.* This is the named, missing artifact. There is no `dkms.conf` anywhere (`find /home/ubuntu -maxdepth 3 -iname "*dkms*"` → empty; `grep -ril dkms` across all kmod evidence dirs → empty). The build is a **plain out-of-tree `Makefile` + `Kbuild`** invoking `make -C /lib/modules/$(uname -r)/build M=$(PWD) modules` — it compiles and loads, but it is not DKMS-packaged (no auto-rebuild on kernel upgrade, no `.deb`/`.rpm` install path). **Closes by:** adding a `dkms.conf` (`PACKAGE_NAME`, `PACKAGE_VERSION`, `BUILT_MODULE_NAME`, `MAKE`/`CLEAN`, `AUTOINSTALL=yes`) and capturing a `dkms add`/`dkms build`/`dkms install` log.
2. *"Observation only" has a minor asterisk.* `cipher_dev.c` already registers a `/dev/cipher` character device (no-op `open`/`release`, `unlocked_ioctl` returns `-ENOTTY`). This is inert (it changes nothing about driver behavior) but it is a forward-looking control surface, slightly beyond a pure observation skeleton. Not a defect; noted for completeness.

**Why PARTIAL not SHIPPED.** The DKMS half of the CP's literal scope was not done. Compile-and-load via raw Makefile is real and verified, but DKMS is explicitly named in the checkpoint and has no evidence.

---

### CP 1.4 — First-load gate: kmod loads on the pod kernel — **SHIPPED**

**Canonical intent.** The module actually loads (`insmod`) on this pod's running kernel without panic or taint change, and unloads cleanly.

**Evidence.**
- *Built for the exact pod kernel.* `strings cipher_kmod.ko | grep vermagic=` →
  `vermagic=6.8.0-1046-nvidia SMP preempt mod_unload modversions`.
  This is a byte-level match to the pod's running kernel and is the strongest single piece of load-gate evidence: a module with a mismatched vermagic would be rejected by `insmod` outright.
- *Behavioral proof the module attached during a real workload.* `diff /home/ubuntu/cipher-phase1-evidence/smoke_with_kmod.log /home/ubuntu/cipher-phase1-evidence/smoke_without_kmod.log` (both 5 252 bytes; md5 `01c9e42447dc1a4b5c8316577c688a19` and `a0fc3bee998dbfdd6bf285bf1b8083a9`) differs in only **three lines**: two non-deterministic (hook address `0x...`, weight-load progress bars) and one the measured `tps` (`71.95` with kmod vs `72.70` without). The model output — `n_compressed=155`, `Counters.marlin=12246`, `gen tokens 80/80` — is byte-identical. The module loaded, observed a full TinyLlama Marlin decode, and did not perturb the result.
- *Report attestation.* `PHASE1_INSIGHT_REPORT.md` §"Final test outcomes" (lines 333–348) records the full gate: `insmod (a)` exit 0, taint `12288 → 12288` (unchanged — the O+E bits were already set by `nvidia.ko`); `dmesg (b)` banner + `/dev/cipher` ready + kprobes-attached + loaded-ok, 0 WARN/oops/BUG; `rmmod (i)` exit 0, taint unchanged, 25 ms unload; 414 s of ambient + smoke load with zero panics.
- Corroborated in Phase 1.5 at a later version: `smoke_with_kmod_v15.log`, `smoke_with_kmod_1.5.7.log`, `smoke_without_kmod_v15.log` all present (5 252 bytes each), confirming the load gate held across `0.1.0 → 0.1.7`.

**Reproducibility.** Re-runnable: `sudo insmod /home/ubuntu/cipher-phase1-evidence/cipher_kmod.ko` then `ls /dev/cipher /proc/cipher/stats` then `sudo rmmod cipher_kmod`. The vermagic match guarantees the `insmod` will be accepted by kernel `6.8.0-1046-nvidia`.

**Honest note.** SHIPPED, with one preserved gap: **the raw `insmod`/`dmesg`/taint capture was not saved as a log file.** A `grep` for `taint` / `loaded ok` / `kprobes attached` / `unloaded clean` across the Phase 1 `.log` files returned empty — those outcomes survive only as a PASS table inside the insight report, not as a primary `dmesg` capture. The gate is SHIPPED on the strength of the vermagic match plus the byte-identical smoke diff (both primary, on-disk, re-verifiable); the absence of the raw load log is a preservation gap, not a gate failure. To fully close: capture and archive `dmesg | grep -i cipher` and `/proc/sys/kernel/tainted` around an `insmod`/`rmmod` cycle.

---

### CP 1.5 — First architectural insight report — **SHIPPED**

**Canonical intent.** Phase 1 produces a written architectural-insight report — the deliverable that justifies the kernel-boundary approach and locks the next phase.

**Evidence.**
- Path: `/home/ubuntu/cipher-phase1-evidence/PHASE1_INSIGHT_REPORT.md`
  (identical copy preserved in `/home/ubuntu/cipher-phase1.5-evidence/PHASE1_INSIGHT_REPORT.md`).
- md5: `93b0575a4f8d7f1fad9a4fedb89c3236` · 351 lines · 17 821 bytes · mtime `2026-05-13 05:27`.
- It is a genuine architectural-insight document, not a status log. The load-bearing insight (§2.2): the Phase 1 plan predicted `NV_ESC_IOCTL_XFER_CMD` (nr 211) would dominate the kernel boundary; the kmod observed **`XFER_CMD = 0` across 1 537 ioctls** during a full TinyLlama Marlin decode. The actual carrier on driver 580.105.08 is the `NV_ESC_RM_*` family (`RM_ALLOC`, `RM_FREE`, `RM_CONTROL`, ~30 entries). A predicted-dominant assumption was empirically overturned — exactly the class of result an insight report exists to surface.
- It carries three workload ioctl signatures (§2.1: `nvidia-smi` 82 ioctls / `torch.zeros` 430 / smoke 1 537), a per-LWP granularity finding (§2.3), a risk register (§5), a Phase 1.5 work list (§4), and a "Phase 2 locked from this evidence" section (§6) — i.e. it does the next-phase-locking job an insight report is meant to do.

**Reproducibility.** The document itself is the deliverable; `md5sum` reproduces the integrity check. The underlying ioctl-signature claims are re-derivable by re-running the three workloads under the module and reading `/proc/cipher/stats`.

**Honest note.** SHIPPED unambiguously. This is the cleanest CP in Phase 1: the named artifact exists, is dated within the phase, is substantive, and was preserved in two locations. (Phase 1.5 later extended the same line of work — `PHASE_1.5_NOTES.md`, `PHASE_1.5.2_OVERHEAD_REPORT.md` — but CP 1.5 is satisfied by the Phase 1 report alone.)

---

## 3. Summary

| CP | Status | One-line basis |
|----|--------|----------------|
| CP 1.1 | PARTIAL | Hook-strategy reasoning shipped inside the insight report; the NVIDIA-open-tree audit half post-dates Phase 1 (cloned May 14, wrong driver 595.71.05) and no standalone hook-strategy report exists. |
| CP 1.2 | SHIPPED | Strategy documented + justified vs LD_PRELOAD in `PHASE1_INSIGHT_REPORT.md` §1/§3; built artifacts corroborate. |
| CP 1.3 | PARTIAL | Skeleton compiled clean (full `.o` set + `.ko` on disk); **no DKMS packaging** — plain Makefile/Kbuild only. |
| CP 1.4 | SHIPPED | Vermagic byte-match to pod kernel + byte-identical smoke diff prove load/observe/unload; raw `dmesg`/taint log not preserved. |
| CP 1.5 | SHIPPED | `PHASE1_INSIGHT_REPORT.md`, md5 `93b0575a4f8d7f1fad9a4fedb89c3236`, 351 lines — substantive, overturns the `XFER_CMD` assumption, locks Phase 2. |

**Most important honest finding:** Phase 1 substantively shipped — a real kernel module that loads on the pod kernel and a genuine architectural insight (the `XFER_CMD`-is-dead discovery). But the canonical CP 1.1–1.5 checkpoint structure was never the team's organizing principle: the work ran as "Phase 1 + Phase 1.5" milestones built around *insight and overhead measurement*, not checkpoints. Mapping the canonical CPs onto that evidence exposes two real gaps that the milestone framing hid — **no NVIDIA-open-tree audit informed the hook decision** (the open-tree was cloned a day later, for a different driver, for Phase 4 work), and **no DKMS packaging exists** (raw Makefile only). Both Phase 1 PARTIALs trace to the same root cause: artifacts the canonical plan names were skipped because the milestone framing did not call for them.
