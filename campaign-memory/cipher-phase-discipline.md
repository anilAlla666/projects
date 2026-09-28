---
name: cipher-phase-discipline
description: "Discipline gates the CIPHER project uses for every phase — plan approval, build verification, .ko version preservation, evidence tarball at close-out"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a1207d0-bd39-48db-a0a9-37165af96920
---

Every CIPHER phase follows the same discipline. The user expects all of these without being reminded:

1. **Plan approval before any code.** Produce a 7-item plan and wait for explicit go. Do not start writing files until approval.
2. **Build verification before insmod.** Run `make` + `modinfo cipher_kmod.ko` and report the version, depends, license, srcversion, md5 BEFORE asking permission to insmod.
3. **Explicit go before each insmod and each rmmod.** Never insmod a freshly-built module without the user saying yes. Same for rmmod when the module is loaded.
4. **Preserve previous .ko as fallback.** Before building a new version, save the existing `cipher_kmod.ko` as `cipher_kmod.ko.v0.X.Y` so we can rmmod-and-roll-back instantly if the new module misbehaves.
5. **Insurance tarballs preserved across phases.** Each phase produces `/home/ubuntu/cipher-phaseN-evidence.tar.gz` and the extracted directory. Older tarballs (may13, phase1, phase1.5, phase2, …) stay on disk; never delete.
6. **PHASE_N_NOTES.md at close-out.** Prose notes (~250 lines, 6 sections) covering outcome, artifacts shipped, test ledger, performance, risk register status, next-phase setup.
7. **Performance budget.** Total per-phase overhead must stay within the budget the spec states (Phase 1.5: 2%, Phase 2: 2%, Phase 3: 2%, measured per the Phase 1.5.2 protocol).

**Why:** Procurement auditors and the user's investor narrative require zero crashes, zero functionality regressions, and clean rollback evidence. The .ko-preserve + tarball pattern is what lets the user demonstrate "we never lose work and we always have a known-good fallback."

**How to apply:** When starting a new phase, treat this list as the implicit acceptance criteria. The user will not enumerate them each time; producing them is your job.
