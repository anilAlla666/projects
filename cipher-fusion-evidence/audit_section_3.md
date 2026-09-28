# CIPHER fusion campaign — anchor + version drift audit

## Section 3 — Anchor + version drift audit

Audit date: 2026-05-16. READ-ONLY inspection. Every md5 below is the literal
output of `md5sum` run on the file at the path given; every srcversion is the
literal output of `modinfo -F srcversion`. No anchor value is asserted without
a command result behind it.

---

### 3.1 — libcipher_v2 anchor drift (86618c30 vs cc0479b8)

Both md5s verified on disk:

```
$ md5sum /home/ubuntu/libcipher_v2/libcipher_v2.so /home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0
cc0479b836e560619e2b286ca1caecb7  /home/ubuntu/libcipher_v2/libcipher_v2.so
86618c30896470b642fcc6985d8dc632  /home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0
```

| md5 | file path | exists? | mtime / size | identity / notes |
|---|---|---|---|---|
| `86618c30896470b642fcc6985d8dc632` | `/home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0` | YES | 2026-05-13 07:31 / 16496 B | **Canonical anchor.** Phase-2-era v0.2.0 build. Byte-identical copies also at `/home/ubuntu/cipher_kmod_phase2_snapshot/libcipher_v2/libcipher_v2.so` and `/home/ubuntu/cipher-phase2-evidence/libcipher_v2/libcipher_v2.so` (both md5 `86618c30`). |
| `cc0479b836e560619e2b286ca1caecb7` | `/home/ubuntu/libcipher_v2/libcipher_v2.so` | YES | 2026-05-13 10:05 / 16968 B | Later **un-anchored** build. Phase 3 Task 5 CUPTI integration (see below). |

**Why it was rebuilt — DETERMINED.** This is not an unexplained rebuild. The
evidence is consistent and decisive that cc0479b8 is a deliberate Phase 3
Task 5 functional successor to 86618c30:

1. **New source file.** `libcipher_v2/` contains `cipher_cupti.c` (mtime
   2026-05-13 10:03) and `cipher_cupti.o` (mtime 2026-05-13 10:05:23 — the
   same minute as the cc0479b8 `.so`). The Phase-2 snapshot directory
   (`cipher_kmod_phase2_snapshot/libcipher_v2/`) has **no** `cipher_cupti.c`
   — only `cipher_inject.c` + `cipher_tenant.c`. The CUPTI source was added
   after the v0.2.0 anchor was cut.

2. **New DT_NEEDED dependency** (`readelf -d`):
   - 86618c30 NEEDED: `libc.so.6` only.
   - cc0479b8 NEEDED: `libc.so.6` **and `libcupti.so.12`**.

3. **Symbol-level diff** (`nm -D`, full dynamic table):
   - Exported (defined `T`) symbols are **identical** in both:
     `InitializeInjection`, `InitializeInjection2`. The injection ABI did not
     change.
   - cc0479b8 adds three new **undefined** imports not present in 86618c30:
     `cuptiSubscribe@libcupti.so.12`, `cuptiEnableCallback@libcupti.so.12`,
     `clock_gettime@GLIBC_2.17`.

4. **Documentation match.** `PHASE_3_NOTES.md` §"Layer C — libcipher_v2 CUPTI
   integration (Task 5)" describes exactly this: `cipher_cupti.c` as a
   "Phase 3 Task 5 addition" that "subscribes a CUPTI callback at cuInit
   time". cc0479b8 IS the Phase 3 Task 5 build; 86618c30 predates it.

**Conclusion:** cc0479b8 got built without an anchor update because the
anchor was set at the Phase-2 v0.2.0 milestone and the Phase 3 Task 5 CUPTI
work that produced cc0479b8 never re-cut a campaign anchor — it was carried
as "Phase 2 housekeeping" rather than promoted. `CP_2_5_REPORT.md` (lines
207-210) and `cp_2_5/PROGRESS.md` (lines 214-218, 308-311) already
acknowledge cc0479b8 as "a later un-anchored build … Phase 2 housekeeping —
not blocking CP 2.5 … flagged for later reconciliation". The shipped
`.deb` correctly bundles the **anchored** 86618c30.

**Recommendation: RE-ANCHOR cc0479b8 (do not discard).** The symbol/dependency
evidence shows cc0479b8 is a functional superset of 86618c30 (same exported
ABI + working CUPTI accounting path), not a stray or broken artifact.
Promote it to a fresh versioned anchor (e.g. `libcipher_v2.so.v0.3.0` with a
recorded md5) and document that the CP 2.5 `.deb` intentionally still ships
86618c30. Discarding would lose the only Phase 3 CUPTI-enabled v2 build on
disk.

---

### 3.2 — libcipher_rt rollback chain

Eight anchors were given for verification. Seven exist on disk; one
(`b1a3424c`) is documented as a never-preserved diagnostic build.

```
$ md5sum cipher_rt_phase4/libcipher_rt.so*  (relevant rows)
c2c5d313e2c24b687ba344cd9fe14a2f  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so
c63b6abc4c1b2fcce7641c11b01c9e40  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so.c63b6abc_instrumented_fixA
caae0bb88650c1121b9f92b1eda00e22  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so.caae0bb8_failed_fix
2f845393d2f254c669fa1e40d107d398  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so.cp2_5_baredrop_BROKEN
a0d6cddacb116f2d51ad1d1f867ef564  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_substrate_gate
55e2e3233c3b92506d1dd2f573b852c4  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so.substrate_gate_diag
$ md5sum /home/ubuntu/libcipher_rt.so.pre_cp2_5
5e304549a7e8a82b39072d11a85e71fb  /home/ubuntu/libcipher_rt.so.pre_cp2_5
```

| md5 anchor | file path | exists? | mtime / size | identity / notes |
|---|---|---|---|---|
| `a0d6cddacb116f2d51ad1d1f867ef564` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_substrate_gate` | YES | 2026-05-15 19:05 / 141400 B | CP 2.4 Marlin build / test-A baseline; rollback point pre substrate-gate. Per `MARLIN_HANG_ROOT_CAUSE.md` and `cp_2_4/PROGRESS.md`. |
| `55e2e3233c3b92506d1dd2f573b852c4` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.substrate_gate_diag` | YES | 2026-05-16 03:03 / 141400 B | Substrate-gate diagnostic (H1 / Option B) — **failed fix**, diagnostic only. |
| `caae0bb88650c1121b9f92b1eda00e22` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.caae0bb8_failed_fix` | YES | 2026-05-16 03:22 / 141896 B | Fix (i) — primary-context quant-path pin only. **Failed fix** (hang moved; GEMM not pinned). |
| `b1a3424c07f8d0ebb36135f3d9130884` | *(none)* | **NO — flagged** | — | **No file on disk.** Instrumented diagnostic build (`caae0bb8` source + logging), used for the H3 hang confirmation. Documented in `cp_2_4/MARLIN_HANG_INSTRUMENTATION.md:139` as "**not a campaign anchor**" and in `MARLIN_HANG_ROOT_CAUSE.md:109` — "All preserved as `libcipher_rt.so.*` except `b1a3424c`". **Absent by design**, not lost: it was deliberately not preserved because its diagnostic value lives in the logged report. Chain integrity is NOT broken. |
| `c63b6abc4c1b2fcce7641c11b01c9e40` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.c63b6abc_instrumented_fixA` | YES | 2026-05-16 06:56 / 150408 B | Instrumented Fix A (quant + dispatch guard). |
| `5e304549a7e8a82b39072d11a85e71fb` | `/home/ubuntu/libcipher_rt.so.pre_cp2_5` | YES | 2026-05-16 14:10 / 142024 B | Clean Fix A — the CP 2.4 campaign anchor; saved as the CP 2.5 rollback point. |
| `2f845393d2f254c669fa1e40d107d398` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.cp2_5_baredrop_BROKEN` | YES | 2026-05-16 14:17 / 150896 B | CP 2.5 bare-drop attempt — **BROKEN** (carried 8 missing symbols per `CP_2_5_REPORT.md:151`). Preserved as dead-end evidence only. |
| `c2c5d313e2c24b687ba344cd9fe14a2f` | `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` | YES | 2026-05-16 14:53 / 150880 B | **Current canonical — CP 2.5 anchor.** Confirmed by sidecar `cipher_rt_phase4/libcipher_rt.so.cp2_5.md5` (`c2c5d313e2c24b687ba344cd9fe14a2f  libcipher_rt.so`) and `CP_2_5_REPORT.md:196,218`. Supersedes `5e304549`. |

**Rollback-chain finding.** The chain `a0d6cdda → 55e2e323 → caae0bb8 →
(b1a3424c diag) → c63b6abc → 5e304549 → 2f845393 → c2c5d313` is **intact**.
Every anchor that was ever meant to be a recoverable artifact is present as a
file on disk. The single absent anchor, `b1a3424c`, is explicitly documented
as a throw-away instrumented diagnostic that was never a campaign anchor and
was deliberately not kept — its absence is expected and does not break
recoverability of any campaign milestone.

**Recommendation:** No action required for the chain. Optionally, **document**
in the CP 2.4 anchor ledger that `b1a3424c` is intentionally file-less so a
future audit does not re-flag it as missing. Current canonical `c2c5d313` is
correctly recorded with a sidecar `.md5`.

---

### 3.3 — kmod anchors

```
$ md5sum /home/ubuntu/cipher_kmod/cipher_kmod.ko /lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko
e2f50452f668859a96b1e25a2cba4e10  /home/ubuntu/cipher_kmod/cipher_kmod.ko
6654d9e5fcbe5f605e7ded216c20c45f  /lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko

$ md5sum /home/ubuntu/cipher-fusion-evidence/cp_2_4/kmod_0.4.7_pre_devnode.ko
2a69f9defd7730665e6b7f9d60e82b43  /home/ubuntu/cipher-fusion-evidence/cp_2_4/kmod_0.4.7_pre_devnode.ko

$ modinfo -F srcversion <each>
/home/ubuntu/cipher_kmod/cipher_kmod.ko                              E427CAFA4E94D548233DC7A
/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko           E427CAFA4E94D548233DC7A
/home/ubuntu/cipher-fusion-evidence/cp_2_4/kmod_0.4.7_pre_devnode.ko 3088289AA293398399B1903
```

| md5 | file path | exists? | mtime / size | identity / notes |
|---|---|---|---|---|
| `e2f50452f668859a96b1e25a2cba4e10` | `/home/ubuntu/cipher_kmod/cipher_kmod.ko` | YES | 2026-05-15 17:57 / 2143200 B | **Reference build, kmod 0.4.8** (devnode-codified). srcversion `E427CAFA4E94D548233DC7A`. Byte-identical to the versioned snapshot `/home/ubuntu/cipher_kmod.ko.v0.4.8` (md5 `e2f50452` confirmed). |
| `6654d9e5fcbe5f605e7ded216c20c45f` | `/lib/modules/6.8.0-1046-nvidia/updates/dkms/cipher_kmod.ko` | YES | 2026-05-16 15:30 / 122987 B | **DKMS-rebuilt** install of the same 0.4.8 source. srcversion `E427CAFA4E94D548233DC7A` — **identical** to `e2f50452`. md5 differs from `e2f50452` and size is much smaller (122987 B vs 2143200 B) — this is **expected and correct**: the DKMS install path strips the module, so the byte image differs while the source identity is unchanged. The differing md5 is NOT version drift. |
| `2a69f9defd7730665e6b7f9d60e82b43` | `/home/ubuntu/cipher-fusion-evidence/cp_2_4/kmod_0.4.7_pre_devnode.ko` | YES | 2026-05-15 17:55 / 2141840 B | **kmod 0.4.7 pre-devnode rollback anchor.** md5 `2a69f9de` confirmed. srcversion `3088289AA293398399B1903` — differs from the 0.4.8 srcversion, which is **expected**: 0.4.7 is the pre-devnode build and 0.4.8 codifies the `/dev/cipher` mode-0666 devnode callback (a real source change). The srcversion delta confirms 0.4.7 and 0.4.8 are genuinely different source trees, exactly as intended for a rollback anchor. |

**kmod findings.**
- Reference build `e2f50452` — **VERIFIED** on disk, matches the v0.4.8 snapshot.
- DKMS `6654d9e5` — **VERIFIED** on disk; srcversion **identical** to the
  reference (`E427CAFA4E94D548233DC7A`), confirming the DKMS module was built
  from the same 0.4.8 source. The md5/size difference is the expected result
  of DKMS stripping and is not drift.
- 0.4.7 pre-devnode anchor `2a69f9de` — **VERIFIED** at
  `cp_2_4/kmod_0.4.7_pre_devnode.ko`. Its distinct srcversion correctly
  reflects that it predates the devnode codification in 0.4.8.

**Recommendation:** No action required. All three kmod anchors verified and
their identity relationships are exactly as expected (reference == DKMS by
srcversion; 0.4.7 deliberately distinct from 0.4.8).

---

### Summary of drift

| Component | Status | Action |
|---|---|---|
| libcipher_v2 | Anchor drift — `cc0479b8` is an un-anchored Phase 3 Task 5 (CUPTI) successor to anchored `86618c30` | **Re-anchor** `cc0479b8` as a fresh version; keep `.deb` on `86618c30` as documented |
| libcipher_rt chain | Intact; 7/8 anchors on disk, `b1a3424c` absent **by design** (never-preserved diagnostic) | **Document** `b1a3424c` as intentionally file-less; no chain repair needed |
| kmod | All 3 anchors verified; reference and DKMS share srcversion; 0.4.7 correctly distinct | None |
