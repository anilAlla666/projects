# CP 4.8 — D6: libcipher_v2 anchor resolution (cc0479b8 vs 86618c30)

**Date:** 2026-05-17. Decision artifact. Resolves the libcipher_v2 anchor
question before the CP 4.8 ship-gate soak, per the scope-adjudication D6.

## The two builds (diff — confirmed, `audit_section_3.md` §3.1 + this pass)

| md5 | file | identity |
|---|---|---|
| `86618c30896470b642fcc6985d8dc632` | `libcipher_v2.so.v0.2.0` | **Canonical anchor.** Phase-2 v0.2.0 build. `DT_NEEDED`: `libc.so.6` only. Bundled in the shipped `cipher-platform_1.0_amd64.deb` (CP 2.5). |
| `cc0479b836e560619e2b286ca1caecb7` | `libcipher_v2.so` (live build target) | Phase-3 Task-5 successor: adds `cipher_cupti.c` (CUPTI launch-stats callback). `DT_NEEDED`: `libc.so.6` **+ `libcupti.so.12`**; 3 new undefined CUPTI imports. Same exported injection ABI (`InitializeInjection`/`InitializeInjection2`). Un-anchored — never promoted to a campaign anchor. |

## Decisive finding — `libcipher_v2.so` is not in the soak runtime

`readelf -d` / `nm -D` on the production runtime `libcipher_rt.so` (c2c5d313):

- **No `DT_NEEDED` on `libcipher_v2.so`.**
- `cipher_v2_cupti_init` and `cipher_v2_tenant_register` are defined `T`
  (statically linked) **inside `libcipher_rt.so` itself**.
- `libcipher_rt.so` `DT_NEEDED`s `libcupti.so.12` **directly** — it carries its
  own CUPTI integration (`cipher_rt_phase4/cipher_cupti.c`).

⇒ The CP 4.8 soak runs `libcipher_rt c2c5d313` via `CUDA_INJECTION64_PATH`.
**The standalone `libcipher_v2.so` — whether `86618c30` or `cc0479b8` — is
never loaded into the soak process image.** The D6 worry ("running the ship
gate against the wrong anchor invalidates the result") is **dissolved by the
linkage fact**: libcipher_v2.so is not in the soak's runtime path. The v2
tenant-register + CUPTID logic that *does* run in the soak is the copy
statically linked into `libcipher_rt c2c5d313` — already anchored and validated
through CP 2.4 / 2.5 / 4.6.

## Decision

1. **CP 4.8 soak anchors: libcipher_rt `c2c5d313`, kmod 0.4.8 `e2f50452`.**
   `libcipher_v2.so` is out of the soak runtime path; it cannot affect the
   ship-gate result.
2. **libcipher_v2 standalone canonical anchor stays `86618c30`** — it is what
   the shipped `cipher-platform_1.0_amd64.deb` bundles. The soak validates the
   shipped stack; the shipped stack's v2 artifact is `86618c30`.
3. **`cc0479b8` preserved, documented, NOT promoted.** Saved as
   `libcipher_v2/libcipher_v2.so.v0.3.0-cupti.unpromoted` (md5 `cc0479b8`,
   non-destructive copy). It is a functional Phase-3 Task-5 CUPTI superset, not
   discarded — but promoting it to the campaign anchor now would force a
   `.deb` re-cut and a re-run of CP 2.5's deployment validation, which is
   **out of CP 4.8 scope**. Promotion is deferred to Phase 5 *iff* a
   standalone-libcipher_v2 CUPTI path is needed (the production runtime already
   has CUPTI linked in, so there is currently no such need).
4. **D6 does not gate D1.** The `tensor_mfu_pct` CUPTI counter work (D1 / B5
   fold-in) is measurement-harness CUPTI in `cipher_workloads`, independent of
   `libcipher_v2`. `cc0479b8`'s CUPTI integration is not required for it.

**D6 RESOLVED.** Anchors for CP 4.8: kmod `e2f50452`, libcipher_rt `c2c5d313`,
libcipher_v2 `86618c30` (bundled fallback, not soak-loaded). Sequence proceeds
to model fetch.
