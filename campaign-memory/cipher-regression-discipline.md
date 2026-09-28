---
name: cipher-regression-discipline
description: When a prior change appears to have broken something, treat it as a regression — audit first, root-cause, fix in source; never work around with a runtime hack.
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

When something a prior CP/change introduced now blocks current work, do
**not** patch around it. Treat it as a regression and stop.

**Why:** during CP 2.4, the DVFS sweep was blocked because `/dev/cipher`
was `0600` (CP 3.3's kmod reload). The instinct to `chmod 0666` for the
measurement was rejected by the user *and* an automated security
classifier — "that's a workaround, not a fix." The user's required
sequence: **anchors/regression check first, decision second, build third.**

**How to apply:**
- On a blocker that traces to an earlier change: run a regression audit
  *before* any further build work. Re-run the prior milestone's binding
  indicators / unit tests / capture-rate checks against the current
  artifact md5s. One report: each check PASS/FAIL, and for each FAIL the
  specific change that caused it. Be precise about whether it is a *code*
  regression vs a latent fragility merely *exposed* by a legitimate change.
- Fix the cause in source (e.g. in the kmod), not in a shell trap / runtime
  chmod / env hack. A runtime workaround that loosens an access control is
  never acceptable as a "fix" — and the harness security classifier will
  block it anyway.
- Do not proceed with the dependent work until the proper fix lands and any
  affected gate is re-verified.
- Privileged operations (GPU clock actuation, device ioctls) legitimately
  need root — running a measurement harness as root is a fix; weakening a
  device ACL for everyone is not.

Linked: [[cipher-fusion-campaign]] (the campaign this discipline governs),
[[cipher-abi-rule]] (kmod ABI is additive — regressions there are serious).
Audit example: `cipher-fusion-evidence/REGRESSION_AUDIT_2026_05_15.md`.
