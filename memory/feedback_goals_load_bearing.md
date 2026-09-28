---
name: CIPHER's three goals are the load-bearing specification
description: Every CIPHER decision must serve all three goals simultaneously — partial delivery is not acceptable
type: feedback
---

**The rule**: CIPHER has exactly three goals, stated by the user. Every decision, plan, and implementation must serve all three. Do not trade one off against another.

1. **New primitive** — CIPHER must be architecturally novel, not a reimplementation of something that already exists. A "drop-in LD_PRELOAD wrapper around existing techniques" does not qualify. The substitution-engine + online-verification + fingerprint-routing combination is what makes it new.

2. **O(1) substitution** — every substituted op must be swappable at call time in constant time. Fingerprint lookup + hash-table hit + pointer swap. No per-call compilation, no per-call SVD, no per-call optimization search. This is a property of the intercept architecture, not a nice-to-have.

3. **MFU 85%+ on GPU clusters** — the real, measured performance claim. Applies to cluster-scale deployments (see `feedback_mfu_target_clusters.md`), not single-GPU batch=1. Cluster scale with tensor-parallelism, pipeline-parallelism, communication overlap.

**Why**: User has stated these three goals at least four times in session 7, explicitly rejects any plan that meets only two. The product pitch is all three together. Dropping any one makes CIPHER "something else".

**How to apply**:
- When proposing a change, ask: does this serve all three goals, or only a subset?
- When measurement shows a mechanism fails at one goal, look for a different mechanism that serves all three. Do not compromise.
- When writing the plan, show how each of the three goals is addressed explicitly.
- The user will notice immediately if I'm sliding into a two-goal framing. Don't.
