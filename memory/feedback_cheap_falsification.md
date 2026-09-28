---
name: Measure-first-then-build discipline is endorsed
description: User explicitly approved the Phase 4.0 cheap-falsification pattern — test the load-bearing assumption before writing expensive code
type: feedback
---

**The rule**: Before committing to a multi-phase implementation with irreversible or costly steps, first run a cheap measurement that can falsify the load-bearing assumption. If the measurement kills the hypothesis, STOP immediately — do not weaken the gate, do not half-land the phase, do not rationalize.

**Why**: User approved this pattern explicitly in session 7 ("option A", "yes go ahead") when I proposed measuring before building Change 6 (graph capture). Phase 4.0 was designed as the cheap falsification step for the allocator-interceptor approach to Instance 4. When the measurement did falsify the hypothesis (PyTorch's caching allocator serves decode-step demand internally; zero libcudart allocator calls per step), the fact that we had only spent ~200 lines of observation-only code on disk — no pool, no graph capture — was exactly the point. User accepted the "hard gate, no weakening" outcome without pushback.

**How to apply**:
- Every multi-phase plan with expensive later steps must have a Phase X.0 that tests the load-bearing assumption cheaply.
- "Cheap" means: observation-only, no behavior change, reversible, measurable within one session.
- Phase X.0 must have a specific binary gate defined upfront — "does metric Y satisfy threshold Z?". No subjective judgment.
- If the gate fails: report the exact numbers, do not weaken the gate, do not proceed, do not half-land. The failure IS the phase succeeding at its job.
- If the gate passes: proceed to the next phase with full commitment.
- The user's "failure policy" phrasing — "nothing should break, nothing should fail, we need to achieve it" — means *the final product must not fail*, not *every measurement must pass*. Cheap falsification measurements that correctly kill bad plans are part of not-failing, not a form of failure.

**When NOT to apply**: Simple refactors, single-file changes, and well-understood additions don't need a falsification step. This discipline is specifically for plans that involve 500+ lines of new code across multiple files with an architectural hypothesis.
