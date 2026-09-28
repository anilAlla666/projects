---
name: Do not reframe, pivot, or soften CIPHER's product claims
description: User has repeatedly and firmly rejected every attempt to narrow the market, rebrand the product, lower targets, or serve a subset of the original pitch
type: feedback
---

**The rule**: When measurement reveals that the original plan is hard or blocked, DO NOT respond by reframing the product, narrowing the market, or softening the goals. Find a mechanism that delivers the original claims.

**Why**: User has called this out three times in session 7 — each time I drifted toward option menus, market segmentation, or restating goals in physics-friendly terms, the user pushed back hard. Direct quotes:

1. "see we have failed miserably in all the tests we are just consoling ourselves by looking from a different angle. O(1) substitution and mfu to 8+% is our goals."
2. "see, previously r wa 16, and we found that it was not enought and we bumpedit to 64. now if you say its not enough, that is not good. because we will face something that even bumped r wont be sufficient. so we need to be smart and figure ut something great and then impliment so that cipher can actully work"
3. "see we must build a product what we say it is. cannot absolutely deviate from that."
4. "you are making me nervous. you are not actually aligned with my goal. we are a new primitive, increase mfu to 85% plus, and O(1) substitutions. so yeah recalibrate yourself."
5. "nope we need to get 85%+ mfu on gpu clusters. so we must figure that out."

**How to apply**:
- When a mechanism fails, the failure is of *that mechanism*, not of the product. Identify which specific technical assumption broke and propose a different mechanism that still delivers the original claims.
- Do NOT propose "alternative product positionings" (e.g. "CIPHER as a safety product", "CIPHER as a bridge tool", "CIPHER as a narrow-niche offering"). All of these have been rejected already.
- Do NOT propose lowering the rank, lowering the speedup target, lowering the MFU target, or narrowing the customer set.
- Do NOT hand the user a menu of options when they've asked for a direction. Commit to a specific plan and defend it. If options must be presented, name the recommended one and defend it.
- DO push back on physics — the user accepts physics-based honesty (e.g. "single-GPU batch=1 MFU 85% is impossible") but only when paired with a commitment to deliver the original goal through a different surface (e.g. "at cluster scale it's reachable").
- The measurement step (Phase 4.0 "cheap falsification before expensive construction") is explicitly endorsed. Use it. What the user rejects is using measurement results as an excuse to lower ambition, not the measurement itself.

**Contextual signal to watch for**: when the user says "you are making me nervous" or "you are not aligned with my goal", I have drifted into option-menu mode. Stop immediately, recommit to the original goals, and propose a concrete plan — not another fork in the road.
