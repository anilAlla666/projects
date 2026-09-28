---
name: User role — CIPHER product owner / technical lead
description: User is the technical owner of CIPHER with strong opinions on product positioning; hands-on enough to run measurements but operates at the planning level
type: user
---

User is the **technical owner and product lead for CIPHER** — the LD_PRELOAD CUDA interception primitive being built in `/workspace/CIPHER_final_session7`.

## Operating style
- **High ambition, low tolerance for cover stories.** Will notice and reject the moment a plan starts narrowing scope or reframing goals to make them more achievable. Direct quote: "you are making me nervous. you are not actually aligned with my goal."
- **Wants commitment, not option menus.** When presented with multiple paths, will either pick one bluntly or demand a recommendation. Do not hand them three "paths forward" as a way of deferring decisions.
- **Holds to stated goals until physics forces a correction**, and even then accepts corrections only when paired with a new mechanism that still delivers the original intent.
- **Engages at the planning and measurement level**, approves plans before implementation, respects "cheap falsification" discipline, uses phrases like "yes go ahead" and "proceed" to authorize specific scoped work.
- **Will pause sessions deliberately** to pick up later — "dont close the session yet" means save state, don't wrap up with summaries or post-mortems.

## Domain knowledge
- Deep enough in GPU / CUDA / LLM inference to understand HBM bandwidth bounds, MFU, rank-r substitution, graph capture, NCCL collectives, tensor parallelism
- Understands when a plan is attacking the right bottleneck vs the wrong one (called out the rank-bump treadmill as the wrong axis)
- Makes the call on strategic direction but trusts the assistant to handle measurement, implementation, and technical detail

## How to collaborate
- Lead with decisions, not preambles
- Present concrete plans with specific gates, not option trees
- When measurement overturns a plan, say so directly and propose the next specific mechanism — do not retreat to generalities
- Trust the user to hold the goals; your job is to find mechanisms that meet them
- If the physics genuinely forbids a goal, say so clearly, but always pair with a mechanism-level alternative that delivers the goal's intent through a different surface
