---
name: Batch is a RANGE 32–256, not a single point
description: CIPHER's production envelope is a batch range 32–256 spanning off-peak to peak traffic; the 85% MFU gate is hit at the upper end and scales gracefully downward; never re-anchor on a single batch point
type: feedback
---

**The rule**: CIPHER's production serving envelope is **continuous batching over a range of batch=32 to batch=256**. The 85% MFU gate is designed to be hit at the **upper end of the range (batch=256)** where fp16 physics permits it, and CIPHER's MFU tracks the theoretical ceiling within 5–7 percentage points across the whole range.

**Why**: User corrected this explicitly on 2026-04-05 after I proposed building for a fixed batch=32. Direct quote:
> "see when i said batch is 32, it should be between 32 and 256."

The range spans:
- **Batch 32** = off-peak traffic, low QPS, hardware budget underused
- **Batch 128** = middle load
- **Batch 256** = peak traffic, GPU cost dominant, where the customer actually pays

Production serving stacks (vLLM, TGI, TRT-LLM-Backend) dynamically adjust batch size based on queue depth — they naturally traverse this range. A single fixed batch point is the wrong mental model.

**How to apply**:

- **Primary gate is at batch=256** (upper end). fp16 pure hits 86.7% MFU ceiling here; CIPHER closes the overhead gap via Instances 4+3 to deliver ≥85%.
- **At lower batch sizes**, MFU ceilings drop (physics): batch=128 → 43%, batch=64 → 22%, batch=32 → 11%. These are HARD ceilings, no software can break them at fp16.
- **CIPHER's goal at lower batch sizes** is to track the ceiling (i.e. at batch=32, deliver ~10% MFU — close to the ~11% ceiling — rather than stock PyTorch's ~5%). The absolute number is unimpressive but the *fraction-of-ceiling* is the real metric below batch=128.
- **At sales conversations the batch=256 number is the headline** because peak traffic is where cost matters most.
- **Secondary gate at batch=128** is unlockable if 2:4 sparsity is approved (doubles ceiling across range).
- DO NOT propose a build plan targeting a single batch point. DO NOT re-introduce literal-batch-1 or literal-batch-32 as the envelope. The range is the product.

**Related memories**:
- `project_operating_envelope.md` — full roofline table across batch range
- `feedback_batch_size_miss.md` — the original batch=1 anchoring mistake
- `feedback_mfu_target_clusters.md` — earlier (partially correct) reframing
- `feedback_run_arithmetic_first.md` — the meta-rule that would have caught all three anchoring mistakes

**Red flag**: if I find myself computing "batch=X MFU" for a single X and proposing a mechanism to hit 85%, stop. The question is always "MFU curve across batch 32–256", not "MFU at batch X".
