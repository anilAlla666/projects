---
name: cipher-t424d-enforcement-fixed
description: "T4.2.4d fixed the GREEN_CTX enforcement gap — persistent cuCtxSetCurrent(green_ctx) per launch makes PyTorch's NULL-stream launches use the green ctx's 8-SM partition"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.2.4d (2026-05-14 afternoon) closed the actuator-cosmetic gap that
T4.2.4c surfaced.

**Root cause:** PyTorch routes 100% of `m.generate(...)` decode/prefill
kernel launches through `stream==NULL` (per-context default stream).
Diagnostic logged 884,493 of 884,493 launches in 30 s went to NULL.
The cuStreamCreate push-on-create approach (T4.2.4a/b/c) only bound
the ~32 explicit streams PyTorch creates, which are not on the hot
path.

**Fix:** persistent `cuCtxSetCurrent(green_cuctx)` at every
cuLaunchKernel ENTER, fast-path skipping the syscall when current
context is already green. NULL stream then resolves to green's
default stream → 8-SM partition.

**Verification (binding diagnostic):** bomb solo prefills/s
- libcipher_v2 baseline: 114.1
- T4.2.4c (cosmetic): ~110-112 (Δ ≈ 0)
- **T4.2.4d (real): 10.4** — **11× slowdown**, well under <40 threshold

**Why:** future P4.2 actuator wiring depends on enforcement actually
working at kernel scheduling, not just API surface. Without this
fix, every actuator built on GREEN_CTX is cosmetic.

**How to apply:**
1. Before claiming any GREEN_CTX-based actuator delivers behavior,
   run the bomb-throughput binding diagnostic. cuGreenCtxGetDevResource
   passing is necessary but NOT sufficient.
2. The fix pattern (persistent cuCtxSetCurrent) has side effects on
   ALL CUDA APIs in the calling thread (cudaMalloc, cudaMemcpy, events).
   Acceptable for current single-tenant-single-context scope. See B13
   in PHASE_4_BACKLOG.md.
3. Doc tension between Programming Guide ("ignore current ctx") and
   Driver API ref ("cuStreamCreate honors cuCtxFromGreenCtx ctx") was
   resolved empirically: Driver API ref is correct, but PyTorch
   bypasses cuStreamCreate via NULL stream anyway.

**Mechanical foothold for product value:**
- For workloads with same-resource contention (two SM-bound tenants):
  predicted QoS isolation (untested).
- For cross-resource pairs (compute-bound bomb + HBM-bound victim,
  this experiment): partition is verified working but hurts victim.
- For "tenant compute budget" use case (limit a tenant to N SMs):
  delivers directly — bomb's power consumption dropped 65% under
  partition. This is the demonstrated capability today.

See PHASE_4_T4_2_4d_REPORT.md and
[[cipher-t424c-partition-not-enforced]] for full chain. The earlier
memory should now be read together with this one.

Build: `libcipher_rt.so.v0.2.0_T4_2_4d` md5 `50414674ddad2689191d13a92377e492`.
