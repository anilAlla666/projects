# Track 2 SC5-2 Phase A — VMM REFCOUNT PROBE

**Date:** 2026-05-19. **The gate** for the SC5 fd-custodian design (SC5 design
memo §1). **Verdict: PROBE PASS — the crux holds; proceed to Phase B.**

## What the probe tested

`sc5_probe_producer.py` creates a 256 MiB CIPHER VMM weight arena, fills it
with a known random pattern, exports the arena fd, prints a checksum, then
waits. `sc5_probe.py` (orchestrator + consumer) imports the arena, reads a
checksum, **SIGKILLs the producer**, re-reads, runs a GPU op, releases.

## Results (`sc5_probe_result.json`)

| claim | measurement | verdict |
|---|---|---|
| consumer mapping survives producer SIGKILL | checksum before kill `-711.76672746`, after kill `-711.76672746`, producer's `-711.766727` — **bit-identical** | **PASS** |
| arena is GPU-usable after producer death | `t*2` GPU op = `-1423.533` ≈ 2 × checksum | **PASS** |
| physical frees on last-handle-drop | producer dead + consumer holding: 877 MiB → consumer releases: 619 MiB → **258 MiB freed** (the 257 MiB arena) | **PASS** |

(First probe run reported a false negative on the memory check — the
orchestrator's torch caching allocator had grown ~256 MiB from the post-kill
GPU-op temporaries, masking the release. Fixed by `torch.cuda.empty_cache()`
before each framebuffer measurement; the arena release is then unambiguous —
258 MiB.)

## Conclusion — the SC5 design-memo §1 crux is confirmed on this H100/CUDA 13

CUDA VMM POSIX-fd shareable handles **are** cross-process reference-counted by
the driver:
1. A consumer's imported mapping **stays valid and GPU-usable after the
   producer process is SIGKILL'd** — the producer's death does not strand or
   corrupt the shared arena.
2. The physical memory **frees once the last handle drops** — here, after the
   producer is dead and the consumer releases, the full 256 MiB returns.

Therefore SC5's kmod is correctly scoped as the **fd custodian** (hold the fd
so consumers can join after the producer exits; the CUDA driver already owns
the physical-memory lifetime) — **not** a memory manager. **Phase B (the kmod
arena registry) proceeds** on the fd-custodian model; no redesign needed.
