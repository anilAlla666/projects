# Track 2 SC3-2 — BUILD LOG

**Date:** 2026-05-19. **Scope:** consumer-side weight-arena import.
cipher_kv_bridge anchor `fca6843d` → `c04b0c39`. kmod `285d102e`,
libcipher_rt `83afd1ca`, libcipher_v2 `86618c30` unchanged.

## Source changes

| file | change |
|---|---|
| `cipher_rt_kv_alloc.h` | +`cipher_rt_weight_arena_import(fd, want_base, bytes, *out_base)` prototype |
| `cipher_rt_kv_alloc.c` | +`cipher_rt_weight_arena_import` — `cuMemImportFromShareableHandle` → `cuMemAddressReserve` (same-VA hint, offset-relative fallback) → `cuMemMap` → `cuMemSetAccess(READ)`; +`imported` flag on `struct weight_arena` |
| `cipher_kv_bridge.cpp` | +`weight_arena_import(fd, want_base, nbytes)` (module fn, symmetric to `weight_arena_create`); +`WeightArena.view(shape, itemsize, dtype, offset)` (aliasing tensor, no copy — symmetric to `.alloc`) |

`cipher_rt_kv_alloc.{c,h}` + `cipher_kv_bridge.cpp` build into
**`cipher_kv_bridge.so` only** (never libcipher_rt — confirmed SC2 report) —
so SC3 rotates `cipher_kv_bridge` and nothing else.

## API note — `weight_arena_import` is a module function

The SC3-1 memo wrote "`WeightArena.import_fd`". As built, **import is a module
function** `cipher_kv_bridge.weight_arena_import(fd, want_base, nbytes)` →
returns a `WeightArena` — symmetric to `weight_arena_create` (also a module
function that *returns* a `WeightArena`). `.view` is the class method
(symmetric to `.alloc`). Same capability, idiomatic naming.

## READ-only consumer mapping

The imported arena is mapped `CU_MEM_ACCESS_FLAGS_PROT_READ` in the consumer —
a consumer that strays into a weight write faults rather than corrupting the
producer's shared copy. (The producer's own arena is READ-WRITE — it `copy_`s
the weights in.)

## New harness (test code)

`sc3_producer.py` (loads TinyLlama, packs the arena, exports fd + JSON
manifest over a UNIX socket `SCM_RIGHTS`, serves multiple consumers,
producer-owned lifetime), `sc3_consumer.py` (imports, meta-loads via
`init_empty_weights(include_buffers=False)`, rebinds every parameter onto
READ-only arena views, forward pass), `sc3_run.py` (driver — bit-identical +
memory + page_info verification, same-VA and offset-relative modes).

## Build

`build_kv_bridge.sh` — clean; built
`cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` md5 **`c04b0c39`**.
`weight_arena_import` + `WeightArena.view` confirmed exported.

## Verification

Integration PASS (`TRACK_2_SC3_INTEGRATION.md`); regression smoke PASS
(`TRACK_2_SC3_REGRESSION.md`). Anchor preservation: `.pre_track2_sc3`
(`fca6843d`), `.track2_sc3` (`c04b0c39`); source `cipher_kv_bridge_src_track2_sc3.tar.gz`
(`5dcd92fe`).
