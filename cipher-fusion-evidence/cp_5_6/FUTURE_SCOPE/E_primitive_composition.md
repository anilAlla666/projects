# FUTURE_SCOPE/E — Substrate Primitive Composition

## Goal

Measure cross-tenant batching **composed with** CIPHER's other substrate
actuators — Marlin INT4, DVFS, speculative decode, KV optimisation — to
determine whether the levers stack, are neutral, or interfere. To date the
batching lift (3.69× / 3.26× substrate-attributable) is measured on **vanilla
FP16** — batching alone earns it; the actuators are unmeasured on the batching
primitive.

## Architectural approach

Three composition studies, each: take the verified static cross-tenant batched
executor, enable one additional actuator, measure substrate-attributable
tok/W + correctness vs the batching-only baseline.

- **Batching + DVFS regime calibration.** DVFS is the *real* half of the
  retracted composed headline (the ~2.3× power cut survived; cf. CP 5.6 P2).
  Does the DVFS clock-lock compose cleanly with batched decode, and what is
  the per-regime optimal clock under a batched load? Honest expectation:
  partly additive (power cut is real), to be measured.
- **Batching + Marlin INT4.** Marlin fails the 99% teacher-forced gate at
  ~90–93% (honest INT4 drift). Composing it with batching must re-clear the
  correctness gate at a defensible threshold; measure the tok/W of
  batched-INT4 vs batched-FP16, honestly accounting for the quant drift.
- **Batching + speculative decode.** Spec decode's apparent CP 2.4 gain was a
  loop artifact; on F1-fixed honest output its accept rate is 0.19–0.77.
  Measure whether spec decode adds anything on top of batching, where, and
  whether it is worth the complexity.
- **Batching + KV optimisation** (CP 4.6/5.1/5.2 KV-dedup / offload) — whether
  cross-tenant KV-dedup composes with cross-tenant batching.

Every study uses the teacher-forced correctness gate; every claim is
substrate-attributable vs the batching-only baseline.

## Build scope + dependencies

- **Depends on:** the verified static batching primitive (built); the
  respective actuators (Marlin, DVFS, spec, KV — all exist in libcipher_rt).
- **Scope:** per-study harness wiring + GPU measurement.

## Time estimate

~2–3 weeks per composition study (4 studies — sequence by expected value:
DVFS first, then Marlin, then KV, then spec).

## Success criteria

- Each study reports, correctness-gated: does the actuator stack with
  batching (additive), is it neutral, or does it interfere — with the measured
  substrate-attributable number.
- No composed claim asserted that is not measured (the campaign's standing
  discipline).
