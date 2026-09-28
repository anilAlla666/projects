# WI-3 FLEET-AGGREGATION SCHEMA END-TO-END — PRE-REGISTRATION (before building)

**Goal:** prove the rc.v1 per-GPU event is fleet-consumable end-to-end, WITHOUT claiming real N-GPU scale.

## Real vs simulated boundary (binding)
- **REAL:** the rc.v1 per-GPU verdict record (`rcv1_event_real.json`, copied read-only from rc_pergpu/inject_verdict.json
  — the validated DEGRADED event with real onset/descriptor/signal-sources).
- **SIMULATED:** the multi-GPU population — produced by REPLAYING the one real event across synthetic `gpu.uuid`s with
  varied onset step, severity, and a healthy/degraded mix. The aggregation LOGIC (collector) is what is being
  validated, NOT real fleet scale.

## Collector design + expected fleet-view outputs
The collector ingests a stream of rc.v1 events keyed by `gpu.uuid` and emits a `fleet_reliability_view`:
- `total_gpus`, `n_healthy`, `n_degraded`, `degraded_fraction`.
- `degraded[]`: per-GPU {uuid, fault_onset (step+wall), severity, corroboration, process_context.descriptor,
  badput_attributed_gpu_seconds}.
- `fleet_badput_avoided_gpu_seconds_total` = Σ over degraded GPUs of WI-1's per-fault badput-avoided (tie to WI-1's
  GPU-seconds-per-fault; STATE the WI-1 scenario used).
- `fleet_effective_training_time` = aggregate progress/(progress+badput+overhead) across the fleet (using WI-1's
  per-GPU model), WITHOUT vs WITH CIPHER.
- Validation: schema-conformance check on every ingested event; aggregation by uuid (uuid is fleet-unique, not ordinal);
  idempotence (re-ingesting the same uuid updates, not double-counts).

## PRE-REGISTERED EXPECTATIONS
1. The collector ingests N replayed events and produces a correct fleet view: n_degraded = (degraded_fraction × N),
   each degraded GPU carrying its real-derived onset+descriptor and a WI-1-tied badput number.
2. Fleet badput-avoided scales linearly with n_degraded (Σ per-GPU). Fleet effective-time WITH > WITHOUT by the WI-1
   per-GPU delta weighted by degraded fraction.
3. Aggregation keys on `gpu.uuid` (validated unique); schema-conformance and idempotence hold.

## Bounds (stated)
- One real event replayed ⇒ the per-GPU SIGNAL diversity is synthetic (varied onset/severity, but the underlying
  detection is the one real DEGRADED event). The COLLECTOR/aggregation logic is validated; real fleet heterogeneity
  (different models, real distinct faults) is not.
- Badput numbers inherit WI-1's bounds (training-goodput definition; covered-op coverage fraction; single-GPU model).
- No real multi-GPU run; no collector-service deployment — the aggregation logic + schema are validated offline.
