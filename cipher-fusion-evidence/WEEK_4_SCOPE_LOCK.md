# Week 4 Scope-Lock — SCOPE LOCKED

**Status:** SCOPE LOCKED.

**Date:** 2026-05-20
**Phase:** v1.2.2 §7 Week 4 (Observability ports + LP-8 retirement + Prometheus extension + Sub-4 measurement)
**Entry anchors:**
  - cipher_rt_phase4: `79c1b4f9` (tag `week-3-step-4-opt2a-dispatch-live-sense`)
  - cipher_kmod: `a21a45ee` (tag `week-3-step-4-opt2a-dsm-propose`)
  - cipher-may13-evidence: `fc8a9ae6` (unchanged)
  - cipher-fusion-evidence: `3ff40701` (WEEK_4_PREFLIGHT.md)

**User adjudications (5):**

| Q | Decision | Effect on scope |
| --- | --- | --- |
| Q1 | SKIP cipher_10ops_impl.cpp port | No 23-file transitive bring-in; stack-construct `CipherRingEntry` pattern continues |
| Q2 | Oracle approach (b): selective init | `cipher_oracle_init(state, NULL, NULL)` ~20 LOC; deferred-init model preserved |
| Q3 | VOLT 1000 MHz LUT preserved | Step 3 v1.2.2-vs-may13 discrepancy formalized in this scope-lock; no actuator edit |
| Q4 | Sub-4 measurement infra → Step 6 | ~65 LOC userspace-only emit to `/tmp/sense_transitions.json` |
| Q5 | Prometheus exporter in scope | PART 0 below; CASE A — source on disk, ready to extend |

---

## PART 0 — Prometheus exporter availability check — CASE A

**Verdict: CASE A — source on disk at `/home/ubuntu/cipher_exporter/`, ready to extend.**

| signal | value |
| --- | --- |
| `/home/ubuntu/cipher_exporter/cipher-exporter.py` | 11037 B, Python 3 stdlib (`http.server` + `urllib`), single-threaded |
| `/home/ubuntu/cipher_exporter/cipher-exporter.service` | systemd unit, ExecStart `/usr/local/bin/cipher-exporter --port 9402`, Requires `cipher-gpustate.service` |
| `/home/ubuntu/cipher_exporter/Makefile` | 1587 B (install target writes to `/usr/local/bin/`) |
| systemd status | `Unit cipher-exporter.service could not be found` (NOT installed) |
| reads | `/proc/cipher/gpu_state`, `/proc/cipher/stats` (existing kmod proc nodes) |
| current metric surface | `cipher_tenant_mfu_pct{tenant=...}` and adjacent gauges |

**Implication for Step 5 budget:** the script is pure stdlib (no `prometheus_client` import); extension is a localized edit (add 4 new metric emitters reading new `/proc/cipher/*` nodes produced by Tier B ports). Install is a one-line `make install` followed by `systemctl daemon-reload && systemctl enable --now cipher-exporter`. Budget remains as scoped (~3-4h).

**Step 5 sub-decomposition:**
- 5.1 Add 4 metric emitters: `cipher_tenant_fairness_quota`, `cipher_tenant_carbon_grams_co2`, `cipher_tenant_receipt_hash`, `cipher_tenant_session_band`
- 5.2 Wire to `/proc/cipher/fairness`, `/proc/cipher/carbon`, `/proc/cipher/receipt`, `/proc/cipher/sense_session` (created in Steps 2-3)
- 5.3 Install via `make install` + `systemctl enable --now`
- 5.4 `curl localhost:9402/metrics | grep cipher_tenant_` smoke test

---

## PART 1 — v1.2.2 §7 W4 + Wave 5 §5.5 W4 reconciliation

### v1.2.2 §7 Week 4 verbatim (prescribed)

> **Week 4** — Port observability shims (12 .cpp). Prometheus exporter for new metrics (`fairness_quota`, `carbon_grams_co2`, `receipt_hash`, `session_band`). 4 tests on observability surface.

**12 .cpp** decomposes into:
- **Tier A (audit-foundation-free):** cipher_loop, cipher_pipeline, cipher_pulse, cipher_continuity (4)
- **Tier B (audit-foundation-dependent — relies on cipher_rt_audit production):** cipher_trace, cipher_receipt, cipher_carbon, cipher_fairness, cipher_fairness_shm, cipher_guard, cipher_comply, cipher_determinism (8)

### Wave 5 §5.5 W4 sweep results (Part 2 of WEEK_4_PREFLIGHT.md, summarized)

| signal | finding |
| --- | --- |
| VERIFIED prescriptions | 4 (Tier A port surface, Tier B port surface, LP-8 retirement, Prometheus extension) |
| HYP prescriptions | 1 (Sub-4 measurement infra — user adjudication moved to in-scope) |
| CONTRADICTIONs | 2 (both self-resolving): C3 cipher_audit.cpp doesn't exist + C4 slot-array naming drift (resolves via LP-8) |
| C3 status | RESOLVED-RETIREMENT-ALREADY-DONE — cipher_audit.cpp NEVER existed in evidence; AUDIT logic inline in cipher_10ops_impl.cpp:341-378; cipher_rt_audit.{c,h} (production) supersedes |
| C4 status | LP-8 retires `cipher_slot[]`-named accessors in favor of `cipher_partition_slot[CIPHER_PARTITION_SLOTS_MAX]` matching cipher_partition_allocator.c:91 |

**Reconciliation:** v1.2.2 §7 + Wave 5 W4 + 5 adjudications fully resolved. Audit foundation (cipher_rt_audit.{c,h}) already exists in production (6789 B + 2902 B at HEAD `79c1b4f9`). Tier B ports compose with the existing audit foundation; no new audit work needed.

---

## PART 2 — 6-step decomposition

Total: ~1315 LOC, ~17-25h. Each step closes with TinyLlama SC6 (~33s); Mistral-7B SC6 gates at Step 1 (oracle integration), Step 3 (Tier B + cipher_sense_get_type linkage stress), Step 6 (closeout).

### Step 1 — Selective oracle init (Q2 approach b, ~20 LOC, ~2h)

- File: `cipher_rt_phase4/cipher_rt_oracle_init.cpp` (NEW)
- Calls `cipher_oracle_init(state, NULL, NULL)` (pure-function variant; no full `cipher_init()` cascade)
- Wires into observer `permit` slot (replaces hardcoded `permit=1` from Week 3 Step 2)
- Symbol `cipher_oracle_init` is provided by `cipher_oracle.cpp` (already in `cipher-may13-evidence/src/`)
- **SC6 gate:** TinyLlama bit-identical (LIVE=1) + Mistral-7B bit-identical (LIVE=1) — load-bearing for oracle-in-permit-path
- Closes Wave 5 W4 oracle-readiness item; unblocks Steps 2-3 with permit signal source-of-truth shifted from hardcode to oracle

### Step 2 — Tier A observability ports (4 files, ~970 source LOC + headers, ~5-7h)

Port from `cipher-may13-evidence/src/` into `cipher_rt_phase4/`:

| file | src LOC | header LOC | non-system undef | resolution |
| --- | ---:| ---:| ---:| --- |
| cipher_loop.cpp | 286 | 47 | 2 | already known clean (system glibc only) |
| cipher_pipeline.cpp | 239 | 30 | 3 | system glibc + cipher_sense_get_type (provided by cipher_sense.cpp, Wave 5 W3 ported) |
| cipher_pulse.cpp | 416 | 52 | 2 | system glibc only |
| cipher_continuity.cpp | 236 | 29 | 3 | system glibc + cipher_sense_get_type (same provider) |

All 4 close cleanly. No new /proc nodes from Tier A (these emit to internal substrate buffers; Tier B and Step 5 consume).

- **SC6 gate:** TinyLlama bit-identical after each file is added to Makefile (4 sub-gates ~33s each = ~2.2 min total gate cost)
- **Makefile delta:** 4 new objects to SRC list; no link order change required (alphabetical insertion)

### Step 3 — Tier B observability ports (8 files, ~1300 source LOC + headers, ~5-7h)

Port from `cipher-may13-evidence/src/` into `cipher_rt_phase4/`:

| file | src LOC | header LOC | proc node emitted |
| --- | ---:| ---:| --- |
| cipher_trace.cpp | 116 | 24 | — (internal buffer; Step 5 consumes) |
| cipher_receipt.cpp | 225 | 25 | `/proc/cipher/receipt` (Step 5.2 read) |
| cipher_carbon.cpp | 169 | 26 | `/proc/cipher/carbon` (Step 5.2 read) |
| cipher_fairness.cpp | 184 | 24 | `/proc/cipher/fairness` (Step 5.2 read) |
| cipher_fairness_shm.cpp | 229 | 38 | — (shm channel; Step 5 reads counters) |
| cipher_guard.cpp | 207 | 25 | — (audit-callback path through cipher_rt_audit) |
| cipher_comply.cpp | 82 | 24 | — (compliance gate; no proc surface in v1) |
| cipher_determinism.cpp | 88 | 24 | — (RNG seed registry) |

All 8 depend on cipher_rt_audit.{c,h} (production), cipher_sense.cpp (W3-ported), cipher_telemetry/cipher_topology (transitively pulled via Step 2 closure). Closure verified clean per the §5.5 W4 sweep.

- **SC6 gate (LOAD-BEARING):** Mistral-7B bit-identical after Step 3 closes — this is the binding stress gate (8 new files; cipher_rt_audit composition; cipher_sense_get_type cross-edge). ~57s.
- **TinyLlama sub-gates:** per-file (8 × ~33s = ~4.4 min)

### Step 4 — LP-8 allocator retirement (rename, ~50 LOC delta, ~1-2h)

Per Wave 5 §5.5 W4 LP-8: rename legacy `cipher_slot[]`-style accessors in observability ports' shared headers to `cipher_partition_slot[CIPHER_PARTITION_SLOTS_MAX]`. Source-of-truth: `cipher_kmod/cipher_partition_allocator.c:91` (`cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` is the kmod-internal name; userspace observability shims must reference the public `cipher_partition_slot` type).

- Pure rename. No semantic change. Verified by `git diff --stat` showing only identifier substitutions.
- **SC6 gate:** TinyLlama bit-identical (~33s) — pure-rename should be a no-op at runtime.

### Step 5 — Prometheus exporter extension (CASE A, ~150 LOC, ~3-4h)

Per PART 0:
- 5.1: Add 4 new emitters (`cipher_tenant_fairness_quota`, `cipher_tenant_carbon_grams_co2`, `cipher_tenant_receipt_hash`, `cipher_tenant_session_band`) to `cipher-exporter.py`
- 5.2: Wire to new `/proc/cipher/*` nodes from Step 3 (receipt, carbon, fairness) + new `/proc/cipher/sense_session` (needs +20 LOC kmod proc node addition; piggyback existing `cipher_proc.c` pattern from W2 Step 5)
- 5.3: `make install` + `systemctl daemon-reload && systemctl enable --now cipher-exporter`
- 5.4: `curl localhost:9402/metrics | grep cipher_tenant_` smoke (4 expected lines per tenant)

### Step 6 — Sub-4 measurement infrastructure (Q4, ~65 LOC, ~1.5-2h)

Userspace-only emit from `cipher_rt_sense_transition.c` (W3 Step 4 II-a invented wrapper). Pure observability, no kmod surface.

- 6.1: Open `/tmp/sense_transitions.json` at first transition (lazy, like the lazy /dev/cipher fd cache pattern in `cipher_rt_classify_observer`)
- 6.2: Append one JSON-line per transition: `{"ts_us":..., "tenant":..., "from":..., "to":..., "conf":..., "reason":"TRANSITION_DETECTED"}`
- 6.3: Atomic flush every 64 transitions (write+fsync; ring-replay if file rotates)
- 6.4: Honor `CIPHER_SENSE_LOG=0` to disable (default on when `CIPHER_SENSE=1`)

This emits the raw data needed to tune N=8/M=4/K=100ms thresholds in a future week from measured production data.

### Step 7 — Closeout (~2-3h)

- `WEEK_4_CLOSEOUT.md` covering Steps 1-6 + ABI delta (1 new ioctl in Step 5 sense_session proc, 0 new ioctls elsewhere) + tag `week-4-complete` on all 3 trees
- SC6 final: TinyLlama + Mistral-7B bit-identical (LIVE=1, CIPHER_SENSE=1) — captures the full Week-4 stack
- VOLT discrepancy (Q3) formalized as a deferred residue with measurement plan referenced

---

## PART 3 — Goal traceability matrix

| v1.2.2 prescription | Step | Status entering W4 | Status at W4 close |
| --- | --- | --- | --- |
| 12 .cpp ports | 2 + 3 | 0/12 | 12/12 |
| Prometheus extension | 5 | exporter source on disk, 4 metrics absent | 4/4 metrics emitted, exporter installed |
| LP-8 retirement | 4 | naming drift in shim headers | rename complete |
| Oracle in permit slot | 1 | hardcoded permit=1 | `cipher_oracle_init` driven |
| Sub-4 measurement (Q4) | 6 | wrapper exists, no measurement emit | JSON-line emit at /tmp/sense_transitions.json |
| VOLT 1000-vs-1200 (Q3) | n/a (deferred) | undocumented discrepancy | documented residue + plan reference |
| 4 observability tests | 7 (closeout SC6 + smokes) | n/a | SC6 + curl smoke + /proc enumeration + dsm_proposals continuity |

---

## PART 4 — SC6 gating cadence + per-step risk

| Step | TinyLlama SC6 | Mistral-7B SC6 | Risk |
| --- | --- | --- | --- |
| 1 — oracle init | YES | YES (oracle in permit hot-path is correctness-relevant) | MEDIUM — first time `cipher_oracle_init` runs in production |
| 2 — Tier A (4 files) | per-file (4×) | NO (Tier A is observability only) | LOW — closure already proven clean |
| 3 — Tier B (8 files) | per-file (8×) | YES (LOAD-BEARING — biggest delta in W4) | MEDIUM-HIGH — cipher_rt_audit + cipher_sense linkage stress |
| 4 — LP-8 rename | YES | NO (pure rename) | LOW |
| 5 — Prometheus | NO (exporter is out-of-process) | NO | LOW — Python edit + systemd install |
| 6 — Sub-4 measure | NO (observability-only) | NO | LOW — JSON-line emit |
| 7 — closeout | YES | YES | LOW — captures full stack |

**Total SC6 gate budget:** 14 TinyLlama (~7.7 min) + 3 Mistral-7B (~2.9 min) = ~10.6 min wall-clock for gates across the week. Acceptable.

---

## PART 5 — Pre-emptive on-disk verification of named targets (drift check)

### Tier A sources (cipher-may13-evidence/src/, headers in include/)

| file | src LOC | header LOC | present |
| --- | ---:| ---:| :---: |
| cipher_loop | 286 | 47 | ✓ |
| cipher_pipeline | 239 | 30 | ✓ |
| cipher_pulse | 416 | 52 | ✓ |
| cipher_continuity | 236 | 29 | ✓ |

### Tier B sources (cipher-may13-evidence/src/, headers in include/)

| file | src LOC | header LOC | present |
| --- | ---:| ---:| :---: |
| cipher_trace | 116 | 24 | ✓ |
| cipher_receipt | 225 | 25 | ✓ |
| cipher_carbon | 169 | 26 | ✓ |
| cipher_fairness | 184 | 24 | ✓ |
| cipher_fairness_shm | 229 | 38 | ✓ |
| cipher_guard | 207 | 25 | ✓ |
| cipher_comply | 82 | 24 | ✓ |
| cipher_determinism | 88 | 24 | ✓ |

### Production audit foundation (cipher_rt_phase4/)

| file | bytes | present |
| --- | ---:| :---: |
| cipher_rt_audit.c | 6789 | ✓ |
| cipher_rt_audit.h | 2902 | ✓ |

### LP-8 targets (cipher_kmod/)

| target | location | confirmation |
| --- | --- | --- |
| `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` | cipher_partition_allocator.c:91 | ✓ array of struct cipher_partition_slot |
| `cipher_dev_request_sm_partition` | cipher_dev.c:107 | ✓ dispatched at line 266 |
| atomic ops on `cipher_slots[].state` | lines 204, 209, 224, 229 | ✓ |

### Closure check (Tier A undef analysis at HEAD)

| file | total undef | non-system undef | unresolved |
| --- | ---:| ---:| ---:|
| cipher_loop.cpp | known | 2 (`__fprintf_chk`, `__stack_chk_fail`) | 0 (both glibc) |
| cipher_pipeline.cpp | known | 3 (those 2 + `cipher_sense_get_type`) | 0 (provided by cipher_sense.cpp, W3-ported) |
| cipher_pulse.cpp | 11 | 2 (same glibc pair) | 0 |
| cipher_continuity.cpp | 9 | 3 (glibc pair + `cipher_sense_get_type`) | 0 |

**All Tier A files close cleanly.** No new cross-TU undef edges require pulling additional `.cpp`. Tier B closure was verified clean by the §5.5 sweep against cipher_rt_audit + cipher_sense providers.

**No drift detected.** SCOPE LOCKED.

---

## PART 6 — Open items + cross-week tracking

### Carried in from Week 3

- **VOLT 1000-vs-1200 MHz discrepancy (Q3):** v1.2.2 §7 prescribes 1200 MHz for B=1 decode; existing LUT in cipher_rt_marlin_actuator preserves may13-measured optimum 1000 MHz. Decision: Q3 = preserve LUT. Formalization: Week 4 closeout will reference this as a deferred-residue item with a measurement plan note (re-run B=1 sweep at 1000/1050/1100/1150/1200 on Mistral-7B in a future week).
- **CIPHER_SENSE thresholds (N=8/M=4/K=100ms):** v1 best-guess. Step 6 measurement emit will produce tunable data.
- **Pattern (a) observability proposals:** /proc/cipher/dsm_proposals queue functional (W3 Step 4 II-a); no auto-action. Pattern (b) actuator gating is future-weeks work.

### New in Week 4

- **Step 5 install side-effect:** writing `/usr/local/bin/cipher-exporter` + enabling systemd unit is a system-state change. Will be done idempotently in Step 5.3. Recorded in WEEK_4_CLOSEOUT.md.
- **New /proc node in Step 5.2:** `/proc/cipher/sense_session` (~20 LOC kmod addition; piggybacks cipher_proc.c pattern). ABI: NO new ioctl. Pure proc-read.

### Deferred to future weeks

- Sub-4 measurement-driven retuning of N/M/K (uses Step 6 JSON data)
- Pattern (b) DSM auto-action (Wave 5 references this in W5+ scope)
- Attn substrate v1.5 (cipher_rt_dispatch_table[1] = ATTENTION → ROUTE_ATTN currently PASS_THROUGH)
- VOLT clock-sweep re-measurement (Q3 deferred residue)
- 100-tenant CP 5.5 benchmark (separate brief)

### Cross-week dependency graph

```
W3 (entry)
  oracle integrate (Step 1) ────→ Step 2 (permit signal source-of-truth)
  Tier A ports (Step 2)    ────→ Step 3 (cipher_sense_get_type satisfied)
  Tier B ports (Step 3)    ────→ Step 5 (/proc/cipher/{receipt,carbon,fairness})
                            ────→ Step 4 (cipher_partition_slot usage in shim headers)
  LP-8 rename (Step 4)     ────→ Step 5 (Prometheus reads consistent /proc surface)
  Step 5                   ────→ Step 7 (closeout SC6 + curl smoke)
  Step 6                   ────→ Step 7 (closeout JSON-line tail check)
                            ────→ (future) threshold retuning
```

---

## Discipline rules in force

- Per-step STOP-and-surface on any spec ambiguity; do NOT propose mitigation
- "No mitigation in this document" applies to all result/closeout docs in Week 4
- WAIT for adjudication between each STEP
- Per-command git env: `GIT_AUTHOR_NAME=Anil GIT_AUTHOR_EMAIL=anil.0666369@gmail.com`
- NEVER skip git hooks; NEVER update git config
- Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com> on all commits
- Preserve fallback .ko outside build dir (kbuild clean wipes *.ko) — same rule as prior weeks
- /dev/cipher ioctl nrs additive only; reserved 2/3/4 still -ENOSYS; new in W4 = NONE (Step 5.2 proc node is not an ioctl)

---

## Entry anchors at scope-lock

| tree | HEAD | tag |
| --- | --- | --- |
| cipher_rt_phase4 | `79c1b4f9` | week-3-step-4-opt2a-dispatch-live-sense |
| cipher_kmod | `a21a45ee` | week-3-step-4-opt2a-dsm-propose |
| cipher-may13-evidence | `fc8a9ae6` | (unchanged) |
| cipher-fusion-evidence | (this commit) | (will be set on commit) |

---

## HEADLINE: SCOPE LOCKED

All 5 user adjudications resolved (Q1-Q5). All 12 source files verified on disk in cipher-may13-evidence. Tier A closure clean. Tier B closure clean against cipher_rt_audit + cipher_sense providers. cipher-exporter source on disk (PART 0 CASE A) — extension is in-script work, install is one `make install`. LP-8 retirement targets concrete and located. SC6 gating cadence sized (~10.6 min wall-clock across the week). No drift detected.

**Week 4 ready to execute.** Awaiting adjudication to begin Step 1 (selective oracle init).
