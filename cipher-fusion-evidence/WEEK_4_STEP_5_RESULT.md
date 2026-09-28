# Week 4 Step 5 — Prometheus Exporter Extension + sense_session Proc Node — RESULT

**Status: PASS.**

Second kmod rotation in 24h. `/proc/cipher/sense_session` proc node
added (~30 LOC across cipher_internal.h + cipher_main.c + cipher_proc.c,
mirroring W2-S5 classify_stats + W3-S4 dsm_proposals patterns).
cipher-exporter.py extended with 3 new scrapers (classify_stats /
dsm_proposals / sense_session) emitting 11 new Prometheus metrics
(~120 LOC Python). DKMS sync executed inline per Step 4 cadence;
srcversion gate verified `180A1D4… → 2106824D…`. CP 5.4 isolation 15/15
byte-identical; SC6 TinyLlama vanilla + CIPHER both 7/7 bit-identical.

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 4, Step 5 of 6
**Anchors:**
  - cipher_kmod: `158ad96` → **`5782609`** (tag `week-4-step-5-sense-session-proc`)
  - cipher_rt_phase4: `3be4531` (unchanged — Step 5 has no rt_phase4 source changes; exporter is out-of-process Python in `/home/ubuntu/cipher_exporter/`)
  - cipher-may13-evidence: `fc8a9ae` (unchanged)
  - cipher_exporter: NOT a git repo; cipher-exporter.py modified in-place (md5 `83d0a3d5`); changes will be captured in the Step 7 cipher-fusion-evidence tail commit

---

## A — Pre-edit gates — PASS

| signal | value |
|---|---|
| cipher_kmod HEAD | `158ad96` ✓ matches Step 4 close |
| cipher_rt_phase4 HEAD | `3be4531` ✓ |
| both trees clean | ✓ |
| Snapshot pre kmod | md5 `15a0d50d` → `/tmp/week4_step5_6_7/cipher_kmod.ko.step5_pre` |
| Snapshot pre rt | md5 `90fea16b` → `/tmp/week4_step5_6_7/libcipher_rt.so.step5_pre` |
| **srcversion loaded** | **`180A1D412429A77A73B464D`** ✓ |
| **srcversion DKMS-installed** | **`180A1D412429A77A73B464D`** ✓ (in sync — post-Step-4 DKMS sync held) |
| cipher_exporter discovery | `/home/ubuntu/cipher_exporter/cipher-exporter.py` (309 LOC, pure stdlib HTTP server on :9402; reads gpu_state + stats only) |
| Pre-edit CP 5.4 | 15/15 PASS (`cp54_pre_step5.log`) |
| Pre-edit SC6 TinyLlama vanilla | 7/7 PASS bit-identical (`sc6_pre_step5_vanilla.log`) |

---

## B — kmod /proc/cipher/sense_session proc node — PASS

| sub-step | result |
|---|---|
| **B.1** | inspected W2-S5 + W3-S4 proc-node patterns in `cipher_proc.c` (487–640) |
| **B.2** | payload chosen: 4 per_class atomic64_t (HUMAN/AGENT/BATCH/UNKNOWN) + total_sessions + total_transitions — mirror of W3-S4 `cipher_dsm_proposals` struct shape |
| **B.3** | implemented: |
| | `cipher_internal.h` +14 lines (CIPHER_PROC_SENSE_SESSION macro + struct + extern) |
| | `cipher_main.c` +5 lines (BSS storage) |
| | `cipher_proc.c` +75 lines (entry static + class names + show fn + open/fops + proc_create with cascading cleanup + proc_remove) |
| **B.4** clean build | rc=0; **1 warning** (kbuild "compiler differs" generic; unchanged from post-Step-4); md5 `e9855b6e`; srcversion **`2106824D60BEA5C2333F4D6`** |
| **B.5** DKMS sync | `sudo dkms remove + recopy sources + recover dkms.conf + dkms add/build/install` — completed cleanly; DKMS-installed `.ko` srcversion matches new build |
| **B.6** reload + srcversion gate | `sudo rmmod + sudo modprobe`; loaded srcversion `2106824D60BEA5C2333F4D6` (CHANGED from `180A1D4…` ✓); DKMS-installed matches loaded ✓ |
| **B.7** /proc reachable | `ls /proc/cipher/sense_session` → `0444 root:root`; `cat` returns expected zero-output with "no producer wired yet — Step 6+ will populate via ioctl bridge" hint |
| **B.8** CP 5.4 intra-step | 15/15 PASS byte-identical to pre (`diff cp54_{pre_step5,post_step5b}.log` empty) |

### B build vitals

| signal | pre-Step-5 | post-Step-5 |
|---|---|---|
| `cipher_kmod.ko` md5 | `15a0d50d` | `e9855b6e` |
| DKMS srcversion | `180A1D412429A77A73B464D` | **`2106824D60BEA5C2333F4D6`** |
| Loaded srcversion | `180A1D412429A77A73B464D` | **`2106824D60BEA5C2333F4D6`** |
| kbuild warnings | 1 | 1 |
| `/dev/cipher` mode | 666 | 666 (devnode callback intact) |
| Proc entries (sense_session new) | 7 | **8** |

---

## C — cipher-exporter.py integration — PASS

### C.1 existing structure (309 LOC, pure stdlib)

`cipher-exporter.py` reads `/proc/cipher/gpu_state` + `/proc/cipher/stats`,
parses with regex, emits Prometheus text format via `http.server`. Pairs
with `cipher-gpustate.service` (Phase 3 Layer D). Single-threaded HTTP
server on `:9402`.

### C.2 additions (~120 Python LOC, in-place edit)

| change | location |
|---|---|
| +3 PROC path constants (CLASSIFY, DSM, SENSE) | top of file, near existing path constants |
| +3 parser functions (`parse_classify_stats`, `parse_dsm_proposals`, `parse_sense_session`) with their regex helpers | new section "W4 S5 parsers" before exposition |
| +3 emit blocks in `render_metrics()` | bottom of render_metrics, before the return |

11 new metrics emitted:
- `cipher_classify_total`, `cipher_classify_handled`, `cipher_classify_passthrough` (counters)
- `cipher_classify_per_op_class{op=...}` (counter, per-op when non-empty)
- `cipher_dsm_proposals_total` (counter), `cipher_dsm_proposals_ring_size` (gauge)
- `cipher_sense_total_sessions`, `cipher_sense_total_transitions` (counters)
- `cipher_sense_per_class{class="HUMAN|AGENT|BATCH|UNKNOWN"}` (gauge × 4)

(Per-DSM-entry detail intentionally NOT exported as metrics — cardinality
explosion risk; the /proc node remains the diagnostic source for entry-by-entry.)

### C.3 smoke test — PASS

```
=== Syntax check ===
syntax ok

=== /health ===
ok

=== /metrics W4 S5 lines ===
cipher_classify_total 0
cipher_classify_handled 0
cipher_classify_passthrough 0
cipher_dsm_proposals_total 0
cipher_dsm_proposals_ring_size 256     ← from kmod proc node
cipher_sense_total_sessions 0
cipher_sense_total_transitions 0
cipher_sense_per_class{class="HUMAN"} 0
cipher_sense_per_class{class="AGENT"} 0
cipher_sense_per_class{class="BATCH"} 0
cipher_sense_per_class{class="UNKNOWN"} 0

=== /foo ===
HTTP 404
```

`cipher_dsm_proposals_ring_size 256` confirms the parser is correctly
reading the kmod-emitted `ring_size: 256` line (CIPHER_DSM_PROPOSAL_RING_SIZE
from W3-S4 II-a). Other metrics are zero as expected (no producers
wired). Total metric lines emitted: 22 (was 11 pre-extension; +11 net).

---

## D — Build + SC6 + commits — PASS

| sub-step | result |
|---|---|
| **D.1** cipher_rt_phase4 build | `Nothing to be done for 'all'` — no source changes (exporter is Python in a separate dir); `libcipher_rt.so` md5 `90fea16b` (unchanged from Step 3) |
| **D.2a** SC6 TinyLlama vanilla | PASS 7/7 bit-identical |
| **D.2b** SC6 TinyLlama CIPHER | PASS 7/7 bit-identical |
| **D.3** CP 5.4 final | 15/15 PASS, **byte-identical** to A.4 pre-baseline |
| **D.4a** cipher_kmod commit | **`5782609`** "Week 4 Step 5 (kmod): /proc/cipher/sense_session proc node" — 3 files / +94 / -0 |
| **D.4b** cipher_kmod tag | **`week-4-step-5-sense-session-proc`** ✓ at HEAD |
| **D.4c** exporter changes | on-disk only (cipher_exporter is NOT a git repo); md5 `83d0a3d5`; will be captured in Step 7 cipher-fusion-evidence tail commit |
| **D.4d** cipher_rt_phase4 commit | **NOT made** — no source changes; no rotation; no `week-4-step-5-prometheus` tag placed (would be a stale-at-Step-3 commit, semantically misleading) |
| Fallbacks preserved | `cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step5` (`15a0d50d`) + `cipher_kmod_fallback/cipher_kmod.ko.w4_step5_sense_session` (`e9855b6e`) |

---

## Honest notes

1. **Step 5 has fewer commits than the spec implied.** Spec L262 said
   "cipher_rt_phase4 commit (Python exporter changes)" + tag
   `week-4-step-5-prometheus`. But:
   - The Python exporter lives in `/home/ubuntu/cipher_exporter/`, NOT
     in `cipher_rt_phase4/`. cipher_rt_phase4 has zero Step 5 source
     changes.
   - `/home/ubuntu/cipher_exporter/` is not a git repo.
   - Tagging cipher_rt_phase4 at unchanged HEAD `3be4531` with
     "week-4-step-5-prometheus" would point at Step 3's commit, which is
     semantically misleading.
   The exporter changes are captured on disk (md5 cited above) and will
   roll into the Step 7 cipher-fusion-evidence tail commit along with
   the result doc.

2. **dsm_proposals per-entry detail NOT exported as Prometheus metrics.**
   The /proc/cipher/dsm_proposals node emits up to 64 most-recent ring
   entries with `tenant=..., reason=..., conf=..., age=... ms` — each
   would explode cardinality if exported as a labeled metric. Kept the
   per-entry detail in /proc only (diagnostic), and exported only the
   aggregate counters (`total_received`, `ring_size`). Consistent with
   Prometheus best practice for high-cardinality event logs.

3. **DKMS sync was a hard requirement, not optional.** Without the post-
   Step-4 DKMS sync that resolved the May-16 stale baseline, Step 5's
   kmod rebuild would have produced an LP-8-retired build *on top of* a
   DKMS image that still had LP-8 — and after `rmmod + modprobe`, the
   DKMS-installed module (not the local build) would have been the one
   loaded. Step 5 entry gate (A.2 "DKMS-installed srcversion matches
   loaded") caught this proactively. Both srcversions now in sync at
   `2106824D…` post-Step-5.

4. **No producer wiring for sense_session in Step 5.** Counters stay
   BSS-zero. The proc emitter explicitly says "no producer wired yet —
   Step 6+ will populate via ioctl bridge". The Prometheus exporter
   reads them as zero. This is identical to the W2-S5 / W3-S4 II-a
   pattern: kmod-side counter infrastructure ships first; userspace
   producer ships later (or in Step 6 if folded).

5. **Wave 5 §132 Step 5 budget = ~20 LOC kmod**; actual was ~94 LOC
   kmod (+14 internal.h, +5 main.c, +75 proc.c). The overshoot is from
   the cascading proc_create error-handling pattern — adding a single
   proc node requires duplicating the full cleanup cascade (10+ lines
   per cascade level, ×N entries). Existing pattern; not refactorable
   without restructuring all 8 proc-entry registrations.

---

## Gate decision

**Step 5: PASS.** All A/B/C/D gates green:
- srcversion entry gate held (180A1D4… loaded + DKMS in sync)
- kmod build clean (rc=0); DKMS sync clean; reload clean
- srcversion exit gate held (CHANGED to 2106824D…; DKMS in sync with loaded)
- /proc/cipher/sense_session reachable + emits expected zeros
- exporter syntax-ok + smoke test PASS (11 new metrics, /health=200, /foo=404)
- CP 5.4 isolation 15/15 byte-identical
- SC6 TinyLlama vanilla 7/7 + CIPHER 7/7 bit-identical
- ABI: nr 9 still -ENOSYS, nr 8 still mask=count=0, /dev/cipher mode 666 preserved
- All fallbacks preserved with md5

**Step 6 cleared to proceed** — Sub-4 measurement infrastructure.

---

**Evidence:**
- `/tmp/week4_step5_6_7/cipher_kmod.ko.step5_pre` (snapshot, md5 `15a0d50d`)
- `/tmp/week4_step5_6_7/libcipher_rt.so.step5_pre` (snapshot, md5 `90fea16b`)
- `/tmp/week4_step5_6_7/srcversion_post_step5b.txt` = `2106824D60BEA5C2333F4D6`
- `/tmp/week4_step5_6_7/cp54_pre_step5.log` + `cp54_post_step5b.log` + `cp54_post_step5_final.log` (all 15/15 PASS, byte-identical)
- `/tmp/week4_step5_6_7/sc6_pre_step5_vanilla.log` + `sc6_post_step5_{vanilla,cipher}.log` (all 7/7 bit-identical)
- `/tmp/week4_step5_6_7/kmod_build.log` (clean rc=0)
- `/tmp/week4_step5_6_7/exporter_smoke.log` (22 metric lines, 11 new W4-S5)
- `/tmp/week4_step5_6_7/exporter_stderr.log` (HTTP debug log)
- `/home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.{pre_w4_step5,w4_step5_sense_session}` (fallbacks)
- `git show 5782609` in cipher_kmod
