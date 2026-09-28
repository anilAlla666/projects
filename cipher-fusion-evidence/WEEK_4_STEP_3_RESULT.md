# Week 4 Step 3 — Tier B Observability Ports — RESULT

**Status: PASS.**

User picked **option C** at the §0 adjudication gate (port all 7 PURE-OBSERVE-ONLY
files + `cipher_topology.cpp` + `cipher_comply.cpp` = **9 ports**). All 9
ports landed bit-identical. SC6 TinyLlama 7/7 PASS after each intra-step;
SC6 Mistral-7B 7/7 PASS both vanilla and CIPHER-injected; CP 5.4 isolation
15/15 byte-identical; build clean rc=0 warning delta 0 vs Step 2 baseline;
+43 new T-symbols; Step 1 + Step 2 anchors preserved; topology CUDA-init
confirmed gated (env-unset SC6 bit-identical, no `/tmp/cipher_topology*`
emissions).

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 4, Step 3 of 6
**Anchors:**
  - cipher_rt_phase4: `b25edf7` → **`3be4531`** (tag `week-4-step-3-tier-b-ports`)
  - cipher_kmod: `a21a45e` (unchanged — userspace-only)
  - cipher-may13-evidence: `fc8a9ae` (unchanged — read-only source)
  - cipher-fusion-evidence: scope-lock `850bd8b`; Step 2 result `b25edf7`

---

## §0 — Closure pre-flight findings + adjudication

(Full §0 contents preserved from initial draft of this doc; see archive
at the bottom of this section.)

**§0 verdict:** 7 PURE-OBSERVE-ONLY (trace / receipt / carbon / fairness /
fairness_shm / guard / determinism) + 1 INFRASTRUCTURE-MISSING
(`cipher_comply.cpp` consumes `cipher_topology_device_count` from unported
`cipher_topology.cpp` which has CUDA-init-time queries gated on env).

**User adjudication:** **option C** — port all 9 (7 + topology + comply).
Rationale: comply's compliance_ok logic does not depend on `topo_devices`;
`cipher_topology_init` is never called from any of the 9 ports; topology's
static-init scan SAFE; CUDA APIs only fire on explicit init + env=on.

---

## A — Pre-edit verification + baseline snapshot — PASS

| signal | value |
|---|---|
| rt_phase4 HEAD | `b25edf7` ✓ matches Step 2 close |
| kmod HEAD | `a21a45e` ✓ unchanged |
| may13-evidence HEAD | `fc8a9ae` ✓ unchanged |
| trees clean (3/3) | ✓ |
| Snapshot pre | md5 `8c81053b` → `/tmp/week4_step3/libcipher_rt.so.step3_pre` |
| Pre-build clean | rc=0, md5 `aa65aa7e` (nvcc non-deterministic), 78 warnings |
| Pre-edit SC6 TinyLlama vanilla | PASS 7/7 bit-identical |

---

## B–J — 9 Intra-step ports (all PASS)

For each port: static-init scan → copy + include-rewrite → Makefile
(OBJS + per-source rule) → clean build → symbol audit → SC6 TinyLlama
CIPHER. All 9 PASS bit-identical.

| step | port | scan | bytes | new T | regression check | SC6 |
|---|---|---|---|---|---|---|
| **B** | `cipher_trace`        | SAFE | 4016 + 673 | **5** | Step-2 (oracle_bridge=2, Tier A 27) intact | PASS 7/7 |
| **C** | `cipher_receipt`      | SAFE | 7984 + 781 | **4** | trace=5, oracle_bridge=2 | PASS 7/7 |
| **D** | `cipher_carbon`       | SAFE | 6012 + 742 | **4** | trace=5 receipt=4, oracle_bridge=2 | PASS 7/7 |
| **E** | `cipher_fairness`     | SAFE | 6687 + 682 | **5** | t/r/c intact, oracle_bridge=2 | PASS 7/7 |
| **F** | `cipher_fairness_shm` | SAFE | (~6 KB) | **7** | shm_open@GLIBC_2.34 added (no -lrt needed); pre/post `/dev/shm/cipher*` unchanged (no init caller) | PASS 7/7 |
| **G** | `cipher_guard`        | SAFE | 7050 + 723 | **5** | sibs intact | PASS 7/7 |
| **H** | `cipher_determinism`  | SAFE | 3124 + 724 (.h pre-staged Week 1 Step 2 v2) | **5** | sibs intact | PASS 7/7 |
| **I** | `cipher_topology`     | SAFE (.init_array=0; CUDA INSIDE init()) | 3228 + 730 | **4** | cudaGetDeviceCount + cudaDeviceCanAccessPeer added to libcipher_rt undef set — gated | PASS 7/7 + **`/tmp/cipher_topology*` absent** |
| **J** | `cipher_comply`       | SAFE | 3309 + 621 | **4** | links cleanly: 8 sibling refs + topology_device_count = all resolved | PASS 7/7 + **`/tmp/cipher_comply_report.json` absent** |

**Per-port build:** rc=0, warning count **78** (delta 0 vs pre-edit
baseline). All 9 intra-step builds were `make clean && make` full rebuilds.

**Per-port SC6 logs:**
`/tmp/week4_step3/sc6_{trace,receipt,carbon,fairness,fairness_shm,guard,determinism,topology_cipher_unset,comply}_cipher.log`

### Step I — Topology safety contract (medium-risk port)

cipher_topology.cpp has `#include <cuda_runtime.h>` and calls
`cudaGetDeviceCount` / `cudaDeviceCanAccessPeer` inside
`cipher_topology_init()`. Three structural guards confirm safety:

1. **Static-init SAFE.** `objdump -h` shows `.init_array=0`, no
   `_GLOBAL__sub_I_*` symbols. `.so` load triggers no topology code.
2. **No caller of `cipher_topology_init` exists in libcipher_rt** — neither
   any pre-existing cipher_rt_phase4 source nor any of the 9 Step-3 ports
   invoke it (verified by `grep -rE 'cipher_topology_init' src/`).
3. **Env-unset SC6 PASS bit-identical** + `ls /tmp/cipher_topology*` empty
   post-run → CUDA APIs were not exercised.

Topology is therefore **compiled-into-libcipher_rt but unreachable at load
time and inert at runtime unless explicitly enabled** (`CIPHER_TOPOLOGY=on`
+ a caller that invokes `cipher_topology_init`). The two new CUDA undefs
(`cudaGetDeviceCount`, `cudaDeviceCanAccessPeer`) resolve to libcudart
which libcipher_rt already links against (`-lcudart`) — no new dependency.

---

## K — Aggregate Mistral-7B SC6 (load-bearing) — PASS

| arm | result | log |
|---|---|---|
| **K.1 vanilla** | **PASS 7/7 bit-identical** | `/tmp/week4_step3/sc6_post_mistral_vanilla.log` |
| **K.2 CIPHER-injected** | **PASS 7/7 bit-identical** | `/tmp/week4_step3/sc6_post_mistral_cipher.log` |

CIPHER-injected arm executed against the post-Step-3 libcipher_rt.so
(md5 `90fea16b`) carrying all 9 new ports + Step 2's Tier A + Step 1's
oracle bridge. **Substrate observe-only confirmed at 7B scale** under
both producer and consumer arms.

---

## L — CP 5.4 regression + side-effect inventory — PASS

### L.1 CP 5.4 isolation: 15/15 byte-identical

```
=== Phase A result: 15 PASS, 0 FAIL ===
```
Log: `/tmp/week4_step3/cp54_isolation.log`. Identical to Step 2 and Step 1
runs — kmod + arbitration substrate untouched.

### L.2 Side-effect inventory

| sink | pre-Step state | post-Step state | finding |
|---|---|---|---|
| `/tmp/cipher_topology*` | absent | **absent** | topology init never fired (CIPHER_TOPOLOGY unset) |
| `/tmp/cipher_comply_report.json` | absent | **absent** | comply report never emitted (CIPHER_COMPLY unset) |
| `/tmp/cipher_pulse*` | absent | **absent** | Step 2 PULSE Stage-2 never fired (no observe caller; consistent with Step 2 finding) |
| `/dev/shm/cipher*` | May-15 `cipher_fairness`, `cipher_sm_demand` | **unchanged** | `cipher_fairness_shm_init` never called; no new shm segments |
| `/tmp/cipher_*` other | May-18/19 pre-existing decode_window artifacts | **unchanged** | nothing dated 2026-05-21 |

**Aggregate read:** every emission gate in the 9 newly-ported ops is
env-conditional and behaviorally observe-only when env is unset. SC6 ran
with no Step-3 env vars set. Nothing fired. The substrate is present and
inert — exactly the Step-3 contract.

### L.3 Topology env-unset confirmation

Verified during Step I; re-confirmed at Step L: `CIPHER_TOPOLOGY` unset
throughout this session; no `/tmp/cipher_topology*` files appeared at any
point; libcipher_rt loads carry the topology TU without invoking any CUDA
driver call.

---

## M — Structural link-graph proof — PASS

Following the Step 2 advisor pattern:

| check | result |
|---|---|
| **M.1** undef refs to any `cipher_(trace\|receipt\|carbon\|fairness\|fairness_shm\|guard\|determinism\|topology\|comply)_*` across the 40 pre-Step-3 .o files | **0** — consumer activation impossible by construction |
| **M.2** comply.o → Tier B siblings (the documented aggregator design) | 8 references all resolve to libcipher_rt exports of this Step (receipt_session_count, carbon_session_count, guard_leak_count + session_count, determinism_count + hash, fairness_overrun_count + tenant_count) + 1 topology_device_count (also exported this Step). All in-Step, no orphan |

The link-graph rules out unintentional consumer activation of the 9 new
surfaces. SC6 confirms the substrate also doesn't perturb runtime via
static-init or library-load side effects. The 9 new TUs only reference
each other in the comply→siblings aggregator pattern (which IS the design),
and that pattern is also observe-only (comply's report fires only on
`CIPHER_COMPLY=on` + an explicit observer caller).

---

## N — Final build state + commit + tag — PASS

| signal | value |
|---|---|
| N.1 final md5 | `90fea16b2f5f2221274bb3f4a0a33405` |
| N.1 pre-Step-3 md5 | `8c81053be2f96aae3413e3affaae5ba2` |
| N.1 total new T symbols | **+43** (TRACE 5 + RECEIPT 4 + CARBON 4 + FAIRNESS 5 + FAIRNESS_SHM 7 + GUARD 5 + DETERMINISM 5 + TOPOLOGY 4 + COMPLY 4) |
| N.1 Step 1 + Step 2 anchors preserved | oracle_bridge=2; loop=6 / pipeline=5 / pulse=11 / continuity=5 — all intact |
| N.1 build warning count | **78** (delta 0 vs pre-edit; same 78 fortified-libc warnings from cipher_may13_stubs.cpp) |
| N.1 diff stat | **18 files changed, 1643 insertions** |
| N.2 commit | **`3be4531a608b574289f5a30a0b38fb06b7156ec5`** |
| N.2 author | Anil &lt;anil.0666369@gmail.com&gt; (per-commit via `git -c`, no global config mutation) |
| N.3 tag | **`week-4-step-3-tier-b-ports`** ✓ points at HEAD |

### Files committed (18)

```
Makefile                                       +38  -0  (9 OBJS, 9 rules, 3 comment lines)
src/may13/cipher_trace.cpp                     +new
src/may13/cipher_receipt.cpp                   +new
src/may13/cipher_carbon.cpp                    +new
src/may13/cipher_fairness.cpp                  +new
src/may13/cipher_fairness_shm.cpp              +new
src/may13/cipher_guard.cpp                     +new
src/may13/cipher_determinism.cpp               +new (header pre-staged Week 1 Step 2 v2)
src/may13/cipher_topology.cpp                  +new
src/may13/cipher_comply.cpp                    +new
include/may13/cipher_trace.h                   +new
include/may13/cipher_receipt.h                 +new
include/may13/cipher_carbon.h                  +new
include/may13/cipher_fairness.h                +new
include/may13/cipher_fairness_shm.h            +new
include/may13/cipher_guard.h                   +new
include/may13/cipher_topology.h                +new
include/may13/cipher_comply.h                  +new
```

(`include/may13/cipher_determinism.h` was already tracked from Week 1 Step
2 v2's header port; my port_one.sh re-rewrote it with bit-identical bytes
so no diff.)

---

## Honest notes

1. **9 ports, not 7-8.** The user-spec mapping listed "TRACE / RECEIPT /
   CARBON / FAIRNESS+SHM / GUARD / COMPLY / DETERMINISM" = 7 names. On
   disk, FAIRNESS+SHM is two files (`cipher_fairness.cpp` +
   `cipher_fairness_shm.cpp`), bringing the natural count to 8. The §0
   adjudication added cipher_topology.cpp as the COMPLY dependency,
   bringing the final count to 9 ports.

2. **No `-lrt` needed.** `cipher_fairness_shm.cpp` calls `shm_open`/`mmap`.
   First instinct was to add `-lrt` to its compile rule (and I did, then
   reverted). Modern glibc (≥2.34) provides `shm_open` in libc; the link
   resolves cleanly as `shm_open@GLIBC_2.34` against libc. Verified by
   nm post-link.

3. **`cipher_determinism.h` was already in HEAD** from Week 1 Step 2 v2
   ("may13 header port, Option B closure"). My port-script overwrote it
   with identical bytes (the existing header was already in may13/-prefix
   form). No diff to track. The .cpp port is new.

4. **Topology's CUDA undefs are inert at load time, gated at runtime.**
   `cudaGetDeviceCount` and `cudaDeviceCanAccessPeer` are added to
   libcipher_rt's undef set in Step I. These resolve to libcudart at load
   (already linked) — no new dependency. They are only **called** if
   `cipher_topology_init()` runs, which requires both `CIPHER_TOPOLOGY=on`
   AND an explicit caller. Neither exists in this build. Step I's
   env-unset SC6 + empty `/tmp/cipher_topology*` empirically confirm the
   contract.

5. **comply's aggregator design.** `cipher_comply.o` has 9 cross-TU
   refs (8 sibling Tier B + 1 topology). This is the documented Op 29
   aggregator pattern — comply pulls per-op counters and writes a
   compliance JSON. Since comply itself is env-gated (`CIPHER_COMPLY=on`)
   and no observer wiring exists yet, all 9 cross-refs are effectively
   inert. Step 4+ or later weeks wire callers.

6. **Substrate observe-only also structurally guaranteed.** Per Step 2's
   advisor strengthening: if any pre-existing .o referenced a new Step-3
   symbol, the pre-Step-3 build would have link-failed. The pre-Step-3
   build succeeded (Step A) → zero pre-existing consumers exist for any
   of the 43 new symbols. Combined with SC6 bit-identity, the
   observe-only claim has both a structural (link-graph) and behavioral
   (SC6) proof.

---

## Goals enabled (substrate present, consumers pending)

| Op | source | enables | consumer pending |
|---|---|---|---|
| **Op 28 TRACE** | `cipher_trace.cpp` | **Goal 5** — deployment auditability; bounded ring of last-N dispatches | a periodic flush driver / Tier C consumer |
| **Op 30 RECEIPT** | `cipher_receipt.cpp` | **Goal 5** — per-session work-receipt persistence with HMAC | a billing/persistence caller (RECEIPT_SECRET env) |
| **Op 31 CARBON** | `cipher_carbon.cpp` | **Goal 1** — tok/W reporting surface (energy attribution per session) | a periodic NVML poller / report driver |
| **Op 24 FAIRNESS** | `cipher_fairness.cpp` | **Goal 2** — per-tenant work quota observer | a quota enforcer / scheduler hook |
| **Op 24b FAIRNESS_SHM** | `cipher_fairness_shm.cpp` | **Goal 2** — cross-process POSIX shm for multi-tenant coordination | the shm registration + yield driver |
| **Op 32 GUARD** | `cipher_guard.cpp` | **Goal 5** — LD_PRELOAD-only safety (leak detection, session count) | a fault-detection observer |
| **Op 33 DETERMINISM** | `cipher_determinism.cpp` | **Goal 5** — reproducibility hooks (event hash chain) | a determinism-mode dispatch wrap |
| **Op 25 TOPOLOGY** | `cipher_topology.cpp` | NVLink/PCIe adjacency surface (COMPLY dependency; multi-GPU future) | explicit init under `CIPHER_TOPOLOGY=on` |
| **Op 29 COMPLY** | `cipher_comply.cpp` | **Goal 5** — compliance event aggregator (bundles RECEIPT+CARBON+GUARD+DETERMINISM+FAIRNESS+TOPOLOGY into one JSON pack) | report driver / external auditor |

---

## Adjudication ask

Step 3 closed cleanly. **Step 4 (LP-8 allocator retirement) is unblocked**
per `WEEK_4_SCOPE_LOCK.md` §2. Step 4 entry pre-flight should:
1. Confirm the LP-8 surface (kmod-side allocator retirement; what gets
   removed, what gets stubbed).
2. Check that no Step 1/2/3 surface references the LP-8 path.
3. Plan the kmod rebuild + DKMS reload cadence (this Step 3 didn't touch
   kmod — Step 4 will be the first kmod rotation since Step 1).

**Rollback:** `git -C /home/ubuntu/cipher_rt_phase4 reset --hard week-4-step-2-tier-a-ports`

---

**Evidence:**
- `/tmp/week4_step3/libcipher_rt.so.step3_pre` (snapshot, md5 `8c81053b`)
- `/tmp/week4_step3/sc6_pre_vanilla.log` (TinyLlama baseline, PASS)
- `/tmp/week4_step3/sc6_{trace,receipt,carbon,fairness,fairness_shm,guard,determinism,topology_cipher_unset,comply}_cipher.log` (9 intra-step PASS)
- `/tmp/week4_step3/sc6_post_mistral_{vanilla,cipher}.log` (Mistral-7B both arms, PASS)
- `/tmp/week4_step3/cp54_isolation.log` (CP 5.4 15/15)
- `/tmp/post_{trace,receipt,carbon,fairness,fairness_shm,guard,determinism,topology,comply}_build.log` (per-port builds)
- `/tmp/pre_step3_build.log` (pre-edit baseline build)
- `/tmp/closure_tierB.py` (Part 0 closure tool, runnable post-hoc against current symbol table)
- `git show 3be4531` (commit)

---

## §0 archive — original closure pre-flight findings + 4-option table (kept for audit trail)

### The 4 options surfaced at the adjudication stop (option C selected)

| | option | scope | risk | recommendation at the time |
|---|---|---|---|---|
| **A** | Port 7 PURE-OBSERVE-ONLY files; defer COMPLY | 7 files (TRACE / RECEIPT / CARBON / FAIRNESS / FAIRNESS_SHM / GUARD / DETERMINISM) | None for the 7; COMPLY's aggregator role deferred but Goal 5 partial cover via the 4 individual ports | Recommended at the time as safest |
| **B** | Port 7 + COMPLY with `cipher_topology_device_count` stub in `cipher_may13_stubs.cpp` | 8 ports; stub returns 0; matches H1 stub precedent | Low; comply's `ok` boolean unchanged | Best correctness/cost trade |
| **C** | Port 7 + cipher_topology.cpp + COMPLY | 9 ports; topology adds CUDA undefs (gated); SC6 must confirm env-unset bit-identical | Medium pre-execution; **post-execution: structurally and behaviorally safe** | **USER SELECTED** — see SC6/topology safety verification in this doc |
| **D** | Stop entirely; re-scope Step 3 | Paperwork only; 0 ports today | High cost (loses 7 clean ports) | Strict spec adherence only |

User picked **C** for the future multi-GPU value of having topology linked
in (even if dormant in v1), accepting the medium pre-execution risk that
the SC6 verification then dispelled (Step I env-unset bit-identical +
empty `/tmp/cipher_topology*` confirmed CUDA APIs unreachable at load).

### Original closure table (verbatim)

| file | category | reason |
|---|---|---|
| `cipher_trace.cpp` | PURE-OBSERVE-ONLY | 1 non-system undef (`__fprintf_chk` libc fortify) |
| `cipher_receipt.cpp` | PURE-OBSERVE-ONLY | HMAC/EVP_sha256 satisfied by `-lcrypto` already in LDFLAGS |
| `cipher_carbon.cpp` | PURE-OBSERVE-ONLY | 1 lib-ok (`cipher_sense_current_session`) + libc fortify |
| `cipher_fairness.cpp` | PURE-OBSERVE-ONLY | 1 lib-ok + libc fortify |
| `cipher_fairness_shm.cpp` | PURE-OBSERVE-ONLY | `shm_open`+`mmap` flagged as SIDE-EFFECT-CAPABLE; satisfied by glibc 2.34 in libc (no -lrt needed); SC6 confirmed no shm segments created |
| `cipher_guard.cpp` | PURE-OBSERVE-ONLY | 1 lib-ok + libc fortify |
| `cipher_determinism.cpp` | PURE-OBSERVE-ONLY | libc fortify only |
| `cipher_comply.cpp` | INFRASTRUCTURE-MISSING | 9 TREE-PULLs: 8 sibling Tier B + 1 `cipher_topology_device_count` from unported topology.cpp |

### Topology-stub equivalence note (honest)

Option **C** (port topology) and option **B** (stub `cipher_topology_device_count`)
converge to **the same runtime value** of `topo_devices` until something
explicitly invokes `cipher_topology_init()`:

- **Option B's stub** returns `0` literally.
- **Option C's compiled-in topology** has `g_device_count` BSS-initialized
  to `0`, only updated by `cipher_topology_init`. With no caller of
  `_init()` in libcipher_rt or in the 9 ports, `cipher_topology_device_count()`
  returns `0` indefinitely. Identical to the stub.

The architectural distinction between B and C is therefore "is topology
code linked in" — real value when multi-GPU adjacency work begins (the
adjacency matrix and the CUDA queries are then directly callable from
inside libcipher_rt), but **zero runtime difference today**. C was the
right pick for forward-compat; the result doc should not overstate
short-term differentiation from B.

