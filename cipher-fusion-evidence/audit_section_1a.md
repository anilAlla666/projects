# CIPHER Code-Reality Audit — Section 1a

## Section 1a — The 12 core ops

### Scope note (read first)

The task brief states the 12 core ops are "implemented in `cipher_10ops_impl.cpp`
+ `cipher_10ops.h`". This is **only true for the Stage 1/2 ops**. That file's own
header comment (lines 5-6) scopes itself to "Stage 1 ... REMEMBER+VALIDATE+AUDIT+
SPECULATE" and "Stage 2 ... ADAPT+ARBITRATE" — it contains **no Stage 0 code**.
The 6 Stage 0 ops live in other files, so this audit was expanded to:

- `src/cipher_dispatch.cpp` — the hot path (CLASSIFY, SUBSTITUTE, oracle-call site)
- `src/cipher_oracle.cpp` + `include/cipher_oracle.h` — the safety oracle
- `include/cipher_classify.hpp` — CLASSIFY geometry fingerprinter
- `src/cipher_intercept.cpp` — RING_WRITE call site, SPECULATE-check call site
- `src/cipher_intercept_cudart.cpp` — SAMPLE ring (`cipher_read_sample` / `maybe_collect_sample`)

**Naming mismatch (load-bearing).** The task's Stage 0 op list
(`CLASSIFY, ORACLE, SUBSTITUTE, COMMIT, SAMPLE, RING_WRITE`) does **not** match
the names the code uses for itself. `cipher_10ops_impl.cpp:971-972` prints the
Stage 0 ops as `CLASSIFY + SPECULATE_CHECK + SUBSTITUTE + ORCHESTRATE + GENERATE
+ RING_WRITE`. There is **no symbol, comment, or marker containing the string
`COMMIT` anywhere in `src/` or `include/`** (`grep -rn COMMIT` → empty). This
audit maps the task names to the closest real code construct and flags every
inference. In particular: the task's "ORACLE" maps to the `cipher_oracle_*`
safety gate (`cipher_oracle.cpp`), while the code's own "Op 2" marker
(`cipher_intercept.cpp:135`) labels the **SPECULATE look-aside check**, a
different thing. "SAMPLE" maps to the EDMD sample ring. "COMMIT" has no clear
code counterpart — see the STUB/MISSING list below.

### Classification table

| Op# | Name | Stage | Class | File:Line evidence | Stress firing | Notes |
|----|------|-------|-------|--------------------|---------------|-------|
| 1 | CLASSIFY | 0 | **WORKING** | `cipher_classify.hpp:94-224`; called `cipher_dispatch.cpp:406-411` & `523-530` | UNKNOWN (no per-op counter); indirect FIRES — `DETERMINISM dispatch_count=36450` in `per_op_validation.json` proves the dispatch/classify path ran | Real geometry fingerprinter: 7-class heuristic on grid/block/shared (`fingerprint()` 94-174) + 512-slot lock-free fn-ptr cache (181-224). Substantive, runs unconditionally on every launch. |
| 2 | ORACLE | 0 | **WORKING** | `cipher_oracle.cpp:293-377` (`cipher_oracle_decide`); called `cipher_dispatch.cpp:496` & `558` | UNKNOWN (no per-op counter; oracle path runs whenever dispatch runs) | Real 5-gate safety logic: phase gate, min-confidence, structural lookup, EMA demotion, N≤4 rule. Phase detector + EMA divergence monitor also real (`cipher_oracle.cpp:125-259`). NOTE: code's own "Op 2" marker at `cipher_intercept.cpp:135` is the **SPECULATE-check**, not this oracle — see scope note. |
| 3 | SUBSTITUTE | 0 | **PARTIAL** | `cipher_dispatch.cpp:200-395` (`apply_recipe`), GEMM case `205-377` | UNKNOWN (no `[L3.2] SUBSTITUTE` markers in `a3_cipher.log`) | GEMM recipe path is substantive (relaunch+cache O(1) path, block-level O(Kr) path). BUT: (a) block-sub init fails — `per_op_validation.log:17-18` "Cannot open /workspace/manifold/V_layer20.npy" so `g_block_sub.ready` is false (the `/workspace` vs `/home/ubuntu` layout bug); (b) recipe types 2/4 (`cipher_dispatch.cpp:379-390`) are passthrough-only `return true` with a "For now / Phase 2" comment; (c) the "substitution" is mostly a real-kernel **relaunch + memcpy cache**, not a neural surrogate. Partial: real machinery, key calibration path broken on this pod. |
| 4 | COMMIT | 0 | **MISSING** | not found | UNKNOWN | No symbol/comment/marker named `COMMIT` exists in `src/` or `include/`. Closest functional analogues: `cipher_oracle_record_substitution` (`cipher_oracle.cpp:383-392`, updates the N≤4 counter) and the `CIPHER_SUBSTITUTED` return committing the result (`cipher_dispatch.cpp:587-589`). Neither is named COMMIT and the brief gives no definition to match against — reported MISSING by name; if "COMMIT" means "record the substitution decision" then that logic exists as `cipher_oracle_record_substitution`. |
| 5 | SAMPLE | 0 | **PARTIAL** | `cipher_intercept_cudart.cpp:2353-2425` (`maybe_collect_sample`, `compute_sample_norm`, `cipher_read_sample`) | UNKNOWN (no per-op counter) | Real: copies up to 256 floats off the C tensor, computes a normalized Frobenius norm, writes a 512-entry ring consumed by ADAPT. PARTIAL because (a) `maybe_collect_sample` hard-stops after 500 launches (`cipher_intercept_cudart.cpp:2409-2411` `if (s_launch_count > 500) return;`) — sampling is a brief warmup burst, not continuous; (b) `CIPHER_SAMPLE_DIM` is 1 — only a scalar norm, not a tensor sample despite the header comment. |
| 6 | RING_WRITE | 0 | **WORKING** | `cipher_10ops.h:83-100` (`cipher_ring_write`); call sites `cipher_intercept.cpp:180-181` & `374-375` | UNKNOWN (no `ring_writes` counter dumped in stress2 evidence) — call site is unconditional on every dispatch when 10ops initialized | Real SPMC disruptor-pattern ring: bounds check vs slowest reader, `memcpy` entry, release-store seq. Increments `ring_writes`. Fires once per launch when `g_cipher_10ops.initialized`. |
| 7 | REMEMBER | 1 | **WORKING** | `cipher_10ops_impl.cpp:428-458` | **NEVER FIRES** in `per_op_validation.log:20` ("Stage 1/2 threads skipped — no observers enabled (lazy-start)") | Real CfC LNN forward pass on a dedicated 64-dim shadow hidden state (`cipher_lnn_forward`, input built by `cipher_lnn_build_input`). Substantive. Only runs inside `stage1_shadow`, which is **only spawned if ≥1 observer env var is set** (`cipher_10ops_impl.cpp:954`); in the stress run none were, so the thread never started. |
| 8 | VALIDATE | 1 | **PARTIAL** | `cipher_10ops_impl.cpp:460-475` | **NEVER FIRES** (`per_op_validation.log:20`, Stage 1 thread not spawned) | Updates per-class Welford stats (`rs_update`) and logs mean/var. BUT it is **observational only**: it never increments `validate_failures` — `grep -rn validate_failures src/` shows the counter is only *read* at `cipher_10ops_impl.cpp:1015`, never written. The 3-sigma anomaly detector `rs_ok` (`333-339`) is defined but **never called**. The runtime report's "anomalies detected" line is therefore structurally always 0. |
| 9 | AUDIT | 1 | **PARTIAL** | `cipher_10ops_impl.cpp:341-378` (`audit_chain_update`), live path `472-474` | **NEVER FIRES** (`per_op_validation.log:20`, Stage 1 thread not spawned) | The HMAC-SHA256 chain function `audit_chain_update` is real and substantive. BUT it is **gated off in the live path**: `cipher_10ops_impl.cpp:472-473` — `// AUDIT: skip HMAC on detached thread (OpenSSL not thread-safe here)` — the `audit_chain_update(...)` call is commented out. Only `audit_entries.fetch_add` runs (line 474). So AUDIT in the running system is a bare counter; the cryptographic chain never executes. |
| 10 | SPECULATE | 1 | **WORKING** | write side `cipher_10ops_impl.cpp:477-505`; check side `cipher_intercept.cpp:127-149`; buffer `cipher_10ops.h:46-69` | **NEVER FIRES** (write side: Stage 1 thread not spawned, `per_op_validation.log:20`). Check side runs on every launch but always misses when Stage 1 is off. | Real: Stage 1 derives a predicted next op-class + confidence from the CfC decision and writes the look-aside buffer; Stage 0 does a ~2ns atomic check and skips the kernel on a hit. Logic is complete and substantive. CAVEAT: the Stage-0 check at `cipher_intercept.cpp:129` compares the prediction against `desc.op_class` which is still `0xFF` (unclassified) at that point — a likely correctness weakness, but the op itself is implemented. |
| 11 | ADAPT | 2 | **PARTIAL** | `cipher_10ops_impl.cpp:597-765` (EDMD/Koopman), KEN `99-285` | **NEVER FIRES** (`per_op_validation.log:20`, Stage 2 thread not spawned; `KOOPMAN` row in `OPS_VALIDATION_REPORT.md:16` also SILENT) | EDMD snapshot collection + `cipher_edmd_solve` Koopman fit + `cipher_lnn_koopman_update` weight nudge are all real, plus a full KEN (Koopman Eigenfunction Network) spectral learner. PARTIAL because: (a) the "Koopman weight update" feeds `h_approx = current h` as **both** h_before and h_after (`cipher_10ops_impl.cpp:693-701`) — the comment admits "h_before is approximated", so the gradient step uses a degenerate (zero-delta) pair; (b) the non-real-sample EDMD input is **synthetic** — `s_out[k][i] = s_inp[k][i]*0.99 + ...` (`663-669`), i.e. a hand-constructed near-identity, not measured dynamics. Real plumbing, semi-synthetic signal. |
| 12 | ARBITRATE | 2 | **PARTIAL** | `cipher_10ops_impl.cpp:767-814` | **NEVER FIRES** (`per_op_validation.log:20`, Stage 2 thread not spawned). `OPS_VALIDATION_REPORT.md:17` lists ARBITRATE SILENT "no exported counter — by design". | Reads/writes a POSIX SHM demand block, computes SPECULATE hit-rate, and on high-hit-rate + SM demand sets `s_shm->sm_available`. BUT the actual Green Context SM rebalance is **not implemented** — `cipher_10ops_impl.cpp:789` "Production: call cuDevSmResourceSplit", `799` "Production: cuGreenCtxDestroy", `803` "Production: recreate Stage 2 Green Context". It updates a shared-memory integer; it never repartitions SMs. Observational/signal-only where an actuator is intended → PARTIAL. |

### Ops classified STUB or MISSING

- **Op 4 COMMIT — MISSING (by name).** No symbol, comment, or marker containing
  `COMMIT` exists anywhere in `src/` or `include/` (`grep -rn COMMIT` returns
  nothing). The task brief supplies no definition of what COMMIT should do, so it
  cannot be matched to code by behavior with confidence. If "COMMIT" is intended
  to mean "record/finalize the substitution decision", that behavior exists but
  under different names: `cipher_oracle_record_substitution`
  (`cipher_oracle.cpp:383-392`) and the `return CIPHER_SUBSTITUTED` at
  `cipher_dispatch.cpp:589`. As an op named COMMIT: **not found**.

No ops met the STUB bar (a named function/branch that exists but does nothing).
The closest were AUDIT (function exists but its call is commented out — classified
PARTIAL because the counter path still runs) and recipe types 2/4 inside SUBSTITUTE
(`cipher_dispatch.cpp:379-390`, bare `return true` passthrough — these are sub-branches,
not whole ops, and folded into SUBSTITUTE's PARTIAL).

### Stress-firing summary

The stress suite `stress2/per_op_validation.{md,json,log}` validates the **21
observer ops** (PREDICT, KOOPMAN, TOPOLOGY, …), **not** the 12 core ops by name —
none of CLASSIFY/ORACLE/SUBSTITUTE/COMMIT/SAMPLE/RING_WRITE/REMEMBER/VALIDATE/
AUDIT/SPECULATE/ADAPT/ARBITRATE appear in its result table. The one load-bearing
fact for the core ops: `per_op_validation.log:20` —
`[CIPHER 10ops] Stage 1/2 threads skipped — no observers enabled (lazy-start)`.
Therefore in that run **all six Stage 1/2 ops (REMEMBER, VALIDATE, AUDIT,
SPECULATE-write, ADAPT, ARBITRATE) never executed**. The Stage 0 ops have no
per-op counters in any stress2 file; their firing is **UNKNOWN** from the cited
evidence, with one indirect signal: `DETERMINISM dispatch_count=36450`
(`per_op_validation.json`) confirms the kernel dispatch path (hence CLASSIFY)
ran. `a3_cipher.log` was also scanned — it contains no `[CIPHER Op*]`, `[CIPHER
ADAPT]`, `[CIPHER ARBIT]`, or `[CIPHER Stage1/2]` markers, consistent with the
background threads being unspawned there too.

### Tally

- WORKING: 5 — CLASSIFY, ORACLE, RING_WRITE, REMEMBER, SPECULATE
- PARTIAL: 6 — SUBSTITUTE, SAMPLE, VALIDATE, AUDIT, ADAPT, ARBITRATE
- STUB: 0
- MISSING: 1 — COMMIT (no op by that name; nearest behavior exists as `cipher_oracle_record_substitution`)

(11 of the 12 task-named ops were located in code; COMMIT could not be. 5+6+0+1 = 12.)
