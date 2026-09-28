# V1 Goal 5 deployment-layer ledger

**Date created:** 2026-05-26 (A.0 paperwork commit, tag `v1-goal5-contract-lock`)
**Owner:** Anil; ledger gate per Anil 2026-05-26 V1 substrate work sequence adjudication
**Scope:** v1 + v1.5 substrate work, phases A through E. Every substrate commit in phases A through E that touches deployment surface MUST update this ledger in the SAME commit per Anil 2026-05-26 ledger-gate discipline.

---

## 1. The locked Goal 5 contract (2026-05-26)

> Customer's existing vLLM serve command runs unchanged with ONE addition: set `LD_PRELOAD=libcipher_rt.so` (or `CUDA_INJECTION64_PATH=libcipher_rt.so` equivalent).
>
> No pip install of `cipher_vllm_plugin`.
> No `--quantization` CLI flag introduced BY CIPHER (customer-set FP8 quantization is the customer's model property, not a CIPHER requirement).
> No `quant_config` Python wrapping.
> No model config field added.
> No Python import-hook required.
>
> Zero application code changes means zero, measured against this contract.

Source: Anil adjudication 2026-05-26, recorded at [[v1-goal5-contract-lock]] memory + `V1_PHASE_A_SCOPE_LOCK.md` §3 decision 1 + `CIPHER_REENGINEERING_PLAN.md:98, 104` (original framing, retained verbatim).

---

## 2. The ledger

| Phase | Customer step | When added/removed | Contract violation? | Status |
|---|---|---|---|---|
| Pre-Phase-A | + `pip install cipher-vllm-kv` | Option 2 close 2026-05-25 | YES (per Goal 5 contract 2026-05-26) | REMOVED at Phase A close 2026-05-26 (plugin OPTIONAL for v1 deployment, INSTALLED for dev/debug only) |
| Phase A close | + set `CUDA_INJECTION64_PATH=libcipher_rt.so` (vLLM V1) OR `LD_PRELOAD=libcipher_rt.so` (other CUDA workloads) | Phase A close 2026-05-26 | NO (contract "or" disjunction preserved; one env var either way) | LANDED |
| Substrate mechanism note | R-A.1 + R-A.2 finding 2026-05-26: vLLM V1 modern CUDA uses cuGetProcAddress lazy resolution which bypasses LD_PRELOAD cuInit interception. CUDA_INJECTION64_PATH is the driver-mediated active-invocation equivalent. Constructor + cuInit-wrapper retained as belt-and-suspenders for non-vLLM-V1 deployments where direct symbol resolution + LD_PRELOAD precedence work. | Phase A.1 Branch B mechanism finding 2026-05-26 | N/A (substrate-internal architecture) | informational |
| Phase B | (no change to customer step; M1 CDI + P1 in-place patch injects all env vars container-side transparently — customer sets ZERO env vars on `docker run --gpus all`) | Phase B B.6''.9.3 close 2026-05-26 (rev6 .deb md5 7c7068ca) | NO (CDI-injected env is substrate-side, not customer-side; if anything Goal 5 surface is BETTER than Phase A) | LANDED — rev6 ships CIPHER_CDI_MARKER + CUDA_INJECTION64_PATH + LD_LIBRARY_PATH + TORCH_CUBLASLT_DISABLE + CIPHER_REGISTER_MODEL=0 via CDI; honest residue items 9+10 (NR 27 KASAN follow-up; cross-model KV-dedup deferred) in `V1_PHASE_B_SCOPE_LOCK_ADDENDUM_B1_DOUBLE_PRIME.md` §15 |
| Phase C | (no change to customer step expected) | n/a | n/a | pending |
| Phase D | (customer sets `--quantization fp8` for FP8 models per their own model choice) | quantization is model property, not CIPHER requirement | NO | pending |

---

## 3. Ledger-gate discipline (Anil 2026-05-26)

Every substrate commit in phases A through E that touches deployment surface MUST update this ledger in the SAME commit. Two binding rules:

1. **No silent additions.** If a substrate change would add a row to this ledger, the row addition lands in the same commit as the substrate code. A substrate commit that fails to update the ledger when the deployment surface changes is a discipline violation.

2. **YES-violation gate.** Any row marked "Contract violation? YES" blocks the substrate commit until resolved per Anil adjudication. Resolution paths are: (a) re-engineer the substrate work to avoid the contract violation, (b) defer the substrate work to v1.5 or later, (c) explicit Anil reversal of the Goal 5 contract for that specific case. Path (c) requires the same explicit reversal discipline as the 2026-05-26 reframe revert (recorded, not silently merged).

Per-phase ledger update checklist (carry into Phase A through E close-out docs):
- Before the substrate commit, identify whether the change adds or removes a customer-side step.
- If yes: edit this ledger in the same working tree; row addition or status change.
- If no: explicitly state "no ledger row added" in the commit message and skip the ledger edit.
- Verify with `grep -nE "Phase [A-E]" V1_GOAL5_DEPLOYMENT_LEDGER.md` that the ledger reflects current state before commit.

---

## 4. Phase-specific gate notes

### Phase A (`v1-substrate-driver-worker-init` at cipher_rt_phase4 `8613812e`)

CLOSED 2026-05-26 via Branch D (a): plugin install REMOVED from customer deployment surface; replaced with `CUDA_INJECTION64_PATH=libcipher_rt.so` for vLLM V1 (driver-mediated active invocation) OR `LD_PRELOAD=libcipher_rt.so` for non-vLLM-V1 CUDA workloads (constructor + cuInit-wrapper belt-and-suspenders).

Mechanism finding row added: R-A.1 + R-A.2 surfaced 2026-05-26 documents why LD_PRELOAD-only does not work for vLLM V1 (modern CUDA cuGetProcAddress lazy resolution bypasses LD_PRELOAD interception); CUDA_INJECTION64_PATH is the equivalent path that does work (driver actively invokes `InitializeInjection2` from the named .so).

A.3 regression measurement: Track 2 SC6 numbers under no-plugin path deferred to Phase B entry per R-A.7-reframed-as-expected (the 76% number depends on the plugin's `_cipher_allocate_kv_cache_tensors` hook; without plugin the savings collapse to substrate-only baseline by mechanism). Phase B entry checklist captures the deferred measurement.

See `V1_PHASE_A_COMPLETE.md` for full close-out.

### Phase B (KV-bridge migration into substrate)

- Anticipated: no customer-side step change. Plugin functionality moves into libcipher_rt.so; customer still only sets LD_PRELOAD.
- Ledger update at Phase B close: confirm "Phase B" row stays at "no change to customer step expected" or update if scope shifts during Phase B implementation.

### Phase C (BF16 cuBLAS dtype-extension for Koopman)

- Anticipated: no customer-side step change. Substrate-internal dtype gate change at `cipher_rt_koopman_engine.cpp:102-105`.
- Ledger update at Phase C close: confirm no change.

### Phase D (FP8 multi-route GOT-patch)

- Anticipated: no CIPHER-imposed customer-side step. Customer setting `--quantization fp8` on vLLM is the customer's model choice independent of CIPHER (the customer would set this for FP8 models whether CIPHER is present or not). This row pre-records the framing so it does not get re-litigated at Phase D close.
- Edge case to surface if it arises: if Phase D substrate work requires a new env gate or CIPHER-specific flag to enable FP8 GOT-patching beyond the existing `CUDA_INJECTION64_PATH`/`LD_PRELOAD` set, that IS a customer-side step addition and the ledger row gets marked "Contract violation? YES" pending Anil adjudication.

### Phase E (CP 5.5 scope-lock + measurement)

- Anticipated: no substrate code change in Phase E (Phase E is the headline benchmark). No deployment-surface change expected.
- Ledger fold into CP 5.5 scope-lock document at Phase E entry.

---

## 5. Related memory

- [[v1-goal5-contract-lock]]: the contract terms and the audit trail of the 2026-05-26 lock
- [[option2-complete]]: REVERT block at line 22 + line 36 records the 2026-05-25 reframe that is rolled back
- [[option2-step0-closed]]: honest residue line 29 REVERTED with reference back here
- [[vllm-v1-worker-subprocess]]: "related" reframe line REVERTED with reference back here
- [[plan-v1.2.3]]: §1 goal-5 text remains at original framing (verified A.0 no-edit-needed)
- [[cipher-fusion-campaign]]: scanned at A.0, no reframe propagation found
- [[pre-cp-5-5-plugin-routed-probe]]: scanned at A.0, no reframe propagation found
- V1_PHASE_A_SCOPE_LOCK.md (cipher-fusion-evidence): Phase A substrate path that restores the original contract

---

**This ledger is the canonical Goal 5 contract artifact for v1 + v1.5 substrate work. Every phase commits its row update here.**
