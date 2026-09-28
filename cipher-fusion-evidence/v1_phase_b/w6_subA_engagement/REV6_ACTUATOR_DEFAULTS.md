# W.6 sub-A Sub-step 1: rev6 actuator engagement defaults audit

**Audit date:** 2026-05-27
**rev6 baseline:** `cipher-platform v2.0` md5 `7c7068ca`; libcipher_rt.so md5 `1f305ce6` (installed)
**rev6 CDI ENV** (`/usr/lib/cipher/cipher_cdi_patch.py:249-255`):
```
CIPHER_CDI_MARKER=v1
CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so
LD_LIBRARY_PATH=/usr/lib/cipher:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
TORCH_CUBLASLT_DISABLE=1
CIPHER_REGISTER_MODEL=0
```

---

## Actuator engagement matrix

| # | Actuator | Gate env var | Current default | File:line | Desired for W.6 sub-A | Correctness risk on auto-engage? |
|---|---|---|---|---|---|---|
| 1 | **VOLT** (DVFS clock lock) | `CIPHER_VOLT=on` + `CIPHER_VOLT_BATCH={1\|8\|32\|64}` OR `CIPHER_VOLT_MHZ=N` | OFF (early return at `cipher_rt_volt.c:272-292`) | `cipher_rt_volt.c:255-292` | **DO NOT auto-engage** | **YES — REGRESSION on 7B+ bf16** |
| 2 | **Marlin** (fp16→INT4 substitution) | `CIPHER_MARLIN=on` | OFF (`cipher_rt_marlin_actuator.c:203-216`) | `CIPHER_MARLIN=on` | safe — dtype gate excludes bf16 + Machete bypasses cuBLAS | NO |
| 3 | **Koopman** (low-rank surrogate) | `CIPHER_KOOPMAN=1` | OFF (`cipher_rt_koopman_engine.cpp:213-223`) | `CIPHER_KOOPMAN=1` | safe — fp16 + empty-registry gates protect bf16 | NO |
| 4 | **KV-dedup** (cross-process prefix) | `CIPHER_KVDEDUP=1` | OFF (`cipher_vllm_kvdedup.py:58`) | `CIPHER_KVDEDUP=1` | safe — xxhash64+memcmp correctness-gated | NO |
| 5 | **REMEMBER** (CfC LNN consumer) | `CIPHER_REMEMBER=1` | OFF (`cipher_rt_remember_consumer.cpp:189`) | `CIPHER_REMEMBER=1` | safe — observability + decision-output to RING; no kernel substitution | NO |
| 6 | **AUDIT chain** (HMAC-SHA256 telemetry) | `CIPHER_AUDIT=1` | OFF (`cipher_rt_audit.c:182`) | `CIPHER_AUDIT=1` | safe — telemetry only | NO |
| 7 | **RECEIPT** (HMAC-signed billing receipts) | `CIPHER_RECEIPT=on` (+ `CIPHER_SENSE` prerequisite) | OFF (`src/may13/cipher_receipt.cpp:107`) | DEFER to W.6 sub-A.2 — requires SENSE prerequisite chain | NO but needs prerequisite |
| 8 | **FAIRNESS** (per-tenant quota) | `CIPHER_FAIRNESS=on` | OFF (`src/may13/cipher_fairness.cpp:70`) | DEFER — observability-only at v1, no consumer | NO |
| 9 | **CARBON** (energy/CO2 telemetry) | `CIPHER_CARBON=on` | OFF (`src/may13/cipher_carbon.cpp:66`) | DEFER — observability-only at v1 | NO |
| 10 | **SENSE** (workload session detector) | `CIPHER_SENSE=on` | OFF (`src/may13/cipher_sense.cpp:159`) | `CIPHER_SENSE=on` (prerequisite for RECEIPT) | safe — observability |
| 11 | **DISPATCH_LIVE** (classifier-driven routing) | `CIPHER_DISPATCH_LIVE=1` | **ON** by default (`cipher_rt_dispatch.cpp:55`) | unchanged | n/a — already on |
| 12 | **SPEC** (speculative decode) | `CIPHER_SPEC=1` | **ON** by default (`cipher_spec_decode.py:18`) | unchanged | n/a — already on |
| 13 | **ATTN_TEST** (diagnostic actuator) | `CIPHER_ATTN_TEST` | OFF | KEEP OFF — diagnostic-only | n/a |
| 14 | **REGISTER_MODEL** (NR 27 userspace path) | `CIPHER_REGISTER_MODEL=0` (gated off due to B.6''.9.1 crash) | **0 explicitly** | KEEP 0 — W.6 sub-B will close the crash | NO |

**Always-on actuators (no env gate; engage automatically):**
- SM_PACKER (cipher_rt_sm_packer.c) — observation-only
- Partition router (cipher_rt_partition_router.c) — per-stream priority
- Per-tenant routing (NR 29 REGISTER_STREAMS) — engaged when plugin registers
- COMMIT atomic primitive — engaged per-launch
- RING_WRITE — engaged per matmul/attn observe
- Stream resolver — engaged per per-tenant routing
- Attn dispatch substrate — passes-through; no substitution actuator registered until W.5
- Auto-repatch + GOT walker — fires on dlopen
- cuBLAS GOT-patch — fires on cuBLAS GEMM (any dtype, any model)

---

## HARD STOP per discipline (g) Memory #11 — VOLT correctness risk

**VOLT CANNOT auto-engage safely without Workload Classifier sequencing.**

Per Memory `cipher-t43-envelope` (locked 2026-05-19 from the T4.3 7-condition envelope study):
> "Mistral-7B B=1 is -14% (no clock recovers positive). Honest claim: '+55% on memory-bandwidth-bound decode only'; 7B+ regime needs recalibration or pivot."

VOLT's `CIPHER_VOLT_BATCH=1|8|32|64` map to clock targets calibrated against TinyLlama-1.1B B=1 (memory-bandwidth-bound). Mistral-7B B=1 and Llama-3-8B B=1 are NOT memory-bandwidth-bound at the same clock; the clock lock REGRESSES throughput by ~14%.

Auto-engaging `CIPHER_VOLT=on CIPHER_VOLT_BATCH=1` in the rev7 CDI patch would:
- IMPROVE TinyLlama-1.1B decode throughput
- REGRESS Mistral-7B-Instruct bf16 default decode by ~14%
- REGRESS Llama-3.1-8B-Instruct bf16 default decode similarly (per envelope study)

This is a direct Memory #25 violation: "the substep's contribution survives on at least one of: {Llama-3-8B bf16 default, Mistral-7B-Instruct bf16 default, TinyLlama-AWQ INT4}". Auto-engaging VOLT would BREAK Llama-3 + Mistral while improving TinyLlama-1.1B (which isn't in the ref set anyway).

**Surfaced for adjudication:**
- Option 1 (RECOMMENDED): keep VOLT default-OFF in rev7. K.1.5 Workload Classifier (when complete) sets VOLT conditionally per workload-class (only on workloads in the bandwidth-bound regime — bandwidth-bound detection signal lives at Classifier).
- Option 2: ship VOLT default-OFF but with a NEW CIPHER_VOLT_AUTO=1 env that asks libcipher_rt.so to read workload signals at runtime and self-enable. Adds substrate complexity at v1 timeline; not recommended.
- Option 3: ship VOLT default-ON despite regression risk. NOT RECOMMENDED — violates Memory #25.

**Decision needed before Sub-step 2** (CDI patch update).

---

## RECEIPT prerequisite chain note

RECEIPT requires SENSE init succeeded (`cipher_rt_receipt.cpp` checks SENSE state). If we enable RECEIPT without SENSE, RECEIPT init returns OFF silently. Cleanest order:
- Add `CIPHER_SENSE=on` first
- Then `CIPHER_RECEIPT=on` (will engage IF SENSE engaged)
- Both safe — observability only

For W.6 sub-A initial scope, RECEIPT/FAIRNESS/CARBON are observability surfaces that the rev6 customer doesn't see without the cipher-platform CLI's audit-verify or Grafana dashboard. **Recommendation: DEFER RECEIPT/FAIRNESS/CARBON env injection** to a later substep that ships the operator-side surface; closing W.6 sub-A on the substitution actuators (Marlin + Koopman + KV-dedup + REMEMBER) is the high-impact opt-in subset.

---

## rev7 CDI patch proposed env additions (pending VOLT adjudication)

```python
CIPHER_ENV = [
    "CIPHER_CDI_MARKER=v1",
    "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so",
    "LD_LIBRARY_PATH=/usr/lib/cipher:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
    "TORCH_CUBLASLT_DISABLE=1",
    "CIPHER_REGISTER_MODEL=0",       # KEEP 0 — W.6 sub-B fix
    # W.6 sub-A additions (substitution actuator auto-engage):
    "CIPHER_MARLIN=on",              # fp16+cuBLAS gates protect bf16/Machete
    "CIPHER_KOOPMAN=1",              # fp16 gate + empty-registry protect bf16
    "CIPHER_KVDEDUP=1",              # xxhash64+memcmp correctness-gated
    "CIPHER_REMEMBER=1",             # observability + CfC LNN consumer
    "CIPHER_AUDIT=1",                # HMAC telemetry only
    "CIPHER_SENSE=on",               # workload session detector (RECEIPT prereq)
    # PENDING ADJUDICATION:
    # "CIPHER_VOLT=on" + "CIPHER_VOLT_BATCH=1": REGRESSION risk on 7B+ bf16 per Memory cipher-t43-envelope
]
```

Customer override mechanism preserved per discipline (negative case):
- Any of the above can be overridden with `=0` or `=off` in the customer's docker run / systemd environment block (CDI ENV is INJECTED at container spawn, but the customer's environment vars OVERRIDE since CDI hook applies env to the container, then customer's `-e CIPHER_KOOPMAN=0` re-overrides).
- Verify override works: customer can disable any actuator via `docker run -e CIPHER_KOOPMAN=0 ... vllm/vllm-openai:v0.21.0`

**HOLD on Sub-step 2 pending VOLT adjudication.** No code changes yet.
