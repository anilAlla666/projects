# V.1 — CP 5.5 soak — ITEM 1 READINESS PRE-FLIGHT — FINDINGS

**Date:** 2026-05-29. **Status:** Item 1 (readiness pre-flight) COMPLETE. **The V.1 soak
(items 2–6) is NOT EXECUTABLE on this pod as-is — PARTIAL on infra/packaging/harness
readiness, NOT a goal failure (per Anil: a readiness gap is a finding, not forced past).
STOP at §4 for Anil.** **Anchors UNCHANGED** (measurement/inspection only; cipher_rt_phase4
`8b5e928`/`9fe23143`, kmod `02fc2d1`, bridge `5a3db034`).

## §1 — Substrate state (loaded vs deployed vs validated)

| Component | On this pod | Validated dev anchor | Gap |
|---|---|---|---|
| kmod | **loaded 0.6.5** (srcversion C36ED68…) | 02fc2d1 = **0.6.6** | loaded kmod lags dev by one ABI minor |
| substrate (.deb) | `/usr/lib/cipher/libcipher_rt.so` = **`1f305ce6`** | `9fe23143` = `build_cuda13/` (container-CUDA-13 build) | **.deb ships an OLDER build** |
| substrate source | — | moved during **W.2 (Machete)** `2a48387` + **W.5 (FlashAttn)** `879c5b7` | **the `1f305ce6` .deb predates the W.2/W.5 actuators** → a stock soak against it would under-report Goals |
| /dev/cipher, /proc/cipher/* | present (0666; classify_stats, flops, sense_session, …) | — | substrate kmod side OK |

**Provenance (resolved, not staleness):** the dev anchor `9fe23143` is the
**container-CUDA-13** build (`build_cuda13/libcipher_rt.so`); W6_SUBC validated it by
bind-mounting it over the CDI path (9/9 PASS). The `.deb` baseline `1f305ce6` is the
**host build / older actuator set**. W6's own recorded follow-up ("fold the `-v
build_cuda13` into the deploy") = **repackage the .deb to `9fe23143` + reload kmod 0.6.6**
is a prerequisite before a representative stock soak.

## §2 — Auto-activation is container/CDI by design (the zero-intervention path)

Per W6_SUBC: CIPHER auto-engages a stock workload **only inside a container** via the CDI
hook injecting `CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so` into the NVIDIA CDI
device spec; the substrate **must be the CUDA-13 container build** (a host build "silently
fails to load via injection in the container"). There is **no bare-metal `vllm serve`
auto-activation path** (CDI is container-only; `/etc/ld.so.preload` empty,
`CUDA_INJECTION64_PATH` unset on the host). A manual `CUDA_INJECTION64_PATH=` injection
would itself **be an intervention** → cannot validate the Memory #19 zero-intervention
claim, so it is NOT used here.

## §3 — Per-goal readiness (capability vs measurability-on-this-pod)

| Goal | Interim gate | Capability status | Measurable on this pod now? |
|---|---|---|---|
| 1 density | ≥10 agents/H100, KL=0 | provider closed (W.4 branch-A); fires in-container (W6) | **NO** — needs the container+CDI soak + weighted-mix harness |
| 2 tok/W | ≥1.5× weighted mix | VOLT counters intact in-container (W6 9/9) | **NO** — needs repackaged substrate + harness |
| 3 MFU | ≥70% weighted-mean | CUPTI MFU (CP 3.3) present | **NO** — needs harness |
| 4 Koopman | handled>0, KL-preserved | koopman counters intact in-container (W6) | **NO** — needs harness + long-context class |
| 5 auto-profile | profile+caps <30s of serve | `cipher-platform status` reports substrate/kmod/plugin/watcher (works); **CDI "applied=unknown" without sudo**; no live profile without a running engaged container | **PARTIAL** — reporter present, but the live <30s-of-`vllm serve` profile needs the engaged container |

**Capability is largely demonstrated** (W6_SUBC: build_cuda13 substrate auto-fires all
actuators in-container via CDI, 9/9, counters intact). **What V.1 adds — the weighted-mix
5-goal engagement gate — is unmeasurable here** because of the four blockers below.

## §4 — Blockers (why the soak can't run on this pod) + VERDICT

1. **Packaging:** the deployed `.deb` (`1f305ce6`) lags the validated substrate
   (`9fe23143`, build_cuda13) and predates W.2/W.5 actuators; loaded kmod 0.6.5 < dev 0.6.6.
   → repackage `.deb` to `9fe23143` + reload kmod 0.6.6 (or bind-mount per W6) before a
   representative soak. **Repackage = a fresh packaging step, NOT folded into V.1 (Mem #16).**
2. **Container runtime:** the zero-intervention path is container/CDI; **docker daemon is
   not usable** on this pod right now (W6_SUBC ran containers here, so it is a restart —
   needs **sudo**), and GPU-CDI for rootless podman is **not configured** (`nvidia-ctk cdi
   list` empty → needs sudo).
3. **CDI patch (sudo):** `/var/run/cdi/nvidia.yaml` is root-only; `cipher-platform verify`
   returns "CDI status unreadable … re-run with sudo." Applying/verifying the cipher CDI
   injection needs sudo.
4. **Harness (net-new):** the V.1 **100-workload weighted-mix soak** harness does not exist
   — W6_SUBC was a **9-cell single-tenant** gate, not the weighted mix. This is real new
   build work regardless of infra.

**§4 VERDICT — PARTIAL (readiness), surfaced not forced.** V.1's 5-goal stock-config soak
**cannot execute on this pod as-is**: blocked on packaging (repackage to the validated
substrate), container/CDI infra (docker restart + GPU-CDI + cipher CDI patch — all
sudo-gated), and a net-new weighted-mix harness. **This is an infra/packaging/harness
readiness gap, NOT a goal failure** — the underlying capabilities auto-fire in-container
(W6_SUBC 9/9). No goal was faked, hand-forced, or bare-metal-injected (which would void the
zero-intervention claim). No anchor moved.

## §5 — Options for Anil

1. **Authorize the infra/packaging restore + build the harness here.** Sudo to: restart
   the container runtime, generate GPU-CDI, apply/verify the cipher CDI patch; repackage
   the `.deb` to `9fe23143` + reload kmod 0.6.6; pull `vllm/vllm-openai:v0.21.0`. Then I
   build the V.1 weighted-mix soak harness and run the 5-goal gate here.
2. **Defer V.1 to a provisioned deployment pod** (the W.7 hardware-deferral pattern,
   [[cipher-hardware-unreachable-gate]]) where the container/CDI stack + the shipped `.deb`
   are the customer-representative config — the most faithful place to test the
   zero-intervention claim.
3. **Scope down** (e.g., a Goal-5-only `cipher-platform status` check + a single-class
   in-container engagement smoke once infra is up), explicitly NOT the full 5-goal soak.

No code, no anchor change, no tag move. Awaiting Anil's authorize/defer/scope decision.
