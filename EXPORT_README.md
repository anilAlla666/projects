# CIPHER — Complete Project Export

**Generated:** 2026-05-18 — Lambda H100 pod (192.222.53.2) full snapshot, pre-disconnect backup.
**Supersedes** the earlier partial export; this one was rebuilt after a full-filesystem
audit that found work in `/workspace`, `/tmp`, the NFS share, and the system install paths.

Excluded only: 148 GB HuggingFace model weights (`models/`) and the public upstream
repos (`ext/`: nouveau, open-gpu-kernel-modules) — freely re-obtainable, not work product.

## Layout

### Current campaign — `/home/ubuntu` work (top level of this archive)
| Path | Contents |
|---|---|
| `campaign-memory/` | **Start here.** `MEMORY.md` = campaign index; one `.md` per closed CP / finding. |
| `cipher_kmod/` + `cipher_kmod.ko.v*` + `cipher_kmod_src_v*.tar.gz` | Kernel module source + every built `.ko`. Current anchor 0.4.8 (`e2f50452`). |
| `cipher_rt_phase4/`, `cipher_rt_phase4_draft/`, `cipher_rt_phase4_src_*.tar.gz`, `libcipher_rt.so.*` | libcipher_rt runtime source + every built lib. Prod anchor `c2c5d313`. |
| `libcipher_v2/` | libcipher v2 work. |
| `cipher_vllm_plugin/` | vLLM KV-cache integration plugin (CP 5.1 / 5.2). |
| `kv-cache-tester/` | KV-cache test harness + `traces/` (629 MB generated traces). |
| `cipher_workloads/` | Workload definitions WL01–WL24. |
| `cipher-*-evidence*` | Per-phase evidence bundles, phase 1 → phase 4 → fusion. |
| `cipher_campaign_state_2026-05-17*.tar.gz` | Consolidated campaign-state snapshots. |
| `cipher_exporter/`, `cipher_gpustate/`, `cipher_measurement/` | Telemetry / measurement tooling. |
| `PHASE_*.md/.csv`, `AUDIT_REPORT_*`, `*_DIAGNOSTICS*`, `STATUS_*`, `RECOVERY_*` | Phase reports, audits, op/workload matrices, T4.x task reports. |
| `cipher_anchors_manifest.txt` | md5-prefix anchor IDs for key artifacts. |

### `workspace/` — the April-era CIPHER build (was in `/workspace`, NOT `/home/ubuntu`)
The earlier build tree. ~140 Python scripts, `src/` `include/` `tests/` `build/`,
`cipher_kmod/` `libcipher_v2/`, the `stress/` + `stress2/` harnesses, ~40 reports
(`INVESTOR_REPORT`, `SCORECARD`, `STAGE13/14`, `STEP1–5`, `monarch_*`, `CIPHER_FLOW_*`),
its own `CLAUDE.md` / `MEMORY.md`, and `.remember/` memory logs.

### `tmp-cipher-files/` + `tmp-cipher-dirs/` — live working files from `/tmp`
Test code (`*.cu`, `*.c`, `*.py`), ~250 result logs/JSON (WL01–24 baselines, density
sweeps, soak runs, VOLT/Marlin runs), the `cipher-deb-build/` package tree, and
`p3audit/`. These were ephemeral — would have been lost on pod termination.

### `installed/` — system-installed copies (from `cipher-platform.deb`)
`usr-src-cipher-kmod-0.4.8/` (deployed kmod source) and `usr-lib-cipher/` (deployed
runtime libs), as installed under `/usr/src` and `/usr/lib`.

### `Anil/` — NFS share snapshots
Campaign-state tarballs incl. the **freshest** ones dated 2026-05-18 (`...0334`, `...0446`).

### `misc/`
The stray Phase-3 plan doc and a copy of the pod's shell history.

## Verify after download
`shasum -a 256 cipher_full_export_20260518.tar.gz` — compare to the value in the
download instructions. Nested `*.tar.gz` with `*.md5`/`*.sha256` siblings self-verify.

## Campaign status at export
Phase 4 CLOSED. Phase 5 in progress (CP 5.1 + 5.2 closed; CP 5.3 partition-aware
Marlin next). Authoritative: `campaign-memory/MEMORY.md`.
