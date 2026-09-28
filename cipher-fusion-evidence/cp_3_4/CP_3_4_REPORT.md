# CP 3.4 — first-light Grafana + ClickHouse dashboard — REPORT

**Date:** 2026-05-15. **Status:** built, gated — **all four gate criteria PASS**
(`gate_result.json`: `all_pass: true`).

CP 3.4 canonical scope: *"Grafana + ClickHouse dashboard."* Built per the
approved design memo (`CP_3_4_DESIGN_MEMO.md`) — the re-scope option was
declined. ClickHouse and Grafana were stood up as rootless podman containers;
a 1 Hz agent ingests `/proc/cipher/flops` (CP 3.3) into ClickHouse; a
provisioned Grafana dashboard renders per-second per-tenant silicon state.

---

## 1. Architecture as built

```
 cipher_flopd (root, CP 3.3 daemon — unchanged)  ──>  /proc/cipher/flops  (1 Hz)
        |
        | cipher_ingest.py  — 1 Hz read, INSERT over ClickHouse HTTP :8123
        v
 ClickHouse  (podman, container `cipher-ch`, host net, v26.4.2)
   ├─ cipher.device_state   105 rows — per-second device silicon state
   └─ cipher.tenant_state   510 rows — per-second per-tenant FLOP/s + MFU
        ^
        | grafana-clickhouse-datasource v4.17.0 (native proto :9000)
 Grafana  (podman, container `cipher-grafana`, host net, v13.0.1)
   └─ dashboard "CIPHER — per-tenant silicon state"  (uid cipher-silicon,
      5 panels, provisioned from JSON on disk)
```

Rootless podman, **no sudo for the dashboard stack**. `--network host` was used
instead of a named podman network — see §4 caveat 1. `cipher_flopd` and
`loadgen` use sudo (they need `/dev/cipher`, 0600 root) — that is the unchanged
CP 3.3 daemon plus the CP 3.4 load generator, not new privilege.

---

## 2. Gate results — all PASS

Gate workload: `loadgen` — 3-tenant asymmetric cuBLAS fp16 tensor-GEMM,
**100 s** (≥90 s as the memo committed). Raw verdict: `gate_result.json`.

| # | Criterion | Result | Verdict |
|---|---|---|---|
| (a) | ClickHouse up, schema applied, **≥90 s × ≥3 tenants** of per-second rows | 105 device rows; 3 gate tenants; **min per-tenant window 99 s** | **PASS** |
| (b) | Grafana up, ClickHouse datasource configured, dashboard with per-tenant panels provisioned | datasource `cipher-ch` + dashboard `cipher-silicon` (5 panels) — Grafana log: *"finished to provision dashboards"*, *"provisioned dashboard … cipher-silicon"* | **PASS** |
| (c) | **"Visible"** — `/api/ds/query` through the dashboard's datasource returns non-empty per-tenant series | 100 time rows; **3 distinct tenant labels** `cipher34-T0/T1/T2`; plugin `status=ok` | **PASS** |
| (d) | Anchors held — kmod, libcipher, ABI, taint unchanged | kmod + libcipher md5 identical pre/post; taint 12288→12288 | **PASS** |

The line that matters: **per-second per-tenant silicon state is visible end to
end** — `/proc/cipher/flops` → ClickHouse → Grafana datasource → dashboard
panels. The CP 3.3 differentiator (real-time per-tenant MFU as a free byproduct
of launch counts) is now a live dashboard, not just a proc file.

**Per-tenant asymmetry captured** (`ch_rows.txt`): under deliberately
asymmetric duty, the dashboard separates T2 (flat-out, **mean MFU 71.5 %**,
706.8 TFLOP/s) from T1 (**11.0 %**, 108.8 TFLOP/s) and T0 (**4.55 %**,
45.0 TFLOP/s). Per-tenant attributed FLOP/s sum (860.6) tracks the device
tensor rate (~887) within the windowing tolerance — sum-consistent.

---

## 3. Build STEP findings

1. **`/proc/cipher/flops` is sufficient — no daemon or kmod change.** The CP 3.3
   proc file carries every column CP 3.4 needs (device tensor TFLOP/s, MFU,
   sm_clock, sm_util, power; per-tenant pid/tgid/tenant/launch_delta/FLOP-s/MFU).
   The ingest agent is 110 LOC of stdlib Python.
2. **ClickHouse plugin works headless.** `grafana-clickhouse-datasource` v4.17.0
   installed at container start via `GF_INSTALL_PLUGINS`, native protocol on
   `:9000`, queries `status=ok` — the full Grafana→ClickHouse path is exercised
   by criterion (c), no browser needed.
3. **`loadgen` is a clean CP-3.4-owned variant**, derived from `cp_3_3/flop_gate`'s
   Phase-B worker. **`cp_3_3/flop_gate.cu` was left byte-unchanged** — no CP 3.3
   artifact was edited.

---

## 4. Honest caveats

1. **Host networking, not a named podman network.** The memo's diagram showed
   containers on a `cipher-net` podman network. Rootless podman's CNI on this
   pod rejects custom-network config (`firewall plugin does not support config
   version "1.0.0"`). `--network host` is the clean, functionally-equivalent
   path: ClickHouse `:8123/:9000` and Grafana `:3000` are distinct host ports,
   Grafana reaches ClickHouse at `127.0.0.1`. Documented, evidenced change —
   not a silent reframe.
2. **Schema applied via `clickhouse-client --multiquery`, not HTTP.** ClickHouse's
   HTTP interface runs one statement per request; the 3-statement schema is fed
   through `clickhouse-client` inside the container. (`clickhouse-client` emits
   a benign DNS warning resolving the pod hostname `192-222-53-2`; the CREATE
   statements still execute — `SHOW TABLES` confirms both tables.)
3. **Gate-(c) verdict computation was corrected before this report.** The first
   `run_cp34.sh` pass counted distinct *data values* in a result column instead
   of distinct *tenant labels* in the frame schema (timeseries-wide format
   carries tenant as a field label, not a column). The raw verdict was
   coincidentally still ≥3 → PASS. It was re-derived honestly from the field
   labels (`grafana_dsquery.json` → `cipher34-T0/T1/T2`, 3), `gate_result.json`
   regenerated, and `run_cp34.sh` fixed. The verdict is a **genuine** PASS, not
   a coincidental one.
4. **Dashboard auto-refresh clamped 1 s → 5 s.** Grafana's `minRefreshInterval`
   default is 5 s; it clamped the dashboard's 1 s setting (log: *"Changing
   refresh interval for provisioned dashboard to minimum"*). The **data
   granularity is per-second** (ClickHouse rows, panel queries) — only the
   browser auto-poll is 5 s. Settable to 1 s via `GF_DASHBOARDS_MIN_REFRESH_INTERVAL`;
   left at default for first-light.
5. **`meta.provisioned: False`** is returned by `/api/dashboards/uid` on this
   Grafana 13.x build. The **provisioning log is authoritative**: it logs the
   datasource and dashboard being inserted "from configuration" and names
   `cipher-silicon` a *"provisioned dashboard"*. The dashboard also lives in the
   `CIPHER` folder the file-provider created — it is provisioned from disk.
6. **No persistent volumes.** First-light data lived in the container writable
   layer and was captured via `SELECT` during the live gate window before
   teardown. Fully reproducible by re-running `run_cp34.sh`. Rollback is total.
7. **No PNG.** The memo declared a renderer PNG optional; criterion (c) stands
   on the `/api/ds/query` response. The `grafana-image-renderer` container was
   not installed.
8. **Per-tenant rows include `pid:<n>` non-workload processes.** Per the memo,
   processes touching `/dev/cipher` without a `CIPHER_TENANT_ID` (here:
   `cipher_flopd`'s own threads) are recorded as `pid:<n>` with 0 FLOP/s rather
   than dropped. They sit at zero in the dashboard; the 3 `cipher34-*` tenants
   carry the signal.

---

## 5. Artifacts

| Artifact | Path | md5 |
|---|---|---|
| load generator (source) | `cp_3_4/loadgen.cu` | `5be70837afca016dc223ad87d5427353` |
| load generator (binary) | `cp_3_4/loadgen` | `d168f8aab8b1f054b36524baee7778ac` |
| ingest agent | `cp_3_4/cipher_ingest.py` | `4121f49cd00502d24646d26a6598d6d3` |
| ClickHouse schema | `cp_3_4/schema.sql` | `cdee7e52fd435b34d4d2ad2c81b318ca` |
| datasource provisioning | `cp_3_4/provisioning/datasources/datasource.yml` | `e0b68bcc3dd6554ab71d984b315aa061` |
| dashboard provider | `cp_3_4/provisioning/dashboards/dashboards.yml` | `38525931c3634e1f7f4fafaeea6e7212` |
| dashboard JSON | `cp_3_4/provisioning/dashboards/cipher_dashboard.json` | `0ca6e639c381e8ea14ae82eabbe9c65d` |
| run script | `cp_3_4/run_cp34.sh` | `9d1a4e20d039c9a3eebdefb5452812a7` |
| teardown script | `cp_3_4/teardown_cp34.sh` | `b30508a9a632df0bd07ced7e3494742d` |
| gate result | `cp_3_4/gate_result.json` | `e500fb6a193b749f7db6abc32dca1b48` |
| ClickHouse row evidence | `cp_3_4/ch_rows.txt` | `13b1deb20e0537bfb15c0572da160e3a` |
| Grafana ds/query (criterion c) | `cp_3_4/grafana_dsquery.json` | `4524b5cc0dd86e21eb9497b0ebe0c957` |
| Grafana datasource list | `cp_3_4/grafana_datasources.json` | `34c038a2cf99660b20a9b0fe6b45ea1e` |
| Grafana dashboard search | `cp_3_4/grafana_search.json` | `5d0f266eca02faff4de3db911b5fe141` |
| Grafana stored dashboard | `cp_3_4/grafana_dashboard_loaded.json` | `f673abe326fce2eac70327ca0c059a68` |
| Grafana provisioning log | `cp_3_4/grafana_provisioning.log` | `c190b73acf8e155ce67bd50a1a5c7053` |
| container image digests | `cp_3_4/container_images.txt` | `bc386d7e4e24eaa9552432cba2c45a54` |
| full run log | `cp_3_4/run.log` | `4ceb2fbc023171ffa054425c93efd3d7` |

Container images (digests in `container_images.txt`):
`clickhouse/clickhouse-server@sha256:6d8f3587…` (v26.4.2),
`grafana/grafana@sha256:2d1f9ae6…` (v13.0.1).

---

## 6. Discipline — anchors held

CP 3.4 is **pure observability — no kmod change, no libcipher change, no ABI
change.** It reads an existing proc file and visualizes it.

- kmod **0.4.7** `2a69f9defd7730665e6b7f9d60e82b43` — **unchanged** pre/post
  (gate criterion d). kmod stays loaded throughout.
- libcipher_v2 anchor `86618c30896470b642fcc6985d8dc632` — **unchanged**.
- kmod anchor `55ab8c0c` — untouched. ABI ioctl nrs untouched; reserved
  2/3/4 untouched.
- Taint **12288 → 12288**.
- No sudo for the dashboard stack (rootless podman). No system service
  installed, no `/etc` touched.
- **Rollback executed and verified:** `teardown_cp34.sh` removed both
  containers and the (unused) network; post-teardown anchors re-checked
  identical. Nothing on the system to revert beyond the containers.

---

## 7. Bottom line

CP 3.4 delivers a first-light Grafana + ClickHouse dashboard: ClickHouse stores
per-second per-tenant silicon state ingested from the CP 3.3 `/proc/cipher/flops`
surface, and a provisioned Grafana dashboard renders it — per-tenant MFU,
per-tenant attributed TFLOP/s, device tensor rate, device clock/util/power, and
a live tenant table. All four gate criteria pass: ≥90 s × 3-tenant per-second
data in ClickHouse, datasource + dashboard provisioned, the per-tenant series
returns through Grafana's `/api/ds/query`, and every anchor held. The honest
deviations — host networking, `clickhouse-client` schema apply, the corrected
gate-(c) computation, the 5 s refresh clamp — are documented, not hidden. No
kmod, libcipher, or ABI change. The re-scope escape hatch was declined; the
real stack stands.
