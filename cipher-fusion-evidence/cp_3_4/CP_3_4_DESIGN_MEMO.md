# CP 3.4 — Design Memo: first-light Grafana + ClickHouse dashboard

**Date:** 2026-05-15. **Status:** DESIGN — awaiting approval. No code, no
containers, no schema is created until this memo is approved.

CP 3.4 canonical scope (verbatim from phase audit `SUMMARY.md`): *"Grafana +
ClickHouse dashboard — Stand up ClickHouse + Grafana — or formally re-scope to
'the Prometheus /metrics exporter is the telemetry surface' and retire the
canonical gate."*

User adjudication for this CP: **"Grafana + ClickHouse dashboard, per-second
per-tenant silicon state visible. Atomic STEP."** → the re-scope option is
declined. We stand up the real ClickHouse + Grafana stack.

---

## 1. Audit before build — what exists, what does not

- **NOT DONE per phase audit.** Filesystem-wide search found zero CIPHER
  Grafana JSON, zero ClickHouse DDL, zero datasource config. Confirmed.
- **What CP 3.4 consumes already exists.** CP 3.3 shipped `/proc/cipher/flops`
  (kmod 0.4.7). Verified live this session with `cipher_flopd` running:

  ```
  device  status=live tensor_tflops=… mfu_pct=… sm_clock_mhz=… sm_util_pct=… power_w=…
  PID  TGID  TENANT  LAUNCH_DELTA  FLOPS_PER_S  MFU%
  ```

  Every column CP 3.4 needs is in that one file — device silicon state **and**
  per-tenant attributed FLOP/s + MFU. **No new kmod surface, no daemon change.**
- **Phase 3 `cipher-exporter.py` (`:9402`, Prometheus `/metrics`)** exists in
  the Phase 3 tarball. It is a *metrics surface*, not a dashboard, and the
  audit correctly ruled it not equivalent to the CP 3.4 gate. See §3 for why
  it is **not** on the CP 3.4 ingest path.
- **Environment verified this session:** podman rootless works (pulled
  `hello-world` with no sudo); 2.5 TB free on `/`; internet reaches
  `hub.docker.com` (HTTP 200) — images are pullable; `/proc/cipher/flops`
  is fresh at 1 Hz when `cipher_flopd` runs.

---

## 2. The atomic STEP

One STEP: **stand up ClickHouse + Grafana as rootless podman containers, wire a
1 Hz ingest agent from `/proc/cipher/flops` into ClickHouse, provision a
Grafana ClickHouse-datasource dashboard with per-tenant panels, run a live
multi-tenant workload, and capture first-light evidence.** Built, gated, one
report. No partial landings.

---

## 3. Architecture as proposed

```
 cipher_flopd (root, CP 3.3)  ──>  /proc/cipher/flops   (1 Hz fresh)
        |
        | cp_3_4/cipher_ingest.py  — reads proc file every 1 s, INSERTs rows
        v
 ClickHouse  (podman rootless, container `cipher-ch`, port 8123/9000)
   ├─ table device_state   (per-second device silicon state)
   └─ table tenant_state   (per-second per-tenant attributed FLOP/s + MFU)
        ^
        | ClickHouse datasource (grafana-clickhouse-datasource plugin)
 Grafana  (podman rootless, container `cipher-grafana`, port 3000)
   └─ dashboard "CIPHER — per-tenant silicon state"  (provisioned from JSON)
```

**Ingest path decision — direct proc-file read, option (A).** A ~80–120 LOC
Python agent (`cipher_ingest.py`, stdlib + `clickhouse` over HTTP `:8123`)
reads `/proc/cipher/flops` once per second and INSERTs one `device_state` row
plus one `tenant_state` row per active tenant.

- **Rejected — option (B):** bridge the Phase 3 `cipher-exporter.py` (`:9402`
  Prometheus) into ClickHouse. Rejected because it puts Prometheus on the
  critical path for no benefit, requires extracting/running a tarball'd
  daemon, and the exporter does not yet emit the CP 3.3 FLOP/MFU series.
  Direct proc-file read is fewer moving parts and per-second cadence is
  trivial. The Phase 3 exporter remains available — it is simply not the
  CP 3.4 ingest path.

**Containers — podman rootless, no sudo.** ClickHouse and Grafana run as the
`ubuntu` user via rootless podman. No system service is installed, no
`/etc` is touched, no sudo is used for the stack. `cipher_flopd` itself needs
root (CUPTI) — that is the unchanged CP 3.3 daemon, not new privilege.

---

## 4. ClickHouse schema (committed — no TBD)

```sql
CREATE DATABASE IF NOT EXISTS cipher;

CREATE TABLE cipher.device_state (
    ts             DateTime64(3),
    tensor_tflops  Float64,
    mfu_pct        Float64,
    sm_clock_mhz   UInt32,
    sm_util_pct    UInt8,
    power_w        Float64
) ENGINE = MergeTree ORDER BY ts;

CREATE TABLE cipher.tenant_state (
    ts             DateTime64(3),
    tenant         String,        -- CIPHER_TENANT_ID; "pid:<n>" fallback if unset
    pid            UInt32,
    tgid           UInt32,
    launch_delta   UInt64,
    flops_per_s    Float64,
    mfu_pct        Float64
) ENGINE = MergeTree ORDER BY (ts, tenant);
```

- **Tenant key is the `TENANT` name** (from `CIPHER_TENANT_ID`), not PID — PID
  is unstable across runs. When a process has no tenant name (`-` in the proc
  file) the agent records `pid:<n>` so the row is never dropped.
- No TTL for first-light (rows are kept; gate window is short). Stated, not
  deferred.

---

## 5. Grafana dashboard (committed panels)

Dashboard **"CIPHER — per-tenant silicon state"**, ClickHouse datasource,
auto-refresh 1 s, default window last 15 min:

1. **Per-tenant MFU %** — time series, one line per `tenant` (`tenant_state.mfu_pct`).
2. **Per-tenant FLOP/s** — time series, one line per `tenant`.
3. **Device tensor TFLOP/s** — time series (`device_state.tensor_tflops`).
4. **Device silicon state** — sm_clock_mhz / sm_util_pct / power_w, time series.
5. **Live tenant table** — latest row per tenant: tenant, MFU %, FLOP/s, launches.

Dashboard JSON, datasource YAML, and dashboard-provisioning YAML are written
to `cp_3_4/provisioning/` and mounted into the Grafana container — the
dashboard is reproducible from disk, not hand-clicked.

---

## 6. Daemon / run lifecycle (gate procedure)

1. Start `cipher_flopd` (root, CP 3.3 daemon — unchanged binary).
2. `podman run` ClickHouse (`cipher-ch`); apply schema (§4).
3. Start `cipher_ingest.py` (1 Hz proc → ClickHouse).
4. `podman run` Grafana (`cipher-grafana`) with provisioning mounted.
5. Run the gate workload: **reuse `cp_3_3/flop_gate` Phase-B** (3 tenants,
   asymmetric duty) — extended to a **≥90 s** window so the dashboard shows a
   sustained per-second multi-tenant series. Reusing the CP 3.3 harness means
   no new load-generator code and a known-good asymmetric-MFU signal.
6. Capture evidence (§7).
7. Tear down: `podman rm -f cipher-ch cipher-grafana`; stop flopd + ingest.

---

## 7. Gate criteria (all four must PASS)

| # | Criterion | Evidence artifact |
|---|---|---|
| (a) | ClickHouse up, schema applied, **≥90 s × ≥3 tenants** of per-second rows present | `SELECT` dump `cp_3_4/ch_rows.txt` + row counts |
| (b) | Grafana up, ClickHouse datasource configured, dashboard with per-tenant panels provisioned | `cp_3_4/provisioning/dashboard.json` on disk + Grafana `/api/dashboards` lists it |
| (c) | **"Visible"** — `/api/ds/query` against the dashboard's ClickHouse datasource *through Grafana* returns non-empty per-tenant series | `cp_3_4/grafana_dsquery.json` (the query response) |
| (d) | Anchors held — kmod, libcipher, ABI, taint all unchanged | md5s recorded pre/post |

**On "visible" (criterion c).** A rendered PNG requires a third
`grafana-image-renderer` container and headless screenshots are flaky. The
*durable, scriptable* proof that the dashboard renders is Grafana's
`/api/ds/query` returning the per-tenant series **through the dashboard's
datasource** — that exercises the exact Grafana→ClickHouse path the panels
use. A PNG via the renderer is an **optional bonus**, not a gate criterion;
if it comes up cheap it is included, otherwise (c) stands on the `ds/query`
response.

---

## 8. Discipline — anchors held (lead claim)

**CP 3.4 is pure observability. No kmod change. No libcipher change. No ABI
change.** It reads an existing proc file and visualizes it.

- Anchors `55ab8c0c` (kmod) / `86618c30` (libcipher_v2) — **frozen, untouched.**
- kmod **0.4.7** `2a69f9defd7730665e6b7f9d60e82b43` — unchanged (recorded
  pre- and post-CP-3.4; md5 equality is gate criterion (d)).
- ABI ioctl nrs untouched; reserved 2/3/4 untouched.
- Taint expected stable at 12288.
- No sudo for the dashboard stack (rootless podman). `cipher_flopd` is the
  unchanged CP 3.3 root daemon.
- **Rollback is total and one-line:** `podman rm -f cipher-ch cipher-grafana`
  then `rm -rf cp_3_4/{ch_data,grafana_data}`. No system state to revert.

---

## 9. New artifacts this CP will produce

| Artifact | Path | Note |
|---|---|---|
| ingest agent | `cp_3_4/cipher_ingest.py` | ~100 LOC, stdlib + HTTP |
| ClickHouse schema | `cp_3_4/schema.sql` | §4 verbatim |
| datasource provisioning | `cp_3_4/provisioning/datasource.yml` | ClickHouse datasource |
| dashboard provisioning | `cp_3_4/provisioning/dashboards.yml` + `dashboard.json` | §5 panels |
| run script | `cp_3_4/run_cp34.sh` | §6 lifecycle, idempotent |
| evidence | `cp_3_4/ch_rows.txt`, `grafana_dsquery.json`, `gate_result.json`, `run.log` | §7 |
| optional PNG | `cp_3_4/dashboard_firstlight.png` | bonus, not gated |

Container images pulled (recorded with digests in the report):
`docker.io/clickhouse/clickhouse-server`, `docker.io/grafana/grafana`.

---

## 10. Open decisions for the user to adjudicate

1. **ClickHouse + Grafana stack confirmed** (re-scope to "exporter is the
   surface" declined) — assumed yes per your CP framing.
2. **Ingest path (A) direct proc-file read** vs (B) Prometheus bridge — memo
   recommends (A).
3. **Criterion (c) "visible" = Grafana `/api/ds/query` response**, PNG
   optional — confirm this is an acceptable definition of "visible" for the
   gate, or require the renderer PNG as mandatory.
4. **Gate workload = reuse `cp_3_3/flop_gate` Phase-B extended to ≥90 s** —
   confirm reuse vs a fresh load generator.

No C, no Python, no container is created until this memo is approved.
