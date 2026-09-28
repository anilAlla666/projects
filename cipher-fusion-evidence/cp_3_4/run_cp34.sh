#!/usr/bin/env bash
# CP 3.4 — build + gate the Grafana + ClickHouse first-light dashboard.
# Atomic STEP. Brings the stack up, ingests >=90 s of per-second per-tenant
# silicon state, captures gate evidence, computes verdicts, tears down.
#
# No sudo for the dashboard stack (rootless podman). sudo is used only for
# cipher_flopd / loadgen, which need /dev/cipher (0600 root) — that is the
# unchanged CP 3.3 daemon + the CP 3.4 load generator, not new privilege.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
CP33="$HERE/../cp_3_3"
KMOD="/home/ubuntu/cipher_kmod/cipher_kmod.ko"
LIBC="/home/ubuntu/libcipher_v2/libcipher_v2.so.v0.2.0"   # anchor 86618c30
CH_IMG="docker.io/clickhouse/clickhouse-server:latest"
GF_IMG="docker.io/grafana/grafana:latest"
GF_USER="admin"; GF_PASS="cipher"
DURATION=100

log() { echo "[cp34 $(date -u +%H:%M:%S)] $*"; }

# ---- 0. pre-anchors -------------------------------------------------------
log "=== CP 3.4 build + gate ==="
KMOD_MD5_PRE=$(md5sum "$KMOD" 2>/dev/null | cut -d' ' -f1)
LIBC_MD5_PRE=$(md5sum "$LIBC" 2>/dev/null | cut -d' ' -f1)
TAINT_PRE=$(cat /proc/sys/kernel/tainted)
log "pre-anchors: kmod=$KMOD_MD5_PRE libcipher=$LIBC_MD5_PRE taint=$TAINT_PRE"

# ---- 1. build loadgen -----------------------------------------------------
log "building loadgen.cu ..."
nvcc loadgen.cu -I/home/ubuntu/cipher_kmod -lcublas \
     -Wno-deprecated-gpu-targets -o loadgen || { log "BUILD FAILED"; exit 1; }
log "loadgen built: $(md5sum loadgen | cut -d' ' -f1)"

# ---- 2. ClickHouse --------------------------------------------------------
# Host networking: rootless-podman CNI on this pod rejects custom-network
# config versions; --network host is the clean path. ClickHouse :8123/:9000
# and Grafana :3000 are distinct host ports, so they coexist; Grafana reaches
# ClickHouse via 127.0.0.1. No system service installed — containers only.
podman rm -f cipher-ch cipher-grafana >/dev/null 2>&1 || true

log "starting ClickHouse container (host net) ..."
podman run -d --name cipher-ch --network host \
  --ulimit nofile=262144:262144 \
  "$CH_IMG" >/dev/null || { log "CH run failed"; exit 1; }

log "waiting for ClickHouse :8123 ..."
for i in $(seq 1 60); do
  [ "$(curl -s http://127.0.0.1:8123/ping 2>/dev/null)" = "Ok." ] && break
  sleep 2
done
[ "$(curl -s http://127.0.0.1:8123/ping 2>/dev/null)" = "Ok." ] \
  || { log "ClickHouse never came up"; exit 1; }
log "ClickHouse up."

# ---- 3. schema ------------------------------------------------------------
# ClickHouse HTTP runs one statement per request; feed the multi-statement
# schema through clickhouse-client --multiquery inside the container.
log "applying schema ..."
podman exec -i cipher-ch clickhouse-client --multiquery < schema.sql \
  || { log "schema apply FAILED"; exit 1; }
podman exec cipher-ch clickhouse-client --query "SHOW TABLES FROM cipher" \
  | tee /tmp/cp34_tables.txt
grep -q tenant_state /tmp/cp34_tables.txt || { log "schema verify FAILED"; exit 1; }
log "schema applied: $(tr '\n' ' ' </tmp/cp34_tables.txt)"

# ---- 4. Grafana -----------------------------------------------------------
log "starting Grafana container (host net; installs ClickHouse datasource plugin) ..."
podman run -d --name cipher-grafana --network host \
  -e GF_SECURITY_ADMIN_PASSWORD="$GF_PASS" \
  -e GF_INSTALL_PLUGINS=grafana-clickhouse-datasource \
  -e GF_AUTH_ANONYMOUS_ENABLED=false \
  -v "$HERE/provisioning/datasources":/etc/grafana/provisioning/datasources:ro \
  -v "$HERE/provisioning/dashboards":/etc/grafana/provisioning/dashboards:ro \
  "$GF_IMG" >/dev/null || { log "Grafana run failed"; exit 1; }

# ---- 5. cipher_flopd (CP 3.3 daemon, unchanged) + ingest -----------------
log "starting cipher_flopd (CP 3.3 daemon) ..."
sudo -n pkill -f cipher_flopd >/dev/null 2>&1 || true
sleep 1
sudo -n "$CP33/cipher_flopd" > /tmp/cp34_flopd.log 2>&1 &
sleep 4
grep -q live /proc/cipher/flops 2>/dev/null \
  && log "flopd live" || log "WARN flopd not yet live"

log "starting ingest agent ..."
python3 cipher_ingest.py > /tmp/cp34_ingest.log 2>&1 &
INGEST_PID=$!

# ---- 6. wait for Grafana health (plugin install can take ~30-60s) --------
log "waiting for Grafana :3000 ..."
for i in $(seq 1 90); do
  curl -s "http://127.0.0.1:3000/api/health" 2>/dev/null | grep -q '"database": *"ok"' && break
  sleep 2
done
curl -s "http://127.0.0.1:3000/api/health" | tee /tmp/cp34_gf_health.txt
echo

# ---- 7. gate workload: 3-tenant asymmetric load, >=90 s ------------------
log "running loadgen — 3 tenants asymmetric, ${DURATION}s ..."
sudo -n ./loadgen "$DURATION" 2>&1 | tee /tmp/cp34_loadgen.log
log "loadgen done; letting ingest flush ..."
sleep 4

# ---- 8. stop ingest + flopd ----------------------------------------------
kill -TERM "$INGEST_PID" 2>/dev/null || true
wait "$INGEST_PID" 2>/dev/null || true
sudo -n pkill -f cipher_flopd >/dev/null 2>&1 || true
cat /tmp/cp34_ingest.log

# ---- 9. capture evidence --------------------------------------------------
log "capturing ClickHouse evidence -> ch_rows.txt ..."
{
  echo "### cipher.device_state ###"
  echo "SELECT count() AS device_rows FROM cipher.device_state" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
  echo "--- first/last 3 device rows ---"
  echo "SELECT * FROM cipher.device_state ORDER BY ts LIMIT 3 FORMAT PrettyCompact" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
  echo "SELECT * FROM cipher.device_state ORDER BY ts DESC LIMIT 3 FORMAT PrettyCompact" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
  echo
  echo "### cipher.tenant_state ###"
  echo "SELECT count() AS tenant_rows, uniqExact(tenant) AS distinct_tenants FROM cipher.tenant_state" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
  echo "--- per-tenant: rows, window seconds, mean/max MFU ---"
  echo "SELECT tenant, count() AS rows, dateDiff('second', min(ts), max(ts)) AS window_s, round(avg(mfu_pct),2) AS mean_mfu, round(max(mfu_pct),2) AS max_mfu, round(avg(flops_per_s)/1e12,1) AS mean_tflops FROM cipher.tenant_state GROUP BY tenant ORDER BY mean_mfu DESC FORMAT PrettyCompact" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
  echo "--- sample: 5 consecutive seconds, all tenants ---"
  echo "SELECT toStartOfSecond(ts) AS sec, tenant, round(mfu_pct,1) AS mfu, round(flops_per_s/1e12,1) AS tflops FROM cipher.tenant_state WHERE tenant LIKE 'cipher34-%' ORDER BY ts LIMIT 15 FORMAT PrettyCompact" \
    | curl -s 'http://127.0.0.1:8123/' --data-binary @-
} | tee ch_rows.txt

# ---- 10. Grafana datasource + dashboard + ds/query -----------------------
log "capturing Grafana datasource list ..."
curl -s -u "$GF_USER:$GF_PASS" "http://127.0.0.1:3000/api/datasources" \
  | tee grafana_datasources.json; echo

log "capturing provisioned dashboard ..."
curl -s -u "$GF_USER:$GF_PASS" "http://127.0.0.1:3000/api/search?query=CIPHER" \
  | tee grafana_search.json; echo

log "criterion (c): /api/ds/query through the dashboard datasource ..."
NOW_MS=$(( $(date +%s) * 1000 ))
FROM_MS=$(( NOW_MS - 3600000 ))
curl -s -u "$GF_USER:$GF_PASS" -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:3000/api/ds/query" \
  -d "{\"queries\":[{\"refId\":\"A\",\"datasource\":{\"type\":\"grafana-clickhouse-datasource\",\"uid\":\"cipher-ch\"},\"editorType\":\"sql\",\"queryType\":\"table\",\"rawSql\":\"SELECT ts, tenant, mfu_pct, flops_per_s FROM cipher.tenant_state WHERE tenant LIKE 'cipher34-%' ORDER BY ts\",\"intervalMs\":1000,\"maxDataPoints\":4000}],\"from\":\"$FROM_MS\",\"to\":\"$NOW_MS\"}" \
  > grafana_dsquery.json
head -c 600 grafana_dsquery.json; echo

# ---- 11. compute gate verdicts -------------------------------------------
log "computing gate verdicts ..."
KMOD_MD5_POST=$(md5sum "$KMOD" 2>/dev/null | cut -d' ' -f1)
LIBC_MD5_POST=$(md5sum "$LIBC" 2>/dev/null | cut -d' ' -f1)
TAINT_POST=$(cat /proc/sys/kernel/tainted)

python3 - "$KMOD_MD5_PRE" "$KMOD_MD5_POST" "$LIBC_MD5_PRE" "$LIBC_MD5_POST" \
            "$TAINT_PRE" "$TAINT_POST" <<'PYEOF'
import json, sys, urllib.request

kmod_pre, kmod_post, libc_pre, libc_post, taint_pre, taint_post = sys.argv[1:7]

def ch(q):
    req = urllib.request.Request("http://127.0.0.1:8123/", data=q.encode())
    return urllib.request.urlopen(req, timeout=10).read().decode().strip()

dev_rows  = int(ch("SELECT count() FROM cipher.device_state"))
ten_rows  = int(ch("SELECT count() FROM cipher.tenant_state"))
# per gate-tenant window + min rows
rows = ch("SELECT tenant, count(), dateDiff('second',min(ts),max(ts)) "
          "FROM cipher.tenant_state WHERE tenant LIKE 'cipher34-%' "
          "GROUP BY tenant ORDER BY tenant FORMAT TSV")
gate_tenants = [r.split("\t") for r in rows.splitlines() if r]
n_tenants = len(gate_tenants)
min_window = min((int(r[2]) for r in gate_tenants), default=0)
all_mfu_pos = ch("SELECT countIf(mfu_pct>0)=count() FROM cipher.tenant_state "
                 "WHERE tenant LIKE 'cipher34-%'") == "1"

# (a) ClickHouse: >=90s x >=3 tenants of per-second rows
gate_a = (n_tenants >= 3 and min_window >= 90 and dev_rows >= 90)

# (b) Grafana: datasource + dashboard provisioned
try:
    with open("grafana_datasources.json") as f: ds = json.load(f)
    ds_ok = any(d.get("uid") == "cipher-ch" and
                d.get("type") == "grafana-clickhouse-datasource" for d in ds)
except Exception:
    ds_ok = False
try:
    with open("grafana_search.json") as f: srch = json.load(f)
    dash_ok = any(d.get("uid") == "cipher-silicon" for d in srch)
except Exception:
    dash_ok = False
gate_b = ds_ok and dash_ok

# (c) /api/ds/query returns non-empty per-tenant series through Grafana.
# tenant count comes from the frame's field LABELS (timeseries-wide format:
# one value field per tenant, tenant carried as a field label) — not from
# a data column.
try:
    with open("grafana_dsquery.json") as f: dq = json.load(f)
    frames = dq["results"]["A"]["frames"]
    fields = frames[0]["schema"]["fields"]
    dq_tenants = len({fd["labels"]["tenant"] for fd in fields
                      if "tenant" in fd.get("labels", {})})
    vals = frames[0]["data"]["values"]
    dq_rows = len(vals[0]) if vals and vals[0] else 0
    gate_c = dq_rows > 0 and dq_tenants >= 3
except Exception as e:
    dq_rows, dq_tenants, gate_c = 0, 0, False

# (d) anchors held
gate_d = (kmod_pre == kmod_post and libc_pre == libc_post
          and taint_pre == taint_post)

res = {
  "cp": "3.4",
  "device_rows": dev_rows, "tenant_rows": ten_rows,
  "gate_tenants": n_tenants, "min_tenant_window_s": min_window,
  "all_tenant_mfu_positive": all_mfu_pos,
  "datasource_provisioned": ds_ok, "dashboard_provisioned": dash_ok,
  "dsquery_rows": dq_rows, "dsquery_distinct_tenants": dq_tenants,
  "kmod_md5_pre": kmod_pre, "kmod_md5_post": kmod_post,
  "libcipher_md5_pre": libc_pre, "libcipher_md5_post": libc_post,
  "taint_pre": int(taint_pre), "taint_post": int(taint_post),
  "gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c, "gate_d": gate_d,
  "all_pass": all([gate_a, gate_b, gate_c, gate_d]),
}
with open("gate_result.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
PYEOF

# ---- 12. container image digests (for the report) ------------------------
log "container image digests ..."
podman images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.Digest}}' \
  | grep -E 'clickhouse|grafana' | tee container_images.txt

log "=== CP 3.4 gate run complete — evidence in $HERE ==="
log "containers still UP for inspection. Teardown: ./teardown_cp34.sh"
