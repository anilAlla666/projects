#!/usr/bin/env bash
# T4.3.1 S1.6 — VOLT TPW A/B measurement.
#
# Runs B=1 TinyLlama decode under two clock conditions:
#   B (default boost):  no clock lock — GPU runs at default ~1830 MHz
#   A (locked 1000):    sudo nvidia-smi -lgc 1000 (operator-level path)
#
# Same workload, same duration. Capture per-second watts and final tok/s.
# Headline metric: tokens-per-watt (tok/W) under each condition.
#
# Usage:
#   bash volt_tpw_ab.sh [duration_s]

set -u
DUR="${1:-60}"
OUT=/home/ubuntu/cipher-phase4-evidence/t4_3_1
mkdir -p "$OUT"

run_one() {
  local label="$1"   # B or A
  local tid="volt_${label,,}"
  local csv="$OUT/${label}_watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"

  rm -f "$csv" "$prog"

  # Background sampler (every 1 s)
  (
    echo "ts_s,power_w,clock_sm_mhz"
    t0=$(date +%s); end=$((t0+DUR+10))
    while [[ $(date +%s) -lt $end ]]; do
      pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
      ck=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
      ts=$(( $(date +%s) - t0 ))
      echo "$ts,${pw:-0},${ck:-0}"
      sleep 1
    done
  ) > "$csv" &
  SAMPLER_PID=$!

  # Run the workload (libcipher_rt in OFF mode; actuation is external).
  CIPHER_TENANT_ID="$tid" WL_DURATION="$DUR" \
    CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
    > "$OUT/${label}_run.log" 2>&1 &
  WL_PID=$!

  wait "$WL_PID"
  WL_RC=$?
  sleep 2
  kill -INT "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null || true

  cp "$prog" "$OUT/${label}_progress.txt"
  echo "[$label] rc=$WL_RC progress tail:"
  tail -2 "$prog"
}

echo "=== ensuring default state (rgc + pl 700) ==="
sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1
sudo nvidia-smi -i 0 -pl 700 >/dev/null 2>&1
nvidia-smi --query-gpu=clocks.sm,power.limit --format=csv,noheader

echo
echo "=== Run B: default boost clock (no lock) ==="
run_one B

echo
echo "=== lock to 1000 MHz ==="
sudo nvidia-smi -i 0 -lgc 1000
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader

echo
echo "=== Run A: locked 1000 MHz ==="
run_one A

echo
echo "=== restore default ==="
sudo nvidia-smi -i 0 -rgc
nvidia-smi --query-gpu=clocks.sm,power.limit --format=csv,noheader

echo
echo "=== analyze ==="
python3 - <<'PY'
import csv, json, statistics, os

OUT = "/home/ubuntu/cipher-phase4-evidence/t4_3_1"
def read_watts(p):
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({"ts": int(row["ts_s"]),
                             "p": float(row["power_w"]),
                             "c": float(row["clock_sm_mhz"])})
            except Exception:
                continue
    return rows

def read_progress(p):
    if not os.path.exists(p): return {}
    end = None
    with open(p) as f:
        for ln in f:
            if " end " in ln:
                end = ln.strip()
    out = {}
    if end:
        for t in end.split():
            if "=" in t:
                k, v = t.split("=", 1)
                try:
                    out[k] = float(v) if "." in v else int(v)
                except ValueError:
                    out[k] = v
    return out

def steady(rows, tail_frac=0.6):
    # Drop the first 40% to skip warm-up; average the rest.
    if not rows: return None, None, None
    cutoff = int(len(rows) * (1 - tail_frac))
    tail = rows[cutoff:]
    if not tail: return None, None, None
    return (round(statistics.mean(r["p"] for r in tail), 2),
            round(statistics.mean(r["c"] for r in tail), 1),
            len(tail))

out = {}
for L in ("B", "A"):
    rows = read_watts(f"{OUT}/{L}_watts.csv")
    prog = read_progress(f"{OUT}/{L}_progress.txt")
    pw, ck, n = steady(rows)
    out[L] = {
        "label": L,
        "watts_steady": pw,
        "clock_steady": ck,
        "samples_steady": n,
        "tokens": prog.get("tokens", 0),
        "iters": prog.get("iters", 0),
        "duration_s_progress": prog.get("t", None),
    }
    # tok/s and tok/W if we can compute
    if prog.get("tokens") and prog.get("t"):
        try:
            secs = float(str(prog["t"]).rstrip("s"))
            tps = prog["tokens"] / secs
            out[L]["tok_per_s"] = round(tps, 2)
            if pw and pw > 0:
                out[L]["tok_per_w"] = round(tps / pw, 4)
        except Exception:
            pass

# Compare
A = out.get("A", {}); B = out.get("B", {})
delta = {}
for k in ("watts_steady", "tok_per_s", "tok_per_w"):
    a, b = A.get(k), B.get(k)
    if a is not None and b is not None and b != 0:
        delta[k + "_pct"] = round((a - b) / b * 100, 2)

result = {"B_default": B, "A_locked_1000": A, "delta_A_vs_B": delta}
print(json.dumps(result, indent=2))
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(result, f, indent=2)
PY
