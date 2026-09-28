#!/usr/bin/env bash
# T4.5.3 measurement — N matched pairs of (CIPHER_MARLIN=off, =on) at
# B=1 TinyLlama decode 60s each. Records watts via 1Hz nvidia-smi and
# tok/s from run_for_duration progress.

set -u
N="${1:-3}"
DUR="${2:-60}"
OUT=/home/ubuntu/cipher-phase4-evidence/t4_5_marlin
mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null || true

run_one() {
  local pair="$1"; local arm="$2"  # off | on
  local tid="t4_5_marlin_${arm}_p${pair}"
  local d="${OUT}/pair${pair}_${arm}"; mkdir -p "$d"
  local prog="/tmp/cipher_tenant_${tid}_progress"; rm -f "$prog"
  local csv="${d}/watts.csv"

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

  CIPHER_TENANT_ID="$tid" WL_DURATION="$DUR" \
    CIPHER_MARLIN="$arm" \
    LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
    > "$d/run.log" 2>&1

  sleep 2
  kill -INT "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null || true
  cp "$prog" "$d/progress.txt" 2>/dev/null || true
  tail -1 "$prog" 2>/dev/null
}

echo "=== baseline state ==="
sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader

for ((i=1; i<=N; i++)); do
  echo
  echo "--- pair $i CIPHER_MARLIN=off ---"
  run_one "$i" off
  echo "--- pair $i CIPHER_MARLIN=on ---"
  run_one "$i" on
done

echo
echo "=== analyze ==="
python3 - <<PY
import os, csv, statistics, json, math
ROOT="$OUT"
def read_watts(p, tail_frac=0.6):
    rows=[]
    if not os.path.exists(p): return None,None
    with open(p) as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                rows.append({"p":float(row["power_w"]),"c":float(row["clock_sm_mhz"])})
            except: continue
    if not rows: return None,None
    cutoff=int(len(rows)*(1-tail_frac))
    tail=rows[cutoff:]
    if not tail: return None,None
    return statistics.mean(r["p"] for r in tail), statistics.mean(r["c"] for r in tail)

def read_prog(p):
    if not os.path.exists(p): return {}
    end=None
    for ln in open(p):
        if " end " in ln: end=ln.strip()
    out={}
    if end:
        for t in end.split():
            if "=" in t:
                k,v=t.split("=",1)
                try: out[k]=float(v) if "." in v else int(v)
                except: out[k]=v
    return out

per={}
for arm in ("off","on"):
    per[arm]=[]
    for i in range(1,$N+1):
        d=f"{ROOT}/pair{i}_{arm}"
        pw,ck=read_watts(f"{d}/watts.csv")
        pr=read_prog(f"{d}/progress.txt")
        try: secs=float(str(pr.get("t","60.0")).rstrip("s"))
        except: secs=60.0
        tps = pr.get("tokens",0)/secs if secs>0 else 0
        tpw = tps/pw if pw and pw>0 else 0
        per[arm].append({"watts":pw,"clock":ck,"tok_s":tps,"tok_w":tpw,
                         "tokens":pr.get("tokens",0),"iters":pr.get("iters",0)})

def stats(rows, k):
    vs=[r[k] for r in rows if r[k] is not None]
    if not vs: return None
    m=statistics.mean(vs); sd=statistics.pstdev(vs)
    return {"mean":round(m,4),"stddev":round(sd,4),"n":len(vs)}

def delta(a,b):
    if not a or not b or b["mean"]==0: return None
    diff=a["mean"]-b["mean"]
    se=math.sqrt(a["stddev"]**2/a["n"]+b["stddev"]**2/b["n"])
    pct=diff/b["mean"]*100
    pct_se=se/b["mean"]*100
    return {"pct_delta":round(pct,2),"pct_se":round(pct_se,2),
            "ci_lo":round(pct-1.96*pct_se,2),"ci_hi":round(pct+1.96*pct_se,2)}

result={}
for k in ("watts","tok_s","tok_w"):
    a=stats(per["on"],k); b=stats(per["off"],k)
    result[k]={"off":b,"on":a,"delta":delta(a,b)}

print("=== per-pair ===")
for i,(o,n) in enumerate(zip(per["off"],per["on"]),1):
    if o["tok_s"] and n["tok_s"]:
        print(f"pair {i}: off tps={o['tok_s']:.2f} W={o['watts']:.1f} tpw={o['tok_w']:.4f}  "
              f"|  on tps={n['tok_s']:.2f} W={n['watts']:.1f} tpw={n['tok_w']:.4f}")
print()
print("=== summary ===")
print(json.dumps(result, indent=2))
with open(f"{ROOT}/summary.json","w") as f: json.dump(result,f,indent=2)
PY
