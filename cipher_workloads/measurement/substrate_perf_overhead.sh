#!/usr/bin/env bash
# T4.5.1 S1.D — substrate perf-overhead probe.
# 3 matched pairs of (baseline, substrate-loaded) at B=1 decode 60s.
# Substrate adds shim+dispatch overhead per cublasGemmEx; goal <1% tok/s drift.

set -u
N="${1:-3}"
DUR="${2:-60}"
OUT=/home/ubuntu/cipher-phase4-evidence/t4_5_substrate
mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null || true

run_one() {
  local pair="$1"; local arm="$2"  # baseline | substrate
  local tid="t4_5_${arm}_p${pair}"
  local d="${OUT}/pair${pair}_${arm}"
  mkdir -p "$d"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  rm -f "$prog"

  if [[ "$arm" == "substrate" ]]; then
    LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    CIPHER_TENANT_ID="$tid" WL_DURATION="$DUR" \
      python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
      > "$d/run.log" 2>&1
  else
    CIPHER_TENANT_ID="$tid" WL_DURATION="$DUR" \
      python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
      > "$d/run.log" 2>&1
  fi
  cp "$prog" "$d/progress.txt" 2>/dev/null || true
  tail -1 "$prog" 2>/dev/null
}

echo "=== ensure baseline pod state ==="
sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1
nvidia-smi --query-gpu=clocks.sm,power.limit --format=csv,noheader

for ((i=1; i<=N; i++)); do
  echo
  echo "--- pair $i baseline ---"
  run_one "$i" baseline
  echo "--- pair $i substrate ---"
  run_one "$i" substrate
done

echo
echo "=== analyze ==="
python3 - <<PY
import os, statistics, json, math
ROOT="$OUT"
def read_prog(p):
    if not os.path.exists(p): return {}
    end=None
    for ln in open(p):
        if " end " in ln: end=ln.strip()
    if not end: return {}
    out={}
    for t in end.split():
        if "=" in t:
            k,v=t.split("=",1)
            try: out[k]=float(v) if "." in v else int(v)
            except: out[k]=v
    return out

base, subs = [], []
for i in range(1, $N+1):
    pb = read_prog(f"{ROOT}/pair{i}_baseline/progress.txt")
    ps = read_prog(f"{ROOT}/pair{i}_substrate/progress.txt")
    secs_b = float(str(pb.get("t","60.0")).rstrip("s"))
    secs_s = float(str(ps.get("t","60.0")).rstrip("s"))
    tps_b = pb.get("tokens",0)/secs_b if secs_b else 0
    tps_s = ps.get("tokens",0)/secs_s if secs_s else 0
    base.append(tps_b); subs.append(tps_s)
    print(f"pair {i}: baseline tok/s={tps_b:.2f}  substrate tok/s={tps_s:.2f}  delta={(tps_s-tps_b)/tps_b*100:+.2f}%")

mb = statistics.mean(base); sb = statistics.pstdev(base)
ms = statistics.mean(subs); ss = statistics.pstdev(subs)
diff = ms - mb
se = math.sqrt(sb**2/$N + ss**2/$N)
pct = diff/mb*100 if mb else 0
pct_se = se/mb*100 if mb else 0
print()
print(f"baseline   mean ± σ : {mb:.3f} ± {sb:.3f}")
print(f"substrate  mean ± σ : {ms:.3f} ± {ss:.3f}")
print(f"Δ% (subs vs base)   : {pct:+.3f}%  ±  {pct_se:.3f}%  (95% CI ≈ [{pct-1.96*pct_se:+.2f}, {pct+1.96*pct_se:+.2f}])")
result = {
  "n_pairs": $N,
  "baseline_tok_s": {"mean":round(mb,3),"stddev":round(sb,3)},
  "substrate_tok_s": {"mean":round(ms,3),"stddev":round(ss,3)},
  "delta_pct": round(pct,3),
  "delta_se_pct": round(pct_se,3),
  "delta_95ci_pct": [round(pct-1.96*pct_se,3), round(pct+1.96*pct_se,3)],
}
with open(f"{ROOT}/perf_summary.json","w") as f: json.dump(result,f,indent=2)
PY
