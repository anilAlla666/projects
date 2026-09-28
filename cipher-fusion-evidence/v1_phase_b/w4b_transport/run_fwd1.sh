#!/usr/bin/env bash
# FWD-1 sweep (generate()-based; supported path). B x COND on Mistral-7B.
#   eager_dynamic  : DynamicCache (W.4b.7-era path)
#   eager_static   : StaticCache (pre-alloc) -- the cache-impl lever
#   compile_static : StaticCache + torch.compile(reduce-overhead) -- the graph lever
# Correctness: token-seq diff vs eager_dynamic (greedy must match). Timing: wall/MAXNEW.
# Native nvidia-smi -lms sampling (no shell sleep) per run.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODELP:-/home/ubuntu/models/Mistral-7B-v0.1}"
MAXNEW="${MAXNEW:-256}"; NREP="${NREP:-3}"; NWARM="${NWARM:-2}"
OUT="${HERE}/fwd1_gen_sweep"; mkdir -p "$OUT"
BSET="${BSET:-1 4 8}"; CONDS="${CONDS:-eager_dynamic eager_static compile_static}"

run_one() {  # $1=B $2=COND
  local B="$1" C="$2" tag="b${1}_${2}"
  local SENT="/tmp/fwd1g_${tag}.win" RJ="${OUT}/${tag}.json"
  rm -f "$SENT"
  nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -lms 100 -i 0 \
    > "${OUT}/${tag}_dmon.csv" 2>/dev/null &
  local SMI=$!
  env WL_MODEL="$MODEL" B="$B" COND="$C" MAXNEW="$MAXNEW" NREP="$NREP" NWARM="$NWARM" \
      SENTINEL="$SENT" RESULT_JSON="$RJ" \
      python3 "${HERE}/fwd1_generate_probe.py" > "${OUT}/${tag}.log" 2>&1
  local rc=$?
  kill $SMI 2>/dev/null
  grep -hE "FWD1gen" "${OUT}/${tag}.log" | tail -1 | sed 's/^/  /'
  [ $rc -ne 0 ] && echo "    (rc=$rc — see ${tag}.log)"
  # modal decode clock (the steady generate window)
  awk -F, '{gsub(/ /,"");print $2}' "${OUT}/${tag}_dmon.csv" | sort -n | uniq -c | sort -rn | head -2 \
    | awk '{print "    sm_MHz="$2" n="$1}'
}

echo "=== FWD-1 generate() sweep: Mistral-7B MAXNEW=$MAXNEW NWARM=$NWARM ==="
for B in $BSET; do
  for C in $CONDS; do
    echo "-- B=$B COND=$C --"; run_one "$B" "$C"
  done
done

echo "=== correctness: token-seq match vs eager_dynamic + decode board power ==="
python3 - "$OUT" "$BSET" <<'PY'
import json, sys, glob, os
OUT=sys.argv[1]; BSET=sys.argv[2].split()
def power_at_modal(tag):
    f=os.path.join(OUT,"%s_dmon.csv"%tag)
    try:
        rows=[l.split(',') for l in open(f) if ',' in l]
        vals=[(float(p),int(float(c))) for p,c in ((r[0],r[1]) for r in rows)]
    except Exception: return None
    if not vals: return None
    from collections import Counter
    modal=Counter(c for _,c in vals).most_common(1)[0][0]
    ps=[p for p,c in vals if c==modal]
    return (modal, round(sum(ps)/len(ps),1))
for B in BSET:
    ref=None
    try: ref=json.load(open(os.path.join(OUT,"b%s_eager_dynamic.json"%B)))["new_tokens"]
    except Exception: pass
    for C in ["eager_dynamic","eager_static","compile_static"]:
        p=os.path.join(OUT,"b%s_%s.json"%(B,C))
        if not os.path.exists(p): print("  B=%s %s: MISSING"%(B,C)); continue
        r=json.load(open(p)); t=r["new_tokens"]
        mt="-"
        if ref is not None:
            tot=sum(len(x) for x in ref); mm=sum(a==b for xx,yy in zip(ref,t) for a,b in zip(xx,yy))
            mt="%d/%d"%(mm,tot)
        pw=power_at_modal("b%s_%s"%(B,C))
        print("  B=%s %-14s ms/tok=%6.3f tok/s=%7.1f match=%-10s clk/W@modal=%s"
              %(B,C,r["ms_per_new_token"],r["tok_s"],mt,pw))
PY
echo "=== results in ${OUT}/ ==="
