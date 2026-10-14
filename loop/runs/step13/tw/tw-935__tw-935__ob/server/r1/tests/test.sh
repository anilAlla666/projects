#!/bin/bash
# Verifier for hardware_audit.sh. Self-contained: needs only bash, coreutils,
# util-linux (lscpu/lsblk), jq (all in the grading image). No network, no pytest.
# Ground truth is recomputed inside the grading container at test time.
SCRIPT="${HW_AUDIT_SCRIPT:-/app/hardware_audit.sh}"
OUT=/tmp/hardware_audit.json
LOGDIR="${HW_AUDIT_LOGDIR:-/logs/verifier}"
mkdir -p "$LOGDIR"
FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS+1)); }
pass() { echo "PASS: $*"; }
finish() {
  if [ "$FAILS" -eq 0 ]; then echo 1 > "$LOGDIR/reward.txt"; echo "RESULT: ACCEPTED";
  else echo 0 > "$LOGDIR/reward.txt"; echo "RESULT: REJECTED ($FAILS failures)"; fi
  exit 0
}

# ---------- 1. script exists, executable; run it fresh ----------
if [ ! -f "$SCRIPT" ]; then fail "script $SCRIPT does not exist"; finish; fi
if [ ! -x "$SCRIPT" ]; then fail "script $SCRIPT is not executable"; finish; fi
pass "script exists and is executable"
# The script must actually query the system (task: "detect and report"); every
# mechanism for each subsystem is listed, so only hard-coded output is caught.
srcchk() { grep -qE "$2" "$SCRIPT" || fail "script contains no way to detect $1 (none of: $2)"; }
srcchk "USB root hubs" "lsusb|/sys/bus/usb|usb-devices|/dev/bus/usb|/sys/kernel/debug/usb|/proc/bus/usb|udevadm|/sys/class/usb|hwinfo|lshw|/sys/devices"
srcchk "CPU information" "lscpu|/proc/cpuinfo|nproc|/sys/devices/system/cpu|getconf|dmidecode|lshw|hwinfo|cpuid"
srcchk "storage devices" "lsblk|/sys/block|/proc/partitions|/sys/class/block|/dev/disk|blockdev|fdisk|lshw|hwinfo|udevadm|/sys/dev/block"
rm -f "$OUT"
( cd /app 2>/dev/null || cd /; timeout 200 "$SCRIPT" ) > /tmp/hw_audit_stdout.txt 2>&1
echo "script exit code: $?"
NOW=$(date +%s)
if [ ! -s "$OUT" ]; then fail "$OUT not created by the script"; finish; fi
if ! jq -e 'type=="object"' "$OUT" >/dev/null 2>&1; then fail "$OUT is not a valid JSON object"; finish; fi
if command -v python3 >/dev/null 2>&1; then
  if ! python3 - "$OUT" <<'PY'
import json,sys
def bad(c): raise ValueError("non-standard constant "+c)
d=json.load(open(sys.argv[1]),parse_constant=bad)
assert isinstance(d,dict)
PY
  then fail "$OUT is not strictly valid JSON"; finish; fi
fi
pass "valid JSON object"
DATA=$(cat "$OUT")

# ---------- 2. ground truth from this container ----------
# USB root hubs: bcdUSB of each root hub (sysfs usbN/version), e.g. "3.10" -> 3.1
USBV=()
for d in /sys/bus/usb/devices/usb*; do
  [ -f "$d/version" ] || continue
  v=$(tr -d ' \n' < "$d/version")
  [ -n "$v" ] && USBV+=("$(awk -v v="$v" 'BEGIN{printf "%.1f", v}')")
done
if [ ${#USBV[@]} -eq 0 ] && command -v lsusb >/dev/null 2>&1; then
  while read -r p; do
    case "$p" in 0001) USBV+=("1.1");; 0002) USBV+=("2.0");; 0003) USBV+=("3.0");; esac
  done < <(lsusb 2>/dev/null | sed -n 's/.*ID 1d6b:\(000[123]\).*/\1/p')
fi
USB_TRUTH=$(printf '%s\n' "${USBV[@]}" | jq -Rn '[inputs|select(length>0)|tonumber] | group_by(.) | map({v:.[0], n:length})')
echo "truth usb root hubs: $(echo "$USB_TRUTH" | jq -c .)"

LSCPU=$(LC_ALL=C lscpu 2>/dev/null)
lsf() { echo "$LSCPU" | sed -n "s/^$1:[[:space:]]*//p" | head -1; }
M1=$(lsf 'Model name'); M2=$(grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-)
T1=$(lsf 'CPU(s)'); T2=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null); T3=$(nproc --all 2>/dev/null); T4=$(nproc 2>/dev/null); T5=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
CPS=$(lsf 'Core(s) per socket'); SOC=$(lsf 'Socket(s)'); CPC=$(lsf 'Core(s) per cluster'); CLU=$(lsf 'Cluster(s)')
C1=""; [[ "$CPS" =~ ^[0-9]+$ && "$SOC" =~ ^[0-9]+$ ]] && C1=$((CPS*SOC))
C2=""; [[ "$CPC" =~ ^[0-9]+$ && "$CLU" =~ ^[0-9]+$ ]] && C2=$((CPC*CLU))
C3=$(awk -F: '/^physical id/{p=$2} /^core id/{print p"-"$2}' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)
[ "$C3" = "0" ] && C3=""
C4=""; TPC=$(lsf 'Thread(s) per core'); [[ "$TPC" =~ ^[0-9]+$ && "$T1" =~ ^[0-9]+$ && "$TPC" -gt 0 ]] && C4=$((T1/TPC))
FLAGS=$( { lsf 'Flags'; grep -E '^(flags|Features)[[:space:]]*:' /proc/cpuinfo 2>/dev/null | cut -d: -f2-; } | tr -s ' \t' '\n' | sed '/^$/d' | sort -u)
CPU_TRUTH=$(jq -n --arg m1 "$M1" --arg m2 "$M2" --arg t "$T1 $T2 $T3 $T4 $T5" --arg c "$C1 $C2 $C3 $C4" --arg f "$FLAGS" '
  def nums: split(" ")|map(select(test("^[0-9]+$"))|tonumber)|map(select(.>0))|unique;
  def norm: gsub("^\\s+|\\s+$";"")|gsub("\\s+";" ");
  {models: ([$m1,$m2]|map(norm)|map(select(length>0))|unique), threads: ($t|nums), cores: ($c|nums),
   flags: ($f|split("\n")|map(select(length>0)))}')
echo "truth cpu: $(echo "$CPU_TRUTH" | jq -c '{models,threads,cores,nflags:(.flags|length)}')"

ALLBLK=$(lsblk -J -b -d -a -o NAME,TYPE,SIZE,ROTA,TRAN 2>/dev/null || echo '{"blockdevices":[]}')
BLK_TRUTH=$(echo "$ALLBLK" | jq '[.blockdevices[]? | {name, ktype:.type, size:((.size//0)|tonumber? // 0),
   rota:(.rota==true or .rota=="1" or .rota==1), tran:(.tran//"")}]')
echo "truth block devices: $(echo "$BLK_TRUTH" | jq -c .)"

# ---------- 3. checks ----------
check() { # name, jq program producing list of error strings
  local name="$1" prog="$2" errs
  errs=$(jq -r --argjson usb "$USB_TRUTH" --argjson cpu "$CPU_TRUTH" --argjson blk "$BLK_TRUTH" \
             "$prog | .[]" <<<"$DATA" 2>&1)
  if [ -n "$errs" ]; then while IFS= read -r e; do fail "$name: $e"; done <<<"$errs"; else pass "$name"; fi
}

check "top-level keys" '[ ("usb","cpu","storage","audit_timestamp") as $k | select(has($k)|not) | "missing key \($k)" ]'

check "usb" '
  def isint: type=="number" and . == floor and . >= 0;
  (.usb // {}) as $u |
  ($usb|map(.v)|max) as $tmax |
  if ($u|type) != "object" then ["usb is not an object"] else
  [
    (if ($u.has_usb3|type) != "boolean" then "has_usb3 must be boolean"
     elif $u.has_usb3 != (($tmax // 0) >= 3) then "has_usb3=\($u.has_usb3) but system max root hub version is \($tmax)" else empty end),
    (if $tmax == null then
        (if ($u.max_version == null) or (($u.max_version|type)=="string" and ((($u.max_version|tonumber?) // null)==null)) then empty
         else "no USB root hubs on this system but max_version=\($u.max_version)" end)
     else
        (if ($u.max_version|type)!="string" then "max_version must be a string, got \($u.max_version)"
         elif (($u.max_version|tonumber?) // -1) != $tmax then "max_version=\($u.max_version) but highest root hub version is \($tmax)" else empty end)
     end),
    (if ($u.root_hubs|type) != "object" then "root_hubs must be an object" else
      ($u.root_hubs|to_entries) as $e |
      ($e[] | select(((.key|tonumber?) // null)==null) | "root_hubs key \(.key) is not a version"),
      ($e[] | select(.value|isint|not) | "root_hubs[\(.key)] must be a non-negative integer"),
      ( [ $e[] | select(((.key|tonumber?) // null)!=null and (.value|isint)) | {v:((.key|tonumber)*10|round/10), n:.value} ]
        | group_by(.v) | map({v:.[0].v, n:(map(.n)|add)}) ) as $got |
      (($usb[] | . as $t | (($got[]|select(.v==$t.v)|.n) // 0) as $n
         | select($n != $t.n) | "root_hubs reports \($n) root hub(s) of version \($t.v), system has \($t.n)"),
       ($got[] | select(.n>0) | . as $g | select(([$usb[]|select(.v==$g.v)]|length)==0)
         | "root_hubs reports \($g.n) root hub(s) of version \($g.v), system has none"))
     end)
  ] end'

check "cpu" '
  (.cpu // {}) as $c |
  def norm: gsub("^\\s+|\\s+$";"")|gsub("\\s+";" ");
  if ($c|type) != "object" then ["cpu is not an object"] else
  [
    (if ($cpu.models|length)>0 then
        (if ($c.model|type)!="string" or (($c.model|norm) as $m | $cpu.models|index($m)) == null
         then "model=\($c.model|tojson), expected one of \($cpu.models|tojson)" else empty end)
     else (if $c.model == null or (($c.model|type)=="string" and ($c.model|length)>0) then empty else "model invalid" end) end),
    (if ($cpu.threads|length)>0 then
        (if ($c.threads|type)!="number" or ($cpu.threads|index($c.threads))==null then "threads=\($c.threads), expected one of \($cpu.threads)" else empty end)
     else empty end),
    (if ($cpu.cores|length)>0 then
        (if ($c.cores|type)!="number" or ($cpu.cores|index($c.cores))==null then "cores=\($c.cores), expected one of \($cpu.cores) (physical cores)" else empty end)
     else empty end),
    (if ($c.cores|type)=="number" and ($c.threads|type)=="number" and $c.threads < $c.cores then "threads < cores" else empty end),
    (if ($c.features|type)!="array" then "features must be an array" else
       ($c.features[] | select(type!="string") | "feature \(tojson) is not a string"),
       (if ($cpu.flags|length)>0 then
          (if ($c.features|length)==0 then "features empty but CPU has flags" else empty end),
          ($c.features[] | select(type=="string") | select(. as $f | $cpu.flags|index($f)|not) | "feature \(.) is not a flag of this CPU"),
          (["aes","avx","avx2","sse4_1","sse4_2"][] | . as $f | select(($cpu.flags|index($f)) and (($c.features|index($f))|not)) | "CPU has flag \(.) but features omits it")
        else empty end)
     end)
  ] end'

check "storage" '
  (.storage // {}) as $s |
  def num: type=="number";
  def gbok($b): num and ((. - ($b/1e9)|fabs) <= 1 or (. - ($b/1073741824)|fabs) <= 1);
  def exptype: if (.name|startswith("nvme")) or .tran=="nvme" then "nvme" elif .rota then "hdd" else "ssd" end;
  if ($s|type) != "object" then ["storage is not an object"] else
  if ($s.devices|type) != "array" then ["devices must be an array"] else
  ($s.devices) as $d |
  [
    ($d[] | select(type!="object" or (.name|type)!="string") | "device entry without a string name: \(tojson)"),
    ([ $d[] | objects | .name ] | group_by(.) | map(select(length>1)) | .[] | "duplicate device \(.[0])"),
    ($blk[] | select(.ktype=="disk" and .size>0) | . as $t |
       ([$d[] | objects | select(.name==$t.name or .name==("/dev/"+$t.name))] | first) as $o |
       if $o == null then "disk \($t.name) missing from devices"
       else
         (if (($o.type|tostring|ascii_downcase)) != ($t|exptype) then "device \($t.name) type=\($o.type), expected \($t|exptype)" else empty end),
         (if ($o.size_gb|gbok($t.size))|not then "device \($t.name) size_gb=\($o.size_gb), disk is \($t.size) bytes" else empty end)
       end),
    ($d[] | objects | select(.name|type=="string") | . as $o | (.name|sub("^/dev/";"")) as $n |
       ([$blk[]|select(.name==$n)]|first) as $t |
       if $t == null then "device \($o.name) is not a block device on this system"
       elif $t.ktype != "disk" or $t.size == 0 then
          (if ($o.size_gb == null) or ($o.size_gb|gbok($t.size)) then empty else "device \($n) size_gb=\($o.size_gb) wrong" end)
       else empty end),
    ([ $d[] | objects | .size_gb | select(type=="number") ] | add // 0) as $sum |
    (if ($s.total_capacity_gb|num|not) then "total_capacity_gb must be a number"
     elif (($s.total_capacity_gb - $sum)|fabs) > ([1, 0.5*($d|length)]|max) then "total_capacity_gb=\($s.total_capacity_gb) but devices sum to \($sum)"
     else empty end),
    (if ($s.type_summary|type)!="object" then "type_summary must be an object" else
       ($s.type_summary) as $ts |
       ((["ssd","hdd","nvme"] + ($ts|keys))|unique[]) as $k |
       ([ $d[] | objects | select((.type|tostring|ascii_downcase)==$k) ] | length) as $n |
       select(($ts[$k] // 0) != $n or ($ts|has($k)|not and ($k=="ssd" or $k=="hdd" or $k=="nvme"))) |
       "type_summary.\($k)=\($ts[$k]) but devices list has \($n) of type \($k)"
     end)
  ] end end'

TS=$(jq -r '.audit_timestamp | if type=="string" then . else "" end' <<<"$DATA")
if [[ "$TS" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2}([.,][0-9]+)?)?(Z|[+-][0-9]{2}(:?[0-9]{2})?)?$ ]]; then
  E=$(date -d "${TS/,/.}" +%s 2>/dev/null)
  if [ -z "$E" ]; then fail "audit_timestamp '$TS' not parseable"
  elif [ $(( E > NOW ? E - NOW : NOW - E )) -gt 900 ]; then fail "audit_timestamp '$TS' is not the time of the audit"
  else pass "audit_timestamp '$TS'"; fi
else
  fail "audit_timestamp '$TS' is not an ISO 8601 date-time"
fi
finish
