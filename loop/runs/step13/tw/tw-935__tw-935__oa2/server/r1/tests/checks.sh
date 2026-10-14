#!/bin/bash
# Checks /tmp/hardware_audit.json against what the grading machine itself reports.
# Uses only bash, coreutils, util-linux, usbutils and jq (all in the task image).
OUT=/tmp/hardware_audit.json
GRACEFUL=/tmp/hardware_audit_graceful.json
SCRIPT=/app/hardware_audit.sh
FAILS=0
fail() { echo "FAILED: $*"; FAILS=$((FAILS+1)); }
pass() { echo "PASSED: $*"; }
# jqcheck <name> <file> <jq-args...> : jq program must output true
jqcheck() {
    local name=$1 file=$2; shift 2
    local r
    r=$(jq -r "$@" "$file" 2>&1)
    if [ "$r" = "true" ]; then pass "$name"; else fail "$name -> $r"; fi
}

# ---------------------------------------------------------------- 1. script + JSON
if [ ! -f "$SCRIPT" ]; then fail "script $SCRIPT missing"; echo "FAILS=$FAILS"; exit 1; fi
[ -x "$SCRIPT" ] && pass "script executable" || fail "script not executable"
if [ ! -f "$OUT" ] || ! jq -e 'type=="object"' "$OUT" >/dev/null 2>&1; then
    fail "$OUT missing or not a JSON object"; echo "FAILS=$FAILS"; exit 1
fi
jqcheck "top-level keys" "$OUT" '(has("usb") and has("cpu") and has("storage") and has("audit_timestamp")
   and (.usb|type)=="object" and (.cpu|type)=="object" and (.storage|type)=="object") // false'

# ---------------------------------------------------------------- 2. USB
sysmap='{}'
for d in /sys/bus/usb/devices/usb*; do
    [ -r "$d/version" ] || continue
    v=$(sed -nE 's/^[[:space:]]*([0-9]+)\.([0-9]).*/\1.\2/p' "$d/version")
    [ -n "$v" ] && sysmap=$(jq -c --arg v "$v" '.[$v] = ((.[$v] // 0) + 1)' <<<"$sysmap")
done
lsmap='{}'
while read -r id; do
    case "$id" in 1d6b:0001) v=1.1;; 1d6b:0002) v=2.0;; 1d6b:0003) v=3.0;; *) continue;; esac
    lsmap=$(jq -c --arg v "$v" '.[$v] = ((.[$v] // 0) + 1)' <<<"$lsmap")
done < <(lsusb 2>/dev/null | sed -nE 's/.*ID ([0-9a-f]{4}:[0-9a-f]{4}).*/\1/p')
if [ "$sysmap" = '{}' ]; then truths="[$lsmap]"; elif [ "$lsmap" = '{}' ]; then truths="[$sysmap]"; else truths="[$sysmap,$lsmap]"; fi
echo "system root hubs (acceptable maps): $truths"
jqcheck "usb section" "$OUT" --argjson truths "$truths" '
  .usb as $u
  | ($u|has("max_version") and has("root_hubs") and has("has_usb3")) as $keys
  | if ($keys|not) then "usb keys missing"
    elif ($u.has_usb3|type) != "boolean" then "has_usb3 not boolean"
    elif ($u.root_hubs|type) != "object" then "root_hubs not an object"
    elif ([$u.root_hubs|keys[]|select(IN("1.0","1.1","2.0","3.0","3.1","3.2")|not)]|length) > 0 then "bad root_hubs key"
    elif ([$u.root_hubs[]|select((type!="number") or (. != floor) or (. < 0))]|length) > 0 then "root_hubs counts must be non-negative integers"
    else
      ($u.root_hubs|with_entries(select(.value > 0))) as $got
      | if ([$truths[]|select(. == $got)]|length) == 0 then "root_hubs \($got) does not match system \($truths)"
        elif ($got|length) > 0 then
          ($got|keys|max_by(tonumber)) as $mx
          | if $u.max_version != $mx then "max_version \($u.max_version) != \($mx)"
            elif $u.has_usb3 != (($mx|tonumber) >= 3) then "has_usb3 inconsistent"
            else true end
        else
          if ($u.max_version == null or (($u.max_version|type)=="string" and ($u.max_version|ascii_downcase)=="unknown")) and $u.has_usb3 == false
          then true else "no root hubs: max_version must be null/unknown and has_usb3 false" end
        end
    end'

# ---------------------------------------------------------------- 3. CPU
LS=$(LC_ALL=C lscpu 2>/dev/null)
field() { sed -nE "s/^$1:[[:space:]]*//p" <<<"$LS" | head -1; }
model=$(field 'Model name')
[ -z "$model" ] && model=$(sed -nE 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)
cps=$(field 'Core\(s\) per socket'); sk=$(field 'Socket\(s\)'); ncpu=$(field 'CPU\(s\)')
cores_opts='[]'
[[ "$cps" =~ ^[0-9]+$ ]] && cores_opts=$(jq -c ". + [$cps * ${sk:-1}]" <<<"$cores_opts" 2>/dev/null || echo "$cores_opts")
pairs=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | grep -c .)
[ "${pairs:-0}" -gt 0 ] && cores_opts=$(jq -c ". + [$pairs]" <<<"$cores_opts")
threads_opts='[]'
for t in "$ncpu" "$(lscpu -p=CPU 2>/dev/null | grep -v '^#' | grep -c .)" "$(nproc 2>/dev/null)" "$(nproc --all 2>/dev/null)" "$(getconf _NPROCESSORS_ONLN 2>/dev/null)"; do
    [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -gt 0 ] && threads_opts=$(jq -c ". + [$t]" <<<"$threads_opts")
done
flags=$( { sed -nE 's/^(flags|Features)[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null; field 'Flags'; } | tr -s ' \t' '\n\n' | grep . | sort -u | jq -R . | jq -sc .)
[ -z "$flags" ] && flags='[]'
echo "system cpu: model='$model' cores in $cores_opts threads in $threads_opts nflags=$(jq length <<<"$flags")"
jqcheck "cpu section" "$OUT" --arg model "$model" --argjson co "$cores_opts" --argjson to "$threads_opts" --argjson flags "$flags" '
  .cpu as $c
  | def norm: gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ");
  if ($c|has("model") and has("cores") and has("threads") and has("features"))|not then "cpu keys missing"
  elif ($model != "") and ((($c.model|type) != "string") or (($c.model|norm) != ($model|norm))) then "model \($c.model) != \($model)"
  elif ($model == "") and ($c.model != null) and ((($c.model|type) != "string") or ($c.model|norm) == "") then "model must be a non-empty string or null"
  elif ($c.cores|type) != "number" or $c.cores != ($c.cores|floor) then "cores not an integer"
  elif ($c.threads|type) != "number" or $c.threads != ($c.threads|floor) then "threads not an integer"
  elif ($co|length) > 0 and ($co|index($c.cores)) == null then "cores \($c.cores) not in \($co)"
  elif ($to|index($c.threads)) == null then "threads \($c.threads) not in \($to)"
  elif $c.threads < $c.cores then "threads < cores"
  elif ($c.features|type) != "array" or ([$c.features[]|select(type!="string")]|length) > 0 then "features must be an array of strings"
  elif ($flags|length) > 0 and ($c.features|length) == 0 then "features empty though CPU reports flags"
  elif ($flags|length) > 0 and ([$c.features[]|select(. as $f | $flags|index($f)|not)]|length) > 0
    then "features not reported by this CPU: \([$c.features[]|select(. as $f | $flags|index($f)|not)])"
  elif ([("aes","avx","avx2","sse4_1")|select(. as $f|($flags|index($f)) and ($c.features|index($f)|not))]|length) > 0
    then "features missing task-named flags this CPU has: \([("aes","avx","avx2","sse4_1")|select(. as $f|($flags|index($f)) and ($c.features|index($f)|not))])"
  else true end'

# ---------------------------------------------------------------- 4. storage
sysdevs=$(lsblk -J -b -o NAME,TYPE,SIZE,ROTA,TRAN 2>/dev/null | jq -c '
   [ (.blockdevices // [])[] | (. + {top:true}), ((.. | objects | .children? // [] | .[]) + {top:false}) ]
   | map({key: .name, value: {type, size: ((.size // 0)|tonumber? // 0), rota, tran, top}})
   | reverse | from_entries' 2>/dev/null)
[ -z "$sysdevs" ] && sysdevs='{}'
echo "system block devices: $sysdevs"
jqcheck "storage section" "$OUT" --argjson sd "$sysdevs" '
  .storage as $s
  | def realdisk($n; $d): $d.top and $d.type == "disk" and ($n | test("^(loop|ram|zram)") | not);
    def exptype($n; $d): if ($n|startswith("nvme")) or ($d.tran == "nvme") then "nvme"
                         elif ($d.rota == true or $d.rota == 1 or $d.rota == "1") then "hdd"
                         elif ($d.rota == false or $d.rota == 0 or $d.rota == "0") then "ssd" else null end;
    def nm: ltrimstr("/dev/");
  if ($s|has("devices") and has("total_capacity_gb") and has("type_summary"))|not then "storage keys missing"
  elif ($s.devices|type) != "array" then "devices not an array"
  elif ([$s.devices[]|select(type != "object" or (has("name") and has("type") and has("size_gb")|not)
         or (.name|type) != "string" or (.type|type) != "string")]|length) > 0 then "device entries need string name, string type, size_gb"
  elif ([$s.devices[]|select($sd[.name|nm] == null)]|length) > 0 then "devices not on this system: \([$s.devices[]|.name|select($sd[nm] == null)])"
  elif ([$s.devices[]|(.name|nm) as $n|select(realdisk($n; $sd[$n]) and exptype($n; $sd[$n]) != null and (.type|ascii_downcase) != exptype($n; $sd[$n]))]|length) > 0
    then "wrong device type: \([$s.devices[]|(.name|nm) as $n|select(realdisk($n; $sd[$n]) and (.type|ascii_downcase) != exptype($n; $sd[$n]))|{name, type, expected: exptype($n; $sd[$n])}])"
  elif ([$s.devices[]|select((.size_gb|type) != "number")]|length) > 0 then "size_gb must be a number"
  elif ([$s.devices[]|$sd[.name|nm].size as $b|select(.size_gb < ($b/1073741824 - 1) or .size_gb > ($b/1000000000 + 1))]|length) > 0
    then "size_gb inconsistent with device size: \([$s.devices[]|$sd[.name|nm].size as $b|select(.size_gb < ($b/1073741824 - 1) or .size_gb > ($b/1000000000 + 1))|{name, size_gb, bytes: $b}])"
  elif ([$sd|to_entries[]|select(realdisk(.key; .value))|.key|select(. as $k|[$s.devices[]|.name|nm]|index($k)|not)]|length) > 0
    then "disks missing from devices: \([$sd|to_entries[]|select(realdisk(.key; .value))|.key|select(. as $k|[$s.devices[]|.name|nm]|index($k)|not)])"
  elif ($s.total_capacity_gb|type) != "number" or $s.total_capacity_gb < 0 then "total_capacity_gb not a non-negative number"
  elif ((($s.total_capacity_gb - ([$s.devices[].size_gb]|add // 0))|fabs) > ([1, ($s.devices|length)]|max))
    then "total_capacity_gb \($s.total_capacity_gb) != sum of device sizes \([$s.devices[].size_gb]|add // 0)"
  elif ($s.type_summary|type) != "object" then "type_summary not an object"
  elif ([("ssd","hdd","nvme") as $k|select(($s.type_summary[$k]|type) != "number" or $s.type_summary[$k] != ([$s.devices[]|select((.type|ascii_downcase) == $k)]|length))]|length) > 0
    then "type_summary \($s.type_summary) does not match the listed devices"
  else true end'

# ---------------------------------------------------------------- 5. timestamp
ts=$(jq -r '.audit_timestamp | if type=="string" then . else "<not a string>" end' "$OUT")
now=$(date -u +%s)
if [[ "$ts" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:?[0-9]{2})?$ ]]; then
    ok=0
    for z in "" UTC; do
        if [ -n "$z" ]; then e=$(TZ=$z date -d "$ts" +%s 2>/dev/null); else e=$(date -d "$ts" +%s 2>/dev/null); fi
        [ -n "$e" ] && [ $(( e > now ? e - now : now - e )) -lt 900 ] && ok=1
    done
    [ $ok = 1 ] && pass "timestamp ISO 8601 and current ($ts)" || fail "timestamp $ts is not the time of the audit"
else
    fail "timestamp not ISO 8601: $ts"
fi

# ---------------------------------------------------------------- 6. graceful handling
if [ -f "$GRACEFUL" ] && jq -e 'type=="object" and has("usb") and has("cpu") and has("storage") and has("audit_timestamp") and ((.usb.has_usb3|type)=="boolean")' "$GRACEFUL" >/dev/null 2>&1; then
    pass "valid output when lsusb is unavailable"
else
    fail "no valid JSON output when lsusb fails"
fi

echo "FAILS=$FAILS"
[ $FAILS -eq 0 ]
