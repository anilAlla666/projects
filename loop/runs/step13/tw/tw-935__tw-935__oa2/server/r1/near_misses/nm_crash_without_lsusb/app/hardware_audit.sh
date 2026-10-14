#!/bin/bash
set -e
usb_raw=$(lsusb)
OUT=/tmp/hardware_audit.json
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
declare -A hubs=( ["1.0"]=0 ["1.1"]=0 ["2.0"]=0 ["3.0"]=0 ["3.1"]=0 ["3.2"]=0 )
found=0
for d in /sys/bus/usb/devices/usb*; do
  [ -r "$d/version" ] || continue
  v=$(tr -d ' ' < "$d/version"); v=${v%0}   # 2.00 -> 2.0, 3.10 -> 3.1
  hubs[$v]=$(( ${hubs[$v]:-0} + 1 )); found=1
done
if [ $found -eq 0 ] && command -v lsusb >/dev/null 2>&1; then
  while read -r id; do
    case "$id" in 1d6b:0001) v=1.1;; 1d6b:0002) v=2.0;; 1d6b:0003) v=3.0;; *) continue;; esac
    hubs[$v]=$(( hubs[$v] + 1 )); found=1
  done < <(lsusb 2>/dev/null | awk '{print $6}')
fi
rh=$(for k in "${!hubs[@]}"; do echo "$k ${hubs[$k]}"; done | jq -Rn '[inputs|split(" ")|{(.[0]):(.[1]|tonumber)}]|add')
max=$(echo "$rh" | jq '[to_entries[]|select(.value>0)|.key]|sort_by(tonumber)|last // null')
has3=$(echo "$max" | jq 'if .==null then false else (tonumber>=3) end')
model=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1); model=${model:-unknown}
cps=$(lscpu | sed -n 's/^Core(s) per socket:[[:space:]]*//p'); sk=$(lscpu | sed -n 's/^Socket(s):[[:space:]]*//p')
cores=$(( ${cps:-1} * ${sk:-1} )); threads=$(lscpu | sed -n 's/^CPU(s):[[:space:]]*//p' | head -1)
feat=$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2 | jq -R 'split(" ")|map(select(length>0))')
[ -z "$feat" ] && feat='[]'
st=$(lsblk -J -b -d -o NAME,TYPE,SIZE,ROTA 2>/dev/null | jq '[.blockdevices[]|select(.type=="disk")|{name, type:(if (.name|startswith("nvme")) then "nvme" elif .rota then "hdd" else "ssd" end), size_gb:((.size/1073741824)|floor)}] as $d | {devices:$d,total_capacity_gb:($d|map(.size_gb)|add // 0),type_summary:{ssd:($d|map(select(.type=="ssd"))|length),hdd:($d|map(select(.type=="hdd"))|length),nvme:($d|map(select(.type=="nvme"))|length)}}')
[ -z "$st" ] && st='{"devices":[],"total_capacity_gb":0,"type_summary":{"ssd":0,"hdd":0,"nvme":0}}'
jq -n --argjson rh "$rh" --argjson max "$max" --argjson h3 "$has3" --arg m "$model" --argjson c "$cores" --argjson t "${threads:-null}" --argjson f "$feat" --argjson s "$st" --arg ts "$ts" \
 '{usb:{max_version:$max,root_hubs:$rh,has_usb3:$h3},cpu:{model:$m,cores:$c,threads:$t,features:$f},storage:$s,audit_timestamp:$ts}' > $OUT
