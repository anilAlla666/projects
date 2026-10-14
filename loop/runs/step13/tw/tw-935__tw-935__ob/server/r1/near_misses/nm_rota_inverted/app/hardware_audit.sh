#!/bin/bash
OUT=/tmp/hardware_audit.json
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
declare -A cnt=( ["1.0"]=0 ["1.1"]=0 ["2.0"]=0 ["3.0"]=0 ["3.1"]=0 ["3.2"]=0 )
for d in /sys/bus/usb/devices/usb*; do
  [ -f "$d/version" ] || continue
  v=$(tr -d ' ' < "$d/version"); v=$(awk -v v="$v" 'BEGIN{printf "%.1f", v}')
  cnt[$v]=$(( ${cnt[$v]:-0} + 1 ))
done
max=null; has=false
for v in 1.0 1.1 2.0 3.0 3.1 3.2; do [ "${cnt[$v]}" -gt 0 ] && max="\"$v\""; done
case "$max" in \"3.*) has=true;; esac
hubs=$(for v in "${!cnt[@]}"; do echo "$v ${cnt[$v]}"; done | jq -Rn '[inputs|split(" ")|{(.[0]):(.[1]|tonumber)}]|add')
model=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
cps=$(lscpu | sed -n 's/^Core(s) per socket:[[:space:]]*//p'); so=$(lscpu | sed -n 's/^Socket(s):[[:space:]]*//p')
thr=$(lscpu | sed -n 's/^CPU(s):[[:space:]]*//p')
flags=$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2)
feat=$(echo $flags | tr ' ' '\n' | jq -R . | jq -s 'map(select(length>0))')
stor=$(lsblk -J -b -d -o NAME,TYPE,SIZE,ROTA | jq '[.blockdevices[]|select(.type=="disk")|{name, type:(if (.name|startswith("nvme")) then "nvme" elif .rota then "ssd" else "hdd" end), size_gb:((.size/1e9*100|round)/100)}] as $d | {devices:$d, total_capacity_gb:($d|map(.size_gb)|add // 0), type_summary:{ssd:($d|map(select(.type=="ssd"))|length),hdd:($d|map(select(.type=="hdd"))|length),nvme:($d|map(select(.type=="nvme"))|length)}}')
jq -n --argjson max "$max" --argjson hubs "$hubs" --argjson has $has --arg model "${model:-unknown}" --argjson cores $((cps*so)) --argjson thr "$thr" --argjson feat "$feat" --argjson stor "$stor" --arg ts "$ts" \
 '{usb:{max_version:$max,root_hubs:$hubs,has_usb3:$has},cpu:{model:$model,cores:$cores,threads:$thr,features:$feat},storage:$stor,audit_timestamp:$ts}' > $OUT
