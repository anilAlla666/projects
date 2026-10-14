#!/bin/bash
OUT=/tmp/hardware_audit.json
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hubs='{}'
for d in /sys/bus/usb/devices/*; do
  [ -f "$d/version" ] || continue
  v=$(tr -d ' ' < "$d/version"); v=$(awk -v v="$v" 'BEGIN{printf "%.1f", v}')
  hubs=$(jq -c --arg v "$v" '.[$v] = ((.[$v] // 0) + 1)' <<<"$hubs")
done
max=$(jq -c 'keys | map(tonumber) | max | if . == null then null else (. * 10 | round / 10 | tostring | if test("\\.") then . else . + ".0" end) end' <<<"$hubs")
has3=$(jq 'keys | map(tonumber) | any(. >= 3)' <<<"$hubs")
model=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1); model=${model:-unknown}
threads=$(lscpu | awk -F: '/^CPU\(s\):/{gsub(/ /,"",$2);print $2;exit}')
cps=$(lscpu | awk -F: '/^Core\(s\) per socket:/{gsub(/ /,"",$2);print $2;exit}')
sk=$(lscpu | awk -F: '/^Socket\(s\):/{gsub(/ /,"",$2);print $2;exit}')
cores=$(( ${cps:-1} * ${sk:-1} ))
feat=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo | cut -d: -f2- | tr -s ' ' '\n' | grep -v '^$' | jq -R . | jq -cs .)
storage=$(lsblk -d -b -J -o NAME,TYPE,SIZE,ROTA,TRAN | jq -c '[.blockdevices[] | select(.type=="disk") | {name, type:(if (.name|startswith("nvme")) or .tran=="nvme" then "nvme" elif .rota then "hdd" else "ssd" end), size_gb:(.size/1000000000|round)}] as $d | {devices:$d, total_capacity_gb:($d|map(.size_gb)|add // 0), type_summary:{ssd:($d|map(select(.type=="ssd"))|length),hdd:($d|map(select(.type=="hdd"))|length),nvme:($d|map(select(.type=="nvme"))|length)}}')
jq -n --argjson hubs "$hubs" --argjson max "$max" --argjson has3 "$has3" --arg model "$model" --argjson cores "$cores" --argjson threads "${threads:-null}" --argjson feat "$feat" --argjson st "$storage" --arg ts "$ts" \
 '{usb:{max_version:$max, root_hubs:$hubs, has_usb3:$has3}, cpu:{model:$model,cores:$cores,threads:$threads,features:$feat}, storage:$st, audit_timestamp:$ts}' > $OUT
