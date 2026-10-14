#!/bin/bash
# Verifier for the nginx package audit report task.
# Pure bash/awk/grep + dpkg (no network, no python needed).
# Ground truth is read live from dpkg on the grading machine.

REPORT="${REPORT_PATH:-/home/user/package_audit_report.txt}"
REWARD_DIR="${REWARD_DIR:-/logs/verifier}"
mkdir -p "$REWARD_DIR" 2>/dev/null
FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS+1)); }
pass() { echo "PASS: $*"; }
finish() {
  if [ "$FAILS" -eq 0 ]; then echo 1 > "$REWARD_DIR/reward.txt"; echo "RESULT: 1";
  else echo 0 > "$REWARD_DIR/reward.txt"; echo "RESULT: 0 ($FAILS failures)"; fi
  exit 0
}

# ---------- 1. report exists ----------
if [ ! -f "$REPORT" ] || [ ! -s "$REPORT" ]; then
  fail "report missing or empty at $REPORT"; finish
fi
pass "report exists"
TMP=$(mktemp -d 2>/dev/null || mktemp -d -p /dev/shm 2>/dev/null || mktemp -d -p "$REWARD_DIR" 2>/dev/null)
if [ -z "$TMP" ] || [ ! -d "$TMP" ]; then fail "cannot create temp dir"; finish; fi
tr -d '\r' < "$REPORT" > "$TMP/report"
R="$TMP/report"

# ---------- ground truth ----------
dpkg -L nginx 2>/dev/null | grep '^/' | grep -v '^/\.$' | sort -u > "$TMP/nginx.list"
dpkg -L nginx-common 2>/dev/null | grep '^/' | grep -v '^/\.$' | sort -u > "$TMP/common.list"
if [ ! -s "$TMP/nginx.list" ]; then fail "nginx not installed on grader"; finish; fi
cat "$TMP/nginx.list" "$TMP/common.list" | sort -u > "$TMP/all.list"
: > "$TMP/dirs.list"; : > "$TMP/nginx.files"
while read -r p; do [ -d "$p" ] && [ ! -L "$p" ] && echo "$p" >> "$TMP/dirs.list"; done < "$TMP/all.list"
while read -r p; do [ -d "$p" ] || echo "$p" >> "$TMP/nginx.files"; done < "$TMP/nginx.list"
printf '/\n/etc\n/usr\n/usr/lib\n/usr/share\n' >> "$TMP/dirs.list"; sort -u -o "$TMP/dirs.list" "$TMP/dirs.list"
conff() { dpkg-query -W -f='${Conffiles}\n' "$1" 2>/dev/null | awk 'NF>=2 && $1 ~ /^\// && $NF != "obsolete" {print $1}'; }
conff nginx | sort -u > "$TMP/cf_nginx"
{ conff nginx; conff nginx-common; } | sort -u > "$TMP/cf_both"
DEPENDS="$(dpkg-query -W -f='${Pre-Depends}, ${Depends}' nginx 2>/dev/null)"

# normalise a path token: strip trailing punctuation/slash, usrmerge aliases
norm() {
  sed -E 's/[]),;:.!?"'"'"'`*]+$//; s#(.)/+$#\1#; s#^/lib(/|$)#/usr/lib\1#; s#^/bin(/|$)#/usr/bin\1#; s#^/sbin(/|$)#/usr/sbin\1#'
}

norm < "$TMP/dirs.list" | sort -u > "$TMP/dirs.norm"

# ---------- 2. sections ----------
# header line: keyword at start of line (after decoration / numbering)
hdr_line() {
  local kw="$1"
  awk -v kw="$kw" '
    { l=tolower($0); s=l; sub(/^[^a-z0-9]*([0-9]+[.)])?[^a-z]*/, "", s) }
    index(s, kw)==1 && substr(s, length(kw)+1, 1) !~ /[a-z]/ {
      if (l ~ /section/) { if (!sec) sec=NR } else { if (!first) first=NR }
    }
    END { if (sec) print sec; else if (first) print first; else print 0 }' "$R"
}
declare -A START
for kw in files dependencies scripts conffiles services; do
  START[$kw]=$(hdr_line "$kw")
  if [ "${START[$kw]}" -eq 0 ]; then fail "no section header for '$kw'"; fi
done
[ "$FAILS" -gt 0 ] && finish
pass "all five section headers present"
TOTAL=$(wc -l < "$R"); TOTAL=$((TOTAL+1))
section() {   # print the text of a section
  local kw="$1" s="${START[$1]}" e=$((TOTAL+1)) k
  for k in files dependencies scripts conffiles services; do
    [ "${START[$k]}" -gt "$s" ] && [ "${START[$k]}" -lt "$e" ] && e=${START[$k]}
  done
  sed -n "${s},$((e-1))p" "$R"
}
for kw in files dependencies scripts conffiles services; do section "$kw" > "$TMP/sec_$kw"; done

# ---------- 3. FILES ----------
F="$TMP/sec_files"
# annotate every line with its current category (from sub-headers)
awk '
  { l=tolower($0); s=l; sub(/^[^a-z]*/, "", s)
    if (l !~ /^[ \t*-]*\//) {
      if (s ~ /^(binar|executable)/) cat="bin";
      else if (s ~ /^config/) cat="conf";
      else if (s ~ /^librar/) cat="lib";
      else if (s ~ /^(doc|man)/) cat="doc";
      else if (s ~ /^other/) cat="other";
    }
    print (cat==""?"none":cat) "\t" $0 }' "$F" > "$TMP/files_cat"
# line-start path tokens  -> "cat<TAB>path"
awk -F'\t' '{ line=$2; sub(/^[ \t]*([-*+•]|[0-9]+[.)])?[ \t]*/, "", line);
  if (substr(line,1,1)=="/") { split(line, a, /[ \t]/); print $1 "\t" a[1] } }' "$TMP/files_cat" > "$TMP/files_tok_raw"
cut -f1 "$TMP/files_tok_raw" > "$TMP/c1"; cut -f2 "$TMP/files_tok_raw" | norm > "$TMP/c2"
paste "$TMP/c1" "$TMP/c2" > "$TMP/files_tok"
want_cat() {
  case "$1" in
    /usr/sbin/*|/usr/bin/*|/sbin/*|/bin/*|/usr/games/*) echo bin;;
    /etc/*) echo conf;;
    /usr/share/doc/*|/usr/share/man/*) echo doc;;
    /usr/lib/*) echo lib;;
    *) echo other;;
  esac
}
ok=1
while read -r p; do
  q=$(printf '%s\n' "$p" | norm)
  if ! awk -F'\t' -v q="$q" '$2==q {f=1} END{exit !f}' "$TMP/files_tok"; then
    fail "FILES section does not list nginx file $p"; ok=0; continue
  fi
  c=$(want_cat "$q")
  if ! awk -F'\t' -v q="$q" -v c="$c" '$2==q && $1==c {f=1} END{exit !f}' "$TMP/files_tok"; then
    fail "nginx file $p is not listed under the '$c' category"; ok=0
  fi
done < "$TMP/nginx.files"
norm < "$TMP/all.list" | sort -u > "$TMP/all.norm"
while IFS=$'\t' read -r c p; do
  [ -z "$p" ] || [ "$p" = "/" ] || [ "$p" = "/." ] && continue
  if ! grep -qxF -- "$p" "$TMP/all.norm"; then
    fail "FILES section lists $p which is not installed by nginx/nginx-common"; ok=0
  fi
done < "$TMP/files_tok"
# the set of non-directory files listed must be exactly nginx's files, or
# nginx + nginx-common files (both readings of "the nginx package" accepted)
while read -r p; do [ -d "$p" ] || echo "$p"; done < "$TMP/common.list" > "$TMP/common.files"
# symlinks (e.g. the dangling /usr/share/nginx/modules) may be listed or omitted
while read -r p; do [ -L "$p" ] && echo "$p"; done < "$TMP/all.list" | norm | sort -u > "$TMP/links"
echo "/nonexistent-placeholder" >> "$TMP/links"
norm < "$TMP/nginx.files" | sort -u | grep -vxF -f "$TMP/links" > "$TMP/fs_strict"
cat "$TMP/nginx.files" "$TMP/common.files" | norm | sort -u | grep -vxF -f "$TMP/links" > "$TMP/fs_broad"
cut -f2 "$TMP/files_tok" | grep -vxF -f "$TMP/dirs.norm" | grep -v "^/\.\?$" | grep -v "^$" | grep -vxF -f "$TMP/links" | sort -u > "$TMP/fs_rep"
if ! cmp -s "$TMP/fs_rep" "$TMP/fs_strict" && ! cmp -s "$TMP/fs_rep" "$TMP/fs_broad"; then
  fail "FILES section file set is neither all nginx files nor all nginx+nginx-common files (partial list)"; ok=0
  diff "$TMP/fs_rep" "$TMP/fs_broad" | head -20
fi
[ $ok = 1 ] && pass "FILES section complete, categorised, no foreign paths"

# ---------- 4. DEPENDENCIES ----------
D="$TMP/sec_dependencies"
ere_escape() { printf '%s' "$1" | sed -E 's/[][\.^$*+?(){}|\/]/\\&/g'; }
ok=1; n=0
IFS=',' read -ra ITEMS <<< "$DEPENDS"
for it in "${ITEMS[@]}"; do
  it=$(echo "$it" | sed -E 's/^ +| +$//g'); [ -z "$it" ] && continue
  it=${it%%|*}; it=$(echo "$it" | sed -E 's/ +$//')
  name=$(echo "$it" | sed -E 's/^([^ (:]+).*/\1/')
  n=$((n+1))
  nre="(^|[^a-z0-9.+-])$(ere_escape "$name")([^a-z0-9.+-]|$)"
  if echo "$it" | grep -q '('; then
    op=$(echo "$it" | sed -E 's/.*\(([<>=]+) *([^)]+)\).*/\1/')
    ver=$(echo "$it" | sed -E 's/.*\(([<>=]+) *([^)]+)\).*/\2/')
    vre="$(ere_escape "$op")[[:space:]]*$(ere_escape "$ver")([^0-9A-Za-z.+~:-]|$)"
    if ! grep -E -- "$nre" "$D" | grep -qE -- "$vre"; then
      fail "DEPENDENCIES: $name with requirement ($op $ver) not stated"; ok=0
    fi
  else
    if ! grep -qE -- "$nre" "$D"; then fail "DEPENDENCIES: $name missing"; ok=0; fi
  fi
done
[ $n -eq 0 ] && { fail "could not read nginx Depends from dpkg"; ok=0; }
[ $ok = 1 ] && pass "DEPENDENCIES: all $n direct dependencies with version requirements"

# ---------- 5. SCRIPTS ----------
S="$TMP/sec_scripts"
if grep -qi 'postinst' "$S" && grep -qi 'nginx' "$S"; then pass "SCRIPTS: postinst listed"
else fail "SCRIPTS: no nginx postinst listed"; fi
if grep -qiE 'start|reload|upgrade|invoke-rc|port 80|systemd|service|daemon|enable' "$S"; then
  pass "SCRIPTS: summary present"
else fail "SCRIPTS: no summary of what the postinst does"; fi

# ---------- 6. CONFFILES ----------
C="$TMP/sec_conffiles"
grep -oE '/[A-Za-z0-9_.+/~-]+' "$C" | norm | grep -v '^/var/lib/dpkg' | grep -v '^/\.\?$' | sort -u > "$TMP/cf_tok0"
grep -vxF -f "$TMP/dirs.norm" "$TMP/cf_tok0" > "$TMP/cf_rep" || true
norm < "$TMP/cf_nginx" | sort -u > "$TMP/cfn"; norm < "$TMP/cf_both" | sort -u > "$TMP/cfb"
if cmp -s "$TMP/cf_rep" "$TMP/cfn" || cmp -s "$TMP/cf_rep" "$TMP/cfb"; then
  pass "CONFFILES: matches dpkg conffiles ($(wc -l < "$TMP/cf_rep") paths)"
else
  fail "CONFFILES: listed set does not match dpkg conffiles of nginx (or nginx+nginx-common)"
  diff "$TMP/cf_rep" "$TMP/cfb" | head -20
fi

# ---------- 7. SERVICES ----------
V="$TMP/sec_services"
if grep -q 'nginx\.service' "$V"; then pass "SERVICES: nginx.service listed"; else fail "SERVICES: nginx.service not listed"; fi
if grep -qiE 'status|active|running|enabled|disabled|dead' "$V"; then pass "SERVICES: status addressed"; else fail "SERVICES: no service status statement"; fi

rm -rf "$TMP"
finish
