#!/bin/bash
# Verifier for /app/shell_report.txt (pure bash + grep/sed, no python needed).
REPORT=/app/shell_report.txt
mkdir -p /logs/verifier
FAILS=0
fail(){ echo "FAIL: $*"; FAILS=$((FAILS+1)); }
finish(){ if [ "$FAILS" -eq 0 ]; then echo PASS; echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi; exit 0; }

if [ ! -f "$REPORT" ] || [ ! -r "$REPORT" ]; then fail "report missing"; finish; fi
if ! grep -q '[^[:space:]]' "$REPORT"; then fail "report empty"; finish; fi

# tokenise a line: split on whitespace and separators, strip leading/trailing . * -
tokens(){ printf '%s\n' "$1" | tr -s ' \t,;:()[]{}'"'"'"`|=<>' '\n' | sed -e 's/^[.*-]*//' -e 's/[.*-]*$//' -e 's#/*$##' | grep -v '^$'; }

# Expected login shells: the grading container's /etc/shells (same image as the solver's).
mapfile -t EXPECTED < <(grep -v '^[[:space:]]*#' /etc/shells | sed 's/[[:space:]]//g' | grep '^/' | awk '!s[$0]++')
EXP=()
for p in "${EXPECTED[@]}"; do [ -x "$p" ] && EXP+=("$p"); done
if [ "${#EXP[@]}" -eq 0 ]; then fail "grading /etc/shells empty"; finish; fi

# All path tokens in the report
ALLTOK=$(while IFS= read -r l || [ -n "$l" ]; do tokens "$l"; done < "$REPORT")

# 1. every login shell listed by absolute path (exact token)
for p in "${EXP[@]}"; do
  grep -qxF -- "$p" <<<"$ALLTOK" || fail "shell $p from /etc/shells not listed by absolute path"
done

# 2. verification that shells exist as executable files
grep -qi 'exist' "$REPORT" && grep -qi 'executable' "$REPORT" || fail "no statement that shells exist as executable files"
# no false claim that an installed shell is missing / not executable
NEG='not[[:space:]]+(found|exist|present|executable|an?[[:space:]]+executable)|does[[:space:]]*n.?t[[:space:]]+exist|missing|non-?executable|fail|no[[:space:]]+such'
while IFS= read -r l || [ -n "$l" ]; do
  if grep -qiE "$NEG" <<<"$l"; then
    lt=$(tokens "$l")
    for p in "${EXP[@]}"; do grep -qxF -- "$p" <<<"$lt" && fail "report falsely says $p missing/not executable: $l"; done
  fi
done < "$REPORT"

# 3. current shell identified as bash ($0, $SHELL and /proc/$$/exe all give bash here)
mapfile -t LINES < "$REPORT"
idx=-1
for i in "${!LINES[@]}"; do
  if grep -qiE 'current[[:space:]]+shell|\$0' <<<"${LINES[$i]}"; then idx=$i; break; fi
done
NAMES=' bash sh dash zsh ksh ksh93 rbash rksh rksh93 mksh tcsh csh fish tmux screen busybox '
if [ "$idx" -lt 0 ]; then
  fail "no current-shell entry"
else
  found=""; seen=0
  for ((j=idx; j<idx+6 && j<${#LINES[@]}; j++)); do
    l=${LINES[$j]}
    if [ "$j" -gt "$idx" ]; then if grep -q "[^[:space:]]" <<<"$l"; then seen=1; elif [ $seen -eq 1 ]; then break; else continue; fi; fi
    while IFS= read -r t; do
      b=${t##*/}; b=${b#-}; b=${b,,}
      [ -n "$b" ] && [[ "$NAMES" == *" $b "* ]] && { found=$b; break; }
    done < <(tokens "$l")
    [ -n "$found" ] && break
  done
  [ "$found" = bash ] || fail "current shell not identified as bash (got '${found:-nothing}')"
fi
finish
