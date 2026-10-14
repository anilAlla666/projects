#!/bin/bash
# Verifier for /app/shell_report.txt. Pure bash + grep (no network, no python needed).
REPORT="${REPORT_PATH:-/app/shell_report.txt}"
REWARD_DIR="${REWARD_DIR:-/logs/verifier}"
mkdir -p "$REWARD_DIR"

# /etc/shells entries produced by the task image (ubuntu:24.04 + bash dash zsh ksh).
# Fixed list: extra entries the solver's harness may add (e.g. tmux) are allowed but not required.
EXPECTED="/bin/sh /usr/bin/sh /bin/bash /usr/bin/bash /bin/rbash /usr/bin/rbash /usr/bin/dash
/bin/ksh93 /usr/bin/ksh93 /bin/rksh93 /usr/bin/rksh93 /bin/zsh /usr/bin/zsh"
SHELLS_RE='(sh|bash|dash|zsh|zsh5|ksh|ksh93|rbash|rksh|rksh93|rzsh|mksh|tcsh|csh|fish)'
NEG_RE="not[[:space:]]+(found|exist|executable|present|installed|available)|does[[:space:]]*n[o']t|doesn't|missing|non-?existent|fail|invalid|✗|❌|\[[[:space:]]*[xX][[:space:]]*\]"

fails=0
fail() { echo "FAIL: $*"; fails=1; }

finish() {
  if [ "$fails" -eq 0 ]; then echo PASS; echo 1 > "$REWARD_DIR/reward.txt"
  else echo 0 > "$REWARD_DIR/reward.txt"; fi
  exit 0
}

if [ ! -f "$REPORT" ]; then fail "report $REPORT missing"; finish; fi
if ! grep -q '[^[:space:]]' "$REPORT"; then fail "report empty"; finish; fi

# path match: not preceded/followed by path characters
path_lines() { grep -E "(^|[^A-Za-z0-9_./-])$(printf '%s' "$1" | sed 's/[./]/\\&/g')([^A-Za-z0-9_/-]|\.([^A-Za-z0-9_/-]|$)|$)" "$REPORT"; }

# 1. all /etc/shells entries listed by absolute path
# 2. none of them reported as missing / not executable
for p in $EXPECTED; do
  l="$(path_lines "$p")"
  if [ -z "$l" ]; then fail "shell path $p (from /etc/shells) not listed"; continue; fi
  if printf '%s\n' "$l" | grep -qiE "$NEG_RE"; then fail "report says existing shell $p is missing/not executable"; fi
done

# 3. existence + executability verification stated
grep -qi 'exist' "$REPORT" || fail "no statement that shells exist"
grep -qi 'executable' "$REPORT" || fail "no statement that shells are executable"

# 4. current shell identified and it is an actual shell
is_shell_text() { printf '%s\n' "$1" | grep -qE "(^|[^A-Za-z0-9_.-])-?$SHELLS_RE([^A-Za-z0-9_./-]|\.?$|\.[^A-Za-z0-9_/-])"; }
label=0; named=0
mapfile -t L < "$REPORT"
n=${#L[@]}
for ((i=0;i<n;i++)); do
  line="${L[$i]}"
  if printf '%s\n' "$line" | grep -qiE 'current[[:space:]]+shell|\$0'; then
    label=1
    after="$(printf '%s\n' "$line" | sed -E 's/.*([Cc][Uu][Rr][Rr][Ee][Nn][Tt][[:space:]]+[Ss][Hh][Ee][Ll][Ll]|\$0)//')"
    if is_shell_text "$after"; then named=1; break; fi
    for ((j=i+1;j<i+4 && j<n;j++)); do
      nl="${L[$j]}"
      [[ "$nl" =~ ^[[:space:]=*#_-]*$ ]] && continue
      if is_shell_text "$nl"; then named=1; fi
      break
    done
    [ $named -eq 1 ] && break
  fi
done
[ $label -eq 1 ] || fail "no 'current shell' / \$0 entry"
[ $label -eq 1 ] && [ $named -eq 0 ] && fail "current shell entry does not name a shell"

finish
