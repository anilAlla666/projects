#!/bin/bash
# Verifier for the bash redirection debugging task. Pure bash + coreutils/diffutils,
# no network, no python needed.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS+1)); }
pass() { echo "PASS: $*"; }

S=/app/process_data.sh
EXP=/app/expected_output.txt
BK=$(mktemp -d)

restore() {
  for f in input_a.txt input_b.txt expected_output.txt; do
    [ -f "$BK/$f" ] && cp -f "$BK/$f" "/app/$f"
  done
}
trap restore EXIT
for f in input_a.txt input_b.txt expected_output.txt; do cp -f "/app/$f" "$BK/$f"; done

# Expected config lines (task: config_summary with parsed configuration values;
# the values are those of the script's here-document / expected_output.txt)
CFG_LINES=("APP_NAME: DataProcessor" "VERSION: 2.0" "MODE: production" "DEBUG: false")
check_cfg() {  # $1 = file, $2 = label
  local f=$1 ok=1
  if [ ! -s "$f" ]; then fail "$2: $f missing or empty"; return; fi
  for l in "${CFG_LINES[@]}"; do
    grep -qxF -- "$l" "$f" || { ok=0; fail "$2: '$l' not found in $f"; }
  done
  [ $ok = 1 ] && pass "$2: config summary has all configuration values"
}

# 1. script exists, bash shebang
if [ ! -f "$S" ]; then
  fail "script missing"; echo "FAILED"; exit 0
fi
first=$(head -n1 "$S" | tr -d '\r' | sed 's/[[:space:]]*$//')
[ "$first" = "#!/bin/bash" ] && pass "shebang" || fail "shebang is '$first'"

# 2. carried-over results
if [ -f /app/output.txt ] && cmp -s /app/output.txt "$EXP"; then pass "submitted output.txt matches expected"
else fail "submitted /app/output.txt missing or differs from expected"; fi
check_cfg /app/config_summary.txt "submitted"

# 3. structural: the redirection operators the task is about are actually used (non-comment code)
CODE=$(grep -v '^[[:space:]]*#' "$S")
grep -q '<<<' <<< "$CODE" && pass "here-string used" || fail "no here-string (<<<) in code"
grep -Eq '(^|[^<])<<-?[[:space:]]*['"'"'"]?[A-Za-z_]' <<< "$CODE" && pass "here-document used" || fail "no here-document in code"
grep -q '<(' <<< "$CODE" && pass "process substitution used" || fail "no process substitution <( ) in code"
grep -q 'expected_output' <<< "$CODE" && fail "script references expected_output.txt"
grep -Eq '(^|[^0-9.])84([^0-9.]|$)' <<< "$CODE" && fail "doubled value 84 hard-coded in script"
grep -Eq 'banana|blueberry' <<< "$CODE" && fail "comparison data hard-coded in script"

# 4. re-run the script fresh on the original inputs
run_script() {
  rm -f /app/output.txt /app/config_summary.txt
  ( cd /app && timeout 60 /bin/bash "$S" </dev/null ) >"$BK/stdout" 2>"$BK/stderr"
  echo $?
}
rc=$(run_script)
[ "$rc" = 0 ] && pass "script exits 0" || fail "script exit code $rc"
if [ -s "$BK/stderr" ]; then fail "script wrote to stderr: $(head -c 500 "$BK/stderr")"; else pass "no stderr"; fi
if [ -f /app/output.txt ] && cmp -s /app/output.txt "$EXP"; then pass "regenerated output.txt matches expected"
else fail "regenerated output.txt differs"; diff /app/output.txt "$EXP" 2>&1 | head -30; fi
check_cfg /app/config_summary.txt "regenerated"

# 5. generality: different input data, expected_output.txt hidden
printf 'kiwi\nlemon\nmango\npapaya\n' > /app/input_a.txt
printf 'kiwi\nmango\nnectarine\npapaya\nquince\n' > /app/input_b.txt
rm -f "$EXP"
NEWDIFF=$(diff /app/input_a.txt /app/input_b.txt)
{
  sed -n '1,/^=== Data Comparison ===$/p' "$BK/expected_output.txt"
  printf '%s\n' "$NEWDIFF"
  printf '\n=== Processing Complete ===\n'
} > "$BK/expected_alt.txt"
rc=$(run_script)
[ "$rc" = 0 ] && pass "alt-input run exits 0" || fail "alt-input run exit code $rc"
if [ -f /app/output.txt ] && cmp -s /app/output.txt "$BK/expected_alt.txt"; then pass "output reflects actual input data"
else fail "output does not reflect changed input data"; diff /app/output.txt "$BK/expected_alt.txt" 2>&1 | head -30; fi
check_cfg /app/config_summary.txt "alt-input"

restore
# leave fresh outputs from original inputs in place
( cd /app && timeout 60 /bin/bash "$S" </dev/null ) >/dev/null 2>&1

if [ $FAILS = 0 ]; then echo 1 > /logs/verifier/reward.txt; echo "ALL PASSED"; else echo "FAILED ($FAILS)"; fi
exit 0
