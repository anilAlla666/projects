#!/bin/bash
# Self-contained verifier (bash + coreutils + diff + bc only; no network).
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

APP=/app
SCRIPT=$APP/process_data.sh
OUT=$APP/output.txt
SUMMARY=$APP/config_summary.txt
EXPECTED=$APP/expected_output.txt
IN_A=$APP/input_a.txt
IN_B=$APP/input_b.txt
FAILS=0
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILS=$((FAILS+1)); }
chk() { local msg=$1; shift; if "$@"; then pass "$msg"; else fail "$msg"; fi; }

W=$(mktemp -d)
cat > "$W/expected" <<'EXP'
=== Configuration Summary ===
APP_NAME: DataProcessor
VERSION: 2.0
MODE: production
DEBUG: false

=== Transformed Values ===
Input value: 42
Doubled value: 84

=== Data Comparison ===
2c2
< banana
---
> blueberry

=== Processing Complete ===
EXP
printf 'apple\nbanana\ncherry\ndate\n' > "$W/orig_a"
printf 'apple\nblueberry\ncherry\ndate\n' > "$W/orig_b"
restore() { cp "$W/orig_a" "$IN_A"; cp "$W/orig_b" "$IN_B"; cp "$W/expected" "$EXPECTED"; }

# bc shim that records what is fed to bc on stdin
REAL_BC=$(command -v bc || echo /usr/bin/bc)
mkdir -p "$W/shim"
cat > "$W/shim/bc" <<SHIM
#!/bin/bash
input=\$(cat)
printf '%s\n' "\$input" >> "$W/bc.log"
printf '%s\n' "\$input" | "$REAL_BC" "\$@"
SHIM
chmod +x "$W/shim/bc"

run_script() {
  rm -f "$OUT" "$SUMMARY"
  PATH="$W/shim:$PATH" timeout 60 /bin/bash "$SCRIPT" </dev/null >"$W/stdout" 2>"$W/stderr"
}

# section HEADER FILE: lines after header until next '=== ' line
section() { awk -v h="$1" 'f && /^=== /{exit} f{print} $0==h{f=1}' "$2"; }

main() {
  if [ ! -f "$SCRIPT" ]; then fail "script exists"; return; fi
  first=$(head -n1 "$SCRIPT" | tr -d '\r' | sed 's/[[:space:]]*$//')
  chk "shebang is #!/bin/bash" [ "$first" = "#!/bin/bash" ]

  grep -v '^[[:space:]]*#' "$SCRIPT" > "$W/code"
  chk "uses a here-string (<<<)" grep -q '<<<' "$W/code"
  chk "uses a here-document (<< WORD)" grep -Eq "<<-?[[:space:]]*['\"]?[A-Za-z_]" <(sed 's/<<<//g' "$W/code")
  chk "uses process substitution <(...)" grep -q '<(' "$W/code"

  restore
  rm -f "$W/bc.log"
  run_script; rc=$?
  chk "script exits 0 (rc=$rc; stderr: $(head -c 300 "$W/stderr"))" [ $rc -eq 0 ]
  chk "output.txt regenerated and identical to expected_output.txt" cmp -s "$OUT" "$W/expected"
  if [ -f "$SUMMARY" ]; then
    pass "config_summary.txt regenerated"
    sed 's/[[:space:]]*$//; s/^[[:space:]]*//' "$SUMMARY" > "$W/summ"
    for l in "=== Configuration Summary ===" "APP_NAME: DataProcessor" "VERSION: 2.0" "MODE: production" "DEBUG: false"; do
      chk "config_summary contains '$l'" grep -Fxq "$l" "$W/summ"
    done
  else
    fail "config_summary.txt regenerated"
  fi
  if grep -Eq '(^|[^0-9])42([^0-9]|$)' "$W/bc.log" 2>/dev/null || grep -q '/bin/bc' "$W/code"; then
    pass "doubled value computed by bc from the input value"
  else
    fail "doubled value computed by bc from the input value"
  fi

  # Run 2: different inputs + decoy expected file: comparison must be computed
  printf 'alpha\nbravo\ncharlie\ndelta\nkilo\n' > "$IN_A"
  printf 'alpha\ncharlie\ndelta\nlima\nmike\n' > "$IN_B"
  echo DECOY > "$EXPECTED"
  cp "$IN_A" "$W/new_a"; cp "$IN_B" "$W/new_b"
  run_script; rc=$?
  chk "script exits 0 on changed inputs" [ $rc -eq 0 ]
  if [ -f "$OUT" ]; then
    diff "$W/new_a" "$W/new_b" > "$W/want_cmp"
    section "=== Data Comparison ===" "$OUT" | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' > "$W/got_cmp"
    chk "comparison section equals diff of the (changed) input files" cmp -s "$W/got_cmp" "$W/want_cmp"
    section "=== Transformed Values ===" "$OUT" > "$W/got_tr"
    section "=== Transformed Values ===" "$W/expected" > "$W/want_tr"
    chk "transform section unchanged" cmp -s "$W/got_tr" "$W/want_tr"
    section "=== Configuration Summary ===" "$OUT" > "$W/got_cf"
    section "=== Configuration Summary ===" "$W/expected" > "$W/want_cf"
    chk "config section unchanged" cmp -s "$W/got_cf" "$W/want_cf"
    chk "ends with completion marker" [ "$(tail -n1 "$OUT")" = "=== Processing Complete ===" ]
  else
    fail "output.txt produced on changed inputs"
  fi
  restore
}

main
echo "FAILURES: $FAILS"
if [ "$FAILS" -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0
