#!/bin/bash
# Verifier for the subshell () vs command-group {} deploy.sh task.
# Pure bash: no network, no package installs.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

FAILS=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILS=$((FAILS+1)); }

APP=/app
SCRIPT=$APP/deploy.sh
CONF=$APP/config/components.conf
WORK=$(mktemp -d)
DEF_IFS=$' \t\n'

# ---------- 1. Result files carried over from the solver ----------
check_outputs() {   # $1 = label, $2.. = extra expected components
    local label="$1"; shift
    local st="$APP/.deploy_state" cl="$APP/config/components.list"
    if [[ -f $st ]] && grep -qx 'DEPLOY_VERSION=2.0' "$st" && grep -qx 'DEPLOY_ENV=production' "$st"; then
        pass "$label: .deploy_state has DEPLOY_VERSION=2.0 and DEPLOY_ENV=production"
    else
        fail "$label: .deploy_state missing or wrong: $(cat "$st" 2>/dev/null)"
    fi
    [[ -e $APP/frontend/.built ]] && pass "$label: frontend/.built" || fail "$label: frontend/.built missing"
    [[ -e $APP/backend/.built ]]  && pass "$label: backend/.built"  || fail "$label: backend/.built missing"
    if [[ -f $cl ]]; then
        local c
        for c in frontend backend config "$@"; do
            grep -qx "$c" "$cl" && pass "$label: components.list has $c" || fail "$label: components.list lacks line '$c': $(cat "$cl")"
        done
        grep -q ',' "$cl" && fail "$label: components.list contains commas (IFS leak?)"
    else
        fail "$label: components.list missing"
    fi
}
check_outputs "solver-output"

if [[ ! -f $SCRIPT ]]; then
    fail "deploy.sh missing"
    echo "FAILED ($FAILS)"; exit 0
fi
bash -n "$SCRIPT" 2>/dev/null && pass "deploy.sh parses" || fail "deploy.sh has syntax errors"

# ---------- 2. Each function behaves correctly in the current shell ----------
# Function definitions only: drop top-level invocations of main.
grep -Ev '^main([[:space:]].*)?$|^main[[:space:]]*;?[[:space:]]*$' "$SCRIPT" > "$WORK/defs.sh"

rm -f "$APP/frontend/.built" "$APP/backend/.built"
mkdir -p "$WORK/elsewhere"
OUT=$(cd "$WORK/elsewhere" && timeout 60 bash -c '
    source "$1" >/dev/null 2>&1
    set +e
    for f in set_deploy_config build_component load_components parse_deploy_info; do
        declare -F "$f" >/dev/null || echo "MISSING_FUNC=$f"
    done
    DEPLOY_VERSION=""; DEPLOY_ENV=""
    set_deploy_config >/dev/null 2>&1
    echo "VAR_VERSION=$DEPLOY_VERSION"
    echo "VAR_ENV=$DEPLOY_ENV"
    start=$(pwd)
    build_component /app/frontend frontend >/dev/null 2>&1; echo "RC_FE=$?"
    echo "CWD1_OK=$([[ $(pwd) == "$start" ]] && echo yes || echo no)"
    build_component /app/backend backend >/dev/null 2>&1; echo "RC_BE=$?"
    echo "CWD2_OK=$([[ $(pwd) == "$start" ]] && echo yes || echo no)"
    COMPONENTS=()
    load_components /app/config/components.conf >/dev/null 2>&1
    echo "ARR=${COMPONENTS[*]}"
    IFS=$'"'"' \t\n'"'"'
    pout=$(parse_deploy_info "server1,server2,server3" 2>&1)
    parse_deploy_info "server1,server2,server3" >/dev/null 2>&1
    echo "IFS_OK=$([[ "$IFS" == $'"'"' \t\n'"'"' ]] && echo yes || echo no)"
    echo "PARSE_OUT=$pout"
' _ "$WORK/defs.sh" 2>&1)
echo "$OUT" | sed 's/^/  | /'

if echo "$OUT" | grep -q '^MISSING_FUNC='; then
    fail "required functions missing: $(echo "$OUT" | grep '^MISSING_FUNC=' | tr '\n' ' ')"
fi
echo "$OUT" | grep -qx 'VAR_VERSION=2.0' && echo "$OUT" | grep -qx 'VAR_ENV=production' \
    && pass "set_deploy_config persists DEPLOY_VERSION/DEPLOY_ENV" \
    || fail "set_deploy_config does not persist variables to caller"
echo "$OUT" | grep -qx 'CWD1_OK=yes' && echo "$OUT" | grep -qx 'CWD2_OK=yes' \
    && pass "build_component does not leak cd" || fail "build_component leaks directory change"
[[ -e $APP/frontend/.built && -e $APP/backend/.built ]] \
    && pass "build_component creates .built in component dir" || fail "build_component did not create .built markers"
[[ -e $WORK/elsewhere/.built ]] && fail "build_component touched .built in caller's directory"
echo "$OUT" | grep -qx 'ARR=frontend backend config' \
    && pass "load_components persists array" || fail "load_components does not persist COMPONENTS"
echo "$OUT" | grep -qx 'IFS_OK=yes' \
    && pass "parse_deploy_info isolates IFS" || fail "parse_deploy_info leaks IFS"
echo "$OUT" | grep -q 'Parsed 3 parts' \
    && pass "parse_deploy_info still parses with ','" || fail "parse_deploy_info no longer splits on ','"

# ---------- 3. Whole script, sourced from /app: ends in /app with clean IFS ----------
rm -f "$APP/.deploy_state" "$APP/config/components.list" "$APP/frontend/.built" "$APP/backend/.built"
END=$(cd "$APP" && timeout 60 bash -c 'source /app/deploy.sh >/dev/null 2>&1; echo "END_PWD=$(pwd)"; [[ "$IFS" == $'"'"' \t\n'"'"' ]] && echo END_IFS=ok' 2>&1)
echo "$END" | sed 's/^/  | /'
echo "$END" | grep -qx 'END_PWD=/app' && pass "script ends in /app" || fail "script does not end in /app"
echo "$END" | grep -qx 'END_IFS=ok' && pass "IFS default after script" || fail "IFS polluted after script"

# ---------- 4. Running the script normally regenerates the outputs ----------
rm -f "$APP/.deploy_state" "$APP/config/components.list" "$APP/frontend/.built" "$APP/backend/.built"
LOG=$(cd "$APP" && timeout 60 bash "$SCRIPT" 2>&1); RC=$?
[[ $RC -eq 0 ]] && pass "deploy.sh exits 0" || fail "deploy.sh exit code $RC"
echo "$LOG" | grep -q 'IFS is corrupted' && fail "script reports IFS corrupted"
echo "$LOG" | grep -q 'No components loaded' && fail "script used fallback component list"
check_outputs "fresh-run"

# ---------- 5. Outputs derive from the config (not hard-coded) ----------
cp "$CONF" "$WORK/conf.bak"
echo "component:docs:static" >> "$CONF"
rm -f "$APP/.deploy_state" "$APP/config/components.list" "$APP/frontend/.built" "$APP/backend/.built"
(cd "$APP" && timeout 60 bash "$SCRIPT" >/dev/null 2>&1)
check_outputs "extra-component-run" docs
cp "$WORK/conf.bak" "$CONF"

if [[ $FAILS -eq 0 ]]; then
    echo "ALL CHECKS PASSED"; echo 1 > /logs/verifier/reward.txt
else
    echo "FAILED ($FAILS)"
fi
