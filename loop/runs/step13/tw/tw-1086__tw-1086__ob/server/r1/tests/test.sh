#!/bin/bash
# Verifier for the bash subshell-vs-command-group deploy.sh task.
# Pure bash (no network, no pytest needed). Writes 1 or 0 to /logs/verifier/reward.txt.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
FAIL=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAIL=1; }

SCRIPT=/app/deploy.sh
DEFAULT_IFS=$' \t\n'

check_outputs() {   # $1 = label
    local label="$1"
    if [[ -f /app/.deploy_state ]]; then
        grep -qx 'DEPLOY_VERSION=2.0' /app/.deploy_state && pass "$label: DEPLOY_VERSION=2.0" || fail "$label: .deploy_state lacks DEPLOY_VERSION=2.0: $(cat /app/.deploy_state)"
        grep -qx 'DEPLOY_ENV=production' /app/.deploy_state && pass "$label: DEPLOY_ENV=production" || fail "$label: .deploy_state lacks DEPLOY_ENV=production: $(cat /app/.deploy_state)"
    else
        fail "$label: /app/.deploy_state missing"
    fi
    [[ -e /app/frontend/.built ]] && pass "$label: frontend/.built" || fail "$label: /app/frontend/.built missing"
    [[ -e /app/backend/.built ]] && pass "$label: backend/.built" || fail "$label: /app/backend/.built missing"
    if [[ -f /app/config/components.list ]]; then
        local c
        for c in frontend backend config; do
            grep -qx "$c" /app/config/components.list && pass "$label: components.list has $c" || fail "$label: components.list lacks line '$c': $(cat /app/config/components.list)"
        done
    else
        fail "$label: /app/config/components.list missing"
    fi
}

# ---- 1. Results left by the solver --------------------------------------
check_outputs "solver output"

if [[ ! -f "$SCRIPT" ]]; then
    fail "/app/deploy.sh missing"
    echo "RESULT: FAIL"; exit 0
fi
bash -n "$SCRIPT" || fail "deploy.sh has a syntax error"

# The config file is not carried over from the solver; make sure it has the
# Dockerfile content.
mkdir -p /app/frontend /app/backend /app/config
printf 'component:frontend:web\ncomponent:backend:api\ncomponent:config:util\n' > /app/config/components.conf

# ---- 2. Unit tests of the four functions (script sourced without main) ---
WORK=$(mktemp -d)
sed -E '/^[[:space:]]*main([[:space:]]+"\$@")?[[:space:]]*;?[[:space:]]*$/d' "$SCRIPT" > "$WORK/funcs.sh"
mkdir -p "$WORK/comp1" "$WORK/elsewhere"
printf 'component:alpha:web\ncomponent:beta:api\nother:skip:x\ncomponent:gamma:util\n' > "$WORK/test.conf"

cat > "$WORK/unit.sh" <<'UNIT'
WORK="$1"
cd /app
source "$WORK/funcs.sh" >/dev/null 2>&1
set +e
for f in set_deploy_config build_component load_components parse_deploy_info; do
    declare -F "$f" >/dev/null || { echo "UNIT_FAIL: function $f not defined"; }
done

# Bug 1: variables must persist
DEPLOY_VERSION=""; DEPLOY_ENV=""
set_deploy_config >/dev/null 2>&1
[[ "$DEPLOY_VERSION" == "2.0" ]] && echo "UNIT_PASS: DEPLOY_VERSION persists" || echo "UNIT_FAIL: DEPLOY_VERSION after set_deploy_config is '$DEPLOY_VERSION'"
[[ "$DEPLOY_ENV" == "production" ]] && echo "UNIT_PASS: DEPLOY_ENV persists" || echo "UNIT_FAIL: DEPLOY_ENV after set_deploy_config is '$DEPLOY_ENV'"

# Bug 2: directory change must not leak; build must happen in the dir
cd "$WORK/elsewhere"
build_component "$WORK/comp1" "comp1" >/dev/null 2>&1
[[ "$PWD" == "$WORK/elsewhere" && "$(pwd -P)" == "$(cd "$WORK/elsewhere" && pwd -P)" ]] && echo "UNIT_PASS: build_component keeps cwd" || echo "UNIT_FAIL: build_component leaked cwd: now $PWD"
[[ -e "$WORK/comp1/.built" ]] && echo "UNIT_PASS: build_component created marker" || echo "UNIT_FAIL: build_component did not create $WORK/comp1/.built"
[[ ! -e "$WORK/elsewhere/.built" ]] || echo "UNIT_FAIL: build_component created .built in caller dir"
cd /app

# Bug 3: array modifications must persist
COMPONENTS=()
load_components "$WORK/test.conf" >/dev/null 2>&1
[[ "${COMPONENTS[*]}" == "alpha beta gamma" ]] && echo "UNIT_PASS: load_components persists array" || echo "UNIT_FAIL: COMPONENTS after load_components is (${COMPONENTS[*]})"
COMPONENTS=()
load_components /app/config/components.conf >/dev/null 2>&1
[[ "${COMPONENTS[*]}" == "frontend backend config" ]] && echo "UNIT_PASS: load_components real conf" || echo "UNIT_FAIL: COMPONENTS from components.conf is (${COMPONENTS[*]})"

# Bug 4: IFS must be isolated, and parsing must still use ','
IFS=$' \t\n'
out=$(parse_deploy_info "s1,s2,s3" 2>&1)
parse_deploy_info "s1,s2,s3" >/dev/null 2>&1
[[ "$IFS" == $' \t\n' ]] && echo "UNIT_PASS: IFS unchanged after parse_deploy_info" || echo "UNIT_FAIL: IFS polluted after parse_deploy_info: $(printf '%q' "$IFS")"
[[ "$out" == *3* ]] && echo "UNIT_PASS: parse_deploy_info parses 3 parts" || echo "UNIT_FAIL: parse_deploy_info did not split on ',': $out"
UNIT

unit_out=$(cd /app && timeout 60 bash "$WORK/unit.sh" "$WORK" 2>&1)
echo "$unit_out"
grep -q '^UNIT_FAIL' <<<"$unit_out" && FAIL=1
[[ $(grep -c '^UNIT_PASS' <<<"$unit_out") -ge 8 ]] || fail "unit tests incomplete"

# ---- 3. Run the fixed script itself, from scratch -------------------------
cat > "$WORK/run.sh" <<'RUN'
__verifier_fin() {
    echo "__FINAL_PWD=$PWD"
    [[ "$IFS" == $' \t\n' ]] && echo __FINAL_IFS_OK
    echo "__FINAL_VER=$DEPLOY_VERSION"
}
trap __verifier_fin EXIT
source /app/deploy.sh
RUN
rm -f /app/.deploy_state /app/frontend/.built /app/backend/.built /app/config/components.list
run_out=$(cd /app && timeout 60 bash "$WORK/run.sh" 2>&1)
rc=$?
echo "$run_out" | tail -n 30
[[ $rc -eq 0 ]] && pass "deploy.sh exits 0" || fail "deploy.sh exited $rc"
grep -qx '__FINAL_PWD=/app' <<<"$run_out" && pass "script ends in /app" || fail "script does not end in /app"
grep -qx "__FINAL_IFS_OK" <<<"$run_out" && pass "global IFS intact at end" || fail "global IFS polluted at end"
grep -qx '__FINAL_VER=2.0' <<<"$run_out" && pass "DEPLOY_VERSION set in main shell" || fail "DEPLOY_VERSION not set in main shell"
grep -q 'IFS is corrupted' <<<"$run_out" && fail "script reports IFS corrupted"
grep -q 'No components loaded' <<<"$run_out" && fail "script used fallback component list"
check_outputs "fresh run"

rm -rf "$WORK"
if [[ $FAIL -eq 0 ]]; then
    echo 1 > /logs/verifier/reward.txt
    echo "RESULT: PASS"
else
    echo "RESULT: FAIL"
fi
