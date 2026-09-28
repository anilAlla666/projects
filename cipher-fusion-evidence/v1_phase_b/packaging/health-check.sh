#!/bin/sh
# cipher-platform standalone health-check.sh
#
# Independent of the cipher-platform CLI - bash + standard utilities only.
# Used as the diagnostic backup when cipher-platform CLI itself is the
# suspect failure point. Exit 0 PASS; non-zero with failure list otherwise.
#
# Five checks:
#   1. libcipher_rt.so present + non-zero size
#   2. cipher_kmod loaded + sane version
#   3. /var/run/cdi/nvidia.yaml exists + CIPHER entries present (grep)
#   4. plugin files present under /usr/lib/python3/dist-packages
#   5. nvidia-container-toolkit version >= 1.17

set -u

CIPHER_LIB=/usr/lib/cipher/libcipher_rt.so
CDI_PATH=/var/run/cdi/nvidia.yaml
PLUGIN_DIR=/usr/lib/python3/dist-packages

FAIL=0
fail() { echo "FAIL: $1" >&2; FAIL=$((FAIL + 1)); }
pass() { echo "PASS: $1"; }

# 1. substrate library
if [ ! -e "$CIPHER_LIB" ]; then
    fail "$CIPHER_LIB not present (is cipher-platform installed?)"
else
    size=$(stat -c '%s' "$CIPHER_LIB" 2>/dev/null || echo 0)
    if [ "$size" -lt 1000 ]; then
        fail "$CIPHER_LIB suspiciously small ($size bytes; expected >100KB)"
    else
        pass "substrate $CIPHER_LIB present ($size bytes)"
    fi
fi

# 2. kmod loaded
if lsmod 2>/dev/null | grep -q '^cipher_kmod'; then
    ver=$(modinfo cipher_kmod 2>/dev/null | awk '/^version:/{print $2}')
    if [ -z "$ver" ]; then
        fail "cipher_kmod loaded but modinfo reports no version"
    else
        pass "cipher_kmod loaded (version $ver)"
    fi
else
    fail "cipher_kmod not loaded (run: sudo modprobe cipher_kmod)"
fi

if [ ! -e /dev/cipher ]; then
    fail "/dev/cipher absent"
else
    pass "/dev/cipher present"
fi

# 3. CDI patch
if [ ! -e "$CDI_PATH" ]; then
    fail "$CDI_PATH not found (is nvidia-container-toolkit configured?)"
elif grep -q 'CIPHER_CDI_MARKER=v1' "$CDI_PATH" 2>/dev/null \
     && grep -q '/usr/lib/cipher/libcipher_rt.so' "$CDI_PATH" 2>/dev/null; then
    pass "$CDI_PATH contains CIPHER patch (marker + substrate mount detected)"
else
    fail "$CDI_PATH missing CIPHER patch entries (run: sudo cipher-platform refresh)"
fi

# 4. plugin files
PLUGIN_FILES="cipher_vllm_kv.py cipher_vllm_kvdedup.py cipher_kv_offload.py cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so cipher_vllm_kv-0.2.0.dist-info/entry_points.txt"
MISSING=""
for f in $PLUGIN_FILES; do
    if [ ! -e "$PLUGIN_DIR/$f" ]; then
        MISSING="$MISSING $f"
    fi
done
if [ -n "$MISSING" ]; then
    fail "plugin chain incomplete under $PLUGIN_DIR; missing:$MISSING"
else
    pass "plugin chain present under $PLUGIN_DIR"
fi

# 5. nvidia-container-toolkit version
nctk_ver=$(dpkg-query -W -f='${Version}' nvidia-container-toolkit 2>/dev/null || echo "")
if [ -z "$nctk_ver" ]; then
    fail "nvidia-container-toolkit not installed (Depends should have caught this)"
else
    # Extract MAJOR.MINOR from e.g. "1.18.1-0lambda0.22.04.1"
    nctk_major=$(echo "$nctk_ver" | awk -F. '{print $1}')
    nctk_minor=$(echo "$nctk_ver" | awk -F. '{print $2}')
    if [ "$nctk_major" -gt 1 ] || \
       { [ "$nctk_major" -eq 1 ] && [ "$nctk_minor" -ge 17 ]; }; then
        pass "nvidia-container-toolkit $nctk_ver (>= 1.17)"
    else
        fail "nvidia-container-toolkit $nctk_ver < 1.17 (CDI mode required)"
    fi
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "cipher-platform health-check.sh: PASS"
    exit 0
else
    echo "cipher-platform health-check.sh: FAIL ($FAIL check(s))"
    exit 1
fi
