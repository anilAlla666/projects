#!/bin/sh
# build_deb.sh - assemble cipher-platform v2.0 .deb from substrate repos + tracked scripts.
#
# Outputs cipher-platform_2.0_amd64.deb in BUILD_DIR (default /tmp/cipher-deb-build/).
#
# Usage: build_deb.sh [build_dir]
set -eu

REPO=$(cd "$(dirname "$0")" && pwd)        # v1_phase_b/packaging
EVIDENCE=$(cd "$REPO/../.." && pwd)         # cipher-fusion-evidence
BUILD_DIR="${1:-/tmp/cipher-deb-build}"
STAGE="$BUILD_DIR/cipher-platform"

# Anchor pins (must match V1 Phase B B.1'' / B.2'' substep 1 + 2 commit messages).
SUBSTRATE_REPO=/home/ubuntu/cipher_rt_phase4
LIBCIPHER_V2_REPO=/home/ubuntu/libcipher_v2
KMOD_REPO=/home/ubuntu/cipher_kmod
PLUGIN_REPO=/home/ubuntu/cipher_vllm_plugin
DIST_INFO_SRC=/home/ubuntu/vllm_env/lib/python3.10/site-packages/cipher_vllm_kv-0.2.0.dist-info
TORCH_LIBS=/home/ubuntu/vllm_env/lib/python3.10/site-packages/torch/lib

# Substrate libcipher_rt.so is now CUDA-13-linked (v1-substrate-cuda13-cublaslt-coverage
# anchor; F-B.3.4 substrate rebuild + F-B.3.6.1 Path B' + F-B.3.6.2 Option B fold).
# Built inside vllm/vllm-openai:v0.21.0 container; preserved at $SUBSTRATE_REPO/build_cuda13/.
# md5 is build-timestamp non-deterministic; DT_NEEDED chain + symbol table are the
# semantic anchor. Verify via post-build ldd inside container (B.6''.8 Gates B/C).
SUBSTRATE_FROM_BUILD_CUDA13=1
LIBCIPHER_V2_MD5=cc0479b836e560619e2b286ca1caecb7
LIBC10_MD5=b814a98fc9420f824bf6a520b6283c92
CIPHER_KV_BRIDGE_MD5=5a3db034ea2b950bf426eb48140ec8fd
CIPHER_KV_BRIDGE_PY312_MD5=00df2811bcf1f15231fd599f14a268ae
CIPHER_VLLM_KV_MD5=d0226d7950bba4e6dfdf11ad5b841e51  # K.1.5 Step 2 Sub-step 0 (rev8): VA pool floor fix

KMOD_VERSION=0.6.5

err() { echo "build_deb: ERROR: $*" >&2; exit 1; }
verify_md5() {
    actual=$(md5sum "$1" | awk '{print $1}')
    [ "$actual" = "$2" ] || err "$1 md5 $actual != expected $2"
}

echo "build_deb: staging into $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/DEBIAN" \
         "$STAGE/usr/bin" \
         "$STAGE/usr/lib/cipher" \
         "$STAGE/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info" \
         "$STAGE/usr/src/cipher-kmod-$KMOD_VERSION" \
         "$STAGE/lib/systemd/system"

# --- DEBIAN/ control + changelog + postinst + prerm + dkms.conf (tracked) ---
cp "$REPO/debian/control"   "$STAGE/DEBIAN/control"
cp "$REPO/debian/changelog" "$STAGE/DEBIAN/changelog"
cp "$REPO/debian/postinst"  "$STAGE/DEBIAN/postinst"
cp "$REPO/debian/prerm"     "$STAGE/DEBIAN/prerm"
chmod 755 "$STAGE/DEBIAN/postinst" "$STAGE/DEBIAN/prerm"

# --- usr/bin/cipher-run (preserved from CP 2.5 v1.0; backward-compat shell wrapper) ---
cat >"$STAGE/usr/bin/cipher-run" <<'EOF'
#!/usr/bin/env bash
# cipher-run - backward-compat launch wrapper from CP 2.5.
# v2.0 deployment is via CDI; this remains for non-container bare-metal use.
set -u
CIPHER_LIB=/usr/lib/cipher
RT="${CIPHER_LIB}/libcipher_rt.so"
if [ ! -e "$RT" ]; then
    echo "cipher-run: ${RT} not found - is cipher-platform installed?" >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    echo "usage: cipher-run <command> [args...]" >&2
    exit 2
fi
if [ ! -e /dev/cipher ]; then
    echo "cipher-run: warning - /dev/cipher absent; is the cipher_kmod" \
         "module loaded? (modprobe cipher_kmod)" >&2
fi
export CUDA_INJECTION64_PATH="$RT"
export LD_LIBRARY_PATH="${CIPHER_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
EOF
chmod 755 "$STAGE/usr/bin/cipher-run"

# --- usr/bin/cipher-platform CLI (substep 5) ---
cp "$REPO/cipher_platform_cli.py" "$STAGE/usr/bin/cipher-platform"
chmod 755 "$STAGE/usr/bin/cipher-platform"

# --- usr/lib/cipher/ substrate libs (substep 1 anchors) ---
# CUDA 13 substrate from build_cuda13/ (built inside vllm-openai container).
cp "$SUBSTRATE_REPO/build_cuda13/libcipher_rt.so" "$STAGE/usr/lib/cipher/libcipher_rt.so"
echo "build_deb: libcipher_rt.so md5: $(md5sum "$STAGE/usr/lib/cipher/libcipher_rt.so" | awk '{print $1}') (CUDA 13)"
cp "$LIBCIPHER_V2_REPO/libcipher_v2.so"  "$STAGE/usr/lib/cipher/libcipher_v2.so"
verify_md5 "$STAGE/usr/lib/cipher/libcipher_v2.so" "$LIBCIPHER_V2_MD5"
cp "$TORCH_LIBS/libc10.so"                "$STAGE/usr/lib/cipher/libc10.so"
verify_md5 "$STAGE/usr/lib/cipher/libc10.so" "$LIBC10_MD5"

# --- usr/lib/cipher/ CDI patch + CLI helpers + runbook + health-check ---
cp "$REPO/cipher_cdi_patch.py" "$STAGE/usr/lib/cipher/cipher_cdi_patch.py"
cp "$REPO/cipher_cdi_patch.sh" "$STAGE/usr/lib/cipher/cipher_cdi_patch.sh"
cp "$REPO/runbook.md"          "$STAGE/usr/lib/cipher/runbook.md"
cp "$REPO/health-check.sh"     "$STAGE/usr/lib/cipher/health-check.sh"
chmod 755 "$STAGE/usr/lib/cipher/cipher_cdi_patch.py" \
          "$STAGE/usr/lib/cipher/cipher_cdi_patch.sh" \
          "$STAGE/usr/lib/cipher/health-check.sh"

# --- usr/lib/python3/dist-packages/ plugin chain (substep 2 manifest) ---
cp "$PLUGIN_REPO/cipher_vllm_kv.py" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_vllm_kv.py"
verify_md5 "$STAGE/usr/lib/python3/dist-packages/cipher_vllm_kv.py" "$CIPHER_VLLM_KV_MD5"
cp "$PLUGIN_REPO/cipher_vllm_kvdedup.py" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_vllm_kvdedup.py"
cp "$PLUGIN_REPO/cipher_kv_offload.py" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_kv_offload.py"
cp "$EVIDENCE/phase_c/cipher_model_fingerprint.py" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_model_fingerprint.py"
cp "$SUBSTRATE_REPO/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so"
verify_md5 "$STAGE/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so" \
           "$CIPHER_KV_BRIDGE_MD5"
# F-B.3.2 remediation: ship cipher_kv_bridge in two ABI flavors (py310 + py312)
# so the same .deb works across vllm-openai images built on different Python
# minor versions. Built inside vllm-openai:v0.21.0 container against its
# /usr/local/lib/python3.12/dist-packages/torch. Output preserved at
# $SUBSTRATE_REPO/build_py312/.
cp "$SUBSTRATE_REPO/build_py312/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so" \
   "$STAGE/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so"
verify_md5 "$STAGE/usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so" \
           "$CIPHER_KV_BRIDGE_PY312_MD5"
for f in METADATA WHEEL entry_points.txt top_level.txt; do
    cp "$DIST_INFO_SRC/$f" \
       "$STAGE/usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/$f"
done

# --- usr/src/cipher-kmod-0.6.5/ DKMS source (substep 1) ---
cp "$REPO/dkms.conf" "$STAGE/usr/src/cipher-kmod-$KMOD_VERSION/dkms.conf"
for f in "$KMOD_REPO"/*.c "$KMOD_REPO"/*.h "$KMOD_REPO"/Makefile "$KMOD_REPO"/Kbuild; do
    base=$(basename "$f")
    # skip build artifacts that aren't kbuild source
    case "$base" in
        cipher_kmod.mod.c|probe_microbench.c) continue ;;
    esac
    cp "$f" "$STAGE/usr/src/cipher-kmod-$KMOD_VERSION/$base"
done

# --- lib/systemd/system/ watcher units (substep 4) ---
cp "$REPO/systemd/cipher-platform-watch.path" \
   "$STAGE/lib/systemd/system/cipher-platform-watch.path"
cp "$REPO/systemd/cipher-platform-watch.service" \
   "$STAGE/lib/systemd/system/cipher-platform-watch.service"

# --- enforce ownership for dpkg-deb --build ---
chmod -R u+rw,go+r "$STAGE"

# --- build the .deb ---
DEB_PATH="$BUILD_DIR/cipher-platform_2.0_amd64.deb"
echo "build_deb: dpkg-deb --build -> $DEB_PATH"
dpkg-deb --build --root-owner-group "$STAGE" "$DEB_PATH"

echo "build_deb: package metadata:"
dpkg-deb --info "$DEB_PATH"
echo "build_deb: package contents (count):"
dpkg-deb --contents "$DEB_PATH" | wc -l
echo "build_deb: package size:"
ls -lh "$DEB_PATH"
echo "build_deb: PASS"
