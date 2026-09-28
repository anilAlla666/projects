#!/usr/bin/env bash
# Assemble cipher-platform_2.0_amd64.deb. Substrate + handlers install byte-identical;
# only cipher_platform/ (CLI/router/gates/report/config) is new. Run from cipher_v2_build/.
set -euo pipefail
ROOT="/home/ubuntu/cipher_v2_build"
PKG="$ROOT/pkgroot"; SRC="$ROOT/src"; H="/home/ubuntu"; SMK="$H/cipher_v2_mock"
RT_SO="$H/cipher_rt_phase4/libcipher_rt.so"
MODEL_DIR="${CIPHER_MODEL_DIR:-/home/ubuntu/models}"
KVER="0.7.0"

rm -rf "$PKG"; mkdir -p "$PKG"/{DEBIAN,usr/lib/cipher,usr/src/cipher-kmod-$KVER,usr/share/cipher/handlers,usr/bin,etc/cipher}
mkdir -p "$PKG/usr/lib/python3/dist-packages/cipher_platform"

echo "[1] substrate -> /usr/lib/cipher"
cp "$RT_SO" "$PKG/usr/lib/cipher/libcipher_rt.so"
[ -f "$H/libcipher_v2/libcipher_v2.so" ] && cp "$H/libcipher_v2/libcipher_v2.so" "$PKG/usr/lib/cipher/" || echo "  (libcipher_v2 not bundled)"
LC10=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libc10.so'))" 2>/dev/null || true)
[ -n "${LC10:-}" ] && [ -f "$LC10" ] && cp "$LC10" "$PKG/usr/lib/cipher/" || echo "  (libc10 from system torch at runtime)"
md5sum "$PKG/usr/lib/cipher/libcipher_rt.so" | awk '{print $1}' > "$PKG/usr/lib/cipher/libcipher_rt.so.md5"

echo "[2] kmod 0.7.0 source -> /usr/src (DKMS)"
# exclude kbuild-generated (*.mod.c) and the standalone probe (not part of the module)
for f in "$H/cipher_kmod/"*.c "$H/cipher_kmod/"*.h; do
  b=$(basename "$f")
  case "$b" in *.mod.c|probe_microbench.c) continue;; esac
  cp "$f" "$PKG/usr/src/cipher-kmod-$KVER/"
done
[ -f "$H/cipher_kmod/Kbuild" ] && cp "$H/cipher_kmod/Kbuild" "$PKG/usr/src/cipher-kmod-$KVER/"
[ -f "$H/cipher_kmod/Makefile" ] && cp "$H/cipher_kmod/Makefile" "$PKG/usr/src/cipher-kmod-$KVER/"
cat > "$PKG/usr/src/cipher-kmod-$KVER/dkms.conf" <<EOF
PACKAGE_NAME="cipher-kmod"
PACKAGE_VERSION="$KVER"
MAKE[0]="make KDIR=/lib/modules/\${kernelver}/build"
CLEAN="make clean KDIR=/lib/modules/\${kernelver}/build"
BUILT_MODULE_NAME[0]="cipher_kmod"
DEST_MODULE_LOCATION[0]="/updates/dkms"
AUTOINSTALL="yes"
EOF

echo "[3] handlers -> /usr/share/cipher/handlers"
for f in cipher_inc4.py v0_phaseA_maxmfu.py v0_phaseC_nf4_cofire.py cipher_engine.py cipher_engine_batched.py cipher_product_report.py; do
  cp "$H/$f" "$PKG/usr/share/cipher/handlers/"
done
for f in smoke_agent_scaled.py smoke_density_real.py mock_handler.py off_byte_identical_real.py; do
  cp "$SMK/$f" "$PKG/usr/share/cipher/handlers/"
done
# portability: smokes import cipher_engine from their own (installed) dir, not /home/ubuntu
sed -i 's#sys.path.insert(0, "/home/ubuntu")#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))#' \
  "$PKG/usr/share/cipher/handlers/smoke_agent_scaled.py" "$PKG/usr/share/cipher/handlers/smoke_density_real.py"

echo "[4] cipher_platform package + CLI"
cp "$SRC/cipher_platform/"*.py "$PKG/usr/lib/python3/dist-packages/cipher_platform/"
cat > "$PKG/usr/bin/cipher" <<'EOF'
#!/usr/bin/env python3
import sys; sys.path.insert(0, "/usr/lib/python3/dist-packages")
from cipher_platform.cli import main
main()
EOF
chmod 0755 "$PKG/usr/bin/cipher"

echo "[5] config -> /etc/cipher/platform.json"
cat > "$PKG/etc/cipher/platform.json" <<EOF
{
  "so_path": "/usr/lib/cipher/libcipher_rt.so",
  "handlers_dir": "/usr/share/cipher/handlers",
  "model_library_dir": "$MODEL_DIR",
  "regimes": {
    "agent":   {"handler":"cipher_inc4.py","quick_handler":"smoke_agent_scaled.py","inject":{"CIPHER_RT_DISABLE_AUTO_INIT":"1","CIPHER_VOLT":"0"},"timeout":1800},
    "density": {"handler":"v0_phaseC_nf4_cofire.py","quick_handler":"smoke_density_real.py","inject":{"CIPHER_RT_DISABLE_AUTO_INIT":"1"},"timeout":600},
    "compute": {"handler":"v0_phaseA_maxmfu.py","quick_handler":"mock_handler.py","inject":{"CUDA_INJECTION64_PATH":"__SO__","CIPHER_FP8":"1"},"timeout":900}
  },
  "accelerate_env": {"CIPHER_KV_ALLOC":"1","CIPHER_KVDEDUP":"1"}
}
EOF

echo "[6] DEBIAN control + maintainer scripts"
RTMD5=$(cat "$PKG/usr/lib/cipher/libcipher_rt.so.md5")
cat > "$PKG/DEBIAN/control" <<EOF
Package: cipher-platform
Version: 2.0
Section: utils
Priority: optional
Architecture: amd64
Depends: dkms, libc6, libstdc++6, libgcc-s1, python3 (>= 3.10)
Recommends: linux-headers-generic
Maintainer: CIPHER Platform <anil.0666369@gmail.com>
Description: CIPHER GPU substrate + Class-D demos/gates + vLLM accelerate
 Single deployable of the CIPHER cross-tenant GPU substrate (libcipher_rt +
 cipher_kmod DKMS) plus the 'cipher' CLI: reproducible Class-D demos and
 discipline gates, and 'cipher accelerate' which injects CIPHER under the
 customer's own vLLM (KV-dedup / weight-share / FP8 / DVFS, zero code change).
 Not a CIPHER-native serving runtime. Single-GPU (H100). The Python ML stack
 and model checkpoints are a host dependency (verified by 'cipher selftest').
EOF
cat > "$PKG/DEBIAN/postinst" <<EOF
#!/bin/sh
set -e
echo "[cipher] verifying substrate integrity (no-regress anchor)"
echo "$RTMD5  /usr/lib/cipher/libcipher_rt.so" | md5sum -c - || { echo "[cipher] FATAL: libcipher_rt.so md5 mismatch"; exit 1; }
if [ -d /usr/src/cipher-kmod-$KVER ]; then
  dkms add -m cipher-kmod -v $KVER 2>/dev/null || true
  dkms build -m cipher-kmod -v $KVER || echo "[cipher] WARN: dkms build failed (need linux-headers + nvidia.ko); module not loaded"
  dkms install -m cipher-kmod -v $KVER 2>/dev/null || true
  modprobe cipher_kmod 2>/dev/null || echo "[cipher] WARN: modprobe cipher_kmod failed (load manually after headers present)"
fi
[ -e /dev/cipher ] && chmod 0666 /dev/cipher 2>/dev/null || true
echo "[cipher] installed. Validate this box:  cipher selftest --quick"
exit 0
EOF
cat > "$PKG/DEBIAN/prerm" <<EOF
#!/bin/sh
set -e
modprobe -r cipher_kmod 2>/dev/null || true
dkms remove -m cipher-kmod -v $KVER --all 2>/dev/null || true
exit 0
EOF
chmod 0755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/prerm"

echo "[7] build .deb"
dpkg-deb --build --root-owner-group "$PKG" "$ROOT/cipher-platform_2.0_amd64.deb"
echo "DONE -> $ROOT/cipher-platform_2.0_amd64.deb"
