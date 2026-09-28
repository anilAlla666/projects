#!/bin/bash
# Build the CIPHER .deb package from the current build artifacts.
#
# Usage: bash package/build_deb.sh
# Output: cipher_1.0.0_amd64.deb in the repo root.
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# Verify the .so files are built.
for so in libcipher_hook.so libcipher_rt.so libcipher_nccl_tuner.so libnccl-tuner-cipher.so; do
    [ -f "$ROOT/$so" ] || { echo "FATAL: $so missing — run make first"; exit 1; }
done

# Stage files into the .deb tree.
STAGE=$ROOT/package/deb
LIBDIR=$STAGE/opt/cipher/lib
BINDIR=$STAGE/opt/cipher/bin
ETCDIR=$STAGE/opt/cipher/etc
mkdir -p "$LIBDIR" "$BINDIR" "$ETCDIR"

cp "$ROOT/libcipher_hook.so"        "$LIBDIR/"
cp "$ROOT/libcipher_rt.so"          "$LIBDIR/"
cp "$ROOT/libcipher_nccl_tuner.so"  "$LIBDIR/"
cp "$ROOT/libnccl-tuner-cipher.so"  "$LIBDIR/"
cp "$ROOT/libcipher_receipt.so"     "$LIBDIR/"
cp "$ROOT/cipher_run.sh"            "$BINDIR/cipher_run.sh"
cp -r "$ROOT/cipher_python"         "$LIBDIR/cipher_python"
# M1.T7 — ship the example policy.json so postinst can install it.
[ -f "$ETCDIR/policy.json.example" ] || \
    cp "$STAGE/opt/cipher/etc/policy.json.example" "$ETCDIR/" 2>/dev/null || true
chmod +x "$BINDIR/cipher_run.sh"
chmod +x "$STAGE/DEBIAN/postinst"

# Patch cipher_run.sh paths for the install layout.
sed -i 's|CIPHER_ROOT=.*|CIPHER_ROOT="/opt/cipher"|' "$BINDIR/cipher_run.sh" || true
# (The script's LD_PRELOAD line builds from $CIPHER_ROOT, so this rewrites
# the install layout transparently. Dev runs from the repo root still work
# because CIPHER_ROOT is computed from $0.)

# dpkg-deb
OUT=$ROOT/cipher_1.0.0_amd64.deb
dpkg-deb --build --root-owner-group "$STAGE" "$OUT"
echo
echo "Built: $OUT"
ls -la "$OUT"
