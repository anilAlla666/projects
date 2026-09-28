#!/usr/bin/env bash
# V1 Phase B.1 R-B.1 POC builder.
# Per V1_PHASE_B_SCOPE_LOCK.md section 4 B.1 + Anil 2026-05-26 B.1 spec.
# Builds cipher-rb1-poc_0.0.1_amd64.deb via dpkg-deb --build mirroring the
# cipher-platform 1.0 raw-dpkg-deb pattern at /tmp/cipher-deb-build/.
set -euo pipefail

cd "$(dirname "$0")"

STAGING=cipher-rb1-poc
OUT=cipher-rb1-poc_0.0.1_amd64.deb

# Ensure staging dir tree exists
test -d "$STAGING/DEBIAN" || { echo "missing $STAGING/DEBIAN" >&2; exit 2; }
test -f "$STAGING/DEBIAN/control" || { echo "missing control" >&2; exit 2; }
test -f "$STAGING/usr/lib/python3/dist-packages/cipher_rb1_poc.py" || \
    { echo "missing cipher_rb1_poc.py" >&2; exit 2; }
test -f "$STAGING/usr/lib/python3/dist-packages/cipher_rb1_poc-0.0.1.dist-info/entry_points.txt" || \
    { echo "missing entry_points.txt" >&2; exit 2; }

# Build the .deb
dpkg-deb --build "$STAGING" "$OUT"

# Echo metadata for evidence
echo "---"
echo "Built: $OUT"
dpkg-deb -I "$OUT"
echo "---"
echo "Contents:"
dpkg-deb -c "$OUT"
