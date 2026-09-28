#!/bin/sh
# cipher_cdi_patch.sh - thin shell wrapper around cipher_cdi_patch.py.
#
# Stable interface for postinst/prerm/cipher-platform CLI to invoke.
# Decouples Python module evolution from the shell-facing entry point.
#
# Usage: cipher_cdi_patch.sh {apply|revert|status} [--cdi-path PATH] [--json]
set -e
exec /usr/bin/python3 /usr/lib/cipher/cipher_cdi_patch.py "$@"
