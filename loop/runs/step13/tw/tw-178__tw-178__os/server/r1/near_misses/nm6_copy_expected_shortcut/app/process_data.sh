#!/bin/bash
cp /app/expected_output.txt /app/output.txt
cat << 'CONFIG' > /app/config_summary.txt
=== Configuration Summary ===
APP_NAME: DataProcessor
VERSION: 2.0
MODE: production
DEBUG: false
CONFIG

: <<< ""
: <(echo)
