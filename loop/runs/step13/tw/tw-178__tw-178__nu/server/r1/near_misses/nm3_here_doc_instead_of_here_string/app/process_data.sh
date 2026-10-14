#!/bin/bash

read_config() {
    cat << CONFIG
APP_NAME: DataProcessor
VERSION: 2.0
MODE: production
DEBUG: false
CONFIG
}

parse_config() {
    local config_data="$1"
    cat << INNER_EOF > /app/config_summary.txt
=== Configuration Summary ===
$config_data
INNER_EOF
}

transform_value() {
    local input_val="$1"
    # Bug: Using here-document (<<) instead of here-string (<<<)
    doubled=$(bc << EOF
$input_val * 2
EOF
)
    echo "Input value: $input_val"
    echo "Doubled value: $doubled"
}

compare_data() {
    diff <(cat /app/input_a.txt) <(cat /app/input_b.txt)
}

main() {
    echo "=== Configuration Summary ===" > /app/output.txt

    config=$(read_config)
    echo "$config" >> /app/output.txt
    echo "" >> /app/output.txt

    parse_config "$config"

    echo "=== Transformed Values ===" >> /app/output.txt
    transform_value 42 >> /app/output.txt
    echo "" >> /app/output.txt

    echo "=== Data Comparison ===" >> /app/output.txt
    compare_data >> /app/output.txt
    echo "" >> /app/output.txt

    echo "=== Processing Complete ===" >> /app/output.txt
}

main
