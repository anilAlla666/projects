#!/bin/bash
# Passes an attempt only if GitHub answers an HTTP request while grading. A bare TCP connect is not evidence:
# Harbor's egress sidecar redirects every connection to a local proxy, so the connect succeeds even when blocked.
mkdir -p /logs/verifier
reply=$(timeout 8 bash -c 'exec 3<>/dev/tcp/github.com/80 && printf "HEAD / HTTP/1.0\r\nHost: github.com\r\n\r\n" >&3 && head -c 12 <&3' 2>/dev/null)
echo "reply: ${reply:-none}"
case "$reply" in HTTP/*) echo 1 ;; *) echo 0 ;; esac > /logs/verifier/reward.txt
