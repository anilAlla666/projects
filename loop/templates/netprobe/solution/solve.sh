#!/bin/sh
# An application reply is the only evidence of reachability: Harbor's egress sidecar may accept a TCP connect
# locally even when the destination is blocked.
python3 - <<'PY'
import json, os, urllib.error, urllib.request
res = {}
for name, url in {"gateway": "https://inference.hud.ai/v1/models", "github": "https://github.com",
                  "raw_github": "https://raw.githubusercontent.com/harbor-framework/terminal-bench/v4.0.0/README.md",
                  "huggingface": "https://huggingface.co", "pypi": "https://pypi.org/simple/"}.items():
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            res[name] = f"reachable (http {r.status})"
    except urllib.error.HTTPError as e:
        res[name] = f"reachable (http {e.code})"
    except Exception as e:
        res[name] = f"blocked ({type(e).__name__})"
os.makedirs("/out", exist_ok=True)
json.dump(res, open("/out/net.json", "w"), indent=1)
print(res)
PY
