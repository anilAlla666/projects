"""V1 Phase B.1 R-B.1 entry-point registration POC.

Bundled via cipher-rb1-poc_0.0.1_amd64.deb at /usr/lib/python3/dist-packages/.
Entry point declared at cipher_rb1_poc-0.0.1.dist-info/entry_points.txt under
[vllm.general_plugins] group as `cipher_rb1_poc = cipher_rb1_poc:register`.

The POC measures whether importlib.metadata.entry_points discovers this entry
post `apt install`. Does NOT validate that vLLM's load_general_plugins
actually invokes register() at engine init (that is B.3 Gate A scope under
full vLLM-on-CIPHER execution). B.1 POC scope is discovery-only.

Per V1_PHASE_B_SCOPE_LOCK.md section 4 B.1 R-B.1 sub-element + Anil
2026-05-26 B.1 spec.
"""
import sys


def register():
    print("[cipher-rb1-poc] POC entry point fired", file=sys.stderr)
