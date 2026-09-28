# cipher-platform v2.0 .deb plugin chain manifest

**Substep:** B.2'' substep 2
**Date staged:** 2026-05-26
**Purpose:** declares which Python plugin chain files ship in the v2.0 .deb under `/usr/lib/python3/dist-packages/`, their source repos + md5 anchors, and importability verification. The .deb build script (`build_deb.sh` at substep 8) copies these files from the canonical source repos at build time per the anchors below; this manifest is the authoritative source of truth.

## File manifest

| Target path in .deb | Source path | md5 | Source repo + anchor |
|---|---|---|---|
| `usr/lib/python3/dist-packages/cipher_vllm_kv.py` | `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py` | `b89a9b6e6a0bd973cedffa2674872e6c` | cipher_vllm_plugin (no git; tracked by md5) |
| `usr/lib/python3/dist-packages/cipher_vllm_kvdedup.py` | `/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kvdedup.py` | `438e40230e700af30661886757e5b7f3` | cipher_vllm_plugin |
| `usr/lib/python3/dist-packages/cipher_kv_offload.py` | `/home/ubuntu/cipher_vllm_plugin/cipher_kv_offload.py` | `d1cd202db246648098eaab498c87cd33` | cipher_vllm_plugin |
| `usr/lib/python3/dist-packages/cipher_model_fingerprint.py` | `/home/ubuntu/cipher-fusion-evidence/phase_c/cipher_model_fingerprint.py` | `15111e0f4bd5555c562cefd370161127` | cipher-fusion-evidence phase_c |
| `usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `5a3db034ea2b950bf426eb48140ec8fd` | cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init` |
| `usr/lib/python3/dist-packages/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so` | `/home/ubuntu/cipher_rt_phase4/build_py312/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so` | `00df2811bcf1f15231fd599f14a268ae` | Built 2026-05-26 inside vllm/vllm-openai:v0.21.0 container against its py312+torch; src cipher_kv_bridge.cpp from cipher_rt_phase4 `8613812e`. F-B.3.2 remediation. |
| `usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/METADATA` | `/home/ubuntu/vllm_env/lib/python3.10/site-packages/cipher_vllm_kv-0.2.0.dist-info/METADATA` | `9fc4459da8e104b5b8869ad9992c7efa` | vllm_env reference install (pip install -e cipher_vllm_plugin output) |
| `usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/WHEEL` | (same dir) | `0be2025a54a57ff313e1e195f0f36c31` | vllm_env reference install |
| `usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/entry_points.txt` | (same dir) | `f97aab12adad156be2e6c77e581a5376` | vllm_env reference install |
| `usr/lib/python3/dist-packages/cipher_vllm_kv-0.2.0.dist-info/top_level.txt` | (same dir) | `62d6be7639c0b39297aac720c5d70034` | vllm_env reference install |

## dist-info contents NOT shipped

These are pip-install-specific and not needed for .deb-shipped dist-info that is dpkg-managed:

- `RECORD` (pip uninstall manifest; dpkg manages files via .deb's data archive)
- `INSTALLER` (tracks which tool installed; dpkg is the installer for .deb)
- `REQUESTED` (pip-explicit-install marker; not applicable)
- `direct_url.json` (pip editable-install marker; not applicable)

## entry_points.txt content

```ini
[vllm.general_plugins]
cipher_vllm_kv = cipher_vllm_kv:register
cipher_vllm_kvdedup = cipher_vllm_kvdedup:register
```

`cipher_vllm_kv:register` is the load-bearing CP 5.1 hook for vLLM KV buffer ownership. `cipher_vllm_kvdedup:register` is the Week 5 cross-tenant KV-dedup hook. `cipher_kv_offload` + `cipher_model_fingerprint` are imported transitively by the above two; not separate entry points.

## Importability verification (2026-05-26, this pod)

Substep 2 verifies the staged dist-packages tree is importable in a clean Python 3.10 process with `PYTHONPATH=/tmp/cipher-deb-build/cipher-platform/usr/lib/python3/dist-packages`. Result:

```
vllm.general_plugins entries:
  cipher_vllm_kv = cipher_vllm_kv:register
  cipher_vllm_kvdedup = cipher_vllm_kvdedup:register

plugin .py + .so importable:
  cipher_vllm_kv         resolved to staged path
  cipher_vllm_kvdedup    resolved to staged path
  cipher_kv_offload      resolved to staged path
  cipher_model_fingerprint   resolved to staged path
  cipher_kv_bridge       resolved to staged path
```

importlib.metadata picks up both entry-points; modules import without error. Substep 2 verification PASS.

## Anchors UNCHANGED through B.2'' substep 2

- cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`
- libcipher_rt.so md5 `1d91e7da`
- cipher_kmod `8c643fc` tag `week-13-14-complete`
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e`
