# Config loader. Real deployable uses /etc/cipher/platform.json (py3.10 target -> JSON,
# not TOML). All hard-coded paths in the validated handlers are overridden via env the
# router injects, sourced from here.
import json, os

DEFAULT = "/etc/cipher/platform.json"

def load(path=None):
    p = path or os.environ.get("CIPHER_PLATFORM_CONFIG", DEFAULT)
    with open(p) as f:
        cfg = json.load(f)
    cfg.setdefault("so_path", "/usr/lib/cipher/libcipher_rt.so")
    cfg.setdefault("handlers_dir", "/usr/share/cipher/handlers")
    cfg.setdefault("model_library_dir", "/home/ubuntu/models")
    return cfg
