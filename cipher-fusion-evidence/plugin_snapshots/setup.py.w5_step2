"""CIPHER CP 5.1 — vLLM KV-cache buffer-ownership plugin (Option A).

Packaging note (Step 3 build): the pod ships setuptools 59.6.0, which predates
the PEP 621 ``[project]`` table (needs >=61) and PEP 660 editable installs
(needs >=64). A ``pyproject.toml``-only package therefore installs as
``UNKNOWN-0.0.0`` with the entry point silently dropped. This explicit
``setup.py`` declares the metadata and the ``vllm.general_plugins`` entry point
the way setuptools 59.6.0 understands, so ``pip install -e`` registers the
hook with no network and no setuptools upgrade. All metadata lives here.
"""
from setuptools import setup

setup(
    name="cipher-vllm-kv",
    version="0.2.0",
    description="CIPHER CP 5.1 + 5.2 + W5 — vLLM KV buffer-ownership + offload + cross-tenant dedup",
    python_requires=">=3.10",
    py_modules=["cipher_vllm_kv", "cipher_kv_offload", "cipher_vllm_kvdedup"],
    # vLLM's load_general_plugins() discovers this group and runs each
    # register() in every process, including the EngineCore subprocess.
    # ORDER MATTERS: kvdedup must register AFTER cipher_vllm_kv per
    # WEEK_5_STEP_1_DESIGN_MEMO.md R-W5.3 + Part D.3. dict iteration order
    # in Python 3.7+ = insertion order; vLLM's load_general_plugins iterates
    # plugins.values() in that order.
    entry_points={
        "vllm.general_plugins": [
            "cipher_vllm_kv      = cipher_vllm_kv:register",       # CP 5.1 (+ chained CP 5.2)
            "cipher_vllm_kvdedup = cipher_vllm_kvdedup:register",  # W5 Step 2 — MUST be after cipher_vllm_kv
        ],
    },
)
