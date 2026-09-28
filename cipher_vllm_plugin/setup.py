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
    version="0.1.0",
    description="CIPHER CP 5.1 — vLLM KV-cache buffer-ownership plugin (Option A)",
    python_requires=">=3.10",
    py_modules=["cipher_vllm_kv", "cipher_kv_offload"],
    # vLLM's load_general_plugins() discovers this group and runs register()
    # in every process, including the EngineCore subprocess.
    entry_points={
        "vllm.general_plugins": [
            "cipher_vllm_kv = cipher_vllm_kv:register",
        ],
    },
)
