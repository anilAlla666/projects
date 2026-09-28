"""Phase 4.0.9 — tenant registration helper.

Sets up CIPHER_TENANT_ID and (implicitly) CUDA_INJECTION64_PATH so that
libcipher_v2 stamps the tenant identity at cuInit. Drivers import and
call register() at process start.

The actual REGISTER_TENANT ioctl is issued by libcipher_v2.so when CUDA
initializes, not from Python directly. Python just sets the env vars
and confirms the tenant ID. Verification (that the registration landed)
happens via /proc/cipher/stats inspection.
"""
import os
import sys


CIPHER_DEV = "/dev/cipher"
LIBCIPHER_V2 = os.environ.get(
    "CIPHER_V2_LIB", "/home/ubuntu/libcipher_v2/libcipher_v2.so"
)


def register(default_tenant=None):
    """Confirm CIPHER_TENANT_ID and CUDA_INJECTION64_PATH are set.
    Returns the tenant_id in use. Aborts cleanly if the kmod isn't loaded."""
    if not os.path.exists(CIPHER_DEV):
        print(f"[tenant_register] {CIPHER_DEV} not present — cipher_kmod not loaded?",
              file=sys.stderr, flush=True)
        return None

    tenant_id = os.environ.get("CIPHER_TENANT_ID")
    if not tenant_id and default_tenant:
        os.environ["CIPHER_TENANT_ID"] = default_tenant
        tenant_id = default_tenant
    if not tenant_id:
        tenant_id = f"wl_{os.getpid()}"
        os.environ["CIPHER_TENANT_ID"] = tenant_id

    if "CUDA_INJECTION64_PATH" not in os.environ:
        os.environ["CUDA_INJECTION64_PATH"] = LIBCIPHER_V2

    print(f"[tenant_register] CIPHER_TENANT_ID={tenant_id} "
          f"CUDA_INJECTION64_PATH={os.environ.get('CUDA_INJECTION64_PATH')} "
          f"pid={os.getpid()}",
          file=sys.stderr, flush=True)

    return tenant_id
