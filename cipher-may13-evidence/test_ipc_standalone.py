#!/usr/bin/env python3
"""Verify the cipher_weight_share IPC mechanism on raw cudaMalloc'd
buffers (no PyTorch, no caching allocator).
"""
import ctypes, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    role = sys.argv[1]  # 'publisher' or 'subscriber'

    cudart = ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice(0)
    rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                     mode=ctypes.RTLD_GLOBAL)
    rt.cipher_weight_share_init.restype = ctypes.c_int
    rt.cipher_weight_share_init()
    rt.cipher_weight_share_export.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    rt.cipher_weight_share_export.restype = ctypes.c_int
    rt.cipher_weight_share_lookup.argtypes = [ctypes.c_void_p]
    rt.cipher_weight_share_lookup.restype = ctypes.c_void_p

    cudart.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemset.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

    SIZE = 1 << 20  # 1 MB
    HASH = 0xCAFEBEEF12345678

    if role == "publisher":
        ptr = ctypes.c_void_p()
        cudart.cudaMalloc(ctypes.byref(ptr), SIZE)
        # Write a recognizable pattern.
        cudart.cudaMemset(ptr, 0xAB, SIZE)
        # Two observes to trigger export — but we'll call export directly.
        slot = rt.cipher_weight_share_export(ptr, SIZE, HASH)
        print(f"[publisher] cudaMalloc ptr={ptr.value:#x} slot={slot}",
              flush=True)
        if slot < 0:
            print("[publisher] export FAILED")
            sys.exit(1)
        print("[publisher] holding for 30s...", flush=True)
        time.sleep(30)
    elif role == "subscriber":
        # Different process — make a SEPARATE cudaMalloc to "shadow" the
        # publisher's pointer.  We'll then export it (which finds the
        # existing slot via content_hash and IMPORTS).
        ptr = ctypes.c_void_p()
        cudart.cudaMalloc(ctypes.byref(ptr), SIZE)
        cudart.cudaMemset(ptr, 0x11, SIZE)  # different content; we'll
                                              # use the same hash to trigger
                                              # the existing-slot path.
        slot = rt.cipher_weight_share_export(ptr, SIZE, HASH)
        print(f"[subscriber] cudaMalloc ptr={ptr.value:#x} slot={slot}",
              flush=True)
        if slot < 0:
            print("[subscriber] import FAILED")
            sys.exit(1)
        # Now lookup should give us the publisher's mapped pointer.
        shared = rt.cipher_weight_share_lookup(ptr)
        print(f"[subscriber] lookup({ptr.value:#x}) -> {shared}", flush=True)
        if not shared:
            print("[subscriber] lookup empty"); sys.exit(1)
        # Read first byte from shared mapping — should be publisher's 0xAB,
        # NOT our 0x11.
        host = (ctypes.c_uint8 * 16)()
        cudart.cudaMemcpy(host, shared, 16, 2)  # 2 = D2H
        print(f"[subscriber] shared[0..16] = "
              f"{[hex(host[i]) for i in range(16)]}", flush=True)


if __name__ == "__main__":
    main()
