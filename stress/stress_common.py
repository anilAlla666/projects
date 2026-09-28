"""Shared helpers for the CIPHER stress tests.

Loads Llama-3.1-8B fp16 on a single CUDA device and (optionally) wires
CIPHER FP8 + fusion the same way step9_llama70b.py does.

Env contract (set BEFORE python imports torch):
    LD_PRELOAD=libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so
    CIPHER_FP8_COMPUTE=on
    CIPHER_SUBSTITUTE_V2=on
    CIPHER_FUSION_KERNELS=on
"""
import ctypes, os, sys, time, threading, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "/home/ubuntu/models/Llama-3.1-8B"


def _setup_alloc_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def init_cipher(use_fp8=True, use_fusion=True):
    """Bind libcipher_rt.so and call init for FP8 / fusion modules.

    Returns (rt_handle_or_None, fp8_stats_fn, fusion_stats_fn).
    """
    if not (use_fp8 or use_fusion):
        return None, lambda: {}, lambda: {}

    rt = ctypes.CDLL(str(ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

    fp8_stats_fn = lambda: {}
    if use_fp8:
        rt.cipher_fp8_compute_init.restype = ctypes.c_int
        rt.cipher_fp8_compute_init()

        class _S(ctypes.Structure):
            _fields_ = [
                ("enabled", ctypes.c_int),
                ("weights_quantized", ctypes.c_uint64),
                ("matmul_calls", ctypes.c_uint64),
                ("passthroughs", ctypes.c_uint64),
                ("correctness_failures", ctypes.c_uint64),
                ("bytes_fp16", ctypes.c_size_t),
                ("bytes_fp8", ctypes.c_size_t),
            ]
        rt.cipher_fp8_compute_stats.argtypes = [ctypes.POINTER(_S)]
        rt.cipher_fp8_compute_stats.restype = ctypes.c_int

        def _fp8():
            s = _S()
            rt.cipher_fp8_compute_stats(ctypes.byref(s))
            return dict(weights=s.weights_quantized, calls=s.matmul_calls,
                        passthroughs=s.passthroughs,
                        failures=s.correctness_failures)
        fp8_stats_fn = _fp8

    fusion_stats_fn = lambda: {}
    if use_fusion:
        class _F(ctypes.Structure):
            _fields_ = [
                ("enabled", ctypes.c_int),
                ("rmsnorm_calls", ctypes.c_uint64),
                ("silu_mul_calls", ctypes.c_uint64),
                ("residual_add_calls", ctypes.c_uint64),
            ]
        rt.cipher_fusion_kernels_stats.argtypes = [ctypes.POINTER(_F)]
        rt.cipher_fusion_kernels_stats.restype = ctypes.c_int

        def _fus():
            s = _F()
            rt.cipher_fusion_kernels_stats(ctypes.byref(s))
            return dict(rmsnorm=s.rmsnorm_calls,
                        silu_mul=s.silu_mul_calls,
                        residual_add=s.residual_add_calls)
        fusion_stats_fn = _fus
        try:
            rt.cipher_fusion_kernels_init.restype = ctypes.c_int
            rt.cipher_fusion_kernels_init()
        except AttributeError:
            pass

    return rt, fp8_stats_fn, fusion_stats_fn


def patch_fusion(rt, model):
    """Mirror step9_llama70b.py's RMSNorm + SiLU.Mul + residual_add patches.

    Argument order matches the C signatures in include/cipher_fusion_kernels.h:
        cipher_fused_rmsnorm(x, weight, out, rows, hidden_dim, eps, stream)
        cipher_fused_silu_mul(gate, up, out, numel, stream)
        cipher_fused_residual_add(x, residual, out, numel, stream)
    Return code 1 is success; anything else falls back to PyTorch.

    Also rebinds accelerate's `_old_forward` per-instance — without that bypass,
    accelerate captures the original `forward` at hook-install time and the
    class-method patch never runs.
    """
    import torch, types
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer)

    rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    rt.cipher_fused_silu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_silu_mul.restype = ctypes.c_int
    rt.cipher_fused_residual_add.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_residual_add.restype = ctypes.c_int

    _orig_rms = LlamaRMSNorm.forward
    _orig_mlp = LlamaMLP.forward

    def _stream_ptr_for(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def _try_fused_rmsnorm(x, weight, eps):
        dev = x.device
        with torch.cuda.device(dev):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            out = torch.empty_like(x)
            out_flat = out.reshape(-1, x.shape[-1])
            rc = rt.cipher_fused_rmsnorm(
                flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                flat.shape[0], flat.shape[-1], float(eps),
                _stream_ptr_for(dev))
        return out, (rc == 1)

    def _try_fused_silu_mul(gate, up):
        dev = gate.device
        if not gate.is_contiguous(): gate = gate.contiguous()
        if not up.is_contiguous():   up   = up.contiguous()
        with torch.cuda.device(dev):
            out = torch.empty_like(gate)
            rc = rt.cipher_fused_silu_mul(
                gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                gate.numel(), _stream_ptr_for(dev))
        return out, (rc == 1)

    def _try_fused_residual_add(x, residual):
        dev = x.device
        if not x.is_contiguous():        x = x.contiguous()
        if not residual.is_contiguous(): residual = residual.contiguous()
        with torch.cuda.device(dev):
            out = torch.empty_like(x)
            rc = rt.cipher_fused_residual_add(
                x.data_ptr(), residual.data_ptr(), out.data_ptr(),
                x.numel(), _stream_ptr_for(dev))
        return out, (rc == 1)

    def _patched_rmsnorm_forward(self, x):
        # Bug-2 fix: CIPHER's NVRTC fusion kernels are fp16-only. bf16 / fp32
        # / int8 inputs would be misinterpreted as fp16 and emit garbage.
        # Fall back to the original PyTorch path for any non-fp16 dtype.
        if x.dtype != torch.float16 or self.weight.dtype != torch.float16:
            return _orig_rms(self, x)
        out, ok = _try_fused_rmsnorm(x, self.weight, self.variance_epsilon)
        if ok: return out
        return _orig_rms(self, x)
    LlamaRMSNorm.forward = _patched_rmsnorm_forward

    def _patched_mlp_forward(self, x):
        if x.dtype != torch.float16:
            return _orig_mlp(self, x)
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        if gate.dtype != torch.float16 or up.dtype != torch.float16:
            mul = torch.nn.functional.silu(gate) * up
            return self.down_proj(mul)
        mul, ok = _try_fused_silu_mul(gate, up)
        if not ok:
            mul = torch.nn.functional.silu(gate) * up
        return self.down_proj(mul)
    LlamaMLP.forward = _patched_mlp_forward

    def _residual_add(x, residual):
        out, ok = _try_fused_residual_add(x, residual)
        if ok: return out
        return x + residual

    # Note: residual_add fusion would require patching LlamaDecoderLayer.forward,
    # whose call signature varies across transformers versions; we skip it
    # here and let PyTorch's fp16 add run unchanged. step9_llama70b.py patches
    # it but the version-coupling makes it brittle for stress runs.

    n_rms = n_mlp = 0
    for m in model.modules():
        if isinstance(m, LlamaRMSNorm) and hasattr(m, '_old_forward'):
            m._old_forward = types.MethodType(_patched_rmsnorm_forward, m)
            n_rms += 1
        elif isinstance(m, LlamaMLP) and hasattr(m, '_old_forward'):
            m._old_forward = types.MethodType(_patched_mlp_forward, m)
            n_mlp += 1
    return dict(n_rms=n_rms, n_mlp=n_mlp)


def load_model(device="cuda:0", patch=True, rt=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.requires_grad_(False)
    model.train(False)
    load_s = time.perf_counter() - t0
    if patch and rt is not None:
        patch_fusion(rt, model)
    return model, tok, load_s


def power_sampler(stop, samples, n_gpus=None, interval=0.15):
    """Append (timestamp, total_watts, total_mem_used_mib) tuples."""
    try:
        import pynvml
        pynvml.nvmlInit()
        if n_gpus is None:
            n_gpus = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]
        while not stop.is_set():
            try:
                w = sum(pynvml.nvmlDeviceGetPowerUsage(h)/1000.0 for h in handles)
                m = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles) // (1024*1024)
                samples.append((time.time(), w, m))
            except Exception:
                pass
            time.sleep(interval)
        try: pynvml.nvmlShutdown()
        except Exception: pass
    except Exception:
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=power.draw,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1)
                w = 0.0; m = 0
                for line in r.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        w += float(parts[0]); m += int(parts[1])
                samples.append((time.time(), w, m))
            except Exception:
                pass
            time.sleep(interval)


def _physical_gpu_for_logical(logical_index=0):
    """Map a CUDA-visible (logical) device index to the physical
    nvidia-smi device index, honoring CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return logical_index
    items = [x.strip() for x in cvd.split(",") if x.strip() != ""]
    if logical_index < len(items):
        try:
            return int(items[logical_index])
        except ValueError:
            return logical_index
    return logical_index


def nvsmi_mem_used_mib(device_index=0):
    """Memory.used in MiB on a logical CUDA device, mapped through
    CUDA_VISIBLE_DEVICES to the right nvidia-smi --id."""
    phys = _physical_gpu_for_logical(device_index)
    r = subprocess.run(
        ["nvidia-smi", f"--id={phys}",
         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=2)
    return int(r.stdout.strip())
