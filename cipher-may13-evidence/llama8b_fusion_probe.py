#!/usr/bin/env python3
"""Per-kernel fusion ablation probe for Llama-3.1-8B fp16.

Tests each fused kernel INDIVIDUALLY (RMSNorm | SiLU·Mul | residual_add | RoPE)
to isolate which one breaks output coherence.  Also measures numerical error
vs PyTorch native after layer 0.

Usage:
    LD_PRELOAD="...libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so" \
        python3 llama8b_fusion_probe.py
"""
import argparse, ctypes, os, sys, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")
import warnings; warnings.filterwarnings("ignore")
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/home/ubuntu/models/Llama-3.1-8B"
PROMPT_BASE = ("Energy efficiency means doing more useful work per watt. "
               "The future of GPU computing is to make every joule count. ")


def make_prompt(tok, length=128):
    ids_one = tok(PROMPT_BASE, return_tensors='pt').input_ids[0]
    n = (length + len(ids_one) - 1) // len(ids_one)
    return ids_one.repeat(n)[:length].unsqueeze(0).to('cuda:0')


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map={'':0})
    return model, tok


def setup_rt():
    rt = ctypes.CDLL(os.path.join(ROOT, 'libcipher_rt.so'),
                     mode=ctypes.RTLD_GLOBAL)
    rt.cipher_fused_rmsnorm.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    rt.cipher_fused_silu_mul.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_silu_mul.restype = ctypes.c_int
    rt.cipher_fused_residual_add.argtypes = [ctypes.c_void_p]*3 + [
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_residual_add.restype = ctypes.c_int
    rt.cipher_substitute_v2_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p]
    rt.cipher_substitute_v2_compile.restype = ctypes.c_ulong
    rt.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
    rt.cipher_substitute_v2_get_function.restype = ctypes.c_void_p
    return rt


def patch_rmsnorm(rt):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    _orig = LlamaRMSNorm.forward
    def _patched(self, x):
        out = torch.empty_like(x)
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        of = out.reshape(-1, x.shape[-1])
        sp = ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        rc = rt.cipher_fused_rmsnorm(
            flat.data_ptr(), self.weight.data_ptr(), of.data_ptr(),
            flat.shape[0], flat.shape[-1], float(self.variance_epsilon), sp)
        return out if rc == 1 else _orig(self, x)
    LlamaRMSNorm.forward = _patched


def patch_silu_mul(rt):
    from transformers.models.llama.modeling_llama import LlamaMLP
    _orig = LlamaMLP.forward
    def _patched(self, x):
        g = self.gate_proj(x); u = self.up_proj(x)
        if not g.is_contiguous(): g = g.contiguous()
        if not u.is_contiguous(): u = u.contiguous()
        out = torch.empty_like(g)
        sp = ctypes.c_void_p(torch.cuda.current_stream(g.device).cuda_stream)
        rc = rt.cipher_fused_silu_mul(
            g.data_ptr(), u.data_ptr(), out.data_ptr(), g.numel(), sp)
        if rc != 1:
            out = torch.nn.functional.silu(g) * u
        return self.down_proj(out)
    LlamaMLP.forward = _patched


def patch_residual_add(rt):
    """Patch LlamaDecoderLayer to use fused residual_add."""
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    def _residual_add(x, residual):
        if not x.is_contiguous(): x = x.contiguous()
        if not residual.is_contiguous(): residual = residual.contiguous()
        out = torch.empty_like(x)
        sp = ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
        rc = rt.cipher_fused_residual_add(
            x.data_ptr(), residual.data_ptr(), out.data_ptr(),
            x.numel(), sp)
        return out if rc == 1 else x + residual
    def _decoder_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_values=None,
                         use_cache=False, cache_position=None,
                         position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs)
        hidden_states = _residual_add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _residual_add(hidden_states, residual)
        return hidden_states
    LlamaDecoderLayer.forward = _decoder_forward


ROPE_SRC = b"""
#include <cuda_fp16.h>
extern "C" __global__ void cipher_fused_rope_qk(
    __half* __restrict__ Q_out, __half* __restrict__ K_out,
    const __half* __restrict__ Q_in, const __half* __restrict__ K_in,
    const __half* __restrict__ cos, const __half* __restrict__ sin,
    int B, int Hq, int Hkv, int S, int D, int cos_batch_stride)
{
    int half_D = D / 2;
    int d = threadIdx.x;
    if (d >= D) return;
    int b = blockIdx.y;
    int row_idx = blockIdx.x;
    int total_q_rows = Hq * S;
    bool is_q = row_idx < total_q_rows;
    int local = is_q ? row_idx : row_idx - total_q_rows;
    int H = is_q ? Hq : Hkv;
    int h = local / S;
    int s = local % S;
    const __half* x_in;
    __half* x_out;
    if (is_q) {
        x_in  = Q_in  + ((b * Hq  + h) * S + s) * D;
        x_out = Q_out + ((b * Hq  + h) * S + s) * D;
    } else {
        x_in  = K_in  + ((b * Hkv + h) * S + s) * D;
        x_out = K_out + ((b * Hkv + h) * S + s) * D;
    }
    int cos_offset = b * cos_batch_stride + s * D + d;
    float c  = __half2float(cos[cos_offset]);
    float si = __half2float(sin[cos_offset]);
    extern __shared__ float shm[];
    shm[d] = __half2float(x_in[d]);
    __syncthreads();
    float x_self = shm[d];
    float x_pair, sign;
    if (d < half_D) { x_pair = shm[d + half_D]; sign = -1.0f; }
    else            { x_pair = shm[d - half_D]; sign =  1.0f; }
    x_out[d] = __float2half(x_self * c + sign * x_pair * si);
}
"""


def patch_rope(rt):
    from transformers.models.llama import modeling_llama as LM
    _orig = LM.apply_rotary_pos_emb
    cubin = rt.cipher_substitute_v2_compile(ROPE_SRC, b"cipher_fused_rope_qk")
    fn = rt.cipher_substitute_v2_get_function(cubin)
    if not fn:
        print("[probe] RoPE compile failed; using orig")
        return
    libcuda = ctypes.CDLL("libcuda.so.1")
    cu_launch = libcuda.cuLaunchKernel
    cu_launch.argtypes = [ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    cu_launch.restype = ctypes.c_int

    def _patched(q, k, cos, sin, unsqueeze_dim=1):
        B, Hq, S, D = q.shape
        _, Hkv, _, _ = k.shape
        cos_c = cos.contiguous(); sin_c = sin.contiguous()
        # PyTorch ships transposed views; force contiguous.
        q_in = q.contiguous(); k_in = k.contiguous()
        cos_bs = 0 if (cos_c.dim() == 2 or cos_c.shape[0] == 1) else S * D
        q_out = torch.empty_like(q_in); k_out = torch.empty_like(k_in)
        args_list = [
            ctypes.c_void_p(q_out.data_ptr()),
            ctypes.c_void_p(k_out.data_ptr()),
            ctypes.c_void_p(q_in.data_ptr()),
            ctypes.c_void_p(k_in.data_ptr()),
            ctypes.c_void_p(cos_c.data_ptr()),
            ctypes.c_void_p(sin_c.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(Hq), ctypes.c_int(Hkv),
            ctypes.c_int(S), ctypes.c_int(D), ctypes.c_int(cos_bs),
        ]
        args_arr = (ctypes.c_void_p * len(args_list))(*[
            ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_list])
        sp = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
        rc = cu_launch(fn, (Hq+Hkv)*S, B, 1, D, 1, 1, D*4, sp, args_arr, None)
        if rc != 0:
            return _orig(q, k, cos, sin, unsqueeze_dim)
        return q_out, k_out
    LM.apply_rotary_pos_emb = _patched


def gen50(model, tok, prompt_ids, label):
    with torch.no_grad():
        out = model.generate(prompt_ids,
                             attention_mask=torch.ones_like(prompt_ids),
                             max_new_tokens=50, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    new = out[0, prompt_ids.shape[1]:].tolist()
    text_full = tok.decode(new, skip_special_tokens=False)
    text_skip = tok.decode(new, skip_special_tokens=True)
    eos_id = tok.eos_token_id
    eos_idx = next((i for i, t in enumerate(new) if t == eos_id), -1)
    print(f"  [{label:18}] new_tokens={len(new)} eos@={eos_idx} "
          f"text_skip={text_skip[:60]!r}", flush=True)
    if eos_idx >= 0 and eos_idx < 5:
        print(f"  [{label:18}] full incl eos: {text_full[:80]!r}")
    return new, eos_idx


def hidden_after_layer0(model, prompt_ids):
    """Capture hidden_states output of layer 0."""
    captured = {}
    def hook(mod, inputs, output):
        # LlamaDecoderLayer returns torch.Tensor.
        captured['out'] = output if not isinstance(output, tuple) else output[0]
    h = model.model.layers[0].register_forward_hook(hook)
    with torch.no_grad():
        _ = model(prompt_ids, attention_mask=torch.ones_like(prompt_ids))
    h.remove()
    return captured['out'].detach().clone()


def numerical_error(a, b, label):
    diff = (a.float() - b.float()).abs()
    rel = diff / (b.float().abs() + 1e-6)
    print(f"  [{label}] max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} "
          f"max_rel={rel.max().item():.3e} "
          f"mean_rel={rel.mean().item():.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "numerical"], default="gen")
    ap.add_argument("--rms",  action="store_true")
    ap.add_argument("--silu", action="store_true")
    ap.add_argument("--add",  action="store_true")
    ap.add_argument("--rope", action="store_true")
    args = ap.parse_args()

    rt = setup_rt()
    print(f"[probe] loading Llama-3.1-8B fp16 on cuda:0...", flush=True)
    t0 = time.time()
    model, tok = load_model()
    print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)
    prompt_ids = make_prompt(tok, length=128)

    # ── Reference: NO patches ──────────────────────────────────────────
    if args.mode == "gen":
        gen50(model, tok, prompt_ids, "BASELINE (no patch)")
    else:
        ref_h = hidden_after_layer0(model, prompt_ids)

    # ── Apply requested patches ────────────────────────────────────────
    label_parts = []
    if args.rms:  patch_rmsnorm(rt);     label_parts.append("rms")
    if args.silu: patch_silu_mul(rt);    label_parts.append("silu")
    if args.add:  patch_residual_add(rt); label_parts.append("add")
    if args.rope: patch_rope(rt);        label_parts.append("rope")
    if not label_parts:
        return
    label = "+".join(label_parts)

    if args.mode == "gen":
        gen50(model, tok, prompt_ids, label)
    else:
        with_h = hidden_after_layer0(model, prompt_ids)
        numerical_error(with_h, ref_h, label)


if __name__ == "__main__":
    main()
