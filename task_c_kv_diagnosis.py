"""TASK C: KV update timing — DynamicCache (torch.cat) vs StaticCache (in-place index_copy_)."""
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
CTX = 2048
N_DECODE = 50

print(f"[load] {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
model.train(False)
cfg = model.config

print(f"\n=== KV update timing: DynamicCache vs StaticCache ===", flush=True)
print(f"  ctx_len={CTX}, n_decode={N_DECODE}, fp16, single decoder layer probed via cuda events", flush=True)
print(f"\n  {'cache_type':<14} {'batch':>5} {'kv_update_us':>14} {'forward_us':>11} {'kv/fwd':>7}", flush=True)
print("  " + "-" * 58, flush=True)

results = []
for cache_type in ["dynamic", "static"]:
    for batch in [1, 8, 32, 64]:
        try:
            torch.manual_seed(0)
            prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(batch, CTX), device="cuda")

            if cache_type == "dynamic":
                cache = DynamicCache()
                with torch.no_grad():
                    out = model(input_ids=prompt, past_key_values=cache, use_cache=True)
                past = out.past_key_values
            else:
                cache = StaticCache(config=cfg, max_batch_size=batch, max_cache_len=CTX + N_DECODE + 10,
                                     device="cuda", dtype=torch.float16)
                with torch.no_grad():
                    cp = torch.arange(CTX, device="cuda", dtype=torch.long)
                    out = model(input_ids=prompt, cache_position=cp,
                                past_key_values=cache, use_cache=True, return_dict=True)
                past = out.past_key_values

            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()

            # Hook layer 16's KV update with cuda events
            kv_times = []
            fwd_times = []
            target = model.model.layers[16]

            orig_attn = target.self_attn.__class__.forward
            def instrumented(self, hidden_states, position_embeddings, attention_mask=None,
                             past_key_values=None, **kwargs):
                from transformers.models.mistral.modeling_mistral import (
                    apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
                )
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None:
                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    k, v = past_key_values.update(k, v, self.layer_idx)
                    e.record()
                    kv_times.append((s, e))
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )
                attn_output, attn_weights = attention_interface(
                    self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling,
                    sliding_window=getattr(self.config, "sliding_window", None), **kwargs)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
            import types
            target.self_attn.forward = types.MethodType(instrumented, target.self_attn)

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    if cache_type == "static":
                        cp = torch.tensor([CTX + len([])], device="cuda", dtype=torch.long)
                        out = model(input_ids=nxt, cache_position=cp,
                                    past_key_values=past, use_cache=True, return_dict=True)
                    else:
                        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()

            kv_times.clear()

            t0 = time.perf_counter()
            for i in range(N_DECODE):
                with torch.no_grad():
                    if cache_type == "static":
                        cp = torch.tensor([CTX + 5 + i], device="cuda", dtype=torch.long)
                        out = model(input_ids=nxt, cache_position=cp,
                                    past_key_values=past, use_cache=True, return_dict=True)
                    else:
                        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            avg_fwd_us = elapsed / N_DECODE * 1e6

            kv_us = [s.elapsed_time(e) * 1000.0 for s, e in kv_times]
            avg_kv_us = sum(kv_us) / len(kv_us) if kv_us else 0
            print(f"  {cache_type:<14} {batch:>5} {avg_kv_us:>14.1f} {avg_fwd_us:>11.1f} {avg_kv_us/avg_fwd_us:>6.1%}", flush=True)
            results.append({"cache": cache_type, "batch": batch, "kv_us": avg_kv_us, "fwd_us": avg_fwd_us})

            target.self_attn.forward = types.MethodType(orig_attn, target.self_attn)
            del prompt, past
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  {cache_type:<14} {batch:>5} OOM", flush=True)
            torch.cuda.empty_cache()
            target.self_attn.forward = types.MethodType(orig_attn, target.self_attn)

print(f"\n=== Verdict ===", flush=True)
print("If StaticCache is dramatically faster than DynamicCache at batch=32, the 'kv_update bug' is", flush=True)
print("just measurement artifact — use StaticCache. No CUDA kernel needed.", flush=True)

import json
with open("/home/ubuntu/op31-prod-fix/task_c_kv_diagnosis.json", "w") as f:
    json.dump(results, f, indent=2)
