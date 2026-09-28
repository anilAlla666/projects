"""ACTION A: Per-operation timing breakdown of one Mistral-7B decoder layer.

Monkey-patches one layer's attention/MLP forward to insert cuda Events around each op.
Runs at batch=1, 8, 32 with 2K KV context. Single decode step (seq=1).
"""
import os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
CTX = 2048
N_REPEAT = 200       # average over many calls
TARGET_LAYER = 16    # mid-stack layer

# Per-event times (us) per repeat
TIMINGS = {}


def make_event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _record(label, start, end):
    TIMINGS.setdefault(label, []).append(start.elapsed_time(end) * 1000.0)  # ms -> us


# Patch ONE attention's forward with instrumented version
def make_instrumented_attn(orig_attn_module, layer_idx):
    """Return a callable that wraps the attention forward and times each op."""
    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
    )
    config = orig_attn_module.config

    def instrumented(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if layer_idx != TARGET_LAYER:
            # Not our layer — call original behavior
            return orig_unwrapped_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q_proj
        s1, e1 = make_event_pair(); s1.record()
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e1.record()

        # k_proj
        s2, e2 = make_event_pair(); s2.record()
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e2.record()

        # v_proj
        s3, e3 = make_event_pair(); s3.record()
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        e3.record()

        # RoPE
        s4, e4 = make_event_pair(); s4.record()
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        e4.record()

        # KV cache update
        s5, e5 = make_event_pair(); s5.record()
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        e5.record()

        # Attention (full SDPA)
        s6, e6 = make_event_pair(); s6.record()
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )
        e6.record()

        # reshape
        s7, e7 = make_event_pair(); s7.record()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        e7.record()

        # o_proj
        s8, e8 = make_event_pair(); s8.record()
        attn_output = self.o_proj(attn_output)
        e8.record()

        # Stash events
        EVENTS_LIST.append([
            ("q_proj", s1, e1), ("k_proj", s2, e2), ("v_proj", s3, e3),
            ("rope", s4, e4), ("kv_update", s5, e5),
            ("sdpa", s6, e6), ("reshape", s7, e7), ("o_proj", s8, e8),
        ])
        return attn_output, attn_weights

    orig_unwrapped_forward = orig_attn_module.__class__.forward
    return instrumented


def make_instrumented_decoder(orig_layer_class, layer_idx):
    """Wrap MistralDecoderLayer.forward to time block-level ops."""
    orig_forward = orig_layer_class.forward

    def instrumented(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_values=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kwargs):
        if layer_idx != TARGET_LAYER:
            return orig_forward(self, hidden_states=hidden_states, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_values=past_key_values,
                                output_attentions=output_attentions, use_cache=use_cache,
                                cache_position=cache_position, position_embeddings=position_embeddings,
                                **kwargs)

        # Match standard Mistral forward but with timing
        residual = hidden_states

        # input_layernorm
        s, e = make_event_pair(); s.record()
        hs = self.input_layernorm(hidden_states)
        e.record()
        EXTRA_EVENTS.append(("input_ln", s, e))

        # Self-attention (uses our instrumented attn forward)
        s, e = make_event_pair(); s.record()
        hs, _ = self.self_attn(hidden_states=hs,
                                position_embeddings=position_embeddings,
                                attention_mask=attention_mask,
                                past_key_values=past_key_values,
                                cache_position=cache_position,
                                **kwargs)
        e.record()
        EXTRA_EVENTS.append(("attn_total", s, e))

        # Residual add
        s, e = make_event_pair(); s.record()
        hs = residual + hs
        e.record()
        EXTRA_EVENTS.append(("attn_residual_add", s, e))

        # Post-attn layernorm
        residual = hs
        s, e = make_event_pair(); s.record()
        hs = self.post_attention_layernorm(hs)
        e.record()
        EXTRA_EVENTS.append(("post_attn_ln", s, e))

        # MLP (we time gate, up, silu*mul, down within mlp)
        # Call the MLP but time its components individually
        s_mlp_total, e_mlp_total = make_event_pair(); s_mlp_total.record()

        s, e = make_event_pair(); s.record()
        gate = self.mlp.gate_proj(hs)
        e.record()
        EXTRA_EVENTS.append(("gate_proj", s, e))

        s, e = make_event_pair(); s.record()
        up = self.mlp.up_proj(hs)
        e.record()
        EXTRA_EVENTS.append(("up_proj", s, e))

        s, e = make_event_pair(); s.record()
        intermediate = F.silu(gate) * up
        e.record()
        EXTRA_EVENTS.append(("silu_mul", s, e))

        s, e = make_event_pair(); s.record()
        mlp_out = self.mlp.down_proj(intermediate)
        e.record()
        EXTRA_EVENTS.append(("down_proj", s, e))

        e_mlp_total.record()
        EXTRA_EVENTS.append(("mlp_total", s_mlp_total, e_mlp_total))

        # Residual add
        s, e = make_event_pair(); s.record()
        hs = residual + mlp_out
        e.record()
        EXTRA_EVENTS.append(("mlp_residual_add", s, e))

        return hs

    return instrumented


EVENTS_LIST = []   # list of [(label, start, end), ...]
EXTRA_EVENTS = []  # decoder-layer-level events


def patch_one_layer(model, layer_idx):
    layer = model.model.layers[layer_idx]
    # Patch this layer's forward
    new_fwd = make_instrumented_decoder(type(layer), layer_idx)
    import types
    layer.forward = types.MethodType(new_fwd, layer)
    # Patch the attention forward
    new_attn_fwd = make_instrumented_attn(layer.self_attn, layer_idx)
    layer.self_attn.forward = types.MethodType(new_attn_fwd, layer.self_attn)


def measure_at_batch(model, tok, batch_size):
    global EVENTS_LIST, EXTRA_EVENTS, TIMINGS
    cfg = model.config
    torch.manual_seed(0)
    prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(batch_size, CTX), device="cuda")
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Reset event list
    EVENTS_LIST.clear()
    EXTRA_EVENTS.clear()
    TIMINGS.clear()

    # Timed run
    for _ in range(N_REPEAT):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Extract timings
    for events in EVENTS_LIST:
        for label, s, e in events:
            _record(label, s, e)
    for label, s, e in EXTRA_EVENTS:
        _record(label, s, e)

    # Print breakdown
    print(f"\n=== batch={batch_size}, ctx={CTX}, layer={TARGET_LAYER}, n_iter={N_REPEAT} ===", flush=True)
    print(f"  {'op':<22} {'mean_us':>10} {'std_us':>9} {'min_us':>9} {'max_us':>9}", flush=True)
    print("  " + "-" * 65, flush=True)

    op_order = [
        "input_ln", "q_proj", "k_proj", "v_proj", "rope", "kv_update",
        "sdpa", "reshape", "o_proj", "attn_total", "attn_residual_add",
        "post_attn_ln", "gate_proj", "up_proj", "silu_mul", "down_proj",
        "mlp_total", "mlp_residual_add",
    ]
    layer_data = {}
    for op in op_order:
        if op not in TIMINGS: continue
        vals = TIMINGS[op]
        m = sum(vals) / len(vals)
        sd = (sum((v-m)**2 for v in vals) / len(vals)) ** 0.5
        layer_data[op] = m
        print(f"  {op:<22} {m:>10.1f} {sd:>9.1f} {min(vals):>9.1f} {max(vals):>9.1f}", flush=True)

    # GEMM vs non-GEMM
    gemm_ops = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    gemm_total = sum(layer_data.get(op, 0) for op in gemm_ops)
    non_gemm_ops = ["input_ln", "rope", "kv_update", "sdpa", "reshape",
                    "attn_residual_add", "post_attn_ln", "silu_mul", "mlp_residual_add"]
    non_gemm_total = sum(layer_data.get(op, 0) for op in non_gemm_ops)
    layer_total = layer_data.get("attn_total", 0) + layer_data.get("mlp_total", 0) + \
                  layer_data.get("input_ln", 0) + layer_data.get("post_attn_ln", 0) + \
                  layer_data.get("attn_residual_add", 0) + layer_data.get("mlp_residual_add", 0)
    print(f"\n  GEMM total:     {gemm_total:>8.1f} us  ({gemm_total/layer_total*100:.1f}% of layer)", flush=True)
    print(f"  non-GEMM total: {non_gemm_total:>8.1f} us  ({non_gemm_total/layer_total*100:.1f}% of layer)", flush=True)
    print(f"  attn block:     {layer_data.get('attn_total', 0):>8.1f} us  ({layer_data.get('attn_total', 0)/layer_total*100:.1f}%)", flush=True)
    print(f"  MLP block:      {layer_data.get('mlp_total', 0):>8.1f} us  ({layer_data.get('mlp_total', 0)/layer_total*100:.1f}%)", flush=True)
    print(f"  SDPA only:      {layer_data.get('sdpa', 0):>8.1f} us  ({layer_data.get('sdpa', 0)/layer_total*100:.1f}%)", flush=True)
    print(f"  layer total:    {layer_total:>8.1f} us", flush=True)

    return layer_data


def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    patch_one_layer(model, TARGET_LAYER)

    all_results = {}
    for B in [1, 8, 32]:
        try:
            d = measure_at_batch(model, tok, B)
            all_results[B] = d
        except torch.cuda.OutOfMemoryError as e:
            print(f"  batch={B}: OOM", flush=True)
            torch.cuda.empty_cache()

    with open("/home/ubuntu/op31-prod-fix/action_a_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] action_a_results.json", flush=True)


if __name__ == "__main__":
    main()
