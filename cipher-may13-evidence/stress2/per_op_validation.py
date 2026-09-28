"""Per-op validation: prove each CIPHER op fires, doesn't crash, output coherent.

For each op: enable env, run 50-tok Llama-3.2-1B, query op's _report() symbol,
parse the JSON it dumps to /tmp/, classify FIRES / SILENT / CRASH.
"""
import os, sys, json, time, ctypes, gc, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()


# (op_name, env_var_to_set, list_of_report_files_or_symbols, optional symbol-counter accessor)
# Each op that has a *_report() function dumps to /tmp/cipher_<op>_report.json
# when that symbol is invoked.  We invoke it then read the file and look for
# nonzero counters.
OPS = [
    # (name, env_to_enable, report_func_name, json_path, key_to_check)
    ("PREDICT",         "CIPHER_PREDICT",         "cipher_predict_report",
                        "/tmp/cipher_predict_report.json",         "observe_calls"),
    ("KOOPMAN",         "CIPHER_ATTN_KOOPMAN",    "cipher_attn_fsm_report",
                        "/tmp/cipher_attn_fsm_report.json",        "saw_qk"),
    ("ARBITRATE",       "CIPHER_ARBITRATE",       None,
                        None,                                       None),  # no report; check via topology
    ("TOPOLOGY",        "CIPHER_TOPOLOGY",        "cipher_topology_report",
                        "/tmp/cipher_topology_report.json",        "n_gpus"),
    ("PIPELINE",        "CIPHER_PIPELINE",        "cipher_pipeline_report",
                        "/tmp/cipher_pipeline_report.json",        "stage_count"),
    ("CONTINUITY",      "CIPHER_CONTINUITY",      "cipher_continuity_report",
                        "/tmp/cipher_continuity_report.json",      "checkpoints"),
    ("LOOP",            "CIPHER_LOOP",            "cipher_loop_report",
                        "/tmp/cipher_loop_report.json",            "loops_detected"),
    ("GUARD",           "CIPHER_GUARD",           "cipher_guard_report",
                        "/tmp/cipher_guard_report.json",           "checks"),
    ("DETERMINISM",     "CIPHER_DETERMINISM",     "cipher_determinism_report",
                        "/tmp/cipher_determinism_report.json",     "observe_calls"),
    ("TRACE",           "CIPHER_TRACE",           "cipher_trace_report",
                        "/tmp/cipher_trace_report.json",           "events"),
    ("CARBON",          "CIPHER_CARBON",          "cipher_carbon_report",
                        "/tmp/cipher_carbon_report.json",          "joules"),
    ("RECEIPT",         "CIPHER_RECEIPT",         "cipher_receipt_report",
                        "/tmp/cipher_receipt_report.json",         "receipts"),
    ("COMPLY",          "CIPHER_COMPLY",          "cipher_comply_report",
                        "/tmp/cipher_comply_report.json",          "audit_lines"),
    ("WEIGHT_SHARE",    "CIPHER_WEIGHT_SHARE",    None,                  None, None),
    ("KV_REDIRECT",     "CIPHER_KV_REDIRECT",     None,                  None, None),
    ("GRAPH_ENGINE",    "CIPHER_GRAPH",           "cipher_graph_report",
                        "/tmp/cipher_graph_report.json",          "captures"),
    ("FUSION_RESIDUAL", "CIPHER_FUSION_KERNELS",  None,                  None, None),
    ("PERSIST_ENGINE",  "CIPHER_PERSIST_ENGINE",  "cipher_persist_engine_report",
                        "/tmp/cipher_persist_engine_report.json",  "register_calls"),
    ("FLOW_PATTERNS",   "CIPHER_FLOW_PATTERNS",   None,                  None, None),
    ("FLOW_RECORDER",   "CIPHER_FLOW_RECORD",     None,                  None, None),
    ("SUBSTITUTE_V2",   "CIPHER_SUBSTITUTE_V2",   None,                  None, None),
]


def run_workload():
    """Run a 50-token Llama-3.2-1B forward.  Returns (coherent, error_str_or_None)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/Llama-3.2-1B")
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "/home/ubuntu/models/Llama-3.2-1B", torch_dtype=torch.float16,
            device_map={"": "cuda:0"})
        model.requires_grad_(False); model.train(False)
        ids = tok("Energy efficiency means", return_tensors="pt").input_ids.to("cuda:0")
        attn = torch.ones_like(ids)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=50,
                                  do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        coherent = ("!!!!" not in text) and len(text.strip()) > 8
        del model, tok
        gc.collect(); torch.cuda.empty_cache()
        return (coherent, text[:80], None)
    except Exception as e:
        return (False, "", f"{type(e).__name__}: {str(e)[:200]}")


def call_report(rt, fn_name):
    if fn_name is None: return None
    try:
        fn = getattr(rt, fn_name)
        fn.restype = None
        fn()
        return True
    except Exception as e:
        return f"err: {e}"


def parse_report(json_path, key):
    if json_path is None: return None
    if not os.path.exists(json_path): return None
    try:
        d = json.load(open(json_path))
        if key in d: return d[key]
        # Search for any nonzero numeric value
        for k, v in d.items():
            if isinstance(v, (int, float)) and v != 0: return d
        return d
    except Exception as e:
        return f"err: {e}"


def main():
    # Set every op's env BEFORE loading the rt so each op's constructor
    # reads its enable flag. Otherwise init() early-returns and the op
    # never wakes up.
    for spec in OPS:
        os.environ[spec[1]] = "on"
    rt = ctypes.CDLL(str(sc.ROOT / "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
    print(f"[per-op] starting validation of {len(OPS)} ops "
          f"(envs set pre-load, all on)", flush=True)
    results = []
    for spec in OPS:
        name, env, fn_name, json_path, key = spec
        # Wipe the json so we can detect fresh writes.
        if json_path and os.path.exists(json_path):
            try: os.remove(json_path)
            except: pass
        # Set the env var and re-init the op (idempotent).
        os.environ[env] = "on"
        # Try to call init for the op (best-effort).
        for init_name in [f"cipher_{name.lower()}_init",
                           f"cipher_attn_koopman_enabled" if name == "KOOPMAN" else None]:
            if init_name and hasattr(rt, init_name):
                try:
                    init_fn = getattr(rt, init_name)
                    init_fn.restype = ctypes.c_int
                    init_fn()
                except Exception: pass
        try:
            ok, text, err = run_workload()
        except Exception as e:
            ok, text, err = False, "", f"{type(e).__name__}: {e}"
        if err:
            results.append(dict(op=name, status="CRASH", err=err))
            continue
        # Call report().
        report_called = call_report(rt, fn_name)
        report_data = parse_report(json_path, key) if fn_name else None
        # Determine fired-ness.
        fired = "UNKNOWN"
        evidence = ""
        if name == "WEIGHT_SHARE":
            try:
                rt.cipher_weight_share_active_count.restype = ctypes.c_int
                cnt = rt.cipher_weight_share_active_count()
                evidence = f"active_count={cnt}"
                fired = "FIRES" if cnt > 0 else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = f"no symbol: {e}"
        elif name == "KV_REDIRECT":
            try:
                rt.cipher_kv_redirect_enabled.restype = ctypes.c_int
                en = rt.cipher_kv_redirect_enabled()
                evidence = f"enabled={en}"
                fired = "FIRES" if en > 0 else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = f"no symbol: {e}"
        elif name == "FUSION_RESIDUAL":
            try:
                # cipher_fused_residual_add is the kernel; the question is
                # whether ANYTHING calls it during the workload.  We don't
                # patch it here, so it'll be SILENT.  Mark accordingly.
                fired = "SILENT"
                evidence = "no Python patch installed"
            except Exception as e:
                fired = "SILENT"; evidence = str(e)
        elif name == "FLOW_PATTERNS":
            try:
                rt.cipher_flow_patterns_recipe_count.restype = ctypes.c_int
                rc = rt.cipher_flow_patterns_recipe_count()
                evidence = f"recipe_count={rc}"
                fired = "FIRES" if rc > 0 else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = str(e)
        elif name == "PERSIST_ENGINE":
            try:
                rt.cipher_persist_engine_enabled.restype = ctypes.c_int
                en = rt.cipher_persist_engine_enabled()
                evidence = f"enabled={en} report_data={report_data}"
                fired = "FIRES" if en > 0 else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = str(e)
        elif name == "SUBSTITUTE_V2":
            try:
                rt.cipher_substitute_v2_enabled.restype = ctypes.c_int
                en = rt.cipher_substitute_v2_enabled()
                evidence = f"enabled={en}"
                fired = "FIRES" if en > 0 else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = str(e)
        elif name == "KOOPMAN":
            try:
                rt.cipher_attn_fsm_saw_qk_count.restype = ctypes.c_uint64
                rt.cipher_attn_fsm_saw_softmax_count.restype = ctypes.c_uint64
                qk = rt.cipher_attn_fsm_saw_qk_count()
                sm = rt.cipher_attn_fsm_saw_softmax_count()
                evidence = f"saw_qk={qk} saw_softmax={sm}"
                fired = "FIRES" if (qk > 0 or sm > 0) else "SILENT"
            except Exception as e:
                fired = "SILENT"; evidence = str(e)
        elif name == "ARBITRATE":
            evidence = "no exported counter"
            fired = "SILENT"
        else:
            # Generic: read report json, look for nonzero counters
            evidence = f"report={report_data}"
            if isinstance(report_data, dict):
                fired = "FIRES" if any(
                    isinstance(v, (int, float)) and v != 0
                    for v in report_data.values()
                ) else "SILENT"
            elif isinstance(report_data, (int, float)) and report_data != 0:
                fired = "FIRES"
            else:
                fired = "SILENT"
        results.append(dict(op=name, env=env, status=fired,
                            coherent=ok, evidence=evidence,
                            text=text[:60]))
        print(f"  {name:<18} {fired:<8} coh={ok} {evidence[:80]}",
              flush=True)
    out_path = os.path.join(os.path.dirname(__file__), "per_op_validation.json")
    with open(out_path, "w") as f:
        json.dump(dict(results=results), f, indent=2)
    print(f"[per-op] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
