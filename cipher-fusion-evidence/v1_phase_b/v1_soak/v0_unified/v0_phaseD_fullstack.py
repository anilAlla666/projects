#!/usr/bin/env python3
# V.0 v2 PHASE D -- the full-stack INSTRUMENTED pass. Arm the actuators (auto-init ON, NOT disabled) under the robust
# CUDA_INJECTION64_PATH injection, drive a real compute-bound + decode workload, then read EVERY exported counter the
# deployed libcipher_rt.so already maintains. NO .so change (read-only counters). This is the "what fires in the
# background across the whole stack" snapshot. Honest about which ops engage given the workload.
import os, sys, ctypes, json, time
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
# Phase D is launched with CUDA_INJECTION64_PATH=$SO + actuator env by the orchestrator; here we read the live counters.
import torch
MODEL=os.environ.get("GO3_MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
lib=ctypes.CDLL(SO)
# ---- bind every scalar counter getter (unsigned long / int). try/except per symbol = robust + honest "not-exported". ----
ULONG=[ # label, symbol
 ("matmul.calls_total","cipher_rt_matmul_calls_total"),("matmul.calls_handled","cipher_rt_matmul_calls_handled"),
 ("matmul.calls_passthrough","cipher_rt_matmul_calls_passthrough"),
 ("classify.calls_total","cipher_rt_classify_calls_total"),("classify.calls_handled","cipher_rt_classify_calls_handled"),
 ("classify.calls_passthrough","cipher_rt_classify_calls_passthrough"),
 ("classify.classifications_count","cipher_workload_classifications_count"),
 ("marlin.is_active","cipher_rt_marlin_is_active"),("marlin.calls_total","cipher_rt_marlin_calls_total"),
 ("marlin.calls_handled","cipher_rt_marlin_calls_handled"),("marlin.weights_quantized","cipher_rt_marlin_weights_quantized"),
 ("marlin.bf16_observed","cipher_rt_marlin_calls_bf16_observed"),("marlin.bf16_substituted","cipher_rt_marlin_calls_bf16_substituted"),
 ("marlin.skipped_by_classifier","cipher_rt_marlin_calls_skipped_by_classifier"),("marlin.weights_count","cipher_rt_marlin_engine_weights_count"),
 ("koopman.is_active","cipher_rt_koopman_is_active"),("koopman.calls_total","cipher_rt_koopman_calls_total"),
 ("koopman.calls_handled","cipher_rt_koopman_calls_handled"),("koopman.calls_skipped","cipher_rt_koopman_calls_skipped"),
 ("koopman.remember_emits","cipher_rt_koopman_remember_emits"),("koopman.bf16_observed","cipher_rt_koopman_bf16_observed"),
 ("fp8.is_active","cipher_rt_fp8_is_active"),("fp8.calls_total","cipher_rt_fp8_calls_total"),
 ("fp8.calls_handled","cipher_rt_fp8_calls_handled"),("fp8.calls_skipped","cipher_rt_fp8_calls_skipped"),
 ("fp8.weights_quantized","cipher_rt_fp8_weights_quantized"),("fp8.max_n","cipher_rt_fp8_max_n"),
 ("attn.calls_total","cipher_rt_attn_calls_total"),("attn.calls_handled","cipher_rt_attn_calls_handled"),
 ("attn.calls_passthrough","cipher_rt_attn_calls_passthrough"),("attn.calls_redirected","cipher_rt_attn_calls_redirected"),
 ("attn.tramp_calls","cipher_rt_attn_tramp_calls"),
 ("geom.attn_intercepts","cipher_rt_geom_attn_intercepts"),("geom.launches","cipher_rt_geom_launches"),("geom.capture","cipher_rt_geom_capture"),
 ("volt.mode","cipher_rt_volt_mode"),("volt.locked_mhz","cipher_rt_volt_locked_mhz"),
 ("volt.classifier_poll","cipher_rt_volt_classifier_poll"),("volt.classifier_engagements","cipher_rt_volt_classifier_engagements"),
 ("volt.classifier_skipped","cipher_rt_volt_classifier_skipped"),
 ("cublas.gemmEx_shim_calls","cipher_rt_cublas_shim_calls"),("cublas.lt_shim_calls","cipher_rt_cublaslt_shim_calls"),
 ("cublas.lt_variant_calls","cipher_rt_cublaslt_variant_calls"),("dlsym.intercepts","cipher_rt_dlsym_intercepts"),
 ("smp.total_launches","cipher_rt_smp_total_launches"),("smp.small_launches","cipher_rt_smp_small_launches"),
 ("smp.longest_streak","cipher_rt_smp_longest_streak"),
 ("green.is_initialized","cipher_rt_green_ctx_is_initialized"),("green.streams_observed","cipher_rt_green_ctx_streams_observed"),
 ("green.sm_count","cipher_rt_green_ctx_sm_count"),
 ("pr.observed_count","cipher_rt_pr_observed_count"),("pr.configured_count","cipher_rt_pr_configured_count"),
 ("audit.count","cipher_rt_audit_count"),("audit.enabled","cipher_rt_audit_enabled"),("commit.total_count","cipher_rt_commit_total_count"),
 ("ring.total_written","cipher_rt_ring_total_written"),("ring.total_throttled","cipher_rt_ring_total_throttled"),
 ("ring.total_dropped","cipher_rt_ring_total_dropped"),
 ("sense.transitions_total","cipher_rt_sense_transition_transitions_total"),("sense.proposals_queued","cipher_rt_sense_transition_proposals_queued"),
 ("sense.proposals_pushed","cipher_rt_sense_transition_proposals_pushed"),("sense.session_count","cipher_sense_session_count"),
 ("pool.distinct_fp_rejected","cipher_rt_pool_distinct_fp_rejected"),("fairness.tenant_count","cipher_fairness_tenant_count"),
 ("workload.class_current","cipher_workload_class_current"),("dispatch.is_live","cipher_rt_dispatch_is_live"),
 ("workload.model_fingerprint","cipher_workload_model_fingerprint"),
]
def read_counter(sym):
    try:
        fn=getattr(lib,sym); fn.restype=ctypes.c_ulong; return int(fn())
    except Exception: return None

def drive_workload():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(MODEL)
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    # compute-bound forwards (n>64 -> FP8 gate; classifier sees large GEMMs; FlashAttn intercept)
    ids=tok("The history of artificial intelligence and high performance computing. "*40,return_tensors="pt").input_ids[:, :1024].cuda()
    with torch.no_grad():
        for _ in range(6): _=m(ids).logits
        torch.cuda.synchronize()
    # decode steps (n<=64 -> Marlin/decode regime; small launches -> SM packer)
    cur=ids[:, :32]
    with torch.no_grad():
        for _ in range(16):
            nt=m(cur).logits[:, -1:].argmax(-1); cur=torch.cat([cur,nt],dim=1)
        torch.cuda.synchronize()
    return m.config

print(f"[v0-D] full-stack instrumented pass: CUDA_INJECTION64_PATH={'set' if os.environ.get('CUDA_INJECTION64_PATH') else 'UNSET'} "
      f"FP8={os.environ.get('CIPHER_FP8')} VOLT={os.environ.get('CIPHER_VOLT')} MARLIN={os.environ.get('CIPHER_MARLIN')} KOOPMAN={os.environ.get('CIPHER_KOOPMAN')}",flush=True)
cfg=drive_workload()
# ---- read EVERY counter ----
counters={}
for label,sym in ULONG:
    counters[label]=read_counter(sym)
# class name
try:
    lib.cipher_workload_class_name.restype=ctypes.c_char_p; lib.cipher_workload_class_name.argtypes=[ctypes.c_int]
    cls=counters.get("workload.class_current") or 0
    counters["workload.class_name"]=lib.cipher_workload_class_name(cls).decode()
except Exception: counters["workload.class_name"]=None
# volt status string
try:
    lib.cipher_rt_volt_status_string.restype=ctypes.c_char_p
    counters["volt.status_string"]=lib.cipher_rt_volt_status_string().decode()
except Exception: counters["volt.status_string"]=None
# profile struct engage flags (the live classifier decision vector)
class Prof(ctypes.Structure):
    _fields_=[("workload_class",ctypes.c_int),("classified_at_ns",ctypes.c_uint64),("confidence",ctypes.c_uint32),
              ("obs",ctypes.c_uint32),("_pad",ctypes.c_uint32),
              ("volt_engage",ctypes.c_uint8),("marlin_engage",ctypes.c_uint8),("koopman_engage",ctypes.c_uint8),
              ("kv_dedup_engage",ctypes.c_uint8),("weight_arena_engage",ctypes.c_uint8),("sm_packer_engage",ctypes.c_uint8),
              ("per_tenant_routing",ctypes.c_uint8),("pool_substrate",ctypes.c_uint8),("flash_attn_intercept",ctypes.c_uint8),
              ("nccl_tuner",ctypes.c_uint8),("persistent_kernel",ctypes.c_uint8),("speculative_decode",ctypes.c_uint8),
              ("graph_engine",ctypes.c_uint8),("flop_telemetry",ctypes.c_uint8),("audit_chain",ctypes.c_uint8)]
try:
    lib.cipher_workload_profile_get.restype=ctypes.POINTER(Prof)
    p=lib.cipher_workload_profile_get()
    if p:
        pr=p.contents
        counters["profile.engage"]={k:int(getattr(pr,k)) for k in ("volt_engage","marlin_engage","koopman_engage","kv_dedup_engage",
            "weight_arena_engage","sm_packer_engage","per_tenant_routing","pool_substrate","flash_attn_intercept","nccl_tuner",
            "persistent_kernel","speculative_decode","graph_engine","flop_telemetry","audit_chain")}
        counters["profile.confidence"]=int(pr.confidence); counters["profile.obs"]=int(pr.obs)
except Exception as e: counters["profile.engage"]=None
print("V0D_JSON "+json.dumps(counters),flush=True)
# human view
print("[v0-D] === LIVE COUNTER SNAPSHOT (non-None) ===",flush=True)
for k,v in counters.items():
    if v not in (None,0,{}) or k.endswith("is_active") or k.endswith("class_name"):
        print(f"    {k:<34} = {v}",flush=True)
sys.stdout.flush(); os._exit(0)
