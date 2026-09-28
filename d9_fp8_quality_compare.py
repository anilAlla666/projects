#!/usr/bin/env python3
# Offline: combine CIPHER-off (ref bf16) and CIPHER-on (fp8) quality runs -> PPL/MMLU delta + true KL.
import json, torch, math
ref=json.load(open("/home/ubuntu/d9_fp8_quality_ref.json"))
fp8=json.load(open("/home/ubuntu/d9_fp8_quality_fp8.json"))
ppl_delta=100.0*(fp8["ppl"]-ref["ppl"])/ref["ppl"]
mmlu_delta=fp8["mmlu"]-ref["mmlu"]
# true KL(p_ref || q_fp8), mean per token over saved windows
a=torch.load("/home/ubuntu/d9_q_ref_logprobs.pt", weights_only=True); b=torch.load("/home/ubuntu/d9_q_fp8_logprobs.pt", weights_only=True)
kl=0.0; n=0
for pa,pb in zip(a,b):
    p=pa.float().exp(); kl+=float((p*(pa.float()-pb.float())).sum(-1).sum()); n+=pa.shape[0]
kl/=max(n,1)
# bar amendment context: locked 0.3% PPL / 0.5pp MMLU; 0.37% is the recorded amendment
ppl_pass03=ppl_delta<=0.3; ppl_pass037=ppl_delta<=0.37; mmlu_pass=abs(mmlu_delta)<=0.5
out={"scheme":"CIPHER per-tensor scalar FP8 E4M3 (transparent, ALL layers, driver-boundary cublasGemmEx)",
     "ref(bf16,CIPHER-off)":{"ppl":ref["ppl"],"mmlu":ref["mmlu"]},
     "fp8(CIPHER-on)":{"ppl":fp8["ppl"],"mmlu":fp8["mmlu"]},
     "ppl_delta_pct":round(ppl_delta,4),"mmlu_delta_abs":round(mmlu_delta,3),"kl_true_nats":round(kl,5),
     "ppl_pass(<=0.3%)":ppl_pass03,"ppl_pass(<=0.37% amended)":ppl_pass037,"mmlu_pass(<=0.5pp)":mmlu_pass,
     "kl_diag(<=0.01)":kl<=0.01,
     "WITHIN_BAR_0.3":ppl_pass03 and mmlu_pass,"WITHIN_BAR_0.37":ppl_pass037 and mmlu_pass,
     "ppl_tokens":ref["ppl_tokens"],"mmlu_n":ref["mmlu_n"]}
json.dump(out,open("/home/ubuntu/d9_fp8_quality_result.json","w"),indent=2)
print(json.dumps(out,indent=2))
print(f"\nPPL {ref['ppl']:.4f}->{fp8['ppl']:.4f} = {ppl_delta:+.4f}% | MMLU {ref['mmlu']:.2f}->{fp8['mmlu']:.2f} = {mmlu_delta:+.3f}pp | KL={kl:.5f}")
print(f"GATE: within 0.3% bar={out['WITHIN_BAR_0.3']} ; within 0.37% amended={out['WITHIN_BAR_0.37']}")
