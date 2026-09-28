#!/usr/bin/env python3
# WI-1 TRAINING-GOODPUT / BADPUT-AVOIDED accounting from the measured runs. Every input traces to a JSON.
import json, os
def L(p): return json.load(open(p))
A=L('A_step.json'); B=L('B_step.json'); C=L('C_step.json'); D=L('D_step.json')
Bc=L('B_counts.json'); Cc=L('C_counts.json')
ckpt=L('wi1_steptime.json').get('ckpt_write_s')   # real LoRA checkpoint write time from the probe

t_s = A['step_time_s_median']                      # baseline step time (fwd+bwd+optimizer), GPU-seconds
# per-check detector cost: a check-step recomputes the step's checked GEMMs ≈ one extra (p90 captures check steps)
per_check = C['step_time_s_p90'] - C['step_time_s_median']
def ovh(N): return per_check/(N*t_s)               # amortized fractional overhead at check-period N
lat = {"N8": 4, "N45": 25}                          # MEASURED detection latency (first_det_step - onset)

# badput-avoided per fault = (discovery_delay_without - N) * t_s  [last-good-checkpoint cancels]
def badput_avoided_gpus(discovery_delay, N): return (discovery_delay - N) * t_s
def breakeven_rate(discovery_delay, N, ovh_): return ovh_/(discovery_delay - N)   # faults/step (size-invariant)

# effective-training-time over W steps, F faults; wasted ~ (delay + C/2) without, (N + C/2) with; C=ckpt cadence.
# coverage fraction p: only p of SDCs land in a covered (fp16 linear-GEMM) op and are caught-within-N; the (1-p)
# uncovered SDCs revert to the WITHOUT discovery cost while STILL paying the continuous detector overhead.
def eff_time(W, F, discovery_delay, N, ovh_, Cstep, p=1.0):
    without = W/(W + F*(discovery_delay + Cstep/2))
    wasted_with = p*F*(N + Cstep/2) + (1-p)*F*(discovery_delay + Cstep/2)
    with_   = W/(W*(1+ovh_) + wasted_with)
    return without, with_

Cstep = 100   # checkpoint cadence (steps); justified: amortizes the (small here, large in real 7B) checkpoint write
out = {
  "goodput_definition": "TRAINING (compute-progress-time / total-time); SDC detection = badput reduction",
  "real_measured": {
    "step_time_s_median": round(t_s,4), "step_time_s_mean": round(A['step_time_s_mean'],4),
    "checkpoint_write_s_LoRA": ckpt, "per_check_cost_s": round(per_check,4),
    "detection_latency_steps": lat, "max_detect_residual": Cc['max_detect_residual'],
    "false_positives_clean": {"detections": L('D_counts.json')['detections'], "max_clean_residual": L('D_counts.json')['max_clean_residual'],
                              "gemms_checked": L('D_counts.json')['gemms_checked']},
    "detector_overhead_per_check_method": {"N8_pct": round(100*ovh(8),2), "N45_pct": round(100*ovh(45),2)},
    "detector_overhead_60step_mean_noisy": {"N8_pct": round(100*(C['step_time_s_mean']/A['step_time_s_mean']-1),1),
                                            "N45_pct": round(100*(B['step_time_s_mean']/A['step_time_s_mean']-1),1)},
  },
  "checkpoint_cadence_steps": Cstep,
  "badput_avoided_per_fault_GPUseconds": {},
  "breakeven_fault_rate_per_step": {},
  "effective_training_time_examples": {},
}
# report across a range of without-discovery-delays (the modeled axis): next-ckpt (~C/2), 1 ckpt (C), silent (10*C)
for label,delay in [("next_checkpoint_~C/2=50", 50), ("one_checkpoint_C=100", 100), ("silent_10C=1000", 1000)]:
    out["badput_avoided_per_fault_GPUseconds"][label] = {
        "N8": round(badput_avoided_gpus(delay,4),2), "N45": round(badput_avoided_gpus(delay,25),2)}
    out["breakeven_fault_rate_per_step"][label] = {
        "N8_1_per_steps": round(1/breakeven_rate(delay,4,ovh(8))), "N45_1_per_steps": round(1/breakeven_rate(delay,25,ovh(45)))}
# effective-time example: W=10000 steps; coverage p=1.0 (covered-op fault) AND p=0.5 (half SDCs uncovered)
for label,(W,F,delay,N,o,p) in {
    "W10k_1fault_silent1000_N8_p1.0": (10000, 1, 1000, 4, ovh(8), 1.0),
    "W10k_5faults_silent1000_N8_p1.0": (10000, 5, 1000, 4, ovh(8), 1.0),
    "W10k_5faults_silent1000_N8_p0.5": (10000, 5, 1000, 4, ovh(8), 0.5),
    "W10k_1fault_nextckpt50_N8_p1.0": (10000, 1, 50, 4, ovh(8), 1.0),
}.items():
    wo,wi = eff_time(W,F,delay,N,o,Cstep,p)
    out["effective_training_time_examples"][label] = {"coverage_p": p, "without": round(wo,4), "with": round(wi,4), "delta_pp": round(100*(wi-wo),3)}
out["coverage_note"] = ("eff_time WIN assumes the fault lands in a COVERED fp16 linear-GEMM op (p). Uncovered SDCs "
  "(attention/lm_head/non-GEMM/mercurial-core) are caught with p=0 and revert to WITHOUT cost + overhead. Realized "
  "fleet win = covered-SDC fraction x the p=1 band; net-negative regime widens as p falls.")
json.dump(out, open('wi1_accounting.json','w'), indent=2)
print(json.dumps(out, indent=2))
