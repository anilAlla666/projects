#!/usr/bin/env python3
# WI-3: fleet-aggregation collector. Ingests rc.v1 per-GPU reliability events keyed by gpu.uuid and emits a
# fleet_reliability_view. The multi-GPU population is SIMULATED by REPLAYING the one REAL rc.v1 event
# (rcv1_event_real.json, copied read-only from rc_pergpu) across synthetic uuids with varied onset step and a
# healthy/degraded mix. The aggregation LOGIC is what is validated; real fleet scale is NOT claimed.
#
# v2 (post-panel): all WI-1 constants now LOADED from wi1_accounting.json (no free-floating numbers); schema check
# tightened + raises ValueError (survives python -O); negative schema tests (4 malformed classes, view-corruption
# check); update-semantics tests (changed-verdict re-ingest, total-level no-double-count); the PREREG-registered
# fleet_effective_training_time is COMPUTED (WITHOUT vs WITH, healthy-GPU overhead included — the term the PREREG
# formula omitted) at the depicted mix, both N=8 and N=45.
import json, copy, hashlib

BASE = '/home/ubuntu/cipher-fusion-evidence'
REAL = json.load(open(f'{BASE}/wi3_fleetagg/rcv1_event_real.json'))
WI1 = json.load(open(f'{BASE}/wi1_trainingbadput/wi1_accounting.json'))

# WI-1 ties (all read from wi1_accounting.json):
WI1_BADPUT_GPUS_PER_FAULT = WI1['badput_avoided_per_fault_GPUseconds']['silent_10C=1000']['N8']   # 138.35
WI1_STEP_S = WI1['real_measured']['step_time_s_median']                                           # 0.1389
WI1_OH = {'N8': WI1['real_measured']['detector_overhead_per_check_method']['N8_pct'] / 100.0,     # 0.0559
          'N45': WI1['real_measured']['detector_overhead_per_check_method']['N45_pct'] / 100.0}   # 0.0099
WI1_LAT = {'N8': WI1['real_measured']['detection_latency_steps']['N8'],                           # 4
           'N45': WI1['real_measured']['detection_latency_steps']['N45']}                         # 25
WI1_EX = WI1['effective_training_time_examples']['W10k_1fault_silent1000_N8_p1.0']                # without .905 / with .9422
WI1_BREAKEVEN = {'N8': WI1['breakeven_fault_rate_per_step']['silent_10C=1000']['N8_1_per_steps'],     # 17820
                 'N45': WI1['breakeven_fault_rate_per_step']['silent_10C=1000']['N45_1_per_steps']}   # 98121
WINDOW_STEPS = 10000  # the WI-1 effective-time example window (W=10k)

REQUIRED_FIELDS = ["schema_version", "event_type", "gpu", "verdict", "severity", "fault_onset", "signal_sources",
                   "corroboration", "process_context", "scope_caveats"]
ALLOWED_SEVERITY = ("critical", "warning", "info")

def schema_ok(ev):
    if not all(k in ev for k in REQUIRED_FIELDS): return False
    if ev["schema_version"] != "rc.v1": return False
    if "uuid" not in ev["gpu"]: return False
    if ev["verdict"] not in ("DEGRADED", "HEALTHY"): return False
    if ev["severity"] not in ALLOWED_SEVERITY: return False
    # a DEGRADED verdict must carry a fault onset (else it would be attributed badput with no onset evidence)
    if ev["verdict"] == "DEGRADED":
        if not isinstance(ev.get("fault_onset"), dict) or "decode_step_detected" not in ev["fault_onset"]:
            return False
    return True

class FleetCollector:
    """Aggregates rc.v1 events by gpu.uuid (uuid is fleet-unique). Idempotent: re-ingest updates, no double-count."""
    def __init__(self): self.by_uuid = {}
    def ingest(self, ev):
        if not schema_ok(ev):                       # explicit raise: survives python -O (assert did not)
            raise ValueError("rc.v1 schema violation")
        self.by_uuid[ev["gpu"]["uuid"]] = ev        # keyed update => idempotent
    def view(self):
        evs = list(self.by_uuid.values())
        degraded = [e for e in evs if e["verdict"] == "DEGRADED"]
        healthy  = [e for e in evs if e["verdict"] == "HEALTHY"]
        deg_list = [{
            "uuid": e["gpu"]["uuid"], "node": e["gpu"].get("node"),
            "fault_onset_step": e["fault_onset"]["decode_step_detected"] if e.get("fault_onset") else None,
            "fault_onset_wall_replayed": e["fault_onset"]["wall_clock_utc"] if e.get("fault_onset") else None,
            "severity": e["severity"], "corroboration": e["corroboration"],
            "descriptor": (e["process_context"]["descriptor"] or {}).get("tgid") if e["process_context"].get("descriptor") else None,
            "descriptor_cmdline": (e["process_context"]["descriptor"] or {}).get("cmdline") if e["process_context"].get("descriptor") else None,
            "badput_avoided_gpu_seconds_gross": WI1_BADPUT_GPUS_PER_FAULT,   # GROSS, WI-1 most-favorable scenario (see label)
        } for e in degraded]
        fleet_badput = round(WI1_BADPUT_GPUS_PER_FAULT * len(degraded), 1)
        n, f = len(evs), (len(degraded) / len(evs) if evs else 0)
        # PREREG-registered fleet effective-training-time, WITHOUT vs WITH, with the healthy-GPU overhead term the
        # PREREG formula omitted. Homogeneity assumptions: every degraded GPU has exactly ONE covered (p=1)
        # silent-1000-discovery fault in the W=10k window; healthy GPUs pay detector overhead continuously.
        # Degraded-GPU WITH at N=45 is derived from the N=8 example by the latency delta (lost steps +21) and the
        # N=45 overhead divisor — same model family as WI-1's example (lost_with(N8) back-solved from the example).
        eff = {}
        lost_with_n8 = WINDOW_STEPS - WI1_EX['with'] * WINDOW_STEPS * (1 + WI1_OH['N8'])
        for Nk in ('N8', 'N45'):
            healthy_with = 1.0 / (1.0 + WI1_OH[Nk])
            lost_with = lost_with_n8 + (WI1_LAT[Nk] - WI1_LAT['N8'])
            deg_with = (WINDOW_STEPS - lost_with) / (WINDOW_STEPS * (1 + WI1_OH[Nk]))
            fleet_without = (1 - f) * 1.0 + f * WI1_EX['without']
            fleet_with = (1 - f) * healthy_with + f * deg_with
            overhead_gpu_h = n * WINDOW_STEPS * WI1_STEP_S * WI1_OH[Nk] / 3600.0
            eff[Nk] = {"fleet_without": round(fleet_without, 4), "fleet_with": round(fleet_with, 4),
                       "delta_pp": round((fleet_with - fleet_without) * 100, 2),
                       "fleet_detector_overhead_gpu_hours_over_window": round(overhead_gpu_h, 1),
                       "breakeven_fault_rate": f"1 per {WI1_BREAKEVEN[Nk]} GPU-steps"}
        gpu_steps = n * WINDOW_STEPS
        return {
            "fleet_reliability_view": {
                "total_gpus": n, "n_healthy": len(healthy), "n_degraded": len(degraded),
                "degraded_fraction": round(f, 4),
                "degraded": deg_list,
                "fleet_badput_avoided_gpu_seconds_total_gross": fleet_badput,
                "fleet_badput_avoided_gpu_hours_total_gross": round(fleet_badput / 3600, 3),
                "wi1_scenario_for_badput": "TRAINING goodput, silent-1000-step discovery, N=8, coverage p=1 — WI-1's "
                    "MOST FAVORABLE row (vs 6.39/13.33 GPU-s for faster-discovery rows) and p=1 is WI-1's labeled "
                    "upper bound; GROSS avoided, NOT net of fleet-wide detector overhead (see fleet_effective_training_time)",
                "fleet_effective_training_time": {
                    "window_steps": WINDOW_STEPS,
                    "assumptions": "homogeneous: each degraded GPU has exactly 1 covered (p=1) silent-1000 fault in "
                        "the window; healthy GPUs pay detector overhead continuously; per-GPU values from "
                        "wi1_accounting.json (N45 degraded-GPU value derived via latency delta, same model family)",
                    "per_N": eff,
                    "fleet_fault_rate_depicted": f"{len(degraded)} faults / {gpu_steps} GPU-steps = 1 per "
                        f"{int(round(gpu_steps / len(degraded))) if degraded else 'inf'} GPU-steps",
                    "net_assessment": "depicted fleet is NET-NEGATIVE at both N (fault rate far below breakeven); "
                        "fleet NET > 0 iff faults per GPU-step exceed the WI-1 breakeven (additive per GPU-step under "
                        "the homogeneity assumptions)",
                    "prereg_note": "PREREG expectation 2 predicted fleet WITH > WITHOUT 'by the per-GPU delta weighted "
                        "by degraded fraction' — FALSIFIED as stated: the PREREG formula omitted the healthy-GPU "
                        "overhead term; at the depicted 2% mix WITH < WITHOUT at both N",
                },
            },
            "provenance": {"real_event_source": "rc_pergpu/inject_verdict.json (rc.v1, validated DEGRADED)",
                           "population": "SIMULATED by replay across synthetic uuids; aggregation logic validated, fleet scale NOT claimed",
                           "onset_step_provenance": "SYNTHETIC (24+7i replay variation; the real event's onset is 64/72); "
                               "wall stamp + descriptor replayed verbatim from the single real event",
                           "wi1_inputs": "all WI-1 constants loaded from wi1_trainingbadput/wi1_accounting.json"},
        }

def synth_uuid(i):  # deterministic synthetic fleet uuid (no Date/random in scripts)
    h = hashlib.sha256(f"fleet-gpu-{i}".encode()).hexdigest()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def replay_fleet(n_gpus, degraded_fraction):
    """Replay the one real rc.v1 event across n_gpus synthetic GPUs; a degraded_fraction are DEGRADED (varied onset
    step ONLY — severity stays the real event's 'critical': PREREG promised varied severity, NOT delivered, disclosed
    as a deviation in the report), the rest HEALTHY (same schema, verdict flipped)."""
    n_deg = int(round(n_gpus * degraded_fraction))
    out = []
    for i in range(n_gpus):
        ev = copy.deepcopy(REAL)
        ev["gpu"] = dict(ev["gpu"]); ev["gpu"]["uuid"] = synth_uuid(i); ev["gpu"]["gpu_index"] = i % 8
        ev["gpu"]["node"] = f"node-{i//8:03d}"
        if i < n_deg:
            ev["verdict"] = "DEGRADED"; ev["severity"] = "critical"
            ev["fault_onset"] = dict(ev["fault_onset"]); ev["fault_onset"]["decode_step_detected"] = 24 + (i*7) % 500
        else:
            ev["verdict"] = "HEALTHY"; ev["severity"] = "info"; ev["fault_onset"] = None
            ev["corroboration"] = "n/a"
        out.append(ev)
    return out

def negative_schema_tests(fc):
    """4 malformed classes MUST be rejected, and rejection must not corrupt the view."""
    view_before = json.dumps(fc.view(), sort_keys=True)
    bad = []
    e1 = copy.deepcopy(REAL); del e1["signal_sources"]; bad.append(("missing_required_field", e1))
    e2 = copy.deepcopy(REAL); e2["gpu"] = {k: v for k, v in e2["gpu"].items() if k != "uuid"}; bad.append(("missing_gpu_uuid", e2))
    e3 = copy.deepcopy(REAL); e3["verdict"] = "BROKEN"; bad.append(("invalid_verdict", e3))
    e4 = copy.deepcopy(REAL); e4["fault_onset"] = None; bad.append(("degraded_with_null_onset", e4))
    rejected = 0
    for _name, ev in bad:
        try:
            fc.ingest(ev)
        except ValueError:
            rejected += 1
    view_after = json.dumps(fc.view(), sort_keys=True)
    return {"malformed_classes_tested": [n for n, _ in bad], "rejected": f"{rejected}/{len(bad)}",
            "view_uncorrupted_by_rejection": view_before == view_after}

def update_semantics_tests(events):
    """Tested on a SCRATCH collector so the emitted main view is untouched.
    (a) identical re-ingest: full view (incl. totals) unchanged; (b) changed-verdict re-ingest: updates, no double-count."""
    fc = FleetCollector()
    for ev in events: fc.ingest(ev)
    base = fc.view()["fleet_reliability_view"]
    fc.ingest(copy.deepcopy(events[0]))                          # identical DEGRADED re-ingest
    after_same = fc.view()["fleet_reliability_view"]
    identical_ok = (after_same["n_degraded"] == base["n_degraded"] and
                    after_same["fleet_badput_avoided_gpu_seconds_total_gross"] == base["fleet_badput_avoided_gpu_seconds_total_gross"])
    flip = copy.deepcopy(events[0])                              # DEGRADED -> HEALTHY for the same uuid
    flip["verdict"] = "HEALTHY"; flip["severity"] = "info"; flip["fault_onset"] = None; flip["corroboration"] = "n/a"
    fc.ingest(flip)
    after_flip = fc.view()["fleet_reliability_view"]
    flip_ok = (after_flip["n_degraded"] == base["n_degraded"] - 1 and
               after_flip["fleet_badput_avoided_gpu_seconds_total_gross"] ==
               round(WI1_BADPUT_GPUS_PER_FAULT * (base["n_degraded"] - 1), 1))
    return {"identical_reingest_totals_unchanged": identical_ok,
            "changed_verdict_updates_not_double_counts": flip_ok,
            "flip_check": {"n_degraded": f'{base["n_degraded"]}->{after_flip["n_degraded"]}',
                           "total": f'{base["fleet_badput_avoided_gpu_seconds_total_gross"]}->{after_flip["fleet_badput_avoided_gpu_seconds_total_gross"]}'}}

if __name__ == "__main__":
    fc = FleetCollector()
    events = replay_fleet(n_gpus=1000, degraded_fraction=0.02)   # 1000-GPU fleet, 2% degraded
    n_bad = 0
    for ev in events:
        try: fc.ingest(ev)
        except ValueError: n_bad += 1
    # idempotence check: re-ingest first 50, total must not change
    pre = len(fc.by_uuid)
    for ev in events[:50]: fc.ingest(ev)
    idempotent = (len(fc.by_uuid) == pre)
    neg = negative_schema_tests(fc)
    upd = update_semantics_tests(events)
    view = fc.view()
    view["validation"] = {"schema_rejections_in_replay": n_bad, "idempotent_reingest_count": idempotent,
                          "all_keyed_by_uuid": len(fc.by_uuid) == len(events),
                          "schema_negative_tests": neg, "update_semantics_tests": upd,
                          "ingest_raises_valueerror_not_assert": True}
    json.dump(view, open(f'{BASE}/wi3_fleetagg/fleet_view.json', 'w'), indent=2)
    v = view["fleet_reliability_view"]
    print("WI3v2", json.dumps({"total_gpus": v["total_gpus"], "n_degraded": v["n_degraded"],
          "degraded_fraction": v["degraded_fraction"],
          "fleet_badput_gpu_hours_gross": v["fleet_badput_avoided_gpu_hours_total_gross"],
          "eff_time": v["fleet_effective_training_time"]["per_N"],
          "fault_rate": v["fleet_effective_training_time"]["fleet_fault_rate_depicted"],
          "validation": view["validation"]}, indent=1))
