#!/usr/bin/env python3
# R.C v1 per-GPU reliability monitor: FUSES (1) the R.A SDC detector signal (in-vLLM cuBLAS shim),
# (2) the W.6 kmod cohort registry (co-residence + per-tgid descriptor), and (3) NVML host telemetry
# (ECC/XID/throttle/power/clock) into a single per-GPU reliability verdict, and emits the fleet
# aggregation schema. DCGM/dcgmi is absent on this pod -> NVML is the consumed host telemetry (stated).
#
# Architecture (advisor item 4): post-hoc CLOCK_MONOTONIC timestamp-join. The monitor samples NVML +
# cohort on a timeline while vLLM runs; the shim appends per-check event records; fusion happens after
# the run by joining on wall/mono time. No live FIFO.
import sys, os, time, json, ctypes, fcntl, threading, subprocess, socket, argparse, signal
import pynvml as N

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "sdc_shim.so")

# ---------- W.6 cohort registry (query-only read; fp=0 => no self-insert) ----------
MAXP = 128
class _Entry(ctypes.Structure):
    _fields_ = [("tgid", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("fp", ctypes.c_uint64)]
class _Query(ctypes.Structure):
    _fields_ = [("caller_fp", ctypes.c_uint64), ("max_entries", ctypes.c_uint32),
                ("n_resident", ctypes.c_uint32), ("entries", _Entry * MAXP)]
def _iowr(t, nr, size): return (3 << 30) | (size << 16) | (t << 8) | nr
_NR_QUERY = _iowr(ord('C'), 31, ctypes.sizeof(_Query))
def cohort_query():
    """Query-only snapshot of the live cohort (does NOT register the monitor)."""
    try:
        fd = os.open("/dev/cipher", os.O_RDWR)
    except OSError:
        return None
    try:
        q = _Query(); q.caller_fp = 0; q.max_entries = MAXP
        fcntl.ioctl(fd, _NR_QUERY, q)
        peers = [(q.entries[i].tgid, q.entries[i].fp) for i in range(min(q.n_resident, MAXP))]
        return {"n_resident": q.n_resident, "peers": peers}
    except OSError:
        return None
    finally:
        os.close(fd)

# ---------- NVML host telemetry ----------
def nvml_static(h):
    return {
        "name": N.nvmlDeviceGetName(h),
        "uuid": N.nvmlDeviceGetUUID(h),
        "pci_bus_id": N.nvmlDeviceGetPciInfo(h).busId.decode() if isinstance(N.nvmlDeviceGetPciInfo(h).busId, bytes) else N.nvmlDeviceGetPciInfo(h).busId,
        "driver": N.nvmlSystemGetDriverVersion(),
    }
def ecc_counts(h):
    out = {}
    for tag, et in (("uncorr", N.NVML_MEMORY_ERROR_TYPE_UNCORRECTED), ("corr", N.NVML_MEMORY_ERROR_TYPE_CORRECTED)):
        try: out["vol_" + tag] = N.nvmlDeviceGetTotalEccErrors(h, et, N.NVML_VOLATILE_ECC)
        except Exception: out["vol_" + tag] = None
        try: out["agg_" + tag] = N.nvmlDeviceGetTotalEccErrors(h, et, N.NVML_AGGREGATE_ECC)
        except Exception: out["agg_" + tag] = None
    return out
def sample_nvml(h, xid_set):
    s = {"mono": time.monotonic(), "wall": time.time()}
    s["ecc"] = ecc_counts(h)
    try: s["power_mw"] = N.nvmlDeviceGetPowerUsage(h)
    except Exception: s["power_mw"] = None
    try: s["sm_clk"] = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
    except Exception: s["sm_clk"] = None
    try: s["throttle"] = N.nvmlDeviceGetCurrentClocksThrottleReasons(h)
    except Exception: s["throttle"] = None
    try:
        procs = N.nvmlDeviceGetComputeRunningProcesses(h)
        s["compute_procs"] = [(p.pid, getattr(p, "usedGpuMemory", None)) for p in procs]
    except Exception:
        s["compute_procs"] = None
    # drain any XID critical events (NVML_ERROR_TIMEOUT on the empty queue is the normal exit)
    xids = []
    if xid_set is not None:
        for _ in range(64):
            try:
                d = N.nvmlEventSetWait(xid_set, 5)
            except N.NVMLError:
                break
            if getattr(d, "eventType", 0) & N.nvmlEventTypeXidCriticalError:
                xids.append(int(getattr(d, "eventData", 0)))
    s["xid_events"] = xids
    c = cohort_query()
    s["cohort"] = c
    return s

# ---------- fusion ----------
def read_jsonl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                try: out.append(json.loads(ln))
                except Exception: pass
    return out

def fuse_process(events_path, counts_path):
    ev = read_jsonl(events_path)
    counts = {}
    if os.path.exists(counts_path):
        try: counts = json.load(open(counts_path))
        except Exception: pass
    checks = sorted(ev, key=lambda e: e["step"])
    dets = [e for e in checks if e.get("is_detection")]
    clean_before = [e for e in checks if not e.get("is_detection")]
    # consecutive-detection persistence across the check sequence (steps spaced by N)
    persist = 0
    if dets:
        steps = sorted(set(e["step"] for e in dets))
        # max run of consecutive check-steps among detections
        ns = counts.get("N", 8)
        best = run = 1
        for i in range(1, len(steps)):
            if steps[i] - steps[i-1] == ns: run += 1
            else: run = 1
            best = max(best, run)
        persist = best
    return {
        "counts": counts,
        "n_checks_logged": len(checks),
        "n_clean_checks_before_onset": len(clean_before),
        "n_detections": len(dets),
        "persistence_consecutive": persist,
        "onset_event": dets[0] if dets else None,
        "confirm_event": dets[1] if len(dets) >= 2 else None,
        "max_residual": max([e["residual"] for e in dets], default=0.0),
        "last_clean_check_step": max([e["step"] for e in clean_before], default=None),
        "first_detection_step": dets[0]["step"] if dets else None,
        "pid": (checks[0]["pid"] if checks else counts.get("pid")),
        "gpu": (checks[0]["gpu"] if checks else counts.get("gpu")),
    }

def ecc_delta(base, fin):
    out = {}
    for k in ("vol_uncorr", "vol_corr", "agg_uncorr", "agg_corr"):
        b, f = base.get(k), fin.get(k)
        out[k + "_delta"] = (f - b) if (b is not None and f is not None) else None
    return out

def descriptor_for(pid, cohort, compute_pids_seen, cmdline_map):
    """DESCRIPTIVE process/context tag: what was running when the SDC fired. Not attributive."""
    cmd = cmdline_map.get(pid)
    fp = None
    if cohort:
        for tgid, f in cohort.get("peers", []):
            if tgid == pid: fp = f
    in_nvml = pid in compute_pids_seen
    in_cohort = fp is not None
    src = "nvml_compute_procs+w6_cohort" if (in_nvml and in_cohort) else \
          ("nvml_compute_procs" if in_nvml else ("w6_cohort" if in_cohort else "sdc_event_only"))
    return {"tgid": pid, "cmdline": cmd, "fingerprint": (hex(fp) if fp else None),
            "descriptor_source": src, "seen_by_nvml": in_nvml, "in_w6_cohort": in_cohort}

# ---------- orchestration ----------
def child_env(idx, tag, inject_gemm, onset, watch, N_, batch, outtok, warm, util, maxlen, model):
    e = dict(os.environ)
    e.pop("CUDA_INJECTION64_PATH", None)        # ensure the substrate runtime is NOT auto-loaded
    e["VLLM_PLUGINS"] = ""                        # no third-party (incl. cipher) vLLM plugins
    e["VLLM_USE_DEEP_GEMM"] = "0"
    e["LD_PRELOAD"] = SHIM
    e["RV_N"] = str(N_); e["RV_VOCAB"] = "32000"; e["RV_GPS"] = "128"
    e["RV_INJECT"] = str(inject_gemm); e["RV_ONSET"] = str(onset); e["RV_WATCH"] = str(watch)
    e["RV_OUT"] = os.path.join(HERE, f"{tag}_sdc_counts_p{idx}.json")
    e["RV_EVENTS"] = os.path.join(HERE, f"{tag}_events_p{idx}.jsonl")
    e["RV_FP"] = str(0xC1DEC0DE00000001 + idx)   # distinct per-context descriptor fingerprint (probe)
    e["RV_MODEL"] = model; e["RV_BATCH"] = str(batch); e["RV_OUT_TOKENS"] = str(outtok)
    e["RV_WARM_TOKENS"] = str(warm); e["RV_UTIL"] = str(util); e["RV_MAXLEN"] = str(maxlen); e["RV_EAGER"] = "1"
    e["RV_RESULT"] = os.path.join(HERE, f"{tag}_driver_p{idx}.json")
    return e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clean", "inject", "probe"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--onset", type=int, default=0)
    ap.add_argument("--inject-gemm", type=int, default=3)   # layer-0 down_proj
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--warm-tokens", type=int, default=64)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--stagger", type=float, default=20.0)   # probe: seconds between the two launches
    args = ap.parse_args()

    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(0)
    static = nvml_static(h)
    static["host"] = socket.gethostname()
    try:
        xid_set = N.nvmlEventSetCreate(); N.nvmlDeviceRegisterEvents(h, N.nvmlEventTypeXidCriticalError, xid_set)
    except Exception as e:
        sys.stderr.write(f"[mon] XID event registration failed: {e}\n"); xid_set = None

    # exclusivity guard
    pre = N.nvmlDeviceGetComputeRunningProcesses(h)
    if pre:
        sys.stderr.write(f"[mon] WARNING: GPU not exclusive at start, procs={[p.pid for p in pre]}\n")
    baseline = sample_nvml(h, xid_set)
    sys.stderr.write(f"[mon] baseline ECC={baseline['ecc']} throttle={hex(baseline['throttle'] or 0)} procs={baseline['compute_procs']}\n")

    timeline = [baseline]
    cmdline_map = {}
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            s = sample_nvml(h, xid_set)
            for pid, _ in (s.get("compute_procs") or []):   # capture cmdline WHILE the process is alive
                if pid not in cmdline_map:
                    try:
                        cmdline_map[pid] = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode().strip()
                    except Exception:
                        cmdline_map[pid] = None
            timeline.append(s)
            stop.wait(0.5)
    th = threading.Thread(target=sampler, daemon=True); th.start()

    # launch driver(s)
    procs = []
    if args.mode == "probe":
        specs = [(0, args.inject_gemm), (1, -1)]   # p0 injected, p1 clean
    elif args.mode == "inject":
        specs = [(0, args.inject_gemm)]
    else:
        specs = [(0, -1)]                          # clean
    for idx, inj in specs:
        # clear stale per-proc artifacts (shim appends to RV_EVENTS) so a re-run is not polluted
        for suf in (f"_events_p{idx}.jsonl", f"_sdc_counts_p{idx}.json"):
            fp_ = os.path.join(HERE, f"{args.tag}{suf}")
            if os.path.exists(fp_): os.remove(fp_)
        env = child_env(idx, args.tag, inj, args.onset, args.inject_gemm, args.N,
                         args.batch, args.out_tokens, args.warm_tokens, args.util, args.maxlen, args.model)
        out = open(os.path.join(HERE, f"{args.tag}_driver_p{idx}.out"), "w")
        err = open(os.path.join(HERE, f"{args.tag}_driver_p{idx}.err"), "w")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "vllm_driver.py"), "decode"],
                             env=env, stdout=out, stderr=err)
        procs.append((idx, inj, p, out, err))
        sys.stderr.write(f"[mon] launched driver p{idx} pid={p.pid} inject={inj}\n")
        if args.mode == "probe" and idx == 0 and args.stagger > 0:
            time.sleep(args.stagger)   # stagger between the two launches

    rc = []
    for idx, inj, p, out, err in procs:
        r = p.wait(); out.close(); err.close(); rc.append((idx, r))
        sys.stderr.write(f"[mon] driver p{idx} exited rc={r}\n")

    time.sleep(1.0); stop.set(); th.join(timeout=3)
    fin = sample_nvml(h, xid_set); timeline.append(fin)

    # save timeline
    with open(os.path.join(HERE, f"{args.tag}_timeline.jsonl"), "w") as f:
        for s in timeline: f.write(json.dumps(s) + "\n")

    # collect any XID seen across the whole run
    all_xids = [x for s in timeline for x in (s.get("xid_events") or [])]
    eccd = ecc_delta(baseline["ecc"], fin["ecc"])
    throttle_seen = sorted(set(s["throttle"] for s in timeline if s.get("throttle") is not None))
    max_cohort = max((s["cohort"]["n_resident"] for s in timeline if s.get("cohort")), default=0)
    compute_pids_seen = sorted(set(pp[0] for s in timeline if s.get("compute_procs") for pp in s["compute_procs"]))
    last_cohort = next((s["cohort"] for s in reversed(timeline) if s.get("cohort") and s["cohort"]["n_resident"]), None)

    # per-process SDC fusion
    procfuse = []
    for idx, inj, p, out, err in procs:
        f = fuse_process(os.path.join(HERE, f"{args.tag}_events_p{idx}.jsonl"),
                         os.path.join(HERE, f"{args.tag}_sdc_counts_p{idx}.json"))
        f["proc_idx"] = idx; f["injected_config"] = inj
        procfuse.append(f)

    # ---- per-GPU verdict ----
    # DEGRADED <=> SDC fired persistently (>=2 consecutive checks) OR a single detection corroborated by
    # a hardware event (ECC-uncorrected delta>0 OR a critical XID). Matches the pre-registered rule.
    degraded = [f for f in procfuse if f["persistence_consecutive"] >= 2 or
                (f["n_detections"] >= 1 and ((eccd.get("vol_uncorr_delta") or 0) > 0 or bool(all_xids)))]
    verdict = "DEGRADED" if degraded else "HEALTHY"
    corrob = "uncorroborated_by_hardware_counters"
    if (eccd.get("vol_uncorr_delta") or 0) > 0: corrob = "corroborated_ecc_uncorrected"
    if all_xids: corrob = "corroborated_xid"

    # onset / descriptor from the (first) degraded process
    onset_rec = None; descriptor = None; observers = []
    if degraded:
        d0 = sorted(degraded, key=lambda f: (f["onset_event"] or {}).get("step", 1e18))[0]
        oe = d0["onset_event"]; ce = d0["confirm_event"]
        onset_rec = {
            "decode_step_detected": oe["step"], "decode_step_confirmed": (ce["step"] if ce else None),
            "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(oe["wall_ns"]/1e9)),
            "mono_ns": oe["mono_ns"], "covered_op": f"linear_gemm:ordinal{oe['gemm_ordinal']}:k{oe['k']}",
        }
        observers = [{"pid": f["pid"], "max_residual": f["max_residual"], "n_detections": f["n_detections"],
                      "persistence": f["persistence_consecutive"],
                      "n_clean_checks_before_onset": f["n_clean_checks_before_onset"]} for f in degraded]
        descriptor = descriptor_for(d0["pid"], last_cohort, set(compute_pids_seen), cmdline_map)

    record = {
        "schema_version": "rc.v1",
        "event_type": "gpu_reliability_event",
        "gpu": {"node": static["host"], "gpu_index": 0, "uuid": static["uuid"],
                "pci_bus_id": static["pci_bus_id"], "name": static["name"], "driver": static["driver"]},
        "verdict": verdict,
        "severity": ("critical" if verdict == "DEGRADED" else "info"),
        "fault_onset": onset_rec,
        "signal_sources": {
            "sdc_detector": {
                "fired": bool(degraded),
                "observers": observers,
                "coverage_note": "linear projections (qkv/o/gate_up/down) via cublasGemmEx; attention(FlashAttn)+lm_head uncovered",
                "threshold": "T=0 (bit-identical clean recompute)",
            },
            "nvml_ecc": eccd,
            "nvml_xid": {"events": all_xids},
            "nvml_throttle_reasons_seen": [hex(t) for t in throttle_seen],
            "nvml_power_clock": {"sm_clk_mhz_final": fin.get("sm_clk"), "power_mw_final": fin.get("power_mw")},
            "telemetry_source": "NVML (DCGM/dcgmi absent on this pod)",
        },
        "corroboration": corrob,
        "process_context": {
            "descriptor": descriptor,
            "co_resident_count": (last_cohort["n_resident"] if last_cohort else max_cohort),
            "co_resident_tgids": ([t for t, _ in last_cohort["peers"]] if last_cohort else []),
            "nvml_compute_pids_seen": compute_pids_seen,
            "note": "DESCRIPTIVE (what was running), NOT attributive (who caused it). Per-GPU verdict; per-tenant attribution is v2.",
        },
        "scope_caveats": [
            "per-GPU not per-tenant (v1)", "ONE GPU validated; multi-GPU scale + aggregation service unbuilt",
            "eager-hosted: cudagraph-on hosting carries the Path-1 +51% eager tax (documented)",
            "deterministic mercurial-core faults BLIND to same-GPU recompute (DCGM RAS may catch some)",
            "attention-internal SDC (Gap-3) + NCCL/collective out of scope",
            "corroborated branch (SDC+ECC/XID) specified but UNEXERCISED here (no real hardware fault injected; software flip => NVML correctly counter-clean)",
        ],
    }
    out_record = os.path.join(HERE, f"{args.tag}_verdict.json")
    with open(out_record, "w") as f: json.dump(record, f, indent=2)
    fuse_summary = {"mode": args.mode, "static": static, "baseline_ecc": baseline["ecc"], "final_ecc": fin["ecc"],
                    "ecc_delta": eccd, "xids": all_xids, "throttle_seen": [hex(t) for t in throttle_seen],
                    "max_cohort_n_resident": max_cohort, "compute_pids_seen": compute_pids_seen,
                    "driver_rc": rc, "per_process": procfuse, "verdict": verdict}
    with open(os.path.join(HERE, f"{args.tag}_fusion.json"), "w") as f: json.dump(fuse_summary, f, indent=2)

    print("VERDICT", json.dumps({"mode": args.mode, "verdict": verdict, "corroboration": corrob,
          "degraded_observers": [o["pid"] for o in observers], "max_cohort": max_cohort,
          "onset": onset_rec, "ecc_delta": eccd, "xids": all_xids}, indent=2))
    N.nvmlShutdown()

if __name__ == "__main__":
    main()
