#!/usr/bin/env python3
# Dispatch-boundary quarantine demo. The SHIM detects SDC at the cuBLAS-interception boundary and records a
# detection signal (events JSONL, is_detection). This serve-gate (SCRATCH code, NOT vLLM source) consumes that
# signal and decides SERVE vs QUARANTINE the response. For a BATCH/non-streaming response the gate refuses to
# serve a flagged response BEFORE it leaves the box (detection at S' < T occurs during generation). The WALL:
# preventing an already-STREAMED token mid-flight, or scrubbing vLLM's in-flight buffer, needs vLLM cooperation.
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def jsonl(p):
    fp=os.path.join(HERE,p); return [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
def toks(p): return json.load(open(os.path.join(HERE,p)))["token_ids"][0]
clean = toks("g_clean.json")
def prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n
OFFSET, ONSET_CUR = 3, 20

def gate(name, run_json, events_file, has_detector):
    w = toks(run_json); T = len(w); f = prefix(clean, w)
    truly_corrupt = (f < T)                        # ground truth: did any token actually diverge?
    if not has_detector:
        decision = "SERVE"; detected_at = None     # no detector => serve whatever (incl. garbage)
    else:
        dets = sorted([e for e in jsonl(events_file) if e.get("is_detection")], key=lambda e: e["step"])
        if dets:
            detected_at = dets[0]["step"] - OFFSET  # detection decode-token
            decision = "QUARANTINE"
        else:
            detected_at = None; decision = "SERVE"  # not detected => served
    served_garbage = (decision == "SERVE" and truly_corrupt)
    return {"config": name, "has_detector": has_detector, "truly_corrupt": truly_corrupt,
            "corruption_starts_token": (f if truly_corrupt else None),
            "detected_at_token": detected_at, "decision": decision,
            "ships_silent_garbage": served_garbage,
            "outcome": ("SHIPS SILENT GARBAGE" if served_garbage else
                        ("quarantined (corrupt response withheld)" if decision=="QUARANTINE" else
                         "served (genuinely clean)"))}

rows = [
    gate("WITHOUT-CIPHER (persistent fault, no detector)", "g_without.json", None, False),
    gate("WITH-CIPHER N=8 (persistent fault)",  "g_with_N8.json",  "g_with_N8_events.jsonl",  True),
    gate("WITH-CIPHER N=45 (persistent fault)", "g_with_N45.json", "g_with_N45_events.jsonl", True),
    gate("CLEAN + detector (no fault)",         "q_clean.json",    "q_clean_events.jsonl",    True),
    gate("TRANSIENT on-check + detector",       "q_tr_oncheck.json","q_tr_oncheck_events.jsonl",True),
    gate("TRANSIENT between-check + detector",  "q_tr_between.json","q_tr_between_events.jsonl",True),
]
summary = {"rows": rows,
  "quarantine_latency_steps": {"persistent_N8": 24-ONSET_CUR, "persistent_N45": 45-ONSET_CUR},
  "false_quarantine": any(r["decision"]=="QUARANTINE" and not r["truly_corrupt"] for r in rows),
  "wall": "Detect+flag at cuBLAS boundary is substrate-legal (scratch serve-gate refuses to serve a flagged "
          "BATCH response before it leaves the box). Scrubbing an already-streamed token / vLLM's in-flight "
          "response buffer needs vLLM cooperation = WALL-WITH-MECHANISM."}
with open(os.path.join(HERE,"quarantine_demo.json"),"w") as fp: json.dump(summary, fp, indent=2)
print(json.dumps(summary, indent=2))
