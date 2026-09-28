#!/usr/bin/env python3
# Goodput analysis: useful = matching-PREFIX length vs clean reference (after first divergence the greedy
# cascade makes later coincidental matches meaningless). WITH-CIPHER quarantines from the detection step S'.
# Holds T fixed (full generate); quarantine is a labeling op. Emits goodput.json for the report.
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def L(p): return json.load(open(os.path.join(HERE, p)))
def toks(d): return d["token_ids"][0]
def jsonl(p):
    out=[]; fp=os.path.join(HERE,p)
    if os.path.exists(fp):
        for ln in open(fp):
            ln=ln.strip()
            if ln: out.append(json.loads(ln))
    return out

clean = L("g_clean.json"); without = L("g_without.json")
withN8 = L("g_with_N8.json"); withN45 = L("g_with_N45.json")
c = toks(clean)
def prefix(a, b):  # length of matching prefix
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n
ONSET_CUR = 20  # RV_ONSET (shim cur_step)
f = prefix(c, toks(without))           # first divergence (decode-token index) in the faulted stream
OFFSET = ONSET_CUR - f                  # prefill->cur_step offset

# advisor assertions
faulted_streams_identical = toks(without)==toks(withN8)==toks(withN45)
determinism_prefix_ok = c[:f]==toks(without)[:f]   # WITHOUT[0,f)==clean[0,f)

def detection_token(events_file):
    ev = sorted(jsonl(events_file), key=lambda e: e["step"])
    dets = [e for e in ev if e.get("is_detection")]
    if not dets: return None, None
    Sc = dets[0]["step"]                # detection cur_step
    return Sc, Sc - OFFSET              # detection decode-token S'

def goodput(run, name, events_file=None):
    T = run["out"]; gt = run["gen_time_s"]; J = run["joules"]
    w = toks(run)
    fr = prefix(c, w)                    # this run's divergence point
    if events_file is None:              # WITHOUT: all T served
        served = T; Sc = Sp = None
    else:                                # WITH: served = [0, S'); quarantine [S', T)
        Sc, Sp = detection_token(events_file)
        served = Sp if Sp is not None else T
    useful = min(fr, served)             # matching-prefix tokens that were actually served
    garbage = max(0, min(served, T) - fr)  # served tokens past divergence (silently wrong)
    quarantined = T - served if events_file is not None else 0
    return {"config": name, "T": T, "gen_time_s": round(gt,3), "joules": round(J,1),
            "divergence_token_f": fr, "detection_cur_step": Sc, "detection_token_Sprime": Sp,
            "served_tokens": served, "useful_tokens": useful, "garbage_served": garbage,
            "quarantined_tokens": quarantined,
            "raw_tok_s": round(T/gt,2), "useful_tok_s": round(useful/gt,3),
            "useful_tok_per_joule": round(useful/J,5),
            "raw_tok_per_joule": round(T/J,4)}

rows = [
    goodput(without, "WITHOUT-CIPHER (fault, served)", None),
    goodput(withN8,  "WITH-CIPHER N=8",  "g_with_N8_events.jsonl"),
    goodput(withN45, "WITH-CIPHER N=45", "g_with_N45_events.jsonl"),
]
clean_row = {"config":"CLEAN (no fault, reference)","T":clean["out"],"gen_time_s":round(clean["gen_time_s"],3),
             "joules":round(clean["joules"],1),"useful_tokens":clean["out"],
             "raw_tok_s":round(clean["out"]/clean["gen_time_s"],2),"useful_tok_s":round(clean["out"]/clean["gen_time_s"],3),
             "useful_tok_per_joule":round(clean["out"]/clean["joules"],5)}

out = {"clean_reference": clean_row, "divergence_token_f": f, "prefill_cur_step_offset": OFFSET,
       "onset_cur_step": ONSET_CUR,
       "assertions": {"faulted_streams_identical_WITHOUT_vs_WITH": faulted_streams_identical,
                      "determinism_WITHOUT_prefix_eq_clean": determinism_prefix_ok},
       "rows": rows,
       "overhead_time_pct": {"N8": round(100*(withN8["gen_time_s"]/without["gen_time_s"]-1),1),
                             "N45": round(100*(withN45["gen_time_s"]/without["gen_time_s"]-1),1)}}
with open(os.path.join(HERE,"goodput.json"),"w") as fp: json.dump(out, fp, indent=2)
print(json.dumps(out, indent=2))
