"""T4.6.3 Phase 2 harness prep — emit 5 trace-window subsets per flavor
for the n=5 statistical bounds the Phase 2 gate requires.

Each CIPHER page = N=2 consecutive 512-token blocks; identical block-id
tuple => content-identical page. Window k = pages [k*M, (k+1)*M). Dedup
accumulates along the trace, so the 5 windows naturally span a range of
dedup ratios — a real spread, not synthetic noise. The harness must
reproduce each window's ratio exactly (real == sim) or the allocator
has a bug.
"""
import json

N = 2
M = 4000
NWIN = 5
TDIR = "/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup/traces"
ODIR = "/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup"
FLAVORS = ["conversation", "toolagent", "synthetic"]

for flavor in FLAVORS:
    allpages = []
    need = M * NWIN
    with open(f"{TDIR}/{flavor}_trace.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hid = json.loads(line)["hash_ids"]
            for i in range(0, len(hid), N):
                allpages.append(tuple(hid[i:i + N]))
            if len(allpages) >= need:
                break
    ratios = []
    for k in range(NWIN):
        win = allpages[k * M:(k + 1) * M]
        if len(win) < M:
            print(f"  {flavor} window {k}: short ({len(win)}) — skipped")
            continue
        idmap, seq = {}, []
        for pg in win:
            if pg not in idmap:
                idmap[pg] = len(idmap)
            seq.append(idmap[pg])
        ratio = (len(seq) - len(idmap)) / len(seq)
        ratios.append(ratio)
        with open(f"{ODIR}/pageseq_{flavor}_{k}.txt", "w") as f:
            f.write(f"{len(seq)} {ratio:.6f}\n")
            f.write("\n".join(str(s) for s in seq) + "\n")
    mean = sum(ratios) / len(ratios)
    print(f"{flavor:13s} 5 windows M={M}: sim_ratio "
          f"mean={mean*100:.2f}% min={min(ratios)*100:.2f}% "
          f"max={max(ratios)*100:.2f}%")
