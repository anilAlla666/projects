#!/usr/bin/env python3
# Join /tmp/cipher_geom_capture.jsonl (fn,geometry,geom_class) with
# /tmp/cipher_kernel_table.json (fn,name,name_class) and answer the 3 questions:
#  Q1 does the geometry classifier's ATTENTION band fire on real vLLM FA kernels?
#  Q2 is there a name-dependence geometry removes (where do geom vs name disagree)?
#  Q3 is nvjet (the cuBLAS-bypass GEMM) geometry-recognizable?
import json, re
OPC = {0:"GEMM",1:"ATTENTION",2:"CONVOLUTION",3:"ELEMENTWISE",4:"REDUCTION",
       5:"MEMCPY_TRANSPOSE",6:"ITERATIVE_CUSTOM",255:"UNCLASSIFIED",0xFF:"UNCLASSIFIED"}
geom = {}
for ln in open("/tmp/cipher_geom_capture.jsonl"):
    ln=ln.strip()
    if not ln: continue
    d=json.loads(ln); geom[d["fn"]]=d
kt = {}
try:
    j=json.load(open("/tmp/cipher_kernel_table.json"))
    for k in j.get("kernels",[]): kt[k["fn"]]=k
except Exception as e:
    print("kernel_table load:", e)

rows=[]
for fn,g in geom.items():
    name = kt.get(fn,{}).get("name","?")
    nclass = kt.get(fn,{}).get("class","?")
    gc = OPC.get(g["geom_class"], str(g["geom_class"]))
    rows.append((name[:48], nclass, gc, g["gx"],g["gy"],g["gz"], g["bx"],g["by"],g["bz"], g["shmem"]))

def is_attn(name,nclass):
    n=name.lower(); return nclass=="FlashAttn" or "flash" in n or "fmha" in n or "mha" in n or "attention" in n or "paged" in n
def is_gemm(name,nclass):
    n=name.lower(); return nclass=="GEMM" or "nvjet" in n or "gemm" in n or "cutlass" in n.replace("flash","")

print("=== ALL captured kernels (name | name_class | GEOM_verdict | grid | block | shmem) ===")
for r in sorted(rows, key=lambda x:(x[2],x[0])):
    print("  %-48s | %-12s | %-12s | g(%d,%d,%d) b(%d,%d,%d) sh=%d" % r)

print("\n=== Q1: ATTENTION kernels — does the GEOMETRY classifier say ATTENTION? ===")
attn=[r for r in rows if is_attn(r[0],r[1])]
for r in attn: print("  %-48s name_class=%-10s GEOM=%-12s shmem=%d grid=(%d,%d,%d) block=(%d,%d,%d)"%(r[0],r[1],r[2],r[9],r[3],r[4],r[5],r[6],r[7],r[8]))
if attn:
    agree=sum(1 for r in attn if r[2]=="ATTENTION")
    print("  -> %d/%d attention-by-name kernels recognized as ATTENTION by geometry"%(agree,len(attn)))
else: print("  (no attention-by-name kernels captured)")

print("\n=== Q3: nvjet / GEMM kernels — geometry verdict? ===")
gemm=[r for r in rows if is_gemm(r[0],r[1])]
for r in gemm: print("  %-48s name_class=%-10s GEOM=%-12s shmem=%d grid=(%d,%d,%d) block=(%d,%d,%d)"%(r[0],r[1],r[2],r[9],r[3],r[4],r[5],r[6],r[7],r[8]))
nvj=[r for r in rows if "nvjet" in r[0].lower()]
print("  -> nvjet kernels captured: %d; geometry verdicts: %s"%(len(nvj),[r[2] for r in nvj]))

print("\n=== Q2: name vs geometry agreement (where they DIFFER = name-dependence) ===")
diff=[r for r in rows if r[1]!="?" and r[1]!="UNKNOWN" and r[2].upper()[:4]!=r[1].upper()[:4]]
print("  total kernels: %d | name_class known: %d"%(len(rows),sum(1 for r in rows if r[1] not in("?","UNKNOWN"))))
for r in diff[:30]: print("  DIFFER %-48s name=%-12s geom=%-12s"%(r[0],r[1],r[2]))
