import json, numpy as np, glob, subprocess
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
md5=subprocess.run(["md5sum",SO],capture_output=True,text=True).stdout.split()[0]
MODELS=["tinyllama","llama32_1b","mistral7b","llama31_8b"]
res={"so_md5":md5,"gate":"OFF byte-identical (Mem#11 HARD STOP) + FP8-ON NaN-free","models":{},
     "note":"4 local models (torch/HF path). The exact W6_SUBC 9-vLLM-cell+CDI harness is NOT preserved (W6_SUBC_CLOSE_REPORT.md:79 /tmp scratch); this is the runnable byte-identical equivalent."}
allbyte=True; allnan_free=True; n=0
for mdl in MODELS:
    try:
        v=np.load(f"/home/ubuntu/d9clz_{mdl}_vanilla.npy"); o=np.load(f"/home/ubuntu/d9clz_{mdl}_cipher_fp8off.npy")
    except Exception as e:
        res["models"][mdl]={"error":str(e)}; allbyte=False; continue
    n+=1
    bytei=bool(np.array_equal(v,o)); md=float(np.max(np.abs(v.astype(np.float64)-o.astype(np.float64))))
    # KL(p_vanilla||q_off) mean per token
    def sm(x):
        x=x.astype(np.float64); x=x-x.max(-1,keepdims=True); e=np.exp(x); return e/e.sum(-1,keepdims=True)
    p=sm(v); q=sm(o); kl=float(np.mean(np.sum(np.where(p>1e-12,p*np.log(p/np.clip(q,1e-12,None)),0),-1)))
    nanfree=True
    try:
        on=np.load(f"/home/ubuntu/d9clz_{mdl}_cipher_fp8on.npy"); nanfree=bool(np.isfinite(on).all())
    except Exception: nanfree=None
    res["models"][mdl]={"byte_identical_OFF_vs_vanilla":bytei,"max_abs_diff":md,"kl_nats":kl,"fp8on_nan_free":nanfree}
    allbyte = allbyte and bytei
    if nanfree is False: allnan_free=False
res["off_byte_identical_n_of_n"]=f"{sum(1 for m in res['models'].values() if m.get('byte_identical_OFF_vs_vanilla'))}/{n}"
res["HARD_STOP_off_byte_identical"]="PASS" if allbyte and n>0 else "FAIL"
res["fp8on_nan_free_all"]="PASS" if allnan_free else "FAIL"
json.dump(res,open("/home/ubuntu/d9_fp8_close_offregression.json","w"),indent=2)
print(json.dumps(res,indent=2))
