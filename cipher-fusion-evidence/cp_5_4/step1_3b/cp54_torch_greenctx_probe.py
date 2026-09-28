"""CP 5.4 Step 1.3b' — torch.cuda.GreenContext SM-selection probe (THROWAWAY).

Option-2 viability question: does torch.cuda.GreenContext.create(num_sms) pick
the LOW-prefix 8-SM groups (groups 0..N-1 in the cuDevSmResourceSplitByCount
ordering measured in PHASE_1_3A_PROBE.md), deterministically?

Method: for num_sms in {8,16,24}, create a torch green context, run a kernel
on its stream that records %smid per block, read back the physical SM set the
green context actually used, and compare to the kmod group ordering. Repeat
4x per size for determinism.
"""
import os, sys
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
import torch
from torch.utils.cpp_extension import load_inline

# kmod / cuDevSmResourceSplitByCount group -> physical SM set (PHASE_1_3A_PROBE.md)
GROUP_SMS = {
    0:{0,1,16,17,32,33,48,49},   1:{2,3,18,19,34,35,50,51},
    2:{4,5,20,21,36,37,52,53},   3:{6,7,22,23,38,39,54,55},
    4:{8,9,24,25,40,41,56,57},   5:{10,11,26,27,42,43,58,59},
    6:{12,13,28,29,44,45,60,61}, 7:{14,15,30,31,46,47,62,63},
    8:{64,65,78,79,92,93,106,107}, 9:{66,67,80,81,94,95,108,109},
    10:{68,69,82,83,96,97,110,111}, 11:{70,71,84,85,98,99,112,113},
    12:{72,73,86,87,100,101,114,115}, 13:{74,75,88,89,102,103,116,117},
    14:{76,77,90,91,104,105,118,119},
}

cpp_src = "void probe_smid(torch::Tensor hit);"
cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void probe_smid_k(int* hit){
    unsigned s; asm volatile("mov.u32 %0, %%smid;":"=r"(s));
    if (s < 256) hit[s] = 1;
}
void probe_smid(torch::Tensor hit){
    probe_smid_k<<<4096, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        hit.data_ptr<int>());
}
'''
ext = load_inline(name='cp54_smid_probe', cpp_sources=[cpp_src],
                  cuda_sources=[cuda_src], functions=['probe_smid'],
                  verbose=False)

torch.zeros(1, device='cuda')          # init CUDA / primary context

def used_sms(num_sms):
    gc = torch.cuda.GreenContext.create(num_sms, 0)
    strm = gc.Stream()
    hit = torch.zeros(256, dtype=torch.int32, device='cuda')
    gc.set_context()
    with torch.cuda.stream(strm):
        ext.probe_smid(hit)
    torch.cuda.synchronize()
    gc.pop_context()
    return sorted((hit == 1).nonzero().flatten().tolist())

overall_ok = True
for n_groups in (1, 2, 3):
    num_sms = n_groups * 8
    expect = set().union(*(GROUP_SMS[g] for g in range(n_groups)))   # low prefix
    runs = []
    for r in range(4):
        sms = used_sms(num_sms)
        runs.append(sms)
        match_low = (set(sms) == expect)
        print(f"create({num_sms:2d})  run{r}  nsm={len(sms):3d}  "
              f"low-prefix-match={match_low}  sms={sms}")
    det = all(r == runs[0] for r in runs)
    low = all(set(r) == expect for r in runs)
    print(f"  -> num_sms={num_sms}: deterministic={det}  low-prefix(groups "
          f"0..{n_groups-1})={low}")
    overall_ok = overall_ok and det and low

print("\n=== OPTION 2 VIABILITY:",
      "PASS — torch picks the low-prefix groups deterministically"
      if overall_ok else
      "FAIL — torch's pick is not the deterministic low prefix", "===")
sys.exit(0 if overall_ok else 1)
