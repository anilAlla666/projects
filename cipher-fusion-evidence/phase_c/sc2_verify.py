# Phase C SC2 verification — producer-side VMM weight arena.
# Loads TinyLlama-1.1B, rebinds every weight tensor into a CIPHER VMM
# weight arena, and runs the six SC2 success-criteria gates.
import sys, os, json
sys.path.insert(0, '/home/ubuntu/cipher_rt_phase4')
import torch
import torch.nn.functional as F
import cipher_kv_bridge as kvb
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/ubuntu/models/TinyLlama-1.1B'
OUT   = '/home/ubuntu/cipher-fusion-evidence/phase_c/sc2_result.json'
res = {}

def dtype_str(dt):
    return {torch.float16:'float16', torch.bfloat16:'bfloat16',
            torch.float32:'float32'}[dt]

torch.manual_seed(0)
kvb.init(64*1024*1024)                      # 64 MiB KV pool — inits CUDA/VMM

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
model.eval()

# teacher-forced gold prefix — a fixed token sequence
prompt = "The history of computing spans several distinct eras, each defined by"
ids = tok(prompt, return_tensors='pt').input_ids.cuda()
print("teacher-forced seq len:", ids.shape[1])

with torch.no_grad():
    stock_logits = model(ids).logits.float().cpu()        # [1,T,V] stock weights

params = list(model.named_parameters())
total = sum(p.numel()*p.element_size() for _,p in params)
res['n_param_tensors']    = len(params)
res['total_weight_bytes'] = int(total)
print(f"params: {len(params)} tensors, {total/1e9:.3f} GB")

# --- criterion 1: VMM-backed weight arena ---
arena = kvb.weight_arena_create(int(total) + 16*1024*1024, 1)
res['arena_base_hex'] = hex(arena.base)

# memcmp sample — capture the largest param's stock bytes BEFORE rebind
big_name, big_p = max(params, key=lambda kv: kv[1].numel())
stock_big = big_p.detach().to('cpu', copy=True).contiguous()

# rebind every weight tensor into the arena
all_equal = True
for name, p in params:
    vt = arena.alloc(list(p.shape), p.element_size(), dtype_str(p.dtype))
    vt.copy_(p.data)
    if not torch.equal(vt, p.data):
        all_equal = False
        print("MISMATCH after copy:", name)
    p.data = vt
res['copy_torch_equal_all'] = bool(all_equal)
print("arena cursor used:", arena.cursor, "of", arena.size)

# --- criterion 3: DtoH memcmp bytes-identical to stock ---
big_after = dict(model.named_parameters())[big_name].detach().cpu().contiguous()
memcmp_ok = (stock_big.numpy().tobytes() == big_after.numpy().tobytes())
res['memcmp_param'] = big_name
res['memcmp_bytes_identical'] = bool(memcmp_ok)
print("memcmp", big_name, "bytes-identical ->", memcmp_ok)

# --- criterion 2: page_info reports VMM-handle presence for weight pages ---
wt = dict(model.named_parameters())[big_name]
pi = kvb.page_info(wt.data_ptr())
res['page_info'] = {k:(hex(v) if k=='vmm_handle' else v) for k,v in pi.items()} if pi else None
print("page_info(weight tensor):", res['page_info'])
junk = torch.zeros(16, device='cuda')       # torch-allocated, not CIPHER
res['page_info_noncipher'] = kvb.page_info(junk.data_ptr())
print("page_info(non-CIPHER ptr):", res['page_info_noncipher'])

# --- criterion 5: producer creates exportable VMM handle ---
fd = arena.export_fd()
fstat_ok = False
try:
    os.fstat(fd); fstat_ok = True
finally:
    if fd >= 0: os.close(fd)
res['export_fd'] = int(fd)
res['export_fd_valid'] = bool(fd >= 0 and fstat_ok)
print("export_fd:", fd, "fstat_ok:", fstat_ok)

# --- criterion 4: real forward pass reads VMM weights — teacher-forced KL ---
with torch.no_grad():
    vmm_logits = model(ids).logits.float().cpu()
ps = F.log_softmax(stock_logits[0], dim=-1)
qs = F.log_softmax(vmm_logits[0],   dim=-1)
kl = (ps.exp() * (ps - qs)).sum(-1)          # KL(stock || vmm) per position
res['kl_mean'] = float(kl.mean())
res['kl_max']  = float(kl.max())
res['logits_max_abs_diff'] = float((stock_logits - vmm_logits).abs().max())
print(f"teacher-forced KL: mean={res['kl_mean']:.3e} max={res['kl_max']:.3e}")
print(f"logits max abs diff: {res['logits_max_abs_diff']:.3e}")

# --- gates ---
res['gate_1_vmm_backed']   = (res['page_info'] is not None and
                              res['page_info'].get('kind') == 'weight')
res['gate_2_page_info']    = (res['page_info'] is not None and
                              res['page_info'].get('vmm_handle','0x0') != '0x0')
res['gate_3_memcmp']       = res['memcmp_bytes_identical']
res['gate_4_kl']           = res['kl_max'] <= 0.1
res['gate_5_export']       = res['export_fd_valid']
res['copy_equal']          = res['copy_torch_equal_all']
res['SC2_PASS'] = all([res['gate_1_vmm_backed'], res['gate_2_page_info'],
                       res['gate_3_memcmp'], res['gate_4_kl'],
                       res['gate_5_export'], res['copy_equal']])
json.dump(res, open(OUT,'w'), indent=2, default=str)
print("\n=== SC2 RESULT ===")
print(json.dumps(res, indent=2, default=str))
print("SC2_PASS:", res['SC2_PASS'])
