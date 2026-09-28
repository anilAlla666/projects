# Track B Path-1 demo: prove a device check-node injected at the cudaGraphInstantiateWithFlags seam
# (1) fires on a REAL torch.cuda.graph-captured graph, (2) executes in replay, (3) host reads its device
# flag post-replay, (4) capture is NOT invalidated. Run under LD_PRELOAD=shim.so.
import ctypes, json, os, torch
shim = ctypes.CDLL(os.path.abspath("gap6_inject_shim.so"))
for fn in ("get_check_flag","get_inject_count","get_last_nodes","get_last_edges","get_last_leaves","get_last_rc"):
    getattr(shim, fn).restype = ctypes.c_int

dev='cuda'; torch.cuda.init()
assert shim.gap6_init()==0, "gap6_init failed"   # pre-allocate device flag cleanly, before any capture
N=2048
static_in  = torch.zeros(N, device=dev)
W = (torch.randn(256,256,device=dev)*0.02)
def body():
    y = static_in*2.0 + 1.0
    z = (W @ W).sum()
    return y + 0.0*z
s=torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3): body()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
inject_before = shim.get_inject_count()
with torch.cuda.graph(g):
    static_out = body()
torch.cuda.synchronize()
inject_after = shim.get_inject_count()
hook_fired = inject_after > inject_before

shim.reset_check_flag()
flag_pre = shim.get_check_flag()
g.replay(); torch.cuda.synchronize()
flag_post = shim.get_check_flag()
print(f"[core] hook_fired={hook_fired} nodes={shim.get_last_nodes()} edges={shim.get_last_edges()} "
      f"leaves={shim.get_last_leaves()} addNode_rc={shim.get_last_rc()} "
      f"flag_pre={flag_pre} flag_post={hex(flag_post & 0xffffffff)} node_ran={flag_post==0x00C1}")

# micro-isolate what (if anything) breaks after injection
def step(label, fn):
    try:
        fn(); torch.cuda.synchronize(); print(f"  [{label}] OK"); return "OK"
    except Exception as e:
        m=str(e).splitlines()[0]; print(f"  [{label}] ERR: {m}"); return m
print("[recap micro-steps]")
r_replay2 = step("second g.replay() same input", lambda: g.replay())
r_alloc   = step("fresh torch.empty alloc",      lambda: torch.empty(N, device=dev).fill_(7.0))
r_copy    = step("static_in.copy_(new)",         lambda: static_in.copy_(torch.full((N,), 3.0, device=dev)))
r_replay3 = step("g.replay() after new input",   lambda: g.replay())
graph_correct=None
try: graph_correct=bool(torch.allclose(static_out, static_in*2.0+1.0, atol=1e-4))
except Exception as e: graph_correct=f"ERR {str(e).splitlines()[0]}"
second_err=None if all(x=="OK" for x in (r_replay2,r_alloc,r_copy,r_replay3)) else f"replay2={r_replay2}|alloc={r_alloc}|copy={r_copy}|replay3={r_replay3}"
print(f"[recap] graph_correct={graph_correct}")

res = dict(hook_fired=bool(hook_fired), inject_count=inject_after,
           last_graph_nodes=shim.get_last_nodes(), last_graph_edges=shim.get_last_edges(),
           last_graph_leaves=shim.get_last_leaves(), addNode_rc=shim.get_last_rc(),
           flag_pre_replay=flag_pre, flag_post_replay=flag_post, flag_post_hex=hex(flag_post & 0xffffffff),
           injected_node_ran=bool(flag_post==0x00C1),
           microsteps=dict(re_replay_same=r_replay2, fresh_alloc=r_alloc, copy_new_input=r_copy,
                           replay_new_input=r_replay3),
           graph_correct_after_new_input=graph_correct, allocator_caveat=second_err)
res["PATH1_NODE_EXECUTES_IN_REPLAY"]=bool(hook_fired and flag_post==0x00C1)
res["CAPTURE_NOT_INVALIDATED"]=bool(graph_correct is True and r_replay2=="OK" and r_replay3=="OK")
json.dump(res, open("/home/ubuntu/cipher-fusion-evidence/ra_keystone/trackB/gap6_path1_result.json","w"), indent=1)
print("PATH1_NODE_EXECUTES_IN_REPLAY:", res["PATH1_NODE_EXECUTES_IN_REPLAY"])
