#!/usr/bin/env python3
"""Track 3 SC3-2 — multi-scenario migration test tenant.

Launched (by sc3_run.py) under CUDA_INJECTION64_PATH=<SC3 libcipher_rt> with
CIPHER_QOS_CLASS=partition + CIPHER_SM_COUNT + scenario env. Runs ONE scenario,
writes a JSON verdict to $SC3_OUT.

Scenarios (env SC3_SCENARIO):
  self_migrate  — call cipher_rt_green_ctx_migrate() directly; kernels must
                  then run on the new SM set (the migrate primitive in
                  isolation, no kmod proposal).
  verify_fault  — CIPHER_SC3_FAULT=verify: L1 verify fails -> migrate returns
                  <0, old context intact and still current.
  destroy_fault — CIPHER_SC3_FAULT=destroy: post-swap destroy fails -> migration
                  still COMMITTED, tenant keeps running on the new context.
  optin         — COMPACT then POLL: PROPOSED iff CIPHER_MIGRATABLE was set
                  (verifies the env -> SUBSCRIBE_MIGRATE plumbing).
  e2e           — TinyLlama decode tenant: handler.step() each round while a
                  controller forces COMPACT; TFGATE KL gate per prompt.
"""
import json
import os
import sys
import time

SCEN = os.environ["SC3_SCENARIO"]
OUT  = os.environ["SC3_OUT"]
sys.path.insert(0, "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3")

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
import torch                                                  # noqa: E402
import cipher_migrate as cm                                   # noqa: E402

res = {"scenario": SCEN, "pid": os.getpid()}


def warmup():
    """Force cuInit + an explicit stream -> green-ctx build."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.ones(8192, device="cuda")
        x = (x * 2.0).relu()
    torch.cuda.synchronize()


def high_target(cur_mask):
    """A count-preserving target: the highest free groups not in cur_mask."""
    k, t, g = bin(cur_mask).count("1"), 0, 14
    while bin(t).count("1") < k and g >= 0:
        if not (cur_mask & (1 << g)):
            t |= (1 << g)
        g -= 1
    return t


# -------- primitive scenarios (no model) --------------------------------
if SCEN in ("self_migrate", "verify_fault", "destroy_fault"):
    warmup()
    h = cm.MigrateHandler()
    res["lib_bound"] = h.lib is not None
    res["lib_error"] = h.lib_error
    old_cur = h.cur_mask()
    old_sms = cm.probe_sms()
    target = high_target(old_cur)
    res.update(old_cur_mask=old_cur, target_mask=target, old_sms=old_sms,
               expected_target_sms=sorted(cm.mask_to_sms(target)))
    rc = h.migrate_primitive(target)
    torch.cuda.synchronize()
    new_sms = cm.probe_sms()
    new_cur = h.cur_mask()
    res.update(primitive_rc=rc, new_sms=new_sms, new_cur_mask=new_cur)

    if SCEN == "self_migrate":
        res["pass"] = bool(
            rc == 0 and new_cur == target and len(new_sms) > 0 and
            set(new_sms) <= cm.mask_to_sms(target) and
            set(new_sms).isdisjoint(old_sms))
    elif SCEN == "verify_fault":
        res["pass"] = bool(
            rc < 0 and new_cur == old_cur and set(new_sms) == set(old_sms))
    else:  # destroy_fault
        post = (torch.ones(4096, device="cuda") * 3.0).sum().item()
        res["post_migration_op_ok"] = bool(post == 4096 * 3)
        res["pass"] = bool(
            rc == 0 and new_cur == target and
            set(new_sms) <= cm.mask_to_sms(target) and
            res["post_migration_op_ok"])
    h.close()

# -------- opt-in plumbing scenario --------------------------------------
elif SCEN == "optin":
    warmup()
    h = cm.MigrateHandler()
    h.compact()
    p = h.poll()
    want_proposed = bool(os.environ.get("CIPHER_MIGRATABLE"))
    got_proposed = (p["state"] == cm.MIG_PROPOSED)
    res.update(migratable_env=os.environ.get("CIPHER_MIGRATABLE"),
               poll_after_compact=p, want_proposed=want_proposed,
               got_proposed=got_proposed)
    res["pass"] = (got_proposed == want_proposed)
    h.close()

# -------- end-to-end decode integration --------------------------------
elif SCEN == "e2e":
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
    MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
    ROUNDS = int(os.environ.get("SC3_ROUNDS", "8"))
    tag = os.environ.get("SC3_TAG", "t")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)
    torch.cuda.synchronize()

    h = cm.MigrateHandler()
    res["lib_bound"] = h.lib is not None
    start_mask = h.cur_mask()
    start_sms = cm.probe_sms()

    gold = {g["prompt"]: g for g in json.load(open(WL01 + "/gold.json"))["prompts"]}
    gold_logits = torch.load(WL01 + "/gold_logits.pt", map_location="cuda")
    n_prompts = len(gold)

    kl_pc, migrations, round_log = [], [], []
    for r in range(ROUNDS):
        pr = r % n_prompts
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]
        # teacher-forced KL gate (cascade-free)
        full = torch.tensor([g["full_ids"]], device="cuda")
        with torch.no_grad():
            lo = model(input_ids=full).logits[0]
        seg = lo[plen - 1:plen - 1 + gen, :].float()
        lp = F.log_softmax(gold_logits[pr].float(), dim=-1)
        lq = F.log_softmax(seg, dim=-1)
        kl_max = float((lp.exp() * (lp - lq)).sum(-1).max())
        kl_pc.append({"round": r, "prompt": pr, "kl_max": kl_max})
        # poll-and-migrate handler at the decode-round boundary
        act = h.step()
        if act["action"] == "migrate":
            migrations.append({"round": r, **act})
        round_log.append({"round": r, "kl_max": kl_max,
                           "handler": act["action"],
                           "cur_mask": h.cur_mask()})
        print("E2E %s round=%d kl_max=%.2e handler=%s cur=0x%x"
              % (tag, r, kl_max, act["action"], h.cur_mask()),
              file=sys.stderr, flush=True)
        time.sleep(0.4)                      # let the controller's COMPACT land

    end_mask = h.cur_mask()
    end_sms = cm.probe_sms()
    kl_max_all = max(x["kl_max"] for x in kl_pc)
    committed = [m for m in migrations if m.get("committed")]
    res.update(
        start_mask=start_mask, end_mask=end_mask,
        start_sms=start_sms, end_sms=end_sms,
        n_migrations=len(migrations), n_committed=len(committed),
        migrations=migrations, kl_max_all=kl_max_all,
        kl_gate_pass=bool(kl_max_all <= 0.1), rounds=round_log)
    # tenant verdict: KL gate holds, and every committed migration had L2 ok
    # and actually moved the green ctx.
    res["pass"] = bool(
        res["kl_gate_pass"] and
        all(m["l2_ok"] and m["primitive_rc"] == 0 for m in committed))
    res["migrated"] = len(committed) > 0
    h.close()

else:
    res["error"] = "unknown SC3_SCENARIO=%s" % SCEN
    res["pass"] = False

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, default=str)
print("SC3_TENANT %s pass=%s -> %s" % (SCEN, res.get("pass"), OUT),
      file=sys.stderr, flush=True)
sys.exit(0 if res.get("pass") else 1)
