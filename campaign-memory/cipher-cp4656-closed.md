---
name: cipher-cp4656-closed
description: CIPHER CP 4.6.5+6 closed 2026-05-16 — Phase 4.6 multi-tenant substrate primitive shipped
metadata: 
  node_type: memory
  type: project
  originSessionId: 48591690-eff3-4458-9d0c-6f3597471e76
---

CP 4.6.5+6 STEP complete 2026-05-16; **Phase 4.6 multi-tenant substrate
primitive CLOSED**. Report `cipher-fusion-evidence/cp_4_6_5_6/CP_4_6_5_6_REPORT.md`.
Anchors unchanged (kmod 0.4.8 e2f50452, libcipher_v2 86618c30, libcipher_rt
c2c5d313) — validation-only CP, no substrate code. Awaiting adjudication.

3-arm T4.6.6 measured results (all from on-disk `t466_*_result.json`):
- **Arm 1** cuIpc dedup substrate scaled to **100/100 procs, no ceiling**;
  m=1.866×, RSS 107.7 MiB/proc. Finding: put throughput 3348→508/s under
  100-way concurrency (kmod lock + cuIpc-import serialise) — Phase 5 item.
- **Arm 2** Llama-3.1-8B 32K fp16 real-decode ceiling **13 tenants** (directive
  est. 16; ~12 GB CUDA/green-ctx/workspace overhead). 2.93 tok/s/tenant,
  p99 ITL 352.5 ms. D4 gate FAILS as specified — §3b is a bandwidth roofline
  B=1 decode can't reach (launch-latency-bound); p99/median=1.036 (tight).
  First run OOM'd at 6 — caching-allocator fragmentation + 8.4 GB logits
  transient, fixed in harness (expandable_segments, logits_to_keep=1).
- **Arm 3** isolation PASS — 3/3 foreign-pointer probes rejected, 0 violations.

Load-bearing gap: substrate dedup is **not wired into PyTorch's live KV
cache** (audit §2) — that's why Arm 2 sees no dedup density benefit. Phase 5
(Q3 2026, 10-14 wk) closes it: KV offload hierarchy + partition-aware Marlin
(Song Han) + arbitration + live-decode wiring → 100 real-decode tenants.
Phase 4.6 closes 16/23 CPs. Supersedes [[cipher-cp25-closed]] as campaign
front; relates to [[cipher-t46-measurement]], [[cipher-audit-2026-05-16]].
