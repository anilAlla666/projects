# CIPHER v2.0 — single deployable `.deb` (Class-D anchored)

**Date:** 2026-06-03 · **Status:** design approved-in-principle; spec for review · **Author:** Claude (with Anil)

## 1. Goal & positioning

Fuse everything CIPHER has built into **one deployable artifact**:
`cipher-platform_2.0_amd64.deb`. Install → the whole substrate lands.

**What it is (honest):** the CIPHER **substrate** (`libcipher_rt.so` + kmod) +
a CLI that (a) runs **reproducible Class-D demos and discipline gates** over the
validated handlers, and (b) provides `cipher accelerate` — the one real
request→tokens path — by **injecting under the customer's own vLLM** (KV-dedup /
weight-share / FP8 / DVFS, zero code change). **It is NOT a CIPHER-native serving
runtime:** the handlers (`cipher_inc4.py` etc.) run fixed synthetic traces and
exit; they have no request intake. The manifest selects *which demo/gate runs*,
it does not carry customer work. (A native request→tokens engine is a separate
future build — §7.)

**Positioning (per the 2026-05-19 Workload-Architecture declaration):** CIPHER v1
is *a cross-tenant GPU substrate for agentic multi-tenant (Class-D) inference.*
Headline metrics are **agents/GPU, fleet tok/W, per-agent p99, density.**
MFU is explicitly **not** a headline; FP8 ships as an optional compute knob.

**Decisions locked (with Anil):** (1) one install package; (2) full validated
set; (3) **substrate + demos/gates; `accelerate` = the real serving path** (not a
CIPHER-native engine); (4) Class-D anchored; (5) cross-tenant batching shipped
but named next-milestone; (6) `cipher demo` is the flagship.

## 2. Discipline invariant (non-negotiable)

The `.so` and handlers install **byte-identical** to the validated artifacts
(md5-checked in postinst against a known-good manifest). The deployed anchor
(`1f305ce6`, Jun-01 `libcipher_rt.so`) is **unchanged**. The *only* new code is
the `cipher_platform/` orchestration layer (CLI, router, manifest, gates,
report, config). `OFF == byte-identical` to a clean baseline. This is what makes
"no regress" mechanically true.

## 3. Package layout

```
/usr/lib/cipher/              libcipher_rt.so (REFRESH -> Jun-01 anchor 1f305ce6), libc10.so, libcipher_v2.so
/usr/src/cipher-kmod-0.7.0/   REFRESH 0.4.8 -> 0.7.0 DKMS source (32 ioctls; NR10 clock-set => DVFS w/o sudo)
/usr/share/cipher/handlers/   cipher_inc4.py (engine+DVFS), v0_phaseC_nf4_cofire.py (density),
                              v0_phaseA_maxmfu.py (FP8, secondary), cipher_engine*.py, cipher_product_report.py
/usr/lib/python3/dist-packages/cipher_platform/  cli.py router.py manifest.py gates.py report.py config.py
/usr/bin/cipher               unified CLI (cipher-run kept as raw injection wrapper)
/etc/cipher/platform.json     so_path, model_library_dir, regime defaults, gate thresholds, tenant config
```

Install mechanics (postinst, extends existing): DKMS build+load `cipher_kmod`
0.7.0 against running kernel → `/dev/cipher` 0666 → optional
`cipher-platform.service` (kmod + exporter) → print "run `cipher selftest`".
`prerm` unloads + DKMS-removes. Dependency: `dkms`, `linux-headers-$(uname -r)`,
loaded `nvidia.ko`. The Python ML stack (torch/transformers/bitsandbytes/pynvml)
+ model checkpoints are a **customer dependency** (checked by `cipher selftest`).

## 4. The `cipher` CLI (substrate + demos/gates; `accelerate` = real serving)

```
cipher demo    [--agents N]           # FLAGSHIP Class-D demo: N agents on 1 GPU -> agents/GPU, fleet tok/W, p99 (wraps cipher_inc4.py)
cipher bench   --manifest jobs.json   # run the reproducible demo/gate suite by declared regime (subprocess per regime)
cipher selftest [--quick]             # discipline gates on THIS box (section 5)
cipher report                         # consolidated Class-D scorecard from last run
cipher accelerate -- <vllm cmd>       # THE REAL serving path: inject under the customer's vLLM (KV-dedup/weight-share/FP8), zero code change
```

`bench`/`demo` run FIXED demo workloads (no request intake — see §1).

**Regime → handler dispatch** (router.py):

| regime | handler | injection state | headline |
|---|---|---|---|
| `agent`/`decode` | cipher_inc4.py | `CIPHER_RT_DISABLE_AUTO_INIT=1`; DVFS via kmod NR10 | yes |
| `density` | v0_phaseC_nf4_cofire.py | auto-init off | yes |
| `compute`/`prefill` | v0_phaseA_maxmfu.py | `CUDA_INJECTION64_PATH`, `CIPHER_FP8=1` | secondary |

The two injection states **conflict by design** (engine needs auto-init off for
capture-safety; FP8 needs the cublasGemmEx actuator); each regime runs as its
own subprocess. All `/home/ubuntu/...` paths read from `platform.json`.

**Lane isolation (`isolate.py`) — load-bearing.** Every gate probe AND every
regime lane runs through one primitive: a fresh subprocess with its own CUDA
context, **GPU-settled + fully reaped before the next unit begins**. A lane crash
is caught (rc + captured stderr) and reported as that-lane-FAILED — it never
poisons the next lane (the V.0 cross-phase poisoning class). Two further rules:
(1) the OFF-byte-identical probe runs **last** in `selftest` (it does full
auto-init whose device state must not precede a capture lane); (2) each lane gets
up to **3 isolated attempts** — the inc-3b *re-capture* mitigation for intrinsic
CUDA-graph capture-invalidation flakiness on torch 2.11 (a fresh re-capture
clears the transient; a genuine fault fails all attempts). Verified: `selftest
--quick` green **5/5**, re-capture used in 3/5 (single retry each).

## 5. Discipline gates, report, honest scope

`cipher selftest` runs the validated gates **on the customer box** and refuses to
claim success without them: **OFF byte-identical** (sum+argmax bit-match);
**engine FAULT=0** (100-agent teacher-forced, or `--quick` ~12-agent);
**energy** (DVFS tok/W); **density** (NF4 KL=0 + GB/model). The negative control
(misroute) must FAULT — a gate that can only pass is rejected.

`cipher report` emits the **Class-D scorecard** (agents/GPU, fleet tok/W,
per-agent p99, density) + a **Tier-B ledger** marked *present-but-not-claimed*:
FP8 quality, cross-tenant batching, KV-dedup/weight-share, Koopman, classifier —
never faked.

**Honest scope (package README):** single-GPU (engine + plugin validated on one
H100; plugin asserts `dev_index==0`). Cross-tenant batching (3–6× tok/W) and
KV-dedup (45 GiB) are validated but path-divergent (socket executor / vLLM
`accelerate` path) — shipped, reachable, **named next-milestone**, not claimed
inside the engine graph.

## 6. Validation already performed (mock + real hardware)

A mock of the orchestration core (`/home/ubuntu/cipher_v2_mock/`) was built and
run, then wired to the **real** validated engine on small cheap workloads:

- **Orchestration mock:** manifest→regime→handler dispatch works; injection-state
  conflict correctly isolated (agent/density get auto-init-off + no injection
  path; compute gets injection path + no auto-init-off); negative control flips
  the gate PASS→FAIL (gate has teeth).
- **Real agent lane** (TinyLlama, 12 agents, 12 coalesced waves): **FAULT=0,
  exact=12, misroute→FAULT**, real p99 short/long **324/390 ms**, real tok/W
  **0.95 @ 89 W**, `auto-init SKIPPED` confirmed.
- **Real density lane** (TinyLlama + Qwen2-7B NF4): **both KL=0** (24/24),
  Qwen2-7B packs to measured **5.46 GB**, 2 models co-resident, live 6.29 GB.
- **Environment drift found & documented:** Llama-3.2-1B `rope_scaling="llama3"`
  breaks CUDA-graph capture on torch 2.11 (validated inc-4 used an older torch).
  Agent lane uses TinyLlama; density uses Qwen2-7B (no llama3 rope). Not a
  product defect — a torch-version note for the `selftest` model set.

## 7. Out of scope (this build)

- Wiring cross-tenant batching into `cipher serve` (it's a socket executor; real
  integration, not packaging — next milestone).
- Multi-GPU (single-GPU validated only).
- Re-proving the substrate (separately validated; this build packages it).
- Bundling torch/models in the `.deb` (customer dependency).

## 8. Build steps

1. Stage payload: refresh `libcipher_rt.so` (Jun-01), kmod 0.7.0 DKMS source,
   handlers, `cipher_platform/` package, `cipher` CLI, `platform.json`.
2. `debian/` control (v2.0, Depends: dkms, linux-headers), postinst (md5-verify
   payload, DKMS build+load, /dev perms), prerm (unload+remove).
3. `dpkg-deb --build` → `cipher-platform_2.0_amd64.deb`; verify `dpkg-deb -c`.
4. `cipher selftest --quick` on a target box as the acceptance gate.
