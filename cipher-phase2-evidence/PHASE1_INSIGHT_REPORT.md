# CIPHER kmod — Phase 1 Insight Report

**Date:** 2026-05-13
**Pod:** Lambda H100 80GB SXM5 · 192.222.53.2
**Kernel:** 6.8.0-1046-nvidia · gcc 11.4.0 · headers
            `/lib/modules/6.8.0-1046-nvidia/build`
**NVIDIA driver:** 580.105.08 · CUDA runtime 13.0 · `nvidia.ko` 14 MB +
                    5 sub-modules
**Author of record:** Anil
**Module:** `cipher_kmod.ko` 0.1.0, 718 384 bytes, GPL,
            `depends:` empty

---

## 1. What we built

`cipher_kmod` is the first piece of CIPHER that lives below the
userspace driver. It is an out-of-tree Linux kernel module — eight
files totalling roughly 700 lines of C — that loads alongside
`nvidia.ko`, registers a kprobe and a kretprobe on
`nvidia_unlocked_ioctl`, and exposes a human-readable view of the
ioctl traffic at `/proc/cipher/stats`.

The build is self-contained. `modinfo cipher_kmod.ko` reports an
empty `depends:` field, meaning we hold no link-time reference to
`nvidia.ko`. The string `"nvidia_unlocked_ioctl"` is embedded in
our module's data segment as a `.symbol_name` field in two
`struct kprobe` instances; the kernel resolves it via its internal
kallsyms machinery at `register_kprobe()` call time. This matters
because it lets us be loaded, unloaded, and rebuilt independently
of the NVIDIA driver, and it confirms the design intent that we
sit beside nvidia.ko, not inside it.

Files at `/workspace/cipher_kmod/`:

| file | role |
|---|---|
| `cipher_internal.h` | shared types, NV_ESC slot constants, externs |
| `cipher_main.c` | `module_init`/`module_exit`, cross-TU global storage |
| `cipher_probe.c` | kprobe + kretprobe handlers, RCU per-PID hashtable |
| `cipher_ioctl_decode.c` | NV_ESC nr to name lookup table |
| `cipher_proc.c` | `/proc/cipher/stats` seq_file emitter |
| `cipher_dev.c` | `/dev/cipher` placeholder character device |
| `Kbuild` | obj-m link list |
| `Makefile` | out-of-tree build wrapper |

The hot path is irq-safe and lock-light. A pre-handler reads
`regs->si` for the ioctl `cmd`, decodes the NV_ESC nr via
`_IOC_TYPE`/`_IOC_NR`, takes `rcu_read_lock()` to look up the calling
PID in a 1024-bucket hashtable, and bumps a per-PID `atomic64_t`
slot plus a global per-cmd `atomic64_t`. New PIDs get inserted
through a slowpath that allocates with `GFP_ATOMIC | __GFP_NOWARN`,
spinlocks, double-checks, and links via `hash_add_rcu`. Per-PID
`comm` is refreshed from `current` on every observation so
`exec()`'d processes show their current name rather than their
initial one. The kretprobe carries `(cmd, pid)` from entry to
return via the `kretprobe_instance->data` blob; on negative
return values it bumps a per-cmd error counter and a per-PID
error counter. On unload we `unregister_kretprobe`, then
`unregister_kprobe`, then `synchronize_rcu` to drain in-flight
readers, then walk-and-`kfree_rcu` the entire hashtable, then
`rcu_barrier` to wait for the deferred frees to complete. Ten
lines of teardown, no leaks.

The placeholder `/dev/cipher` exposes a no-op `open`/`release` and
returns `-ENOTTY` from `unlocked_ioctl`. That control surface is
reserved for Phase 2 onward when userspace will use it to register
tenant identity, request snapshots, and reset counters. For
Phase 1 it exists so that a future CIPHER stack can rely on the
node being present without a second kmod load.

The full Phase 1 build-and-validate cycle ran end to end without
incident. Build: clean, zero warnings. `insmod`: clean, taint
unchanged at 12288 (the O+E bits already set by `nvidia.ko`). 414
seconds of ambient and smoke-test load: clean. `rmmod`: clean,
25 ms unload time, taint unchanged. Re-run of the smoke test
without the kmod loaded: produced byte-identical model output to
the with-kmod run, confirming that the attach/detach cycle did not
perturb the runtime in any way that survived unload.

## 2. What the kernel boundary showed us

The point of Phase 1 was not to fix anything. It was to put eyes
on a layer of the stack we had been guessing about. Three
workloads — `nvidia-smi`, a one-line `torch.zeros(8, device='cuda').sum()`,
and the existing CIPHER `smoke_marlin_tinyllama.py` — each
produced a distinct ioctl signature, and the signatures together
overturned a load-bearing assumption from the original Phase 1
plan.

### 2.1 The three signatures

| workload | duration | total ioctls | named NV_ESC | OTHER (RM_*) | OTHER share |
|---|---:|---:|---:|---:|---:|
| `nvidia-smi` (one shot) | seconds | 82 | 7 | 75 | **91.5 %** |
| `torch.zeros(8).sum()` | one call | 430 | 32 | 398 | **92.6 %** |
| `smoke_marlin_tinyllama.py` | 370 s | 1 537 | 84 | 1 453 | **94.5 %** |

The names we had built into the decoder — the 14 NV_ESC_*
ioctls in nrs 200..218 — accounted for 5 to 9 percent of the
ioctl traffic across all three workloads. The remaining 90+
percent went to nrs outside that range. Our handler counted them
correctly into the OTHER bucket, which is what told us the bulk
of CUDA's kernel boundary is somewhere we had not been looking.

### 2.2 The XFER_CMD assumption was wrong

Our agreed plan predicted that `NV_ESC_IOCTL_XFER_CMD` (nr 211)
would dominate. It is the ioctl name that older NVIDIA RM ABI
documentation pointed at as the "carry batched commands across
the kernel boundary" entry point, and the Phase 1 plan inherited
that prediction.

After the smoke test, the per-PID summary showed
`XFER_CMD = 0` for every PID, including the python3 process that
just ran a full TinyLlama decode with Marlin patching 155 linears
and firing 12 246 Marlin GEMM calls. Zero `XFER_CMD`s across
1 537 observed ioctls.

The actual carrier on driver 580.105.08 is the
`NV_ESC_RM_*` family — `NV_ESC_RM_ALLOC` (0x29 = 41),
`NV_ESC_RM_FREE` (0x29-adjacent), `NV_ESC_RM_CONTROL` (0x2A = 42),
and roughly thirty more entries spanning nrs 0x21..0x4D
(33..77 decimal). These are issued through the same
`nvidia_unlocked_ioctl` handler, but their nrs sit below the
NV_ESC name range our decoder knew about, which is why they all
landed in OTHER. The XFER_CMD path appears to be a legacy ABI
that is no longer the primary carrier on this driver
generation.

This is exactly the class of insight Phase 1 was built to surface.
At LD_PRELOAD we never see it: by the time a userspace shim
intercepts a `cuLaunchKernel` call, the kernel-side ioctl has
already been issued, dispatched, and returned, and the calling
convention you see is CUDA's, not the driver's. The kernel
boundary tells you what NVIDIA is actually doing under the
covers, not what the user-mode driver is asking it to do at the
CUDA layer.

### 2.3 Per-thread granularity earns its place

The smoke test's `/proc/cipher/stats` showed seven distinct PIDs
with two distinct TGIDs. The main python3 process (PID 9617,
TGID 9617) issued 1 492 ioctls; four worker LWPs sharing TGID
9617 (PIDs 9661, 9649, 9664, 9650) issued 27, 10, 6, and 2
ioctls respectively. Two earlier PIDs from steps d and f
(nvidia-smi at 9467, the test torch process at 9494) were still
present in the table because Phase 1 retains entries for the
module's lifetime — a known and intentional tradeoff documented
under the Phase 1.5 work list.

Aggregating to TGID would have collapsed those workers into the
parent and we would have lost the signal that CIPHER's runtime
spawns helper threads that touch the driver. That signal is going
to matter when Phase 2 tries to associate ioctls with tenant
identity: tenants are TGIDs, but contention shows up at the LWP
level, and a per-LWP table makes it possible to see that
distinction.

`get_task_comm()` was called on every observation per Modification A.
The cost is one task spinlock acquisition plus a 16-byte copy per
ioctl, charged into the IRQ-disabled kprobe context. Across
1 537 smoke-test ioctls we saw zero allocation failures and zero
errors, and the workload tps came in at 71.95 with the kmod
loaded versus 72.70 without — a one-percent delta that lives
inside TinyLlama's natural run-to-run variance and is not large
enough to attribute to the probe with any confidence. We will get
a bounded number for the per-ioctl cost in Phase 1.5.

## 3. What this enables that LD_PRELOAD cannot

The prototype CIPHER stack lives entirely above the userspace
driver. `libcipher_rt.so` and `libcipher_hook.so` intercept
`cuLaunchKernel`, `cublasGemmEx`, and a handful of related
entry points; they observe and substitute at the CUDA API
boundary. That layer is the right place to swap a Marlin GEMM in
for a cuBLAS GEMM, but it is not the right place to enforce
multi-tenant policy or to produce an audit trail that a
sovereign-cloud security team will sign off on. Three things
become structurally available below the userspace driver that are
not available above it.

**PID and namespace identity at ioctl entry.** The kprobe runs in
the kernel context of the calling thread, before any allocation or
mapping has crossed the PCIe bus. We have `current->pid`,
`current->tgid`, `current->nsproxy`, and (with the obvious
extension) the cgroup the thread is pinned into. A userspace
LD_PRELOAD shim sees only the `cuLaunch*` arguments by the time
they have been laundered through libnvidia-rmconfig, and it has no
trustworthy view of which container or which user the caller
represents — the calling process can lie, and the shim has no
ground to push back. The kernel does not have that problem. The
foundation we just laid will let Phase 2 attach a tenant identity
to every ioctl, with the kernel's word for it.

**Per-tenant cmd histograms for SLA enforcement.** The Phase 1
proc output already shows per-PID counts; a small extension gives
us per-tenant counts. Once the histogram exists, rate-limiting,
quota enforcement, and SLA telemetry become arithmetic over the
counters. A noisy-neighbor tenant generating ten times the ioctl
rate of its peers is identifiable from the kernel side without
asking the userspace driver for cooperation.

**The trust shape that enterprise procurement asks for.** A
signed kernel module installed via `.deb` or `.rpm` and
registered through `nvidia-modprobe`-style infrastructure is the
shape that regulated deployments — sovereign clouds, regulated
finance, healthcare, defense — will accept. LD_PRELOAD is a
disqualifier in those procurement processes because the user
running the workload can override it. A kernel-rooted module
cannot be overridden from userspace. The CIPHER compliance
observers (CARBON, TRACE, RECEIPT, FAIRNESS, COMPLY) get
hardware-rooted trust from this position instead of trust that
relies on the user not cheating.

**The integration substrate for Phase 2.** Phase 2 replaces
`LD_PRELOAD=libcipher_hook.so` with `CUDA_INJECTION64_PATH=`
pointing at a `libcipher.so` v2 that registers itself with the
CUDA runtime through NVIDIA's documented injection interface.
That injection path needs a kernel-side counterpart to attribute
its calls to a tenant; cipher_kmod is that counterpart. The
ioctl interface on `/dev/cipher` (currently a `-ENOTTY`
placeholder) is where the tenant identity will flow.

**The integration substrate for Phase 3.** Phase 3 wires the GPU
state spine — MMIO reads of GPU clock and power, PMU programming
for counter capture, possibly green-context partition enforcement.
Those operations require the kernel-mode PCIe BAR mappings that
nvidia.ko owns. cipher_kmod is the host module those mappings
will eventually live alongside.

## 4. Phase 1.5 work list

Concrete, prioritized, none of these in scope tonight.

1. **Extend `cipher_ioctl_decode.c` with the `NV_ESC_RM_*` family.**
   Approximately thirty entries spanning nrs 0x21..0x4D. With this
   in place, the OTHER bucket should drop from 90+ percent to
   single-digit percent on a real CUDA workload. This is the single
   change that converts the Phase 1 evidence from "we know there's
   a lot in OTHER" to "we know exactly which RM_CONTROL paths
   CUDA actually uses."

2. **Probe `do_exit` to reap dead PID entries.** Phase 1 retains
   per-PID entries for the module's lifetime. Two PIDs from earlier
   test steps were still in the post-smoke table 6 minutes after the
   processes exited. For a module that may stay loaded for days, this
   is a leak. A second kprobe on `do_exit` with a per-PID lookup
   plus `hash_del_rcu` plus `kfree_rcu` clears it. Watch for the
   PID-reuse case: same PID re-entering after exit should look like
   a fresh entry with new `first_seen_jiffies`, not an old one with
   accumulated counts.

3. **Measure kprobe overhead empirically.** The smoke comparison
   gave us 71.95 vs 72.70 tps, a 1.03 % delta that is within run
   noise. A proper measurement is ten runs each with and without the
   kmod, reporting median and inter-quartile range for both tps and
   tok/W. This number has to be defensible before we expand the hot
   path with anything more expensive.

4. **Add per-tenant cmd histogram to `/proc` output.** The data is
   already collected per PID. Aggregating per TGID and per (eventual)
   tenant tag is one extra map. Useful for diagnosis well before
   Phase 2's tenant identity flow lands.

5. **Lockless ring buffer for last N ioctls.** Deferred from the
   Phase 1 mock per Modification C. When we want to debug a specific
   tenant's ioctl sequence (which RM_CONTROL did they call, in what
   order, with what args) the histogram is not enough; a ring of the
   last few thousand observations is. Per-CPU ring or single-shared
   ring with `cmpxchg` head; both viable.

## 5. Risk register update

| risk | Phase 1 outcome | residual |
|---|---|---|
| Kernel panic from probe handler | 0 panics across insmod + 414 s of load + smoke + rmmod | low; design is irq-safe and tested |
| `nvidia.ko` interaction (unload-while-attached, version drift) | not exercised; symbol resolution held cleanly | medium; needs `nvidia.ko` rmmod test in Phase 1.5 |
| rt perturbation | byte-identical smoke output with vs without kmod; tps within run noise | very low |
| Memory leak | `alloc_failures=0` across run; teardown freed entire table; `rcu_barrier` confirmed quiescence | low |
| License / kernel API misuse | GPL declared, kprobes are documented and approved API | very low |
| Forward compatibility across NVIDIA driver versions | symbol `nvidia_unlocked_ioctl` resolved on 580.105.08 | medium; the symbol name is stable historically but not contractual; Phase 1.5 should test against >=1 other driver version |
| Misleading decode of OTHER ioctls | reported correctly; predicted-dominant XFER_CMD turned out to be wrong; this is the insight, not a defect | n/a |

The realized risk in Phase 1 was zero on every operationally
meaningful axis. The medium-residual items (nvidia.ko unload race,
driver version drift) are testing gaps rather than defects.

## 6. Phase 2, locked from this evidence

The Phase 1 result is what justifies the Phase 2 plan. Three
elements are now committed.

**Replace LD_PRELOAD with CUDA_INJECTION64_PATH.** NVIDIA's
documented injection interface accepts a `libcipher.so` v2 that
the CUDA runtime loads at `cuInit` time and offers proper hooks
for. This is the trust shape that procurement will accept and the
shape that integrates cleanly with the kernel-side identity flow.
The existing `libcipher_hook.so` LD_PRELOAD path remains for
local development and CI but stops being the production deployment
shape.

**Tenant identity flows through `/dev/cipher`.** `libcipher.so` v2
opens `/dev/cipher`, identifies the tenant via an ioctl that
the kmod stamps onto the calling TGID's stats entry, and from that
point every ioctl from that TGID is attributable to a tenant
without further userspace cooperation. The Phase 1 placeholder
`/dev/cipher` is the surface this lands on; the `-ENOTTY`
becomes a real ioctl table.

**The first three integration APIs are designed against the
tenant-aware foundation.** Specifically:

- `CIPHER_IOC_REGISTER_TENANT(uuid, sla_class)` — userspace
  declares its tenant identity; kmod stamps the calling TGID's
  hashtable entry.
- `CIPHER_IOC_SNAPSHOT(buffer, size)` — userspace requests a
  consistent snapshot of the per-tenant histogram; kmod copies
  out under RCU.
- `CIPHER_IOC_RESET(scope)` — clear counters globally, per
  tenant, or per PID.

Beyond these three, Phase 2 also plans a netlink path for the
high-rate stream (per-ioctl events) since ioctl is the wrong
shape for a streaming firehose. That decision can wait for
Phase 2 design; the three synchronous APIs above are what
`libcipher.so` v2 needs to ship.

---

## Final test outcomes (this session, end to end)

| step | check | result |
|---|---|---|
| build | `make` clean, 0 warnings | PASS |
| .ko verification | `depends:` empty, no nvidia symbol references, all expected symbols present | PASS |
| insmod (a) | exit 0, taint 12288 to 12288, refcount 0 | PASS |
| dmesg (b) | banner + `/dev/cipher` ready + kprobes-attached + loaded-ok; 0 WARN/oops/BUG | PASS |
| fs presence (c) | `/dev/cipher` and `/proc/cipher/stats` both present | PASS |
| nvidia-smi (d) | normal output, no perturbation | PASS |
| stats post-d (e) | 82 ioctls captured from nvidia-smi, 7 named, 0 errors | PASS |
| torch CUDA op (f) | `torch.zeros(8).sum() = 0.0` on cuda:0 | PASS |
| stats post-f (g) | 512 ioctls total, 2 PIDs, 0 errors, 0 alloc failures; XFER_CMD assumption found wrong | PASS w/ insight |
| smoke (h) | 155 compressed, 12246 marlin counters, 80/80 tokens, coherent text, exit 0 | PASS |
| rmmod (i) | exit 0, taint unchanged, "unloaded cleanly" in dmesg, fs entries gone, alloc_failures=0 | PASS |
| nvidia-smi post-rmmod (i.1) | normal output | PASS |
| smoke post-rmmod (i.2) | byte-identical output to step h, tps 72.70 vs 71.95 (noise) | PASS |

Phase 1 complete. Foundation laid. The architectural inflection
from LD_PRELOAD-only to kernel-rooted observation is real on this
pod, and the Phase 2 plan now has a substrate to build on.
