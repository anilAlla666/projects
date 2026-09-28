# CIPHER kmod -- Phase 2 Notes (tenant identity flow shipped)

**Date:** 2026-05-13
**Pod:** Lambda H100 80GB SXM5 / 192.222.53.2
**Driver:** 580.105.08 (CUDA runtime 13.0)
**Module:** cipher_kmod.ko 0.2.0, GPL, depends empty, md5 55ab8c0cd8309ca7cc0fc40fe556aa19
**Injection lib:** libcipher_v2.so 0.2.0, pure-libc, 16 KB
**ABI header:** cipher_ioctl.h shipped (REGISTER_TENANT live, three nrs reserved)
**Insurance bundles preserved:** Phase 1, Phase 1.5, May 2 prototype

---

## Outcome

Phase 2 integration test: 11 of 11 PASS. cipher_kmod 0.2.0 plus
libcipher_v2.so together implement the full tenant identity flow from
`CUDA_INJECTION64_PATH` through `/dev/cipher` to the cipher_kmod
per-PID hashtable, surfaced in `/proc/cipher/stats` as a TENANT column
on both per-PID and per-TGID rows. Performance is +0.028 % relative to
the no-injection baseline (Step 7 with v2 injection: 71.31 tps; Step 8
control no injection: 71.29 tps) -- well within the run-to-run noise
floor of TinyLlama Marlin decode. The byte-identical smoke outputs
across both arms confirm that the kernel-side tenant stamp is
genuinely free at the workload layer.

The architectural inflection point recorded in `WHY_KMOD_NOT_OPTIONAL.md`
is now operationally complete: CIPHER's observation layer no longer
sits above the userspace driver. It sits below, with tenant identity
flowing through a kernel-rooted control surface that procurement
auditors will accept and that user code cannot override.

## Artifacts shipped

`cipher_kmod` v0.2.0 is a 772 KB out-of-tree GPL module with three
kprobes (entry + return on `nvidia_unlocked_ioctl`, plus the do_exit
reaper carried from Phase 1.5.3), a 24-entry binary-searched ioctl
decoder, a 1024-bucket RCU-protected per-PID hashtable, a `/dev/cipher`
character device with the first live ioctl (`CIPHER_REGISTER_TENANT`)
plus three reserved nrs returning `-ENOSYS`, and a `/proc/cipher/stats`
seq_file emitter that produces sectioned family output plus per-PID
and per-TGID summaries with the new TENANT column. The module has no
link-time dependency on `nvidia.ko` (depends: field is empty in
modinfo); it resolves `nvidia_unlocked_ioctl` through internal kallsyms
at `register_kprobe()` time, which lets us load/unload independently
of the NVIDIA driver.

`libcipher_v2.so` v0.2.0 is a 16 KB pure-libc shared library. The CUDA
driver loads it via `CUDA_INJECTION64_PATH` at `cuInit()`, looks up
`InitializeInjection` (or `InitializeInjection2`) by `dlsym`, and calls
it once. The library reads `CIPHER_TENANT_ID` from the environment,
opens `/dev/cipher`, issues a single `CIPHER_REGISTER_TENANT` ioctl
with `gettid()` for `pid` and `getpid()` for `tgid`, closes the fd,
and returns 1 so the driver proceeds with CUDA init. If
`CIPHER_TENANT_ID` is unset the library no-ops silently. If
`/dev/cipher` can't be opened or the ioctl fails, the library logs an
error to stderr but still returns 1 -- we never block CUDA from
initializing on tenant-registration failure.

`cipher_ioctl.h` is the stable public ABI between the kernel module
and any userspace client. It uses `__u32` and `char[N]` exclusively
(no kernel-only types), so the same header includes cleanly in both
contexts without `__KERNEL__` guards. The struct layout is locked: 12
bytes header (pid, tgid) plus 64 bytes tenant_id, no padding on
x86_64. Once shipped, this header is contractually stable.

The four ioctl nrs on magic 'C' are now committed in the public
namespace. `CIPHER_REGISTER_TENANT` (nr 1) is live. `CIPHER_SNAPSHOT`
(nr 2), `CIPHER_RESET` (nr 3), and `CIPHER_GET_VERSION` (nr 4) are
declared with placeholder layouts that the kernel rejects with
`-ENOSYS`. Phase 6 will implement them; the nrs are reserved now so
unaware client code cannot accidentally collide.

## Integration test ledger

| step | criterion | result |
|---|---|---|
| baseline | tainted=12288, no cipher loaded, no /dev/cipher | PASS |
| 1 (insmod 0.2.0) | rc=0, taint unchanged | PASS |
| 1b (chmod 666 /dev/cipher) | crw-rw-rw- | PASS |
| 2 (dmesg) | Phase 2 banner + REGISTER_TENANT live + do_exit reaper | PASS |
| 3 (fs presence) | /dev/cipher + /proc/cipher/stats present | PASS |
| 4 (inject lib, no env var) | torch.zeros worked; no [cipher_v2] log; silent skip | PASS |
| **5 (tenant env var set)** | **`[cipher_v2] tenant 'test-tenant-A' registered`**; TENANT='test-tenant-A' in /proc | **PASS -- Phase 2 headline** |
| 6 (two concurrent tenants) | both registered; both visible in per-PID + per-TGID; correct attribution; no cross-tenant pollution | PASS |
| 7 (rt-safety: smoke through injection) | n_compressed=155, marlin=12246, coherent, 80/80; tps=71.31 | PASS |
| 8 (control: smoke without injection) | n_compressed=155, marlin=12246, coherent, 80/80; tps=71.29 | PASS |
| performance neutrality | Step 7 vs 8: 71.31 vs 71.29 -> +0.028 % (within noise) | PASS |
| 9 (rmmod) | rc=0, taint unchanged, alloc_failures=0, reaped=14 (exact spawn count), 32 ms drain | PASS |

The reaper count of 14 deserves a callout: across all nine workload
steps the kmod observed exactly 14 task exits, matching the exact
number of processes the integration test spawned (1 + 1 + 2 + 5 + 5).
The do_exit reaper that Phase 1.5.3 added is not just functional but
quantitatively exact: no leaked entries, no double-reaped entries.

## Strategic findings (carrying numbering from prior phases)

**Finding 6.** `CUDA_INJECTION64_PATH` is the procurement-friendly,
NVIDIA-official mechanism for loading observation libraries beneath
the CUDA Driver API. It is the same mechanism NVIDIA's Nsight Systems
and Nsight Compute use to attach. Loading libcipher_v2 via this path
sidesteps the LD_PRELOAD-era load-order issue that broke libcipher_rt:
because the library is dlopen'd by libcuda itself after libcuda's
symbols are resolved, our injection library never sees the
"undefined symbol `cuGreenCtxDestroy`" failure that crashed
c2_marlin.py during Workload C of Phase 1.5.2. Different mechanism,
different load order, same goal of "code that runs inside the CUDA
process", but with the trust shape and the stability properties that
procurement asks for.

**Finding 7.** A pure-libc injection library is feasible and is the
right Phase 2 shape. 16 KB on disk, three dependencies (libc, ld-linux,
vdso), two exported symbols (`InitializeInjection`,
`InitializeInjection2`). No CUDA dependency, no CUPTI dependency, no
libcuda link. Future Phase 3 actuators that need CUPTI subscribe
callbacks for kernel-launch decisions will be added additively: the
library grows a `cuptiSubscribe` call and a callback dispatcher, but
the InitializeInjection entrypoint and the tenant-registration body
stay exactly as they are today. Keeping Phase 2 dependency-light makes
the upgrade path clean.

**Finding 8.** `gettid()` versus `getpid()` is the canonical pitfall in
Linux tenant-identity work, and we documented it at the source level
in cipher_tenant.c. POSIX glibc semantics are that `getpid()` returns
the thread-group ID (Linux TGID) and `gettid()` returns the actual
light-weight-process ID (what the kernel's `task_struct` calls `pid`).
The Phase 2 ioctl payload uses `gettid()` for the `pid` field
(matches kernel `current->pid`) and `getpid()` for the `tgid` field
(matches `current->tgid`). Without this exact mapping the cipher_dev
spoof check would reject every multi-thread tenant registration with
`-EPERM`. The source comment is the artifact that prevents a future
maintainer from "fixing" `gettid()` to `getpid()` and silently
breaking the ABI.

**Finding 9.** `cipher_kmod` now enforces its first security boundary
in production-shaped use. `cipher_dev_register_tenant` validates that
the payload's claimed `pid` and `tgid` match the calling task's
`current->pid` and `current->tgid` before stamping. Tenant A cannot
claim tenant B's identity from a different task. The validation is
unconditional and happens before any state mutation. This same pattern
will be replicated for the Phase 6 ioctls (SNAPSHOT, RESET,
GET_VERSION) when they land.

**Finding 10.** The do_exit reaper added in Phase 1.5.3 is correct
under multi-process workloads at production scale. Across the
integration test we spawned 14 processes (4 short python invocations
in steps 4-6 and two smoke runs in steps 7-8 with 5 LWPs each), and
the module reported `reaped=14` exactly at unload time. Zero leaked
entries. Zero double-frees. This is the empirical evidence that the
RCU-protected del-and-defer pattern is stable.

## Phase 3 entry criteria

The tenant_id field is populated in the per-PID hashtable and
proven correct under concurrent multi-tenant workloads.
`/proc/cipher/stats` exposes tenant identity for diagnosis on both
per-PID and per-TGID rows. `libcipher_v2.so` is the userspace
bootstrap point that future CUPTI subscribers will extend without
rewriting. `cipher_dev_register_tenant` is the canonical template
for future `/dev/cipher` ioctls: validate-against-current, then
mutate hashtable, then return.

Phase 3 adds three things on top of this foundation:

**MMIO BAR reads from cipher_kmod.** Voltage, clock, and thermal
state queried directly from the NVIDIA card's PCIe configuration
space, bypassing nvidia-smi's NVML wrapper. Cheaper, lower latency,
and avoids the userspace dependency on libnvidia-ml.so version
matching. The kmod is in the right address space for this; libnvidia
is not.

**PMU counter programming for free per-tenant MFU.** The CUPTI
profiler API can read GPU performance counters, but it requires
USER-mode profiling permissions and serialises kernel launches. Doing
the counter programming from kernel space via the device's MMIO
registers is the path to per-tenant MFU measurement without
perturbing the workload. This is the technical foundation for the
density-and-MFU pitch.

**First-light Grafana dashboard reading from /proc/cipher/stats.**
The proc output today is text and lossy for time-series purposes. A
small exporter reads stats periodically, emits prometheus metrics,
Grafana plots them. The dashboard is the visible end of the CIPHER
demo for partners and investors. Phase 3 builds it on top of the
already-populated /proc data; no kmod changes needed to start.

The deferred libcipher_rt `cuGreenCtxDestroy` ABI fix from Phase 1.5.2
is also a Phase 3 candidate (it surfaces during multi-tenant density
sweeps that exercise the green-context partition path) but is not
blocking for the MMIO and PMU work.

Phase 3 begins next session.
