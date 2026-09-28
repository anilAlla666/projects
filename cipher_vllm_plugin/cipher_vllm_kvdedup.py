"""CIPHER Week 5 Step 2 — vLLM cross-tenant KV-dedup plugin.

Registered under the ``vllm.general_plugins`` entry-point group, runs in the
EngineCore subprocess (same lifecycle as cipher_vllm_kv / cipher_kv_offload).

What it does: walks the CIPHER-VMM-backed KV pages allocated by CP 5.1's
``cipher_vllm_kv`` (which monkey-patches ``GPUModelRunner._allocate_kv_cache_tensors``
to source from ``cipher_kv_bridge.vmm_zeros``), and on operator demand
calls ``cipher_rt_kv_dedup_alias`` on each 2 MiB page. The substrate
primitive (added in Week 5 Step 1b — commit ``ec0e005``) does an in-place
``cuMemUnmap`` + ``cuMemRelease`` + ``cuMemMap`` swap on the caller's VA
so cross-tenant same-content pages share one physical page in HBM
without the caller's tensor data_ptr changing.

Trigger model — **explicit flush** (per WEEK_5_STEP_1_DESIGN_MEMO.md Part A.5b):
- Operator (test harness in W5 Step 3; production wiring in W6+) calls
  ``cipher_vllm_kvdedup.dedup_now()`` at a quiescent point (post-prefill
  before decode, per Step 1 memo Part E.4).
- The cold-path latency budget (T4.6.4 memo §Call model: ~400 µs/HIT) makes
  per-decode-step trigger unacceptable; explicit-flush is the W5
  call-model choice.

Ordering — **MUST register AFTER cipher_vllm_kv** (per R-W5.3 in scope-lock).
setup.py lists this entry point after ``cipher_vllm_kv``; vLLM's
``load_general_plugins`` iterates entry-point discovery in dict insertion
order (Python 3.7+), so the order is preserved. The wrapper installed here
wraps the already-CP-5.1-wrapped ``_allocate_kv_cache_tensors``; CP 5.1's
patch returns CIPHER-VMM-backed tensors, then this wrapper records each
returned tensor's VMM range for later ``dedup_now()`` flush.

Env gates:
  CIPHER_KVDEDUP        1|0   (default 0)  — opt-in master enable
  CIPHER_TENANT_NUM     int   (default 0)  — tenant id (shared convention
                                             with cipher_vllm_kv)
  CIPHER_RT_DIR         path  (default /home/ubuntu/cipher_rt_phase4) — holds
                                             cipher_kv_bridge.so

Test entry point: ``dedup_now()`` returns a dict with pages_processed +
pre/post STATS counters (puts/hits/misses) for the test harness to assert.
"""
import ctypes
import json
import os
import signal
import sys
import threading
import time

_PREFIX = "[cipher-vllm-kvdedup]"
_PAGE_SIZE = 2 * 1024 * 1024  # 2 MiB — kmod kvdedup ABI is fixed at this granularity


def _log(msg):
    print(f"{_PREFIX} {msg}", file=sys.stderr, flush=True)


def _enabled():
    return os.environ.get("CIPHER_KVDEDUP", "0") not in ("0", "off", "no", "")


_RT_DIR = os.environ.get("CIPHER_RT_DIR", "/home/ubuntu/cipher_rt_phase4")
_TENANT = int(os.environ.get("CIPHER_TENANT_NUM", "0"))

# ----- ctypes binding to cipher_kv_bridge.so kv_dedup C surface ---------
# cipher_kv_bridge.so is loaded into the Python process by cipher_vllm_kv's
# `import cipher_kv_bridge` (via sys.path[+/=_RT_DIR]). We re-open it with
# ctypes.CDLL to bind the plain-C symbols that are not exposed via pybind11.
if _RT_DIR not in sys.path:
    sys.path.insert(0, _RT_DIR)

_LIB = None  # lazy-loaded in _ensure_lib()


class _DedupStats(ctypes.Structure):
    _fields_ = [
        ("puts",              ctypes.c_uint64),
        ("hits",              ctypes.c_uint64),
        ("misses",            ctypes.c_uint64),
        ("physical_pages",    ctypes.c_uint64),
        ("virtual_pages",     ctypes.c_uint64),
        ("hash_collisions",   ctypes.c_uint64),
        ("refcount_releases", ctypes.c_uint64),
    ]


def _ensure_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    import cipher_kv_bridge  # noqa: F401 — loads the .so into the address space
    so_path = os.path.join(_RT_DIR, "cipher_kv_bridge.so")
    _LIB = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    _LIB.cipher_rt_kv_dedup_init.restype  = ctypes.c_int
    _LIB.cipher_rt_kv_dedup_init.argtypes = []
    _LIB.cipher_rt_kv_dedup_alias.restype  = ctypes.c_int
    _LIB.cipher_rt_kv_dedup_alias.argtypes = [ctypes.c_uint64,
                                              ctypes.POINTER(ctypes.c_int)]
    _LIB.cipher_rt_kv_dedup_get_stats.restype  = None
    _LIB.cipher_rt_kv_dedup_get_stats.argtypes = [ctypes.POINTER(_DedupStats)]
    return _LIB


# ----- per-runner-instance page tracking --------------------------------
# Keyed by GPUModelRunner instance id (id() of the Python object). Each
# entry is a list of (devptr, size_bytes, alloc_time_monotonic) tuples
# captured at the wrap site below. dedup_now() iterates them.
_runner_pages: dict = {}


def _read_stats():
    s = _DedupStats()
    _ensure_lib().cipher_rt_kv_dedup_get_stats(ctypes.byref(s))
    return s


def dedup_now():
    """Walk all tracked KV pages and call cipher_rt_kv_dedup_alias on
    each 2 MiB page within them. Returns a dict with pre/post STATS
    deltas and per-call outcomes (hits vs misses on THIS flush).

    Called by the test harness at a quiescent point (post-prefill,
    before decode). Not on the hot path."""
    lib = _ensure_lib()
    pre = _read_stats()
    pages_processed = 0
    hits_on_flush = 0
    misses_on_flush = 0
    rebind_errors = 0

    # Iterate over all tracked runners' tracked tensors.
    for runner_id, entries in list(_runner_pages.items()):
        for devptr_base, size_bytes, _t in entries:
            n_pages = size_bytes // _PAGE_SIZE
            for i in range(n_pages):
                page_devptr = devptr_base + i * _PAGE_SIZE
                was_deduped = ctypes.c_int(0)
                rc = lib.cipher_rt_kv_dedup_alias(
                    ctypes.c_uint64(page_devptr),
                    ctypes.byref(was_deduped))
                if rc != 0:
                    rebind_errors += 1
                    if rebind_errors <= 3:
                        _log(f"alias failed at va=0x{page_devptr:x} rc={rc}")
                    continue
                pages_processed += 1
                if was_deduped.value:
                    hits_on_flush += 1
                else:
                    misses_on_flush += 1

    post = _read_stats()
    return {
        "pages_processed":     pages_processed,
        "hits_on_flush":       hits_on_flush,
        "misses_on_flush":     misses_on_flush,
        "rebind_errors":       rebind_errors,
        "pre_puts":            pre.puts,
        "pre_hits":            pre.hits,
        "pre_misses":          pre.misses,
        "post_puts":           post.puts,
        "post_hits":           post.hits,
        "post_misses":         post.misses,
        "puts_delta":          post.puts - pre.puts,
        "hits_delta":          post.hits - pre.hits,
        "misses_delta":        post.misses - pre.misses,
        "post_physical_pages": post.physical_pages,
        "post_virtual_pages":  post.virtual_pages,
    }


_dedup_initialized = False


def _ensure_dedup_init():
    """Lazy dedup_init. cipher_rt_kv_dedup_init requires
    cipher_rt_kv_alloc_init to have been called first; the latter is
    called by CP 5.1's _ensure_bridge_init() at the first vmm_zeros
    invocation. So we defer dedup_init until just AFTER CP 5.1's
    wrapper returns (when alloc is guaranteed to be initialized).

    Runs in the EngineCore subprocess (the wrapper that calls us
    executes there). This is the right place to install the
    auto-flush threads — Python threads do NOT survive fork, so the
    threads spawned from register() (driver process) are dead in the
    subprocess where _runner_pages actually lives. Installing here
    after dedup_init succeeds puts them in the right process."""
    global _dedup_initialized
    if _dedup_initialized:
        return True
    lib = _ensure_lib()
    rc = lib.cipher_rt_kv_dedup_init()
    if rc != 0:
        _log(f"cipher_rt_kv_dedup_init failed rc={rc} "
             "(alloc still not initialized? CP 5.1 may have been disabled)")
        return False
    _dedup_initialized = True
    _log("cipher_rt_kv_dedup_init: ok (lazy-init after CP 5.1 bridge alloc)")
    _install_auto_flush_threads()
    return True


def _make_kvdedup_wrapper(cp51_wrapped):
    """Wrap CP 5.1's _allocate_kv_cache_tensors patch. CP 5.1's wrapper
    returns CIPHER-VMM-backed tensors; we record each tensor's
    (devptr, size_bytes) for later dedup_now() flush. We DO NOT call
    cipher_rt_kv_dedup_alias here — at allocation time the pages are
    all zeros, which would degenerately dedupe to a single zero-page
    across tenants. Real dedup opportunity is post-prefill when shared
    system-prompt KV content has been written; that's what dedup_now()
    targets.

    Lazy-inits the dedup substrate at first call (after CP 5.1's
    bridge alloc has run)."""
    def kvdedup_wrapped(self, kv_cache_config):
        tensors = cp51_wrapped(self, kv_cache_config)
        # tensors is dict[str, torch.Tensor]. Each tensor is CIPHER-VMM
        # backed (raw int8 from cipher_kv_bridge.vmm_zeros).
        if not _ensure_dedup_init():
            # dedup substrate unavailable; skip tracking, return tensors
            # untouched (CP 5.1's path is unaffected).
            return tensors
        runner_id = id(self)
        if runner_id not in _runner_pages:
            _runner_pages[runner_id] = []
        seen = set()  # dedupe alias-tensors (CP 5.1 aliases shared_by layers)
        for layer_name, t in tensors.items():
            devptr = t.data_ptr()
            if devptr in seen:
                continue
            seen.add(devptr)
            size_bytes = t.numel() * t.element_size()
            _runner_pages[runner_id].append((devptr, size_bytes, time.monotonic()))
        _log(f"kvdedup tracked {len(seen)} tensor(s) "
             f"({sum(s for _, s, _ in _runner_pages[runner_id]) / (1 << 20):.0f} MiB) "
             f"on runner id={runner_id}")
        # Write the PID file *from the wrapper* — vLLM v1 runs us in an
        # EngineCore subprocess where _runner_pages actually lives. The
        # outer driver process also registers (and writes a stale PID),
        # so this re-write claims the file for the process that does
        # the real work. Test harness reads this file to know who to
        # send SIGUSR1 to.
        try:
            pid_file = f"/tmp/cipher_kvdedup_pid_t{_TENANT}.txt"
            with open(pid_file, "w") as f:
                f.write(f"{os.getpid()}\n")
            _log(f"claimed pid_file {pid_file} (this is the tracking process)")
        except OSError as e:
            _log(f"could not write pid_file: {e}")
        return tensors
    return kvdedup_wrapped


# ----- Week 6 auto-flush (time-based + pressure-based) ------------------
# Both modes are OFF by default so the Week 5 SIGUSR1-orchestrated
# regression harnesses (sc_kvdedup_n2.py, sc_kvdedup_n4_*, etc.) keep
# their flush-order invariants (assertions like t2_hits > t1_hits require
# the orchestrator to flush t1 first). B1/B2 unattended runs opt in via
# the env vars below.
#
# Env vars:
#   CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC          float (default 0 = off)
#     time-based flush every N seconds.
#   CIPHER_KVDEDUP_PRESSURE_THRESHOLD          float (default 0 = off)
#     pressure ratio = substrate.physical_pages / MAX_PHYSICAL_PAGES;
#     fires when ratio > threshold AND physical_pages changed since last
#     poll (which in single-tenant only happens via our own flushes, so
#     this mode is primarily useful for multi-tenant where other tenants'
#     flushes grow the shared kmod table).
#   CIPHER_KVDEDUP_PRESSURE_MAX_PHYSICAL_PAGES int   (default 8192)
#     denominator for the pressure ratio. 8192 pages × 2 MiB = 16 GiB.
#
# Deviation from W6 spec: spec referenced stats.pages_resident /
# stats.max_pages; the substrate's _DedupStats exposes physical_pages /
# virtual_pages / puts / hits / misses / hash_collisions /
# refcount_releases. physical_pages is the closest analog to
# "pages_resident", and max_physical_pages is configured by env.

_flush_lock = threading.Lock()
_last_flush_monotonic = 0.0
_FLUSH_DEBOUNCE_SEC = 1.0
_auto_flush_threads_installed = False


def _try_auto_flush(reason):
    """Debounced trigger of dedup_now() via SIGUSR1 to self.

    Why SIGUSR1 instead of calling dedup_now() directly: the
    cipher_rt_kv_dedup_alias path does a cuMemcpyDtoH which requires
    a current CUDA context on the calling thread. Python threads
    spawned in the EngineCore subprocess do NOT inherit the main
    thread's CUDA context; calling alias from them fails with
    `cuMemcpyDtoH -> 201 invalid device context`. Routing via
    SIGUSR1 hands the work to the main thread (Python delivers
    signals on the main thread by default), which holds the
    context vLLM/PyTorch established.

    Returns True if SIGUSR1 was sent (the actual flush completes
    asynchronously when the main thread reaches a bytecode boundary),
    False if debounced or substrate not initialized."""
    global _last_flush_monotonic
    if not _dedup_initialized:
        return False
    with _flush_lock:
        now = time.monotonic()
        if now - _last_flush_monotonic < _FLUSH_DEBOUNCE_SEC:
            return False
        _last_flush_monotonic = now
    try:
        _log(f"auto-flush ({reason}): SIGUSR1 -> main thread")
        os.kill(os.getpid(), signal.SIGUSR1)
        return True
    except Exception as ex:
        _log(f"auto-flush ({reason}) error: {ex}")
        return False


def _time_flush_loop(interval_sec):
    while True:
        time.sleep(interval_sec)
        _try_auto_flush(f"time/{interval_sec:.1f}s")


def _pressure_flush_loop(threshold, max_physical):
    poll_sec = 1.0
    last_physical = -1
    while True:
        time.sleep(poll_sec)
        if not _dedup_initialized:
            continue
        try:
            s = _read_stats()
        except Exception as ex:
            _log(f"pressure poll error: {ex}")
            continue
        pressure = s.physical_pages / max(max_physical, 1)
        if pressure > threshold and s.physical_pages != last_physical:
            _try_auto_flush(
                f"pressure/{pressure:.2f} "
                f"phys={s.physical_pages}/{max_physical}")
        last_physical = s.physical_pages


def _install_auto_flush_threads():
    global _auto_flush_threads_installed
    if _auto_flush_threads_installed:
        return  # idempotent — _ensure_dedup_init may run more than once
    _auto_flush_threads_installed = True

    flush_interval = float(os.environ.get(
        "CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC", "0"))
    pressure_threshold = float(os.environ.get(
        "CIPHER_KVDEDUP_PRESSURE_THRESHOLD", "0"))
    pressure_max = int(os.environ.get(
        "CIPHER_KVDEDUP_PRESSURE_MAX_PHYSICAL_PAGES", "8192"))

    if flush_interval > 0:
        threading.Thread(
            target=_time_flush_loop,
            args=(flush_interval,),
            name="cipher-kvdedup-time-flush",
            daemon=True,
        ).start()
        _log(f"time-based auto-flush installed: every {flush_interval}s")
    else:
        _log("time-based auto-flush disabled (CIPHER_KVDEDUP_FLUSH_INTERVAL_SEC=0)")

    if pressure_threshold > 0:
        threading.Thread(
            target=_pressure_flush_loop,
            args=(pressure_threshold, pressure_max),
            name="cipher-kvdedup-pressure-flush",
            daemon=True,
        ).start()
        _log(f"pressure auto-flush installed: threshold={pressure_threshold} "
             f"max_physical_pages={pressure_max}")
    else:
        _log("pressure auto-flush disabled (CIPHER_KVDEDUP_PRESSURE_THRESHOLD=0)")


def register():
    """vLLM general-plugin entry point. Installs the kvdedup wrap atop
    CP 5.1's already-installed _allocate_kv_cache_tensors patch. Per
    R-W5.3 + setup.py ordering, this register fires AFTER
    cipher_vllm_kv:register, so the current attribute on GPUModelRunner
    is already CP 5.1's wrapper."""
    if not _enabled():
        _log("disabled via CIPHER_KVDEDUP (default off) — kvdedup wrapper "
             "not installed; CP 5.1 / CP 5.2 unaffected")
        return

    try:
        _ensure_lib()
    except Exception as e:  # pragma: no cover
        _log(f"could not load cipher_kv_bridge.so ({e}); kvdedup NOT installed")
        return

    # Substrate dedup_init is DEFERRED to first wrapper call (see
    # _ensure_dedup_init). At register-time CP 5.1's alloc may not yet
    # be initialized — the alloc is initialized lazily by CP 5.1's own
    # _ensure_bridge_init() which fires inside CP 5.1's wrapper at the
    # first _allocate_kv_cache_tensors call.

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover
        _log(f"could not import GPUModelRunner ({e}); kvdedup NOT installed")
        return

    current = GPUModelRunner._allocate_kv_cache_tensors
    if getattr(current, "_cipher_kvdedup_hooked", False):
        _log("kvdedup hook already installed (idempotent)")
        return  # idempotent — load_general_plugins may run more than once

    wrapped = _make_kvdedup_wrapper(current)
    wrapped._cipher_kvdedup_hooked = True
    wrapped._cipher_kvdedup_orig = current
    # Preserve CP 5.1's existing marker so a second cipher_vllm_kv:register
    # call still detects the hook chain is in place.
    if getattr(current, "_cipher_hooked", False):
        wrapped._cipher_hooked = True
        wrapped._cipher_orig = getattr(current, "_cipher_orig", current)
    GPUModelRunner._allocate_kv_cache_tensors = wrapped
    _log(f"kvdedup hook installed atop CP 5.1 (tenant={_TENANT}, "
         f"page={_PAGE_SIZE} B); explicit-flush trigger via dedup_now()")

    # ---- SIGUSR1 IPC: external trigger for dedup_now() ----------------
    # vLLM v1 runs the wrapper in a separate EngineCore subprocess. The
    # _runner_pages dict lives there, so dedup_now() must be invoked
    # from inside that process. We install a SIGUSR1 handler that calls
    # dedup_now() and writes the JSON result to a tenant-scoped file
    # that the test harness reads.
    pid_file  = f"/tmp/cipher_kvdedup_pid_t{_TENANT}.txt"
    result_file = f"/tmp/cipher_kvdedup_result_t{_TENANT}.json"
    try:
        with open(pid_file, "w") as f:
            f.write(f"{os.getpid()}\n")
    except OSError as e:
        _log(f"could not write {pid_file}: {e}")

    def _sigusr1_handler(signum, frame):
        try:
            r = dedup_now()
            with open(result_file, "w") as f:
                json.dump(r, f)
            _log(f"SIGUSR1: dedup_now() -> {result_file} "
                 f"(pages={r['pages_processed']} hits={r['hits_on_flush']} "
                 f"misses={r['misses_on_flush']} hits_delta={r['hits_delta']})")
        except Exception as ex:
            _log(f"SIGUSR1 handler error: {ex}")

    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        _log(f"SIGUSR1 trigger installed; pid={os.getpid()} -> {pid_file}, "
             f"result -> {result_file}")
    except Exception as e:
        _log(f"could not install SIGUSR1 handler: {e}")
    # NOTE: auto-flush threads are NOT installed here. They are installed
    # in _ensure_dedup_init() which runs in the EngineCore subprocess where
    # _runner_pages and _dedup_initialized actually live. Threads do not
    # survive fork; installing here would spawn dead threads in the driver.
