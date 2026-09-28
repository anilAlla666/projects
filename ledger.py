"""
cipher_runtime.ledger — DEP.2.2 Substitution Ledger
Neural Dynamics, Inc.

Per-kernel audit trail for every substitution CIPHER makes.
Each entry records: kernel identity, layer, recipe used,
predicted TFLOPS saved, timestamp.

This is the audit trail the customer verifies independently.
Stored as newline-delimited JSON (NDJSON) for streaming writes.

USAGE:
    from cipher_runtime.ledger import CipherLedger

    ledger = CipherLedger("cipher_run.ndjson")
    ledger.record(
        kernel="cublas_gemm_4096x4096",
        layer=5,
        recipe="gemm_roofline",
        tflops_saved=12.4,
        latency_ns=1797,
    )
    summary = ledger.close()
    summary.print()
"""

import os
import json
import time
import threading
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


# ── Ledger entry ──────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """One substitution event. Written immediately to disk."""
    seq:           int    # monotonic sequence number
    ts:            float  # Unix timestamp
    kernel:        str    # kernel name / hash
    layer:         int    # transformer layer index
    recipe:        str    # recipe used (gemm_roofline, chebyshev, edmd, ...)
    op_class:      str    # GEMM / ATTN / EW / REDUCE / CONV
    grid_x:        int    # kernel grid dims
    grid_y:        int
    grid_z:        int
    block_size:    int
    shmem_bytes:   int
    latency_ns:    int    # forward pass latency
    tflops_saved:  float  # estimated TFLOPS recovered
    confidence:    float  # LNN confidence score
    error_bound:   float  # Barron error bound (from oracle)
    phase:         str    # warmup / convergence / finetune


@dataclass
class LedgerSummary:
    """Aggregate summary across all ledger entries."""
    run_id:              str
    device_name:         str
    start_time:          str
    end_time:            str
    total_entries:       int
    total_tflops_saved:  float
    avg_tflops_per_sub:  float
    avg_confidence:      float
    avg_latency_ns:      float
    p50_latency_ns:      float
    p99_latency_ns:      float
    by_recipe:           Dict[str, int]   # recipe → count
    by_op_class:         Dict[str, int]   # op_class → count
    by_layer:            Dict[str, int]   # layer → count
    error_bound_max:     float
    error_bound_avg:     float

    def print(self):
        print("\n" + "=" * 60)
        print("  CIPHER DEP.2.2 — Substitution Ledger Summary")
        print("=" * 60)
        print(f"\n  Run:           {self.run_id}")
        print(f"  Device:        {self.device_name}")
        print(f"  Period:        {self.start_time} → {self.end_time}")
        print(f"\n  Total substitutions:  {self.total_entries:,}")
        print(f"  TFLOPS saved total:   {self.total_tflops_saved:.2f}")
        print(f"  Avg TFLOPS per sub:   {self.avg_tflops_per_sub:.3f}")
        print(f"  Avg confidence:       {self.avg_confidence:.3f}")
        print(f"  Avg latency:          {self.avg_latency_ns:.0f} ns")
        print(f"  P50/P99 latency:      {self.p50_latency_ns:.0f} / {self.p99_latency_ns:.0f} ns")
        print(f"  Max error bound:      {self.error_bound_max:.4f}")
        print(f"\n  By recipe:")
        for r, c in sorted(self.by_recipe.items(), key=lambda x: -x[1]):
            pct = c / max(1, self.total_entries) * 100
            print(f"    {r:<20} {c:>6,}  ({pct:.1f}%)")
        print(f"\n  By op class:")
        for op, c in sorted(self.by_op_class.items(), key=lambda x: -x[1]):
            pct = c / max(1, self.total_entries) * 100
            print(f"    {op:<20} {c:>6,}  ({pct:.1f}%)")
        print("=" * 60)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ── Ledger ────────────────────────────────────────────────────

class CipherLedger:
    """
    Streaming audit ledger for CIPHER substitutions.

    Writes one JSON line per substitution to an NDJSON file.
    Never holds entries in memory — safe for long training runs.
    Thread-safe via lock.
    """

    def __init__(self, path: str,
                 run_id: Optional[str] = None,
                 device_name: str = "unknown",
                 append: bool = False):
        self._path        = path
        self._run_id      = run_id or f"cipher_{int(time.time())}"
        self._device_name = device_name
        self._lock        = threading.Lock()
        self._seq         = 0
        self._start_time  = time.time()

        # In-memory stats only (not the entries themselves)
        self._total_tflops = 0.0
        self._confidences: List[float] = []
        self._latencies:   List[int]   = []
        self._error_bounds: List[float] = []
        self._by_recipe:   Dict[str, int] = {}
        self._by_op_class: Dict[str, int] = {}
        self._by_layer:    Dict[str, int] = {}

        mode = "a" if append else "w"
        self._file = open(path, mode, buffering=1)  # line-buffered

        # Write header comment
        header = {
            "_type":       "cipher_ledger_header",
            "run_id":      self._run_id,
            "device":      device_name,
            "start_time":  datetime.datetime.utcnow().isoformat() + "Z",
            "format":      "ndjson",
            "version":     "1.0",
        }
        self._file.write(json.dumps(header) + "\n")
        self._file.flush()

        print(f"[CIPHER DEP.2.2] Ledger opened: {path}")

    def record(self,
               kernel:       str   = "",
               layer:        int   = 0,
               recipe:       str   = "unknown",
               op_class:     str   = "GEMM",
               grid_x:       int   = 0,
               grid_y:       int   = 0,
               grid_z:       int   = 1,
               block_size:   int   = 256,
               shmem_bytes:  int   = 0,
               latency_ns:   int   = 0,
               tflops_saved: float = 0.0,
               confidence:   float = 1.0,
               error_bound:  float = 0.0,
               phase:        str   = "convergence") -> int:
        """
        Record one substitution event. Thread-safe.
        Returns the sequence number.
        """
        with self._lock:
            seq = self._seq
            self._seq += 1

            entry = LedgerEntry(
                seq=seq, ts=time.time(),
                kernel=kernel, layer=layer,
                recipe=recipe, op_class=op_class,
                grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
                block_size=block_size, shmem_bytes=shmem_bytes,
                latency_ns=latency_ns, tflops_saved=tflops_saved,
                confidence=confidence, error_bound=error_bound,
                phase=phase,
            )

            # Write immediately
            self._file.write(json.dumps(asdict(entry)) + "\n")

            # Update in-memory stats
            self._total_tflops += tflops_saved
            self._confidences.append(confidence)
            self._latencies.append(latency_ns)
            self._error_bounds.append(error_bound)
            self._by_recipe[recipe]   = self._by_recipe.get(recipe, 0) + 1
            self._by_op_class[op_class] = self._by_op_class.get(op_class,0)+1
            layer_key = str(layer)
            self._by_layer[layer_key] = self._by_layer.get(layer_key, 0) + 1

        return seq

    def flush(self):
        """Force flush to disk."""
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> LedgerSummary:
        """Close the ledger and return a summary."""
        end_time = time.time()

        with self._lock:
            n = self._seq
            # Compute latency percentiles
            lats = sorted(self._latencies)
            p50  = lats[int(len(lats) * 0.50)] if lats else 0
            p99  = lats[int(len(lats) * 0.99)] if lats else 0

            summary = LedgerSummary(
                run_id              = self._run_id,
                device_name         = self._device_name,
                start_time          = datetime.datetime.utcfromtimestamp(
                                        self._start_time).isoformat() + "Z",
                end_time            = datetime.datetime.utcfromtimestamp(
                                        end_time).isoformat() + "Z",
                total_entries       = n,
                total_tflops_saved  = self._total_tflops,
                avg_tflops_per_sub  = self._total_tflops / max(1, n),
                avg_confidence      = (sum(self._confidences) /
                                       max(1, len(self._confidences))),
                avg_latency_ns      = (sum(self._latencies) /
                                       max(1, len(self._latencies))),
                p50_latency_ns      = p50,
                p99_latency_ns      = p99,
                by_recipe           = dict(self._by_recipe),
                by_op_class         = dict(self._by_op_class),
                by_layer            = dict(self._by_layer),
                error_bound_max     = max(self._error_bounds, default=0.0),
                error_bound_avg     = (sum(self._error_bounds) /
                                       max(1, len(self._error_bounds))),
            )

            # Write footer
            footer = {
                "_type":   "cipher_ledger_footer",
                "entries": n,
                "end":     datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._file.write(json.dumps(footer) + "\n")
            self._file.flush()
            self._file.close()

        print(f"[CIPHER DEP.2.2] Ledger closed: {self._path}  ({n:,} entries)")
        return summary

    @property
    def entry_count(self) -> int:
        with self._lock:
            return self._seq

    @property
    def total_tflops_saved(self) -> float:
        with self._lock:
            return self._total_tflops

    @staticmethod
    def replay(path: str):
        """
        Replay a ledger file for audit.
        Yields (header, entries, footer) in order.
        """
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except json.JSONDecodeError:
                    continue
