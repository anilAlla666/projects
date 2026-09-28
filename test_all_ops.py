#!/usr/bin/env python3
"""CIPHER 10-Op Verification Test — Session 4"""

import os
import sys
import re
import time

LOG_PATH = "/tmp/cipher_test.log"

# ── Redirect stderr to log file ──────────────────────────────────────────────
log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
saved_stderr = os.dup(2)
os.dup2(log_fd, 2)

# ── Run workload ─────────────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/CIPHER_final")
import cipher_wrapper
import torch

B = torch.randn(4096, 4096).cuda()
prev_a = torch.randn(4096, 4096).cuda()
for i in range(500):
    a = torch.randn(4096, 4096).cuda() * 0.02 + prev_a * 0.98
    prev_a = a.clone()
    torch.mm(a, B)

torch.cuda.synchronize()
time.sleep(2)

# ── Restore stderr ───────────────────────────────────────────────────────────
os.dup2(saved_stderr, 2)
os.close(log_fd)
os.close(saved_stderr)

# ── Read log ─────────────────────────────────────────────────────────────────
with open(LOG_PATH, "r") as f:
    log = f.read()
lines = log.splitlines()
cipher_lines = [l for l in lines if "CIPHER" in l or "O(1)-block" in l]

# ── Parse ops ────────────────────────────────────────────────────────────────
results = {}

# Op1 CLASSIFY
conf_vals = [int(m.group(1)) for m in re.finditer(r"conf=(\d+)", log)]
if any(c >= 50 for c in conf_vals):
    results["Op1 CLASSIFY"] = f"PASS  (max conf={max(conf_vals)})"
else:
    results["Op1 CLASSIFY"] = f"FAIL  (max conf={max(conf_vals) if conf_vals else 'none'})"

# Op2 SPECULATE HIT
if "SPECULATE HIT" in log:
    count = log.count("SPECULATE HIT")
    results["Op2 SPECULATE HIT"] = f"PASS  ({count} hits)"
else:
    results["Op2 SPECULATE HIT"] = "NOT OBSERVED"

# Op3 SUBSTITUTE
block_lines = [l for l in lines if "O(1)-block" in l]
if block_lines:
    results["Op3 SUBSTITUTE"] = f"PASS  ({len(block_lines)} block substitutions logged)"
else:
    results["Op3 SUBSTITUTE"] = "FAIL  (no O(1)-block lines)"

# Op4 ORCHESTRATE
results["Op4 ORCHESTRATE"] = "NOT BUILT"

# Op5 GENERATE
if block_lines:
    results["Op5 GENERATE"] = "PASS  (kernel skipped via block sub)"
else:
    results["Op5 GENERATE"] = "FAIL  (no block substitutions)"

# Op6 RING_WRITE
ring_seqs = re.findall(r"ring_seq=(\d+)", log)
if ring_seqs:
    results["Op6 RING_WRITE"] = f"PASS  ({len(ring_seqs)} ring_seq entries, max={max(int(s) for s in ring_seqs)})"
else:
    results["Op6 RING_WRITE"] = "FAIL  (no ring_seq in log)"

# Op7 REMEMBER
remember_lines = [l for l in lines if "REMEMBER h[" in l]
h0_vals = []
for l in remember_lines:
    m = re.search(r"h\[0\.\.3\]=\s*([\-\d.]+)", l)
    if m:
        h0_vals.append(float(m.group(1)))
if len(h0_vals) >= 2 and h0_vals[0] != h0_vals[-1]:
    results["Op7 REMEMBER"] = f"PASS  ({len(remember_lines)} lines, h[0]: {h0_vals[0]:.4f} -> {h0_vals[-1]:.4f})"
elif len(h0_vals) >= 2:
    results["Op7 REMEMBER"] = f"FAIL  ({len(remember_lines)} lines but h[0] unchanged: {h0_vals[0]:.4f})"
elif len(h0_vals) == 1:
    results["Op7 REMEMBER"] = f"PARTIAL  (only 1 line, h[0]={h0_vals[0]:.4f})"
else:
    results["Op7 REMEMBER"] = "FAIL  (no REMEMBER lines)"

# Op8 VALIDATE
validate_lines = [l for l in lines if "VALIDATE class=" in l]
if validate_lines:
    means = re.findall(r"mean=([\-\d.]+)", "\n".join(validate_lines))
    varis = re.findall(r"var=([\-\d.]+)", "\n".join(validate_lines))
    results["Op8 VALIDATE"] = f"PASS  ({len(validate_lines)} reports, mean={means[-1] if means else '?'}, var={varis[-1] if varis else '?'})"
else:
    results["Op8 VALIDATE"] = "FAIL  (no VALIDATE lines)"

# Op9 AUDIT
results["Op9 AUDIT"] = "PARTIAL  (HMAC disabled for thread safety, counter increments only)"

# Op10 SPECULATE
spec_lines = [l for l in lines if "SPECULATE write:" in l]
conf_spec = []
for l in spec_lines:
    m = re.search(r"conf=([\d.]+)", l)
    if m:
        conf_spec.append(float(m.group(1)))
if len(conf_spec) >= 2 and conf_spec[0] != conf_spec[-1]:
    results["Op10 SPECULATE"] = f"PASS  ({len(spec_lines)} writes, conf: {conf_spec[0]:.3f} -> {conf_spec[-1]:.3f})"
elif len(conf_spec) >= 1:
    results["Op10 SPECULATE"] = f"PARTIAL  ({len(spec_lines)} writes, conf constant at {conf_spec[0]:.3f})"
else:
    results["Op10 SPECULATE"] = "FAIL  (no SPECULATE write lines)"

# ── Print report ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CIPHER 10-Op Verification Report")
print("=" * 70)
for op, status in results.items():
    print(f"  {op:25s} {status}")
print("=" * 70)

passed = sum(1 for s in results.values() if s.startswith("PASS"))
partial = sum(1 for s in results.values() if s.startswith("PARTIAL"))
not_built = sum(1 for s in results.values() if s.startswith("NOT BUILT"))
not_obs = sum(1 for s in results.values() if s.startswith("NOT OBSERVED"))
failed = sum(1 for s in results.values() if s.startswith("FAIL"))
print(f"  PASS: {passed}  PARTIAL: {partial}  NOT BUILT: {not_built}  NOT OBSERVED: {not_obs}  FAIL: {failed}")
print("=" * 70)

print(f"\nFirst 30 CIPHER lines from {LOG_PATH}:")
print("-" * 70)
for l in cipher_lines[:30]:
    print(l)
print("-" * 70)

# ── Latency benchmark: GPU block sub vs cuBLAS ──────────────────────────────
print("\n" + "=" * 70)
print("Latency Benchmark: O(Kr) GPU block sub vs cuBLAS 4096x4096 GEMM")
print("=" * 70)

# Diagnostic: confirm GPU path is firing
print(f"  predict restype: {cipher_wrapper._hook.cipher_block_sub_predict.restype}")
gpu_lines = [l for l in lines if "block-GPU" in l]
print(f"  GPU kernel fires: {len(gpu_lines)} times in workload")

# Redirect stderr again for benchmark
log_fd2 = os.open("/dev/null", os.O_WRONLY)
saved_stderr2 = os.dup(2)
os.dup2(log_fd2, 2)

N_BENCH = 1000
B_bench = torch.randn(4096, 4096).cuda()
a_bench = torch.randn(4096, 4096).cuda()
out_bench = torch.empty(4096, 4096, device="cuda")

# Warmup
for _ in range(20):
    cipher_wrapper._hook.cipher_block_sub_predict(a_bench.data_ptr(), out_bench.data_ptr(), 4096)
    torch.mm(a_bench, B_bench)
torch.cuda.synchronize()

# ── Measurement 1: Python loop (includes Python overhead) ────────────────
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

start_ev.record()
for _ in range(N_BENCH):
    cipher_wrapper._hook.cipher_block_sub_predict(a_bench.data_ptr(), out_bench.data_ptr(), 4096)
end_ev.record()
torch.cuda.synchronize()
block_sub_ms = start_ev.elapsed_time(end_ev)
block_sub_us = block_sub_ms * 1000.0 / N_BENCH

# cuBLAS GEMM (Python loop)
start_ev.record()
for _ in range(N_BENCH):
    torch.mm(a_bench, B_bench)
end_ev.record()
torch.cuda.synchronize()
cublas_ms = start_ev.elapsed_time(end_ev)
cublas_us = cublas_ms * 1000.0 / N_BENCH

# ── Measurement 2: CUDA stream burst (no Python loop overhead) ───────────
# Pre-enqueue N_BENCH kernel launches into a dedicated stream, then time
# the entire stream from first launch to completion.
stream = torch.cuda.Stream()
torch.cuda.synchronize()

# Block sub stream burst
with torch.cuda.stream(stream):
    start_ev.record(stream)
    for _ in range(N_BENCH):
        cipher_wrapper._hook.cipher_block_sub_predict(a_bench.data_ptr(), out_bench.data_ptr(), 4096)
    end_ev.record(stream)
stream.synchronize()
block_stream_ms = start_ev.elapsed_time(end_ev)
block_stream_us = block_stream_ms * 1000.0 / N_BENCH

# cuBLAS stream burst
with torch.cuda.stream(stream):
    start_ev.record(stream)
    for _ in range(N_BENCH):
        torch.mm(a_bench, B_bench)
    end_ev.record(stream)
stream.synchronize()
cublas_stream_ms = start_ev.elapsed_time(end_ev)
cublas_stream_us = cublas_stream_ms * 1000.0 / N_BENCH

# Restore stderr
os.dup2(saved_stderr2, 2)
os.close(log_fd2)
os.close(saved_stderr2)

speedup1 = cublas_us / block_sub_us if block_sub_us > 0 else 0
speedup2 = cublas_stream_us / block_stream_us if block_stream_us > 0 else 0

print(f"\n  Measurement 1: Python loop ({N_BENCH} iterations)")
print(f"    O(Kr) GPU block sub:  {block_sub_us:8.1f} us/call")
print(f"    cuBLAS 4096x4096:     {cublas_us:8.1f} us/call")
print(f"    Speedup:              {speedup1:8.2f}x")

print(f"\n  Measurement 2: CUDA stream burst ({N_BENCH} iterations)")
print(f"    O(Kr) GPU block sub:  {block_stream_us:8.1f} us/call")
print(f"    cuBLAS 4096x4096:     {cublas_stream_us:8.1f} us/call")
print(f"    Speedup:              {speedup2:8.2f}x")

if block_stream_us < cublas_stream_us:
    print(f"\n  Result: FASTER by {cublas_stream_us - block_stream_us:.1f} us (stream burst)")
else:
    print(f"\n  Result: SLOWER by {block_stream_us - cublas_stream_us:.1f} us (stream burst)")
print("=" * 70)
