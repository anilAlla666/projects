"""bench_llm.py — Week-6 industry-standard LLM bench harness.

Replaces /tmp/postclose/bench_mfu.py. Methodology mapped to:
  - MLPerf Inference v5.1 LLM-server scenario (TTFT/TPOT/ITL + p99 tails)
  - PaLM/Karpathy MFU formula: 2*N + 4*L*H*Q*T per token forward, phase-split
  - TokenPowerBench convention: pynvml NVML-direct sampler @ 100 Hz,
    trapezoidal energy integration, phase-attributed
  - vLLM bench_serving.py: per-token streaming hooks (AsyncLLMEngine)

CLI (see argparse below). Output:
  /tmp/bench_v2/results/<run_id>/result.json     full per-request data
  /tmp/bench_v2/results/<run_id>/summary.md      human-readable table
  /tmp/bench_v2/results/<run_id>/telemetry.json  100 Hz NVML samples
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# H100 SXM5 dense Tensor-Core BF16/FP16 peak. No sparsity. Authoritative
# for the FLOP-ceiling denominator in MFU.
H100_PEAK_BF16_TFLOPS = 989.0

# Target locked clocks per CIPHER campaign convention.
TARGET_GR_CLOCK_MHZ = 1980
TARGET_MEM_CLOCK_MHZ = 2619

# -----------------------------------------------------------------------------
# HardwareLock — verify + lock clocks, persistence, power; record full state.
# -----------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _nvsmi_q(field: str) -> str:
    rc, out, _ = _run(["nvidia-smi", f"--query-gpu={field}",
                       "--format=csv,noheader,nounits", "-i", "0"])
    if rc != 0:
        raise RuntimeError(f"nvidia-smi --query-gpu={field} failed")
    return out.split("\n")[0].strip()


class HardwareLock:
    """Verify + lock GPU state for reproducible measurement."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def lock_and_verify(self) -> None:
        # Persistence + clock lock require sudo. Best-effort; if no sudo,
        # we still verify and surface a fix command.
        rc, _, _ = _run(["sudo", "-n", "true"])
        have_sudo = rc == 0

        if have_sudo:
            _run(["sudo", "-n", "nvidia-smi", "-i", "0", "-pm", "1"])
            _run(["sudo", "-n", "nvidia-smi", "-i", "0",
                  "-lgc", str(TARGET_GR_CLOCK_MHZ)])
            _run(["sudo", "-n", "nvidia-smi", "-i", "0",
                  "-lmc", str(TARGET_MEM_CLOCK_MHZ)])

        # Verify max clocks (device variant check).
        max_gr = int(float(_nvsmi_q("clocks.max.gr")))
        max_mem = int(float(_nvsmi_q("clocks.max.mem")))
        power_limit = float(_nvsmi_q("power.limit"))
        persistence = _nvsmi_q("persistence_mode")
        ecc_mode = _nvsmi_q("ecc.mode.current")
        gpu_name = _nvsmi_q("name")
        driver = _nvsmi_q("driver_version")

        problems: list[str] = []
        if max_gr != TARGET_GR_CLOCK_MHZ:
            problems.append(
                f"max gr clock {max_gr} != {TARGET_GR_CLOCK_MHZ} "
                f"(wrong device variant?)")
        if max_mem != TARGET_MEM_CLOCK_MHZ:
            problems.append(
                f"max mem clock {max_mem} != {TARGET_MEM_CLOCK_MHZ}")
        if persistence != "Enabled":
            problems.append(
                f"persistence={persistence}; fix: "
                f"sudo nvidia-smi -i 0 -pm 1")

        self.state = {
            "gpu_name": gpu_name,
            "driver_version": driver,
            "max_gr_clock_mhz": max_gr,
            "max_mem_clock_mhz": max_mem,
            "power_limit_w": power_limit,
            "persistence_mode": persistence,
            "ecc_mode": ecc_mode,
            "lock_attempted": have_sudo,
            "target_gr_clock_mhz": TARGET_GR_CLOCK_MHZ,
            "target_mem_clock_mhz": TARGET_MEM_CLOCK_MHZ,
        }
        if problems:
            sys.stderr.write("HardwareLock FAILED:\n")
            for p_ in problems:
                sys.stderr.write(f"  - {p_}\n")
            sys.stderr.write(
                f"\nTo fix locking:\n"
                f"  sudo nvidia-smi -i 0 -pm 1\n"
                f"  sudo nvidia-smi -i 0 -lgc {TARGET_GR_CLOCK_MHZ}\n"
                f"  sudo nvidia-smi -i 0 -lmc {TARGET_MEM_CLOCK_MHZ}\n")
            raise SystemExit(2)


# -----------------------------------------------------------------------------
# TelemetrySampler — pynvml direct, dedicated thread @ 100 Hz, pinned to CPU 0.
# -----------------------------------------------------------------------------

class TelemetrySampler:
    """100 Hz NVML sampler; energy via trapezoidal integration."""

    def __init__(self, interval_s: float = 0.010, cpu_affinity: int = 0,
                 device_idx: int = 0) -> None:
        import pynvml
        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self.interval_s = interval_s
        self.cpu_affinity = cpu_affinity
        self.samples: list[tuple[int, int, int, int, int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._missed = 0
        self._sampled = 0

    def _run(self) -> None:
        try:
            os.sched_setaffinity(0, {self.cpu_affinity})
        except Exception:
            pass
        nvml = self.pynvml
        h = self.handle
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic_ns()
            try:
                pwr = nvml.nvmlDeviceGetPowerUsage(h)        # mW
                mem = nvml.nvmlDeviceGetMemoryInfo(h)
                util = nvml.nvmlDeviceGetUtilizationRates(h)
                sm_mhz = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)
                self.samples.append(
                    (now, pwr, mem.used >> 20, util.gpu, util.memory, sm_mhz))
                self._sampled += 1
            except Exception:
                self._missed += 1
            next_t += self.interval_s
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind — drop catch-up to avoid burst sampling.
                self._missed += int(-sleep_for / self.interval_s)
                next_t = time.monotonic()

    def start(self) -> None:
        self.samples = []
        self._missed = 0
        self._sampled = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def trapezoidal_energy_j(self, t_start_ns: int | None = None,
                             t_end_ns: int | None = None) -> float:
        """Energy in J between [t_start_ns, t_end_ns] over (t,P) samples."""
        if not self.samples:
            return 0.0
        s = self.samples
        if t_start_ns is None:
            t_start_ns = s[0][0]
        if t_end_ns is None:
            t_end_ns = s[-1][0]
        e_mws = 0.0  # mW·s
        for i in range(len(s) - 1):
            t0, p0 = s[i][0], s[i][1]
            t1, p1 = s[i + 1][0], s[i + 1][1]
            if t1 < t_start_ns or t0 > t_end_ns:
                continue
            tt0 = max(t0, t_start_ns)
            tt1 = min(t1, t_end_ns)
            dt_s = (tt1 - tt0) / 1e9
            # linearly interpolate power at the clipped endpoints
            def lerp(t: int) -> float:
                if t1 == t0:
                    return p0
                return p0 + (p1 - p0) * (t - t0) / (t1 - t0)
            p_a = lerp(tt0)
            p_b = lerp(tt1)
            e_mws += (p_a + p_b) / 2.0 * dt_s
        return e_mws / 1000.0  # J

    def stats_in_window(self, t0: int, t1: int) -> dict[str, Any]:
        ps = [s[1] for s in self.samples if t0 <= s[0] <= t1]
        hbm = [s[2] for s in self.samples if t0 <= s[0] <= t1]
        clks = [s[5] for s in self.samples if t0 <= s[0] <= t1]
        if not ps:
            return {"n_samples": 0}
        return {
            "n_samples": len(ps),
            "missed_samples": self._missed,
            "miss_rate_pct": 100.0 * self._missed
                             / max(1, self._missed + self._sampled),
            "mean_power_w": statistics.mean(ps) / 1000.0,
            "peak_power_w": max(ps) / 1000.0,
            "p50_power_w": statistics.median(ps) / 1000.0,
            "p95_power_w": _pct(ps, 0.95) / 1000.0,
            "p99_power_w": _pct(ps, 0.99) / 1000.0,
            "mean_hbm_mib": statistics.mean(hbm),
            "peak_hbm_mib": max(hbm),
            "sm_clock_mhz_p50": statistics.median(clks),
            "sm_clock_mhz_min": min(clks),
            "sm_clock_mhz_max": max(clks),
            "energy_j": self.trapezoidal_energy_j(t0, t1),
        }


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(q * (len(s) - 1))
    return s[k]


# -----------------------------------------------------------------------------
# WorkloadGenerator — sharegpt | agentic with length distributions.
# -----------------------------------------------------------------------------

AGENTIC_SYSTEM = (
    "You are an autonomous engineering agent operating in a Linux shell. "
    "You can read files, execute commands, edit code, and reason about the "
    "results. Always justify your plan before acting. Tools you have access "
    "to are documented below. When you receive an error message, do not "
    "retry blindly — interpret the error, diagnose the cause, and adjust. "
    "Surface critical findings explicitly. Avoid speculative refactoring. "
    "Prefer reading existing source over inferring its behavior. "
)

AGENTIC_TOOLS = (
    "Tool: shell. Description: executes shell commands. Args: command (str), "
    "cwd (optional str). Returns: stdout, stderr, exit_code.\n"
    "Tool: read_file. Description: reads a file. Args: path (str), offset "
    "(optional int), limit (optional int). Returns: content (str).\n"
    "Tool: write_file. Description: overwrites or creates a file. Args: "
    "path (str), content (str). Returns: bytes_written.\n"
    "Tool: edit_file. Description: surgical string replacement. Args: path, "
    "old_string, new_string. Returns: applied (bool).\n"
    "Tool: grep. Description: ripgrep over the working tree. Args: pattern, "
    "glob (optional), case_sensitive (optional bool). Returns: matches.\n"
)

AGENTIC_USER_TEMPLATE = (
    "Investigate the following symptom in the codebase and propose a fix: "
    "{symptom}. Read at least three relevant files before suggesting changes. "
    "Cite the file:line locations for every claim. If you find evidence the "
    "symptom is intentional, surface that explicitly. Walk me through your "
    "reasoning step by step."
)

SYMPTOMS = [
    "the request queue grows unbounded under sustained load even though the "
    "backpressure flag is set",
    "intermittent 502s from the edge proxy when the upstream is healthy and "
    "responsive on direct probes",
    "memory residence climbs by ~40 MiB/hour in the worker pool with no "
    "matching allocation growth in tcmalloc heap profiles",
    "rate-limiter bucket refills appear to skip the first millisecond of "
    "every minute boundary, producing visible request bursts at minute marks",
    "logs from the canary instance lack the request_id field even though the "
    "logging middleware is identical to baseline",
    "kafka consumer lag oscillates between 0 and 60s every ~7 minutes despite "
    "constant inbound rate",
    "the auth service returns a stale signing key for ~200ms after rotation "
    "even after the cache invalidation broadcast",
    "p99 latency on the search path doubled after a no-op deploy and snapped "
    "back after a rolling restart",
]


class WorkloadGenerator:
    def __init__(self, tokenizer, kind: str, input_len: int,
                 seed: int = 1337) -> None:
        self.tok = tokenizer
        self.kind = kind
        self.input_len = input_len
        self.rng = random.Random(seed)

    def _to_ids(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return ids

    def _truncate_or_pad(self, ids: list[int]) -> list[int]:
        target = self.input_len
        bos = self.tok.bos_token_id or 1
        if len(ids) >= target:
            return [bos] + ids[: target - 1]
        # Pad by repeating last chunk
        pad_id = ids[-1] if ids else 1
        return [bos] + ids + [pad_id] * (target - 1 - len(ids))

    def make_agentic_prompt(self, i: int) -> list[int]:
        sym = SYMPTOMS[i % len(SYMPTOMS)]
        rng_tail = self.rng.randint(1_000_000, 9_999_999)
        text = (
            AGENTIC_SYSTEM
            + "\n\n## Available tools\n"
            + AGENTIC_TOOLS
            + "\n## Task\n"
            + AGENTIC_USER_TEMPLATE.format(symptom=sym)
            + f"\n\n(trace_id: {rng_tail}-{i})"
        )
        ids = self._to_ids(text)
        return self._truncate_or_pad(ids)

    def make(self, n: int) -> list[list[int]]:
        if self.kind == "agentic":
            return [self.make_agentic_prompt(i) for i in range(n)]
        if self.kind == "sharegpt":
            # Not used in default headline; fallback to agentic if file absent.
            sg = Path("/tmp/bench_v2/sharegpt.json")
            if not sg.exists():
                return [self.make_agentic_prompt(i) for i in range(n)]
            data = json.loads(sg.read_text())
            out: list[list[int]] = []
            for i in range(n):
                text = data[i % len(data)]
                ids = self._to_ids(text)
                out.append(self._truncate_or_pad(ids))
            return out
        raise ValueError(f"unknown workload kind: {self.kind}")


# -----------------------------------------------------------------------------
# FlopAccountant — PaLM formula, phase-split. Reads N, L, H, Q from config.
# -----------------------------------------------------------------------------

@dataclass
class ModelDims:
    n_non_embed: int
    n_total: int
    layers: int
    heads_q: int
    heads_kv: int
    head_dim: int
    vocab: int
    hidden: int
    intermediate: int
    dtype: str


def model_dims_from_config(hf_cfg) -> ModelDims:
    L = hf_cfg.num_hidden_layers
    H_q = hf_cfg.num_attention_heads
    H_kv = getattr(hf_cfg, "num_key_value_heads", H_q)
    Q = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // H_q)
    hidden = hf_cfg.hidden_size
    inter = hf_cfg.intermediate_size
    vocab = hf_cfg.vocab_size
    embed = vocab * hidden
    # GQA-aware per-layer non-embedding param count
    per_layer = (
        hidden * H_q * Q                # q_proj
        + hidden * H_kv * Q             # k_proj
        + hidden * H_kv * Q             # v_proj
        + H_q * Q * hidden              # o_proj
        + 3 * hidden * inter            # gate, up, down
        + 2 * hidden                    # 2 RMSNorms
    )
    final_norm = hidden
    lm_head = vocab * hidden
    n_total = embed + L * per_layer + final_norm + lm_head
    n_non_embed = n_total - embed
    return ModelDims(
        n_non_embed=n_non_embed,
        n_total=n_total,
        layers=L,
        heads_q=H_q,
        heads_kv=H_kv,
        head_dim=Q,
        vocab=vocab,
        hidden=hidden,
        intermediate=inter,
        dtype=str(getattr(hf_cfg, "torch_dtype", "bfloat16")),
    )


def flops_per_token(N_non_embed: int, L: int, H_q: int, Q: int, T: int) -> int:
    """PaLM/Karpathy decoder-only forward FLOPs at position T."""
    return 2 * N_non_embed + 4 * L * H_q * Q * T


def total_prefill_flops(md: ModelDims, prompt_len: int) -> int:
    P = prompt_len
    attn = 2 * md.layers * md.heads_q * md.head_dim * P * (P + 1)
    return 2 * md.n_non_embed * P + attn


def total_decode_flops(md: ModelDims, prompt_len: int, n_decode: int,
                       batch: int = 1) -> int:
    total = 0
    for i in range(n_decode):
        T = prompt_len + i
        total += flops_per_token(md.n_non_embed, md.layers, md.heads_q,
                                 md.head_dim, T)
    return total * batch


# -----------------------------------------------------------------------------
# StreamingRunner — AsyncLLMEngine, per-token timestamps via async generator.
# -----------------------------------------------------------------------------

@dataclass
class RequestTrace:
    request_id: str
    prompt_len: int
    t_submit_ns: int = 0
    t_first_token_ns: int = 0
    t_each_token_ns: list[int] = field(default_factory=list)
    output_tokens: int = 0

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token_ns - self.t_submit_ns) / 1e6

    @property
    def decode_window_ns(self) -> int:
        if not self.t_each_token_ns:
            return 0
        return self.t_each_token_ns[-1] - self.t_first_token_ns

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens < 2 or self.decode_window_ns <= 0:
            return float("nan")
        return self.decode_window_ns / (self.output_tokens - 1) / 1e6

    @property
    def itl_ms(self) -> list[float]:
        return [
            (self.t_each_token_ns[i] - self.t_each_token_ns[i - 1]) / 1e6
            for i in range(1, len(self.t_each_token_ns))
        ]

    @property
    def output_speed_tps(self) -> float:
        if self.output_tokens < 2 or self.decode_window_ns <= 0:
            return float("nan")
        return (self.output_tokens - 1) / (self.decode_window_ns / 1e9)


class StreamingRunner:
    def __init__(self, model: str, max_model_len: int,
                 gpu_mem_util: float, enforce_eager: bool) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        args = AsyncEngineArgs(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=enforce_eager,
            enable_prefix_caching=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = None  # set lazily

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = self.engine.get_tokenizer()
        return self.tokenizer

    async def _drive_one(self, prompt_ids: list[int], rid: str,
                         max_tokens: int) -> RequestTrace:
        from vllm import SamplingParams, TokensPrompt
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                            ignore_eos=True)
        trace = RequestTrace(request_id=rid, prompt_len=len(prompt_ids))
        trace.t_submit_ns = time.monotonic_ns()
        last_n = 0
        async for ro in self.engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids), sp, rid):
            now = time.monotonic_ns()
            tok_ids = ro.outputs[0].token_ids
            n = len(tok_ids)
            if n > last_n:
                if trace.t_first_token_ns == 0:
                    trace.t_first_token_ns = now
                # If multiple new tokens arrived in one yield (rare in
                # streaming), evenly distribute timestamps in the gap.
                if n - last_n == 1:
                    trace.t_each_token_ns.append(now)
                else:
                    prev_t = (trace.t_each_token_ns[-1]
                              if trace.t_each_token_ns
                              else trace.t_first_token_ns)
                    for j in range(n - last_n):
                        trace.t_each_token_ns.append(
                            int(prev_t + (now - prev_t)
                                * (j + 1) / (n - last_n)))
                last_n = n
        trace.output_tokens = last_n
        return trace

    async def run_batch(self, prompts: list[list[int]],
                        max_tokens: int) -> list[RequestTrace]:
        rid_prefix = f"req-{time.monotonic_ns():x}"
        coros = [
            self._drive_one(p, f"{rid_prefix}-{i}", max_tokens)
            for i, p in enumerate(prompts)
        ]
        return await asyncio.gather(*coros)


# -----------------------------------------------------------------------------
# MetricsAggregator — percentiles + phase-split MFU + power/HBM rollups.
# -----------------------------------------------------------------------------

def aggregate_iteration(traces: list[RequestTrace], telemetry_stats: dict,
                        md: ModelDims, batch_size: int,
                        iter_window_ns: tuple[int, int],
                        peak_tflops: float) -> dict[str, Any]:
    # Per-request scalars
    ttfts = [t.ttft_ms for t in traces if t.t_first_token_ns]
    tpots = [t.tpot_ms for t in traces if not math.isnan(t.tpot_ms)]
    speeds = [t.output_speed_tps for t in traces
              if not math.isnan(t.output_speed_tps)]
    itls_all = [v for t in traces for v in t.itl_ms]

    # Phase windows: prefill = [t_submit_min, t_first_token_max]
    #                decode  = [t_first_token_min, t_last_token_max]
    if not traces or not all(t.t_first_token_ns for t in traces):
        return {"error": "incomplete traces"}
    t_submit_min = min(t.t_submit_ns for t in traces)
    t_first_max = max(t.t_first_token_ns for t in traces)
    t_first_min = min(t.t_first_token_ns for t in traces)
    t_last_max = max(t.t_each_token_ns[-1] for t in traces
                     if t.t_each_token_ns)

    prefill_s = max(1e-9, (t_first_max - t_submit_min) / 1e9)
    decode_s = max(1e-9, (t_last_max - t_first_min) / 1e9)

    # FLOPs: prefill totals over batch; decode totals over batch
    prefill_flops = sum(
        total_prefill_flops(md, t.prompt_len) for t in traces)
    decode_flops = sum(
        total_decode_flops(md, t.prompt_len, t.output_tokens, batch=1)
        for t in traces)

    peak_flops_s = peak_tflops * 1e12
    prefill_mfu = (prefill_flops / prefill_s) / peak_flops_s * 100
    decode_mfu = (decode_flops / decode_s) / peak_flops_s * 100

    # Aggregate TPS over the iteration window (per spec: tokens / wall)
    iter_s = max(1e-9, (iter_window_ns[1] - iter_window_ns[0]) / 1e9)
    total_out = sum(t.output_tokens for t in traces)
    agg_tps = total_out / iter_s

    # Energy: phase-attributed via trapezoidal integration
    # (use TelemetrySampler.trapezoidal_energy_j externally; passed in
    # telemetry_stats already)
    return {
        "batch_size": batch_size,
        "n_requests": len(traces),
        "iter_wall_s": iter_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "ttft_ms_p50": _pct(ttfts, 0.50) if ttfts else float("nan"),
        "ttft_ms_p90": _pct(ttfts, 0.90) if ttfts else float("nan"),
        "ttft_ms_p95": _pct(ttfts, 0.95) if ttfts else float("nan"),
        "ttft_ms_p99": _pct(ttfts, 0.99) if ttfts else float("nan"),
        "tpot_ms_p50": _pct(tpots, 0.50) if tpots else float("nan"),
        "tpot_ms_p90": _pct(tpots, 0.90) if tpots else float("nan"),
        "tpot_ms_p99": _pct(tpots, 0.99) if tpots else float("nan"),
        "itl_ms_p50": _pct(itls_all, 0.50) if itls_all else float("nan"),
        "itl_ms_p99": _pct(itls_all, 0.99) if itls_all else float("nan"),
        "per_req_output_speed_p50": _pct(speeds, 0.50) if speeds else float("nan"),
        "aggregate_tps": agg_tps,
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "prefill_mfu_pct": prefill_mfu,
        "decode_mfu_pct": decode_mfu,
        "total_output_tokens": total_out,
        **telemetry_stats,
        "tok_per_w": (agg_tps / telemetry_stats["mean_power_w"])
                    if telemetry_stats.get("mean_power_w", 0) > 0 else float("nan"),
        "tok_per_j": (total_out / telemetry_stats["energy_j"])
                    if telemetry_stats.get("energy_j", 0) > 0 else float("nan"),
        "j_per_tok": (telemetry_stats["energy_j"] / total_out)
                    if total_out > 0 else float("nan"),
    }


def aggregate_iterations(iters: list[dict]) -> dict[str, Any]:
    keys_to_pct = [
        "ttft_ms_p50", "ttft_ms_p90", "ttft_ms_p99",
        "tpot_ms_p50", "tpot_ms_p99", "itl_ms_p99",
        "aggregate_tps", "decode_mfu_pct", "prefill_mfu_pct",
        "mean_power_w", "peak_power_w", "p99_power_w",
        "energy_j", "tok_per_w", "tok_per_j", "j_per_tok",
        "mean_hbm_mib", "peak_hbm_mib",
        "sm_clock_mhz_p50", "sm_clock_mhz_min", "sm_clock_mhz_max",
    ]
    out: dict[str, Any] = {"n_iterations": len(iters)}
    for k in keys_to_pct:
        vals = [it[k] for it in iters
                if k in it and isinstance(it[k], (int, float))
                and not math.isnan(it[k])]
        if not vals:
            out[k + ".mean"] = float("nan")
            continue
        out[k + ".mean"] = statistics.mean(vals)
        out[k + ".stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[k + ".min"] = min(vals)
        out[k + ".max"] = max(vals)
    return out


# -----------------------------------------------------------------------------
# SignificanceTester — bootstrap 95% CI of (cipher - vanilla) delta.
# -----------------------------------------------------------------------------

def bootstrap_ci_of_delta(a: list[float], b: list[float],
                          n_resample: int = 1000, seed: int = 7) -> dict:
    rng = random.Random(seed)
    if not a or not b:
        return {"mean_delta": float("nan"), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "cohens_d": float("nan"),
                "significant": False}
    deltas: list[float] = []
    for _ in range(n_resample):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(statistics.mean(sb) - statistics.mean(sa))
    deltas.sort()
    lo = deltas[int(0.025 * n_resample)]
    hi = deltas[int(0.975 * n_resample)]
    mean_delta = statistics.mean(b) - statistics.mean(a)
    pooled_var = (
        (statistics.pvariance(a) * (len(a) - 1)
         + statistics.pvariance(b) * (len(b) - 1))
        / max(1, len(a) + len(b) - 2)
    ) if len(a) > 1 and len(b) > 1 else 0.0
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    cd = mean_delta / pooled_sd if pooled_sd > 0 else float("nan")
    return {
        "mean_delta": mean_delta,
        "ci95_low": lo,
        "ci95_high": hi,
        "cohens_d": cd,
        "significant": bool((lo > 0) or (hi < 0)),
    }


# -----------------------------------------------------------------------------
# Reporter — JSON + markdown summary.
# -----------------------------------------------------------------------------

def write_results(run_id: str, payload: dict[str, Any]) -> Path:
    outdir = Path(f"/tmp/bench_v2/results/{run_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    md = render_summary_md(payload)
    (outdir / "summary.md").write_text(md)
    return outdir


def render_summary_md(p: dict[str, Any]) -> str:
    lines = [
        f"# {p['run_id']}",
        "",
        "## Hardware state",
        "```json",
        json.dumps(p.get("hardware_state", {}), indent=2),
        "```",
        "",
        "## Args",
        "```json",
        json.dumps(p.get("args", {}), indent=2),
        "```",
        "",
        "## Per-batch results (mean over iters)",
        "",
        "| B | n_iter | ttft_p50_ms | ttft_p99_ms | tpot_p50_ms | tpot_p99_ms |"
        " agg_tps | dec_MFU% | pre_MFU% | mean_W | peak_W | en_J | tok/W | tok/J |"
        " peakHBM_MiB | smClk_p50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p.get("per_batch", []):
        a = r["aggregate"]
        lines.append(
            f"| {r['batch_size']} | {a['n_iterations']} | "
            f"{a.get('ttft_ms_p50.mean', float('nan')):.1f} | "
            f"{a.get('ttft_ms_p99.mean', float('nan')):.1f} | "
            f"{a.get('tpot_ms_p50.mean', float('nan')):.2f} | "
            f"{a.get('tpot_ms_p99.mean', float('nan')):.2f} | "
            f"{a.get('aggregate_tps.mean', float('nan')):.2f} | "
            f"{a.get('decode_mfu_pct.mean', float('nan')):.3f} | "
            f"{a.get('prefill_mfu_pct.mean', float('nan')):.2f} | "
            f"{a.get('mean_power_w.mean', float('nan')):.1f} | "
            f"{a.get('peak_power_w.mean', float('nan')):.1f} | "
            f"{a.get('energy_j.mean', float('nan')):.1f} | "
            f"{a.get('tok_per_w.mean', float('nan')):.3f} | "
            f"{a.get('tok_per_j.mean', float('nan')):.3f} | "
            f"{a.get('peak_hbm_mib.mean', float('nan')):.0f} | "
            f"{a.get('sm_clock_mhz_p50.mean', float('nan')):.0f} |"
        )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch-sizes", type=str, default="1",
                   help="comma-separated, e.g. 1,8,16,32,64")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--input-length-distribution", type=str,
                   default="fixed:1024",
                   help="fixed:N | normal:M,S | realistic")
    p.add_argument("--workload", choices=["agentic", "sharegpt"],
                   default="agentic")
    p.add_argument("--no-cipher", action="store_true")
    p.add_argument("--cipher", action="store_true")
    p.add_argument("--run-id", default="")
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--telemetry-interval-ms", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def parse_input_len(spec: str, rng: random.Random) -> int:
    if spec.startswith("fixed:"):
        return int(spec.split(":")[1])
    if spec.startswith("normal:"):
        m, s = spec.split(":")[1].split(",")
        v = int(rng.gauss(float(m), float(s)))
        return max(128, min(4096, v))
    if spec == "realistic":
        return rng.choice([256, 512, 1024, 2048])
    raise ValueError(spec)


async def main_async(args: argparse.Namespace) -> None:
    # 1. Hardware lock
    hw = HardwareLock()
    hw.lock_and_verify()

    # CIPHER plugin env gate (must be set before EngineCore subprocess fork)
    if args.no_cipher and not args.cipher:
        os.environ["CIPHER_KV_ALLOC"] = "0"
        os.environ["CIPHER_KVDEDUP"] = "0"
        os.environ["CIPHER_OFFLOAD"] = "0"
        cipher_state = "off"
    elif args.cipher:
        os.environ["CIPHER_KV_ALLOC"] = "1"
        os.environ["CIPHER_KVDEDUP"] = "1"
        # CUDA_INJECTION64_PATH for libcipher_rt actuator
        if "CUDA_INJECTION64_PATH" not in os.environ:
            so = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
            if Path(so).exists():
                os.environ["CUDA_INJECTION64_PATH"] = so
        cipher_state = "on"
    else:
        cipher_state = "default(on)"
    print(f"[bench_llm] CIPHER state: {cipher_state}  "
          f"KV_ALLOC={os.environ.get('CIPHER_KV_ALLOC','?')}  "
          f"KVDEDUP={os.environ.get('CIPHER_KVDEDUP','?')}  "
          f"CUDA_INJECTION64_PATH={os.environ.get('CUDA_INJECTION64_PATH','')}")

    # 2. Run id
    run_id = args.run_id or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{Path(args.model).name.lower().replace('-', '_').replace('.', '_')}_"
        f"{'cipher' if args.cipher else 'vanilla'}_"
        f"{args.workload}"
    )

    # 3. Engine
    rng = random.Random(args.seed)
    input_len = parse_input_len(args.input_length_distribution, rng)
    max_model_len = input_len + args.output_tokens + 16
    print(f"[bench_llm] run_id={run_id}  input_len={input_len}  "
          f"max_model_len={max_model_len}")
    runner = StreamingRunner(args.model, max_model_len,
                             args.gpu_mem_util, args.enforce_eager)
    tok = runner.get_tokenizer()

    # 4. FLOP accountant
    hf_cfg = runner.engine.vllm_config.model_config.hf_config
    md = model_dims_from_config(hf_cfg)
    print(f"[bench_llm] model_dims: N_non_embed={md.n_non_embed:,}  "
          f"L={md.layers}  H_q={md.heads_q}  Q={md.head_dim}  "
          f"dtype={md.dtype}")

    # 5. Workload
    wlg = WorkloadGenerator(tok, args.workload, input_len, seed=args.seed)

    # 6. Sweep
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    sampler = TelemetrySampler(interval_s=args.telemetry_interval_ms / 1000.0)
    per_batch_results: list[dict] = []

    for B in batch_sizes:
        prompts = wlg.make(B)

        # Warmup
        for _ in range(args.warmup_iters):
            await runner.run_batch(prompts, args.output_tokens)

        iter_aggs: list[dict] = []
        iter_traces: list[list[RequestTrace]] = []
        for it in range(args.iterations):
            sampler.start()
            t0 = time.monotonic_ns()
            traces = await runner.run_batch(prompts, args.output_tokens)
            t1 = time.monotonic_ns()
            sampler.stop()
            tstats = sampler.stats_in_window(t0, t1)
            agg = aggregate_iteration(traces, tstats, md, B, (t0, t1),
                                       H100_PEAK_BF16_TFLOPS)
            iter_aggs.append(agg)
            iter_traces.append(traces)
            print(f"[bench_llm] B={B} iter={it+1}/{args.iterations}  "
                  f"agg_tps={agg.get('aggregate_tps', float('nan')):.1f}  "
                  f"decode_MFU={agg.get('decode_mfu_pct', float('nan')):.3f}%  "
                  f"mean_W={tstats.get('mean_power_w', float('nan')):.0f}  "
                  f"peakHBM={tstats.get('peak_hbm_mib', 0):.0f} MiB  "
                  f"smClk_p50={tstats.get('sm_clock_mhz_p50', 0):.0f}MHz")

        per_batch_results.append({
            "batch_size": B,
            "input_len": input_len,
            "iterations_raw": [
                {k: v for k, v in a.items() if k != "telemetry"}
                for a in iter_aggs
            ],
            "aggregate": aggregate_iterations(iter_aggs),
            "traces_summary": [
                [{
                    "rid": t.request_id,
                    "prompt_len": t.prompt_len,
                    "output_tokens": t.output_tokens,
                    "ttft_ms": t.ttft_ms,
                    "tpot_ms": t.tpot_ms,
                    "output_speed_tps": t.output_speed_tps,
                } for t in trs]
                for trs in iter_traces
            ],
        })

    payload = {
        "run_id": run_id,
        "args": vars(args),
        "hardware_state": hw.state,
        "model_dims": asdict(md),
        "peak_bf16_tflops": H100_PEAK_BF16_TFLOPS,
        "per_batch": per_batch_results,
        "harness_md5": _self_md5(),
    }
    outdir = write_results(run_id, payload)

    # Telemetry trace serialized separately (can be large)
    (outdir / "telemetry.json").write_text(json.dumps({
        "interval_s": sampler.interval_s,
        "missed_samples": sampler._missed,
        "sampled": sampler._sampled,
    }, indent=2))

    print(f"[bench_llm] DONE  results: {outdir}")


def _self_md5() -> str:
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
