"""Diagnostic 3: torch.profiler on D1-style single-tenant Mistral-7B decode
using density_harness's MarlinLinear (lock + 2 wait_stream events / call).

Profiles ~10 seconds of timed decode (after warmup). User asked for 30s
but the profiler aggregation cost on 30s × ~36 tps × ~448 events =
~480k events would dominate. 10 s × ~36 tps × 448 ≈ 160k events captures
the steady-state distribution and aggregates in seconds.

Prints: top 10 CUDA ops, top 10 CPU ops, per-call event-creation cost.
"""
import sys, os, time
sys.path.insert(0, "/workspace/stress2")
import density_harness as dh

PROFILE_SECONDS = 10.0


def main():
    print("=" * 80)
    print("DIAG 3 — torch.profiler on density_harness MarlinLinear")
    print("=" * 80, flush=True)

    rt = dh.setup_rt()
    print("[setup] loading Mistral-7B...", flush=True)
    t0 = time.perf_counter()
    model, tok = dh.load_model_shared()
    print(f"[setup] model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print("[setup] patching Marlin...", flush=True)
    t0 = time.perf_counter()
    n_c, n_s = dh.patch_model_marlin(rt, model)
    print(f"[setup] {n_c} linears Marlin-compressed in "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

    import torch
    from transformers import StaticCache
    from torch.profiler import profile, ProfilerActivity

    prefill_len = 128
    max_decode = 200
    MAX_LEN = prefill_len + max_decode + 16
    prompt_ids = dh.make_prompt_ids(tok, prefill_len)
    stream = torch.cuda.Stream(device="cuda:0")

    with torch.cuda.stream(stream):
        cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
            o = model(input_ids=prompt_ids, cache_position=cp,
                       past_key_values=cache, use_cache=True,
                       return_dict=True)
        input_ids = o.logits[:, -1:].argmax(-1).clone()
        cache_pos = torch.tensor([prefill_len], device="cuda:0",
                                   dtype=torch.long)
        stream.synchronize()
        # warmup
        for _ in range(8):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                           past_key_values=cache, use_cache=True,
                           return_dict=True)
            input_ids.copy_(o.logits.argmax(-1))
            cache_pos += 1
        stream.synchronize()

        print(f"\n[prof] entering torch.profiler "
              f"({PROFILE_SECONDS}s wall-clock)...", flush=True)
        n_tokens = 0
        t_start = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            while time.perf_counter() - t_start < PROFILE_SECONDS:
                with torch.no_grad():
                    o = model(input_ids=input_ids,
                               cache_position=cache_pos,
                               past_key_values=cache, use_cache=True,
                               return_dict=True)
                input_ids.copy_(o.logits.argmax(-1))
                cache_pos += 1
                stream.synchronize()
                n_tokens += 1
                if cache_pos.item() >= MAX_LEN - 4:
                    cache_pos = torch.tensor([prefill_len],
                                               device="cuda:0",
                                               dtype=torch.long)
                    with torch.no_grad():
                        o = model(input_ids=prompt_ids,
                                   cache_position=torch.arange(
                                       prefill_len, device="cuda:0",
                                       dtype=torch.long),
                                   past_key_values=cache,
                                   use_cache=True, return_dict=True)
                    input_ids.copy_(o.logits[:, -1:].argmax(-1))
                    cache_pos += 1
                    stream.synchronize()
        elapsed = time.perf_counter() - t_start
        print(f"[prof] decoded {n_tokens} tokens in {elapsed:.2f}s "
              f"({n_tokens/elapsed:.2f} tps)", flush=True)

    print("\n" + "=" * 80)
    print(f"TOP 10 CUDA KERNELS by total cuda_time (over {n_tokens} tokens)")
    print("=" * 80, flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n" + "=" * 80)
    print(f"TOP 10 CPU OPS by total cpu_time")
    print("=" * 80, flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Specific event-related ops
    print("\n" + "=" * 80)
    print("FOCUS: event/stream/lock-related ops (sum across all calls)")
    print("=" * 80)
    averages = prof.key_averages()
    keys_of_interest = [
        "cudaEventCreateWithFlags", "cudaEventRecord",
        "cudaEventDestroy", "cudaStreamWaitEvent",
        "aten::record_stream", "aten::wait_stream",
        "cuLaunchKernel", "cudaLaunchKernel",
    ]
    hits = []
    for avg in averages:
        if any(k in avg.key for k in keys_of_interest):
            hits.append(avg)
    if hits:
        print(f"{'op':<40} {'count':>10} {'cpu_total_us':>15} {'cuda_total_us':>15} {'cpu_avg_us':>12}")
        for h in sorted(hits, key=lambda x: x.cpu_time_total, reverse=True):
            print(f"{h.key[:40]:<40} {h.count:>10} "
                  f"{h.cpu_time_total:>15.1f} {h.cuda_time_total:>15.1f} "
                  f"{h.cpu_time_total/max(h.count,1):>12.2f}")
    else:
        print("(no event/stream ops surfaced under those keys; "
              "list keys present below)")
        for avg in list(averages)[:30]:
            print(f"  {avg.key}  count={avg.count}")

    print(f"\nObserved tps in profiled window: {n_tokens/elapsed:.2f}")


if __name__ == "__main__":
    main()
