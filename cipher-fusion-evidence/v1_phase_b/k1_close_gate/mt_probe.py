#!/usr/bin/env python3
"""W.6 sub-C multi-tenant probe: long-lived continuous light inference.

Argv: <tag> <model> <gpu_util> <run_secs>
Loads the model and runs continuous small generate() calls for run_secs so the
classifier keeps observing -> keeps querying the kmod cohort registry (~1/sec).
Two concurrent instances overlap long enough for each to observe the other in
the co-residence count. The co_resident signal is read from the substrate's
[cipher_v2] CLASSIFY-CORES stderr lines (captured by docker run)."""
import sys, time, traceback

def main():
    tag = sys.argv[1]; model = sys.argv[2]
    util = float(sys.argv[3]); run_secs = float(sys.argv[4])
    print(f"=== MT_PROBE {tag} model={model} util={util} secs={run_secs} ===", flush=True)
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, gpu_memory_utilization=util, enforce_eager=True,
                  max_model_len=512, enable_prefix_caching=False, disable_log_stats=True)
        sp = SamplingParams(max_tokens=16, temperature=0.0)
        prompts = ["Hello there, tell me a short story about a robot."] * 4
        t0 = time.time(); i = 0
        while time.time() - t0 < run_secs:
            llm.generate(prompts, sp, use_tqdm=False)
            i += 1
        print(f"MT_PROBE {tag} DONE iters={i} elapsed={time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"MT_PROBE {tag} ERROR {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
