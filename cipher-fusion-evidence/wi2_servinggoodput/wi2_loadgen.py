#!/usr/bin/env python3
# WI-2: serving-goodput load generator. Poisson arrivals at rate lambda against a vLLM OpenAI server (streaming),
# records per-request TTFT (time to first token) and TPOT (mean inter-token time), computes SLO-attainment and
# SERVING goodput = (requests meeting BOTH TTFT and TPOT SLOs) / wall-time, per GPU.
import asyncio, time, json, random, sys, argparse
from openai import AsyncOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, required=True)        # offered request rate (req/s)
ap.add_argument("--n", type=int, default=60)               # number of requests
ap.add_argument("--out-tokens", type=int, default=128)
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--ttft-slo", type=float, default=1.0)     # relaxed 7B-eager SLO (s)
ap.add_argument("--tpot-slo", type=float, default=0.05)    # s/token (=20 tok/s/req)
ap.add_argument("--ttft-slo-strict", type=float, default=0.25)  # DistServe chatbot
ap.add_argument("--tpot-slo-strict", type=float, default=0.10)
ap.add_argument("--result", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="x")
PROMPT = "Explain the history of computing in detail, covering mechanical calculators, vacuum tubes, transistors, and modern accelerators. "

async def one_request(rid, results):
    t0 = time.perf_counter(); ttft = None; n_tok = 0; t_first = None; t_last = None
    try:
        stream = await client.completions.create(model="mistralai/Mistral-7B-v0.1", prompt=PROMPT,
                    max_tokens=args.out_tokens, temperature=0.0, stream=True)
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.choices and chunk.choices[0].text:
                if ttft is None: ttft = now - t0; t_first = now
                n_tok += 1; t_last = now
    except Exception as e:
        results.append({"rid": rid, "error": str(e)[:120]}); return
    tpot = ((t_last - t_first)/(n_tok-1)) if (n_tok and n_tok>1 and t_first) else None
    results.append({"rid": rid, "ttft": ttft, "tpot": tpot, "n_tok": n_tok, "wall": time.perf_counter()-t0})

async def main():
    results = []; tasks = []
    t_start = time.perf_counter()
    for rid in range(args.n):
        tasks.append(asyncio.create_task(one_request(rid, results)))
        await asyncio.sleep(random.expovariate(args.lam))   # Poisson inter-arrival
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start
    ok = [r for r in results if "error" not in r and r["ttft"] is not None and r["tpot"] is not None]
    def attain(tt, tp):
        m = [r for r in ok if r["ttft"]<=tt and r["tpot"]<=tp]
        return len(m), (len(m)/len(ok) if ok else 0)
    n_relaxed, a_relaxed = attain(args.ttft_slo, args.tpot_slo)
    n_strict, a_strict = attain(args.ttft_slo_strict, args.tpot_slo_strict)
    import statistics as st
    ttfts = sorted(r["ttft"] for r in ok); tpots = sorted(r["tpot"] for r in ok)
    res = {"lam": args.lam, "n": args.n, "n_ok": len(ok), "n_err": len(results)-len(ok), "wall_s": round(wall,2),
           "achieved_rate": round(len(results)/wall,3),
           "ttft_p50": round(st.median(ttfts),3) if ttfts else None, "ttft_p90": round(ttfts[int(0.9*len(ttfts))-1],3) if ttfts else None,
           "tpot_p50": round(st.median(tpots),4) if tpots else None, "tpot_p90": round(tpots[int(0.9*len(tpots))-1],4) if tpots else None,
           "relaxed_slo": {"ttft": args.ttft_slo, "tpot": args.tpot_slo, "attainment": round(a_relaxed,3),
                           "goodput_req_s": round(n_relaxed/wall,3)},
           "strict_slo": {"ttft": args.ttft_slo_strict, "tpot": args.tpot_slo_strict, "attainment": round(a_strict,3),
                          "goodput_req_s": round(n_strict/wall,3)}}
    json.dump(res, open(args.result,"w"), indent=2)
    print("WI2", json.dumps(res))

asyncio.run(main())
