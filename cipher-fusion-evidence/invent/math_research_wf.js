export const meta = {
  name: 'math-mfu-levers-research',
  description: 'Deep paper research on mathematical model-level MFU levers and whether CIPHER can deliver a quality-preserving 2x',
  phases: [{ title: 'Research' }, { title: 'Assess' }, { title: 'Synthesize' }],
}
const FIND_SCHEMA = {
  type: 'object', required: ['area', 'summary', 'methods'],
  properties: {
    area: { type: 'string' }, summary: { type: 'string' },
    methods: { type: 'array', items: { type: 'object',
      required: ['name', 'math', 'measured_speedup', 'measured_quality', 'needs_model_codesign', 'posthoc_possible', 'hardware', 'sources'],
      properties: {
        name: { type: 'string' }, math: { type: 'string' },
        measured_speedup: { type: 'string' }, measured_quality: { type: 'string' },
        needs_model_codesign: { type: 'string', enum: ['no-pure-posthoc', 'light-calibration-only', 'needs-finetune', 'needs-pretrain'] },
        posthoc_possible: { type: 'string' }, hardware: { type: 'string' },
        sources: { type: 'array', items: { type: 'string' } },
      } } },
  },
}
const ASSESS_SCHEMA = {
  type: 'object', required: ['method', 'verdict', 'honest_gain', 'delivery', 'reasoning'],
  properties: {
    method: { type: 'string' },
    verdict: { type: 'string', enum: ['real-2x-hardware', 'real-but-needs-finetune', 'flop-reduction-not-throughput', 'overstated', 'quality-killer'] },
    honest_gain: { type: 'string' }, delivery: { type: 'string' }, reasoning: { type: 'string' },
  },
}
const COMMON = [
  'You are a research analyst for CIPHER, a CUDA-driver-interposition substrate (intercepts cuBLAS/cutlass GEMM under unmodified vLLM; does NOT retrain models; CAN host a modified checkpoint the way it hosts a quantized one). H100 / 7B-dense (Mistral/Llama).',
  '',
  'The FOUNDING research already tried these math levers POST-HOC (frozen model, no retrain) and they FAILED for that reason -- build past, do not repeat:',
  '- Monarch/Kronecker factorization: 23/32 layers full-rank, no structure post-hoc.',
  '- Low-rank/Koopman projection: full-layer rank-k gave KL=13 mode-collapse; only 9/32 O-proj layers worked.',
  '- Activation sparsity: SwiGLU too dense (0.52 sparsity, 0.39 Jaccard) -- dead post-hoc.',
  '- W4A16 INT4: 1.38x at B=8 but only fires at M=1; kernel is cutlass/marlin-optimal.',
  '- single-GPU dense MFU ceiling ~67% (compute-bound); decode MFU ~0.2-23% (memory-bound).',
  '',
  'KEY QUESTION: the founding failed because it imposed structure POST-HOC on a frozen model. Does 2024-2026 SOTA in your area achieve a QUALITY-PRESERVING speedup when structure is CO-DESIGNED (prune+calibration / light finetune)? Report REAL measured speed AND quality AND model-side work. Be brutally honest about FLOP-reduction-vs-MFU: reducing FLOPs lowers the numerator -- only HARDWARE-accelerated structure (2:4 sparse tensor cores = 2x peak) actually raises tokens/sec at fixed quality.',
  '',
  'TOOLS: WEB SEARCH + WEB FETCH via ToolSearch (query "select:WebSearch,WebFetch"). 5-8 searches, fetch 2-4 best sources, extract real numbers. No invented gains; if unverified, say so.',
].join('\n')

phase('Research')
const AREAS = [
  { key: 'structured_sparsity', prompt: COMMON + '\n\nAREA: STRUCTURED SPARSITY with hardware support. NVIDIA 2:4 structured sparsity (Sparse Tensor Cores, 2x peak FLOPs on Ampere/Hopper), SparseGPT, Wanda, post-hoc 2:4 pruning + quality recovery, TensorRT 2:4, double-sparse. CRITICAL: does 2:4 deliver a REAL measured ~2x tokens/sec on 7B LLMs on H100 (hardware sparse GEMM) at <1% quality loss after light finetune, or only paper-FLOPs? What model-side work (one-shot prune vs finetune) keeps quality? Can a substrate host a 2:4-pruned checkpoint?' },
  { key: 'lowrank_factorization', prompt: COMMON + '\n\nAREA: LOW-RANK / FACTORIZATION SOTA beyond the founding KL=13 failure. SVD-LLM, ASVD, Palu, FWSVD, LoRD, Basis-sharing, activation-aware SVD, layer-wise low-rank with calibration. Does calibrated/activation-aware low-rank achieve real speedup at preserved quality where naive projection failed? Measured compression vs PPL on 7B. Post-hoc (calibration) or needs finetune? Does it actually speed up AND map to MFU/throughput, or memory-only?' },
  { key: 'structured_matrices', prompt: COMMON + '\n\nAREA: STRUCTURED MATRICES as trained layers. Monarch / Monarch Mixer M2, Butterfly, Block-Tensor-Train BTT, Pixelated Butterfly -- structured linear layers replacing dense GEMMs with sub-quadratic FLOPs, quality WHEN TRAINED IN vs founding post-hoc failure. Measured FLOP/speed reduction + quality at transformer scale. Honest: do these need PRETRAINING/finetune -- can a substrate ever deliver, or model-architecture-only?' },
  { key: 'attention_kv_math', prompt: COMMON + '\n\nAREA: ATTENTION & KV-CACHE MATH. Multi-head Latent Attention (MLA, DeepSeek low-rank KV), GQA/MQA, KV low-rank (Palu/Eigen), linear/SSM attention (Mamba hybrids), sliding-window, KV quantization SOTA (KIVI, KVQuant beyond founding 2-bit). For long-context decode (KV-dominated): real measured throughput gain at preserved quality, model-codesign vs post-hoc. Is MLA retrofittable to a trained GQA model?' },
  { key: 'frontier_2026', prompt: COMMON + '\n\nAREA: THE 2025-2026 FRONTIER the founding predates. NEWEST measured-2x+ inference-efficiency: FP4/NVFP4 + QAT, microscaling MXFP, speculative+quant co-design, EAGLE-3 on quantized models, dynamic/learned sparsity (TEAL/CATS quality), MoE-ification of dense (upcycling), Medusa/MTP retrofit, math+hardware co-design claiming real H100 throughput at preserved quality. Single most promising NEW lever (2025-2026), measured gain+quality, model-side cost.' },
]
const research = await pipeline(AREAS,
  (a) => agent(a.prompt, { label: 'research:' + a.key, phase: 'Research', schema: FIND_SCHEMA }),
  (res, a) => {
    if (!res) return null
    const top = (res.methods || []).filter(function (m) { return m.needs_model_codesign !== 'needs-pretrain' }).slice(0, 3)
    log('research:' + a.key + ' -- ' + (res.methods || []).length + ' methods, ' + top.length + ' assessed')
    if (!top.length) return { area: a.key, summary: res.summary, methods: res.methods, assessments: [] }
    return parallel(top.map(function (m) {
      return function () {
        return agent(COMMON + '\n\nADVERSARIAL ASSESS with WEB SEARCH. Verify SKEPTICALLY: (1) is the speedup REAL on H100 hardware, not paper-FLOP-reduction that does not raise tokens/sec? (2) is the quality cost honest (post-hoc vs requires-finetune)? (3) what would CIPHER actually ship: host a co-designed checkpoint (feasible, like quant) or invent it (infeasible)? (4) honest expected gain for H100/7B-dense/preserved-quality. Search critiques/reproductions.\n\nMETHOD: ' + m.name + '\nMATH: ' + m.math + '\nCLAIMED SPEED: ' + m.measured_speedup + '\nCLAIMED QUALITY: ' + m.measured_quality + '\nMODEL-CODESIGN: ' + m.needs_model_codesign,
          { label: 'assess:' + a.key + ':' + m.name.slice(0, 24), phase: 'Assess', schema: ASSESS_SCHEMA })
          .then(function (v) { return { method: m, assessment: v } })
      }
    })).then(function (as) { return { area: a.key, summary: res.summary, methods: res.methods, assessments: as.filter(Boolean) } })
  })
const out = research.filter(Boolean)
phase('Synthesize')
const dossier = JSON.stringify(out, null, 1).slice(0, 95000)
const SYN = COMMON + '\n\nSYNTHESIS. Research+assessment on mathematical model-level MFU levers is below. ONE honest answer needed: is there a math lever that genuinely lifts MFU/throughput ~2x at PRESERVED QUALITY on H100/7B-dense, and CAN CIPHER DELIVER IT?\n' +
  '1. THE LEVER (if any): most promising, honest measured gain+quality, model-side cost. If 2:4-sparse-tensor-cores is the answer, say so with the number.\n' +
  '2. DELIVERY PATH: can CIPHER ship it by hosting a co-designed checkpoint (like a quantized model)? Or does it need pretraining (= a model company, not a substrate)? Explicit.\n' +
  '3. HONEST VERDICT: is "lift MFU 2x at preserved quality" achievable as a CIPHER product, or does every real lever require becoming a model company? Distinguish "substrate hosts a 2:4/low-rank checkpoint someone else made" (feasible) vs "substrate invents the math" (founding proved infeasible post-hoc).\n' +
  '4. The 1-2 week prove/kill experiment on H100/Mistral-7B (e.g. measure a 2:4-pruned Mistral real decode tok/s + PPL vs dense).\n' +
  '5. KILL CRITERIA.\n' +
  'Brutally honest -- 14 months of inflated numbers behind them. No method without a real cited number. If the only quality-preserving 2x needs a finetuned/co-designed model the customer must supply, say that plainly.\n\nDOSSIER:\n' + dossier
const product = await agent(SYN, { label: 'synthesis:math-product', phase: 'Synthesize' })
return { research: out, product }
