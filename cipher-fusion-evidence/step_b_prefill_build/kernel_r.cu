// Fork (a) -- r-half: fused ABFT reference r[m] = sum_k A[m,k]*wref[k] injected into
// the CUTLASS sm90 WGMMA collective mainloop, reusing A while it is resident in smem.
// Strategy (per advisor): subclass the builder-generated collective, override ONLY mma();
// the cooperative driver / load() / Params / SharedStorage all inherit unchanged, so the
// baseline tensor-core config is preserved.  r computed UNCONDITIONALLY by every CTA to
// scratch (per-CTA work sets wall time -> global index dissolves for the cost measurement).
// Runtime flag g_ck_on gives a clean A/B (r-on vs r-off) on the SAME binary.
// READ-ONLY spike: production .so untouched.
#include <cuda_runtime.h>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

// ---- scratch device globals (set from host via cudaMemcpyToSymbol; no Params plumbing) ----
__device__ float* g_wref;   // [K] reference vector = sum_n W[n,k], fp32, precomputed once
__device__ float* g_rout;   // scratch >= BLK_M floats
__device__ int    g_ck_on;  // 0 = r-off (baseline-equiv), 1 = r-on

using ElementA      = cutlass::half_t;
using ElementB      = cutlass::half_t;
using ElementC      = cutlass::half_t;
using ElementAcc    = float;
using ElementCompute= float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;
using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;
using TileShape    = Shape<_128,_256,_64>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute, ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using BaseMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass, ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// ---- subclass: override mma() with the r-injection; everything else inherited ----
struct CkMainloop : BaseMainloop {
  using BaseMainloop::BaseMainloop;

  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(typename BaseMainloop::MainloopPipeline pipeline,
      typename BaseMainloop::PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      typename BaseMainloop::TensorStorage& shared_tensors,
      typename BaseMainloop::Params const& mainloop_params) {
    using namespace cute;
    using DispatchPolicy = typename BaseMainloop::DispatchPolicy;
    using SmemLayoutA = typename BaseMainloop::SmemLayoutA;
    using SmemLayoutB = typename BaseMainloop::SmemLayoutB;
    using TiledMma    = typename BaseMainloop::TiledMma;
    constexpr int NumThreadsPerWarpGroup = cutlass::NumThreadsPerWarpGroup;
    constexpr int K_PIPE_MMAS = BaseMainloop::K_PIPE_MMAS;
    constexpr int BLK_M = size<0>(TileShape{});
    constexpr int BLK_K = size<2>(TileShape{});

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));
    Tensor tCsA = thread_mma.partition_A(sA);
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    // ---- r-injection state ----
    // mode1 = correct per-row scalar (WG0, one row/thread, swizzled logical read sA(m,k))
    // mode2 = COST-FLOOR diagnostic: efficient vectorized linear physical smem read, all 256 threads
    //         (k-pairing scrambled -> wrong r value, faithful traffic only; for floor timing)
    const int  mode  = g_ck_on;
    const bool do_r  = (mode == 1) && (warp_group_idx == 0) && (thread_idx < BLK_M);
    const int  r_row = thread_idx;          // 0..BLK_M-1 for WG0
    float r_acc = 0.f;
    int   kt = 0;                            // absolute k-tile index
    float* wref = g_wref;
    constexpr int TILE_ELTS = BLK_M * BLK_K;          // 8192 half per stage
    constexpr int N4 = TILE_ELTS / 8;                 // 1024 int4 (128-bit) per stage
    const int4* smemA4 = reinterpret_cast<const int4*>(shared_tensors.smem_A.data());

    PipelineState smem_pipe_release = smem_pipe_read;
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      if (do_r) {
        CUTLASS_PRAGMA_UNROLL
        for (int kl = 0; kl < BLK_K; ++kl) r_acc += float(sA(r_row, kl, read_stage)) * wref[kt*BLK_K + kl];
      } else if (mode == 2) {
        const int4* b4 = smemA4 + read_stage * N4;
        for (int i = thread_idx; i < N4; i += 256) {
          int4 v = b4[i]; const __half2* h = reinterpret_cast<const __half2*>(&v);
          int k0 = kt*BLK_K + ((i*8) & (BLK_K-1));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) { float2 f = __half22float2(h[j]); r_acc += f.x*wref[k0] + f.y*wref[k0]; }
        }
      }
      ++kt;
      ++smem_pipe_read;
    }
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();
      if (do_r) {
        CUTLASS_PRAGMA_UNROLL
        for (int kl = 0; kl < BLK_K; ++kl) r_acc += float(sA(r_row, kl, read_stage)) * wref[kt*BLK_K + kl];
      } else if (mode == 2) {
        const int4* b4 = smemA4 + read_stage * N4;
        for (int i = thread_idx; i < N4; i += 256) {
          int4 v = b4[i]; const __half2* h = reinterpret_cast<const __half2*>(&v);
          int k0 = kt*BLK_K + ((i*8) & (BLK_K-1));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) { float2 f = __half22float2(h[j]); r_acc += f.x*wref[k0] + f.y*wref[k0]; }
        }
      }
      ++kt;
      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    k_tile_count -= prologue_mma_count;
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();
      // ---- INJECT r in the overlap window: after WGMMA issue/commit, BEFORE warpgroup_wait ----
      if (do_r) {
        CUTLASS_PRAGMA_UNROLL
        for (int kl = 0; kl < BLK_K; ++kl) r_acc += float(sA(r_row, kl, read_stage)) * wref[kt*BLK_K + kl];
      } else if (mode == 2) {
        const int4* b4 = smemA4 + read_stage * N4;
        for (int i = thread_idx; i < N4; i += 256) {
          int4 v = b4[i]; const __half2* h = reinterpret_cast<const __half2*>(&v);
          int k0 = kt*BLK_K + ((i*8) & (BLK_K-1));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) { float2 f = __half22float2(h[j]); r_acc += f.x*wref[k0] + f.y*wref[k0]; }
        }
      }
      ++kt;
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_read;
      ++smem_pipe_release;
    }
    warpgroup_fence_operand(accum);

    if (do_r || mode == 2) g_rout[r_row & (BLK_M-1)] = r_acc;   // mode1: correct rows (single-CTA validation); mode2: keep r_acc live
  }
};

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CkMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

extern "C" int set_ck_ptrs(void* wref, void* rout, int ck_on) {
  cudaError_t e1 = cudaMemcpyToSymbol(g_wref, &wref, sizeof(void*));
  cudaError_t e2 = cudaMemcpyToSymbol(g_rout, &rout, sizeof(void*));
  cudaError_t e3 = cudaMemcpyToSymbol(g_ck_on, &ck_on, sizeof(int));
  if (e1||e2||e3) { printf("set_ck_ptrs cudaMemcpyToSymbol failed\n"); return 9; }
  return 0;
}

extern "C" int run_gemm(const void* A, const void* B, void* C,
                        int M, int N, int K, void* stream_v) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    { reinterpret_cast<const ElementA*>(A), sa, reinterpret_cast<const ElementB*>(B), sb },
    { {ElementCompute(1.0f), ElementCompute(0.0f)},
      reinterpret_cast<const ElementC*>(C), sc, reinterpret_cast<ElementC*>(C), sd }
  };
  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { printf("can_implement: %s\n", cutlass::cutlassGetStatusString(st)); return 1; }
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_workspace_size) { if (g_workspace) cudaFree(g_workspace); cudaMalloc(&g_workspace, ws); g_workspace_size = ws; }
  st = gemm.initialize(args, g_workspace, stream);
  if (st != cutlass::Status::kSuccess) { printf("initialize: %s\n", cutlass::cutlassGetStatusString(st)); return 2; }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) { printf("run: %s\n", cutlass::cutlassGetStatusString(st)); return 3; }
  return 0;
}
