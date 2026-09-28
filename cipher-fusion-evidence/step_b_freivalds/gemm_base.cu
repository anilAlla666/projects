// Baseline CUTLASS sm90 tensor-core GEMM, fp16 in / fp32 accum.
// Computes C[M,N] = A[M,K] @ B[K,N], with B passed as the PyTorch Linear weight
// W[N,K] row-major (== B[K,N] column-major).  C row-major [M,N].
// This is the un-instrumented baseline whose throughput must land near cuBLAS
// for the fused-checksum deltas (added later) to transfer.  READ-ONLY spike: no .so touched.
#include <cuda_runtime.h>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

using ElementA      = cutlass::half_t;
using ElementB      = cutlass::half_t;
using ElementC      = cutlass::half_t;
using ElementAcc    = float;
using ElementCompute= float;

using LayoutA = cutlass::layout::RowMajor;     // A [M,K] row-major
using LayoutB = cutlass::layout::ColumnMajor;  // B [K,N] col-major == W[N,K] row-major
using LayoutC = cutlass::layout::RowMajor;     // C [M,N] row-major

constexpr int AlignA = 8;   // 128-bit / fp16
constexpr int AlignB = 8;
constexpr int AlignC = 8;

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;

using TileShape    = Shape<_128,_256,_64>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementC, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

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
    { reinterpret_cast<const ElementA*>(A), sa,
      reinterpret_cast<const ElementB*>(B), sb },
    { {ElementCompute(1.0f), ElementCompute(0.0f)},
      reinterpret_cast<const ElementC*>(C), sc,
      reinterpret_cast<ElementC*>(C), sd }
  };

  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    printf("can_implement FAILED: %s\n", cutlass::cutlassGetStatusString(st)); return 1;
  }
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_workspace_size) {
    if (g_workspace) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, ws); g_workspace_size = ws;
  }
  st = gemm.initialize(args, g_workspace, stream);
  if (st != cutlass::Status::kSuccess) {
    printf("initialize FAILED: %s\n", cutlass::cutlassGetStatusString(st)); return 2;
  }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) {
    printf("run FAILED: %s\n", cutlass::cutlassGetStatusString(st)); return 3;
  }
  return 0;
}
