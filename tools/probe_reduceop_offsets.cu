#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <cstdio>
#include <cstddef>

int main() {
    using R = at::native::ReduceOp<float,
                                    at::native::MeanOps<float, float, float, float>,
                                    unsigned int, float, 4>;
    printf("sizeof(R)               = %zu\n", sizeof(R));
    printf("offsetof(R, ops)        = %zu\n", offsetof(R, ops));
    printf("offsetof(R, ident)      = %zu\n", offsetof(R, ident));
    printf("offsetof(R, config)     = %zu\n", offsetof(R, config));
    printf("offsetof(R, input_calc) = %zu\n", offsetof(R, input_calc));
    printf("offsetof(R, output_calc)= %zu\n", offsetof(R, output_calc));
    printf("offsetof(R, src)        = %zu\n", offsetof(R, src));
    printf("offsetof(R, dst)        = %zu\n", offsetof(R, dst));
    printf("offsetof(R, acc_buf)    = %zu\n", offsetof(R, acc_buf));
    printf("offsetof(R, cta_buf)    = %zu\n", offsetof(R, cta_buf));
    return 0;
}
