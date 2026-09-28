#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <cstdio>
#include <cstddef>

int main() {
    using R = at::native::ReduceOp<float,
                                    at::native::MeanOps<float, float, float, float>,
                                    unsigned int, float, 4>;
    using RC = at::native::ReduceConfig;
    printf("sizeof(ReduceConfig)            = %zu\n", sizeof(RC));
    printf("offsetof(ReduceConfig, num_inputs)  = %zu\n", offsetof(RC, num_inputs));
    printf("offsetof(ReduceConfig, num_outputs) = %zu\n", offsetof(RC, num_outputs));
    printf("offsetof(R, config)             = %zu\n", offsetof(R, config));
    printf("→ R.config.num_inputs  is at byte  = %zu\n",
        offsetof(R, config) + offsetof(RC, num_inputs));
    printf("→ R.config.num_outputs is at byte  = %zu\n",
        offsetof(R, config) + offsetof(RC, num_outputs));
    return 0;
}
