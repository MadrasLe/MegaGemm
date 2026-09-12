#include <torch/extension.h>

#include "../src/mlp_prefill_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "cublaslt_bf16_algorithm_count_cuda",
        &cublaslt_bf16_algorithm_count_cuda,
        "Return available cuBLASLt BF16 heuristics for input @ weight.T"
    );
    module.def(
        "cublaslt_bf16_linear_cuda",
        &cublaslt_bf16_linear_cuda,
        "cuBLASLt BF16 linear with an explicit heuristic index"
    );
}
