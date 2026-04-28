#include <torch/library.h>

#include "fpsample_cuda_kernel.cuh"

using torch::Tensor;

std::tuple<Tensor, Tensor> sample_cuda(const Tensor &x, int64_t k,
                                       c10::optional<int64_t> h,
                                       c10::optional<int64_t> start_idx) {
    return sample_cuda_impl(x, k, h, start_idx);
}

TORCH_LIBRARY_IMPL(torch_fpsample, CUDA, m) { m.impl("sample", &sample_cuda); }
