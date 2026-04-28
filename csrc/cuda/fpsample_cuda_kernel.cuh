#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> sample_cuda_impl(
    const torch::Tensor &x, int64_t k, c10::optional<int64_t> h,
    c10::optional<int64_t> start_idx);
