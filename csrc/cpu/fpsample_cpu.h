#pragma once

#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> sample_cpu(const torch::Tensor &x, int64_t k, torch::optional<int64_t> h,
                                                    torch::optional<int64_t> start_idx, c10::string_view backend);
