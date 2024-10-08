#pragma once

#include <torch/extension.h>

inline std::tuple<torch::IntArrayRef, torch::Tensor>
bnorm_reshape(const torch::Tensor &t) {
    if (t.dim() > 2) {
        // reshape to (..., rows, cols)
        return {t.sizes(), t.view({-1, t.size(-2), t.size(-1)})};
    } else if (t.dim() == 2) {
        // reshape to (1, rows, cols)
        return {t.sizes(), t.view({1, t.size(0), t.size(1)})};
    } else {
        TORCH_CHECK(false,
                    "x must have at least 2 dims, but got size: ", t.sizes());
    }
}
