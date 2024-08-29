#if __cplusplus < 201703L
#error "C++17 is required"
#endif

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "cpu/fpsample_cpu.h"

#define STR_(x) #x
#define STR(x) STR_(x)

TORCH_LIBRARY(torch_fpsample, m) {
    m.def("sample(Tensor self, int k, int? h=None, int? start_idx=None, str "
          "backend=\"bucket\") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_fpsample, CPU, m) { m.impl("sample", &sample_cpu); }

PYBIND11_MODULE(_core, m) {
    m.attr("CPP_VERSION") = __cplusplus;
    m.attr("PYTORCH_VERSION") = STR(TORCH_VERSION_MAJOR) "." STR(
        TORCH_VERSION_MINOR) "." STR(TORCH_VERSION_PATCH);
    m.attr("PYBIND11_VERSION") = STR(PYBIND11_VERSION_MAJOR) "." STR(
        PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);
}
