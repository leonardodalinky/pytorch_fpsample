// Refactored by AyajiLin on 2023/9/16.

#pragma once
#include <cstddef>
#include <type_traits>

namespace quickfps {
using ssize_t = std::make_signed_t<size_t>;

template <typename T>
inline constexpr T powi(const T base, const size_t exponent) {
    // (parentheses not required in next line)
    return (exponent == 0) ? 1 : (base * powi(base, exponent - 1));
}
} // namespace quickfps
