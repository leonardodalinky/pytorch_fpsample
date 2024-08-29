#pragma once
#include <cstddef>
#include <cstdint>

void bucket_fps_kdline(const float *raw_data, size_t n_points, size_t dim,
                       size_t n_samples, size_t start_idx, size_t height,
                       int64_t *sampled_point_indices);
