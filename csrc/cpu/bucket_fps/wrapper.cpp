#include "wrapper.h"
#include "dynamic/KDLineTree.h"
#include "static/KDLineTree.h"
#include <array>
#include <memory>
#include <torch/extension.h>
#include <utility>
#include <vector>

#ifndef BUCKET_FPS_MAX_DIM
#define BUCKET_FPS_MAX_DIM 8
#endif
constexpr size_t max_dim = BUCKET_FPS_MAX_DIM;

using quickfps::KDLineTree;
using quickfps::Point;

template <typename T, typename S>
using DynPoint = quickfps::dynamic::Point<T, S>;

//////////////////
//              //
//    Static    //
//              //
//////////////////

template <typename T, size_t DIM, typename S>
std::vector<Point<T, DIM, S>> raw_data_to_points(const float *raw_data,
                                                 size_t n_points, size_t dim) {
    std::vector<Point<T, DIM, S>> points;
    points.reserve(n_points);
    for (size_t i = 0; i < n_points; i++) {
        const float *ptr = raw_data + i * dim;
        points.push_back(Point<T, DIM, S>(ptr, i));
    }
    return points;
}

template <typename T, size_t DIM, typename S = T>
void kdline_sample(const float *raw_data, size_t n_points, size_t dim,
                   size_t n_samples, size_t start_idx, size_t height,
                   int64_t *sampled_point_indices) {
    auto points = raw_data_to_points<T, DIM, S>(raw_data, n_points, dim);
    auto sampled_points = std::make_unique<Point<T, DIM, S>[]>(n_samples);
    KDLineTree<T, DIM, S> tree(points.data(), n_points, height,
                               sampled_points.get());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

///////////////////
//               //
//    Dynamic    //
//               //
///////////////////

template <typename T, typename S>
std::vector<DynPoint<T, S>>
raw_data_to_points_varlen(const float *raw_data, size_t n_points, size_t dim) {
    std::vector<DynPoint<T, S>> points;
    points.reserve(n_points);
    for (size_t i = 0; i < n_points; i++) {
        const float *ptr = raw_data + i * dim;
        points.push_back(DynPoint<T, S>(std::vector<T>(ptr, ptr + dim), i));
    }
    return points;
}

template <typename T, typename S = T>
void kdline_sample_varlen(const float *raw_data, size_t n_points, size_t dim,
                          size_t n_samples, size_t start_idx, size_t height,
                          int64_t *sampled_point_indices) {
    auto points = raw_data_to_points_varlen<T, S>(raw_data, n_points, dim);
    auto sampled_points =
        std::vector<DynPoint<T, S>>(n_samples, DynPoint<T, S>(dim));
    quickfps::dynamic::KDLineTree<T, S> tree(points.data(), n_points, height,
                                             sampled_points.data());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

////////////////////////////////////////
//                                    //
//    Compile Time Function Helper    //
//                                    //
////////////////////////////////////////
using KDLineFuncType = void (*)(const float *, size_t, size_t, size_t, size_t,
                                size_t, int64_t *);

template <typename T, size_t Count, typename M, size_t... I>
constexpr std::array<T, Count> mapIndices(M &&m, std::index_sequence<I...>) {
    std::array<T, Count> result { m.template operator()<I + 1>()... };
    return result;
}

template <typename T, size_t Count, typename M>
constexpr std::array<T, Count> map(M m) {
    return mapIndices<T, Count>(m, std::make_index_sequence<Count>());
}

template <typename T, typename S = T> struct kdline_func_helper {
    template <size_t DIM> KDLineFuncType operator()() {
        return &kdline_sample<T, DIM, S>;
    }
};

/////////////////
//             //
//     API     //
//             //
/////////////////

void bucket_fps_kdline(const float *raw_data, size_t n_points, size_t dim,
                       size_t n_samples, size_t start_idx, size_t height,
                       int64_t *sampled_point_indices) {
    TORCH_CHECK(dim > 0, "dim should be larger than 0");
    TORCH_CHECK(n_points != 0, "n_points should be larger than 0");
    TORCH_CHECK(n_samples != 0, "n_samples should be larger than 0");
    TORCH_CHECK(height != 0, "height should be larger than 0");
    TORCH_CHECK(start_idx < n_points,
                "start_idx should be smaller than n_points");
    if (dim <= max_dim) {
        auto func_arr =
            map<KDLineFuncType, max_dim>(kdline_func_helper<float>{});
        func_arr[dim - 1](raw_data, n_points, dim, n_samples, start_idx, height,
                          sampled_point_indices);
    } else {
        // var dim
        kdline_sample_varlen<float>(raw_data, n_points, dim, n_samples,
                                    start_idx, height, sampled_point_indices);
    }
}
