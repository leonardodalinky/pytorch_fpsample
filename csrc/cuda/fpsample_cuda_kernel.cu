#include "fpsample_cuda_kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils.h"

namespace cg = cooperative_groups;

namespace {

constexpr float kInitialDistance = 1.0e20f;
constexpr int kBuildBlock = 256;
constexpr int kSampleBlock = 512;

struct BoundsSum {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    float sum_x, sum_y, sum_z;
};

__device__ __forceinline__ BoundsSum emptyBoundsSum() {
    return {1.0e30f, 1.0e30f, 1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ BoundsSum pointBoundsSum(const float4 &p) {
    return {p.x, p.y, p.z, p.x, p.y, p.z, p.x, p.y, p.z};
}

__device__ __forceinline__ BoundsSum mergeBoundsSum(const BoundsSum &a, const BoundsSum &b) {
    return {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y), fminf(a.min_z, b.min_z),
            fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z),
            a.sum_x + b.sum_x, a.sum_y + b.sum_y, a.sum_z + b.sum_z};
}

__device__ __forceinline__ float coordOf(const float4 &p, const int dim) {
    return dim == 0 ? p.x : (dim == 1 ? p.y : p.z);
}

__device__ __forceinline__ float dist2(const float4 a, const float4 b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float bboxMinDist2(const float4 p, const float3 boxMax, const float3 boxMin) {
    const float dx_hi = fmaxf(p.x, boxMax.x) - boxMax.x;
    const float dx_lo = boxMin.x - fminf(p.x, boxMin.x);
    const float dy_hi = fmaxf(p.y, boxMax.y) - boxMax.y;
    const float dy_lo = boxMin.y - fminf(p.y, boxMin.y);
    const float dz_hi = fmaxf(p.z, boxMax.z) - boxMax.z;
    const float dz_lo = boxMin.z - fminf(p.z, boxMin.z);
    return dx_hi * dx_hi + dx_lo * dx_lo + dy_hi * dy_hi + dy_lo * dy_lo + dz_hi * dz_hi + dz_lo * dz_lo;
}

__device__ __forceinline__ bool betterPair(const float lhsDist, const int lhsIdx,
                                           const float rhsDist, const int rhsIdx) {
    return lhsIdx >= 0 && (rhsIdx < 0 || lhsDist > rhsDist);
}

template <int BLOCK_SIZE>
__device__ void reduceBlockMax(float &bestDist, int &bestIdx) {
    __shared__ float sharedDist[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];
    const int tid = threadIdx.x;
    sharedDist[tid] = bestDist;
    sharedIdx[tid] = bestIdx;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && betterPair(sharedDist[tid + stride], sharedIdx[tid + stride], sharedDist[tid], sharedIdx[tid])) {
            sharedDist[tid] = sharedDist[tid + stride];
            sharedIdx[tid] = sharedIdx[tid + stride];
        }
        __syncthreads();
    }
    bestDist = sharedDist[0];
    bestIdx = sharedIdx[0];
}

__global__ void packPointsKernel(const float *x, float4 *points, int batch_size, int point_count, int dim) {
    const int total = batch_size * point_count;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int local = i % point_count;
        const int base = i * dim;
        points[i] = make_float4(x[base], x[base + 1], x[base + 2], static_cast<float>(local));
    }
}

__global__ void initRootKernel(int *bucket_offset, int *bucket_length, int batch_size, int point_count, int bucket_count) {
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < batch_size) {
        bucket_offset[batch * bucket_count] = batch * point_count;
        bucket_length[batch * bucket_count] = point_count;
    }
}

__global__ void snapshotKernel(const int *bucket_offset, const int *bucket_length, int batch_size,
                               int bucket_count, int active_buckets, int *old_offset, int *old_length) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * active_buckets;
    if (task >= total) return;
    const int batch = task / active_buckets;
    const int bucket = task - batch * active_buckets;
    const int slot = batch * bucket_count + bucket;
    old_offset[slot] = bucket_offset[slot];
    old_length[slot] = bucket_length[slot];
}

__global__ void splitMetadataKernel(const float4 *points, const int *bucket_offset, const int *bucket_length,
                                    int batch_size, int bucket_count, int active_buckets,
                                    int *split_dim, float *split_value) {
    extern __shared__ unsigned char raw[];
    BoundsSum *shared = reinterpret_cast<BoundsSum *>(raw);
    const int task = blockIdx.x;
    const int batch = task / active_buckets;
    const int bucket = task - batch * active_buckets;
    if (batch >= batch_size) return;
    const int slot = batch * bucket_count + bucket;
    const int offset = bucket_offset[slot];
    const int len = bucket_length[slot];
    BoundsSum local = emptyBoundsSum();
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        local = mergeBoundsSum(local, pointBoundsSum(points[offset + i]));
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = mergeBoundsSum(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const BoundsSum v = shared[0];
        const float rx = v.max_x - v.min_x;
        const float ry = v.max_y - v.min_y;
        const float rz = v.max_z - v.min_z;
        int dim = 0;
        float split = len > 0 ? v.sum_x / static_cast<float>(len) : 0.0f;
        if (ry > rx && ry > rz) {
            dim = 1;
            split = len > 0 ? v.sum_y / static_cast<float>(len) : 0.0f;
        }
        if (rz > rx && rz > ry) {
            dim = 2;
            split = len > 0 ? v.sum_z / static_cast<float>(len) : 0.0f;
        }
        split_dim[slot] = dim;
        split_value[slot] = split;
    }
}

__global__ void countSplitKernel(const float4 *points, const int *bucket_offset, const int *bucket_length,
                                 int batch_size, int bucket_count, int active_buckets,
                                 const int *split_dim, const float *split_value, int *left_count) {
    extern __shared__ int shared[];
    const int task = blockIdx.x;
    const int batch = task / active_buckets;
    const int bucket = task - batch * active_buckets;
    if (batch >= batch_size) return;
    const int slot = batch * bucket_count + bucket;
    const int offset = bucket_offset[slot];
    const int len = bucket_length[slot];
    const int dim = split_dim[slot];
    const float value = split_value[slot];
    int local = 0;
    for (int i = threadIdx.x; i < len; i += blockDim.x) local += coordOf(points[offset + i], dim) <= value ? 1 : 0;
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int left = shared[0];
        if (len > 1) left = max(1, min(left, len - 1));
        left_count[slot] = left;
    }
}

__global__ void prepareChildrenKernel(const int *old_offset, const int *old_length, int batch_size,
                                      int bucket_count, int active_buckets, const int *left_count,
                                      int *bucket_offset, int *bucket_length, int *left_cursor, int *right_cursor) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * active_buckets;
    if (task >= total) return;
    const int batch = task / active_buckets;
    const int bucket = task - batch * active_buckets;
    const int slot = batch * bucket_count + bucket;
    const int right_slot = slot + active_buckets;
    const int offset = old_offset[slot];
    const int len = old_length[slot];
    const int left = left_count[slot];
    bucket_offset[slot] = offset;
    bucket_length[slot] = left;
    bucket_offset[right_slot] = offset + left;
    bucket_length[right_slot] = len - left;
    left_cursor[slot] = offset;
    right_cursor[slot] = offset + left;
}

__global__ void scatterSplitKernel(const float4 *src, float4 *dst, const int *old_offset, const int *old_length,
                                   int batch_size, int bucket_count, int active_buckets,
                                   const int *split_dim, const float *split_value,
                                   int *left_cursor, int *right_cursor) {
    const int task = blockIdx.x;
    const int batch = task / active_buckets;
    const int bucket = task - batch * active_buckets;
    if (batch >= batch_size) return;
    const int slot = batch * bucket_count + bucket;
    const int offset = old_offset[slot];
    const int len = old_length[slot];
    const int dim = split_dim[slot];
    const float value = split_value[slot];
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const float4 p = src[offset + i];
        const bool left = coordOf(p, dim) <= value;
        const int out = left ? atomicAdd(&left_cursor[slot], 1) : atomicAdd(&right_cursor[slot], 1);
        dst[out] = p;
    }
}

__global__ void bboxKernel(const float4 *points, const int *bucket_offset, const int *bucket_length,
                           int total_buckets, float3 *bbox_max, float3 *bbox_min) {
    extern __shared__ unsigned char raw[];
    BoundsSum *shared = reinterpret_cast<BoundsSum *>(raw);
    const int slot = blockIdx.x;
    if (slot >= total_buckets) return;
    const int offset = bucket_offset[slot];
    const int len = bucket_length[slot];
    BoundsSum local = emptyBoundsSum();
    for (int i = threadIdx.x; i < len; i += blockDim.x) local = mergeBoundsSum(local, pointBoundsSum(points[offset + i]));
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = mergeBoundsSum(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const BoundsSum v = shared[0];
        bbox_min[slot] = make_float3(v.min_x, v.min_y, v.min_z);
        bbox_max[slot] = make_float3(v.max_x, v.max_y, v.max_z);
    }
}

__global__ void findStartKernel(const float4 *points, int batch_size, int point_count, int start_idx, int *start_positions) {
    const int batch = blockIdx.x;
    if (batch >= batch_size) return;
    const int point_base = batch * point_count;
    for (int i = threadIdx.x; i < point_count; i += blockDim.x) {
        if (static_cast<int>(points[point_base + i].w) == start_idx) {
            start_positions[batch] = point_base + i;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void sampleKernel(const float4 *points, const int *bucket_offset, const int *bucket_length,
                             const float3 *bbox_max, const float3 *bbox_min, int batch_size,
                             int point_count, int bucket_count, int sample_count, int start_idx,
                             const int *start_positions,
                             int blocks_per_batch, float *temp, int *bucket_far_idx,
                             float *bucket_far_dist, int *next_idx, int64_t *out_idx) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int batch = blockIdx.x / blocks_per_batch;
    const int batch_block = blockIdx.x - batch * blocks_per_batch;
    if (batch >= batch_size) return;
    const int point_base = batch * point_count;
    const int bucket_base = batch * bucket_count;
    for (int i = batch_block * BLOCK_SIZE + tid; i < point_count; i += blocks_per_batch * BLOCK_SIZE) {
        temp[point_base + i] = kInitialDistance;
    }
    for (int b = batch_block * BLOCK_SIZE + tid; b < bucket_count; b += blocks_per_batch * BLOCK_SIZE) {
        const int slot = bucket_base + b;
        bucket_far_idx[slot] = bucket_length[slot] > 0 ? bucket_offset[slot] : -1;
        bucket_far_dist[slot] = bucket_length[slot] > 0 ? kInitialDistance : -1.0f;
    }
    if (batch_block == 0 && tid == 0) {
        next_idx[batch] = start_positions[batch];
        out_idx[batch * sample_count] = start_idx;
    }
    grid.sync();

    for (int sample = 1; sample < sample_count; ++sample) {
        const int origin_idx = next_idx[batch];
        const float4 origin = points[origin_idx];
        for (int b = batch_block; b < bucket_count; b += blocks_per_batch) {
            const int slot = bucket_base + b;
            const int len = bucket_length[slot];
            if (len <= 0) continue;
            const int far_idx = bucket_far_idx[slot];
            const float last_dist = bucket_far_dist[slot];
            bool need_scan = true;
            if (far_idx >= 0 && last_dist < kInitialDistance * 0.5f) {
                const float cur_dist = dist2(origin, points[far_idx]);
                const float bound_dist = bboxMinDist2(origin, bbox_max[slot], bbox_min[slot]);
                need_scan = (cur_dist <= last_dist || bound_dist < last_dist);
            }
            if (need_scan) {
                const int offset = bucket_offset[slot];
                float best_dist = -1.0f;
                int best_idx = -1;
                for (int local = tid; local < len; local += BLOCK_SIZE) {
                    const int point_idx = offset + local;
                    const float updated = fminf(dist2(points[point_idx], origin), temp[point_idx]);
                    temp[point_idx] = updated;
                    if (betterPair(updated, point_idx, best_dist, best_idx)) {
                        best_dist = updated;
                        best_idx = point_idx;
                    }
                }
                reduceBlockMax<BLOCK_SIZE>(best_dist, best_idx);
                if (tid == 0) {
                    bucket_far_dist[slot] = best_dist;
                    bucket_far_idx[slot] = best_idx;
                }
                __syncthreads();
            }
            __syncthreads();
        }
        grid.sync();
        if (batch_block == 0) {
            float final_dist = -1.0f;
            int final_idx = -1;
            for (int b = tid; b < bucket_count; b += BLOCK_SIZE) {
                const int slot = bucket_base + b;
                if (betterPair(bucket_far_dist[slot], bucket_far_idx[slot], final_dist, final_idx)) {
                    final_dist = bucket_far_dist[slot];
                    final_idx = bucket_far_idx[slot];
                }
            }
            reduceBlockMax<BLOCK_SIZE>(final_dist, final_idx);
            if (tid == 0) {
                next_idx[batch] = final_idx;
                out_idx[batch * sample_count + sample] = static_cast<int64_t>(points[final_idx].w);
            }
        }
        grid.sync();
    }
}

void check_cuda(cudaError_t err, const char *msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> sample_cuda_impl(
    const torch::Tensor &x, int64_t k, c10::optional<int64_t> h,
    c10::optional<int64_t> start_idx) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor, but found on ", x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);
    TORCH_CHECK(x.size(-1) >= 3, "CUDA FPS currently requires point dim >= 3, but got ", x.size(-1));
    TORCH_CHECK(k <= x.size(-2), "k must be <= number of points");
    const auto old_size = x.sizes().vec();
    auto x_reshaped_raw = x.dim() > 2 ? x.view({-1, x.size(-2), x.size(-1)}) : x.view({1, x.size(0), x.size(1)});
    auto x_reshaped = x_reshaped_raw.to(torch::kFloat32).contiguous();
    const int batch_size = static_cast<int>(x_reshaped.size(0));
    const int point_count = static_cast<int>(x_reshaped.size(1));
    const int dim = static_cast<int>(x_reshaped.size(2));
    const int tree_high = static_cast<int>(h.value_or(4));
    TORCH_CHECK(tree_high >= 0 && tree_high <= 10, "CUDA FPS tree height must be in [0, 10]");
    const int bucket_count = 1 << tree_high;
    const int start = static_cast<int>(start_idx.value_or(0));
    TORCH_CHECK(start >= 0 && start < point_count, "start_idx out of range");

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto float_opts = x_reshaped.options().dtype(torch::kFloat32);
    auto int_opts = x_reshaped.options().dtype(torch::kInt32);
    auto long_opts = x_reshaped.options().dtype(torch::kLong);

    auto points = torch::empty({batch_size, point_count, 4}, float_opts);
    auto bucket_offset = torch::empty({batch_size, bucket_count}, int_opts);
    auto bucket_length = torch::empty({batch_size, bucket_count}, int_opts);
    auto bbox_max_t = torch::empty({batch_size, bucket_count, 3}, float_opts);
    auto bbox_min_t = torch::empty({batch_size, bucket_count, 3}, float_opts);
    auto temp_points = torch::empty_like(points);
    auto split_dim = torch::empty({batch_size, bucket_count}, int_opts);
    auto split_value = torch::empty({batch_size, bucket_count}, float_opts);
    auto left_count = torch::empty({batch_size, bucket_count}, int_opts);
    auto left_cursor = torch::empty({batch_size, bucket_count}, int_opts);
    auto right_cursor = torch::empty({batch_size, bucket_count}, int_opts);
    auto old_offset = torch::empty({batch_size, bucket_count}, int_opts);
    auto old_length = torch::empty({batch_size, bucket_count}, int_opts);

    const int total_points = batch_size * point_count;
    packPointsKernel<<<(total_points + 255) / 256, 256, 0, stream>>>(
        x_reshaped.data_ptr<float>(), reinterpret_cast<float4 *>(points.data_ptr<float>()),
        batch_size, point_count, dim);
    initRootKernel<<<(batch_size + 255) / 256, 256, 0, stream>>>(
        bucket_offset.data_ptr<int>(), bucket_length.data_ptr<int>(), batch_size, point_count, bucket_count);

    float4 *src = reinterpret_cast<float4 *>(points.data_ptr<float>());
    float4 *dst = reinterpret_cast<float4 *>(temp_points.data_ptr<float>());
    bool src_is_points = true;
    const std::size_t bounds_shared = kBuildBlock * sizeof(BoundsSum);
    for (int level = 0; level < tree_high; ++level) {
        const int active = 1 << level;
        const int active_slots = batch_size * active;
        snapshotKernel<<<(active_slots + 255) / 256, 256, 0, stream>>>(
            bucket_offset.data_ptr<int>(), bucket_length.data_ptr<int>(), batch_size, bucket_count, active,
            old_offset.data_ptr<int>(), old_length.data_ptr<int>());
        splitMetadataKernel<<<active_slots, kBuildBlock, bounds_shared, stream>>>(
            src, old_offset.data_ptr<int>(), old_length.data_ptr<int>(), batch_size, bucket_count, active,
            split_dim.data_ptr<int>(), split_value.data_ptr<float>());
        countSplitKernel<<<active_slots, kBuildBlock, kBuildBlock * sizeof(int), stream>>>(
            src, old_offset.data_ptr<int>(), old_length.data_ptr<int>(), batch_size, bucket_count, active,
            split_dim.data_ptr<int>(), split_value.data_ptr<float>(), left_count.data_ptr<int>());
        prepareChildrenKernel<<<(active_slots + 255) / 256, 256, 0, stream>>>(
            old_offset.data_ptr<int>(), old_length.data_ptr<int>(), batch_size, bucket_count, active,
            left_count.data_ptr<int>(), bucket_offset.data_ptr<int>(), bucket_length.data_ptr<int>(),
            left_cursor.data_ptr<int>(), right_cursor.data_ptr<int>());
        scatterSplitKernel<<<active_slots, kBuildBlock, 0, stream>>>(
            src, dst, old_offset.data_ptr<int>(), old_length.data_ptr<int>(), batch_size, bucket_count, active,
            split_dim.data_ptr<int>(), split_value.data_ptr<float>(), left_cursor.data_ptr<int>(), right_cursor.data_ptr<int>());
        std::swap(src, dst);
        src_is_points = !src_is_points;
    }
    if (!src_is_points) {
        check_cuda(cudaMemcpyAsync(points.data_ptr<float>(), src, points.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                   "copy final axis-split points failed");
        src = reinterpret_cast<float4 *>(points.data_ptr<float>());
    }
    bboxKernel<<<batch_size * bucket_count, kBuildBlock, bounds_shared, stream>>>(
        src, bucket_offset.data_ptr<int>(), bucket_length.data_ptr<int>(), batch_size * bucket_count,
        reinterpret_cast<float3 *>(bbox_max_t.data_ptr<float>()),
        reinterpret_cast<float3 *>(bbox_min_t.data_ptr<float>()));

    int device = x.get_device();
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, device);
    TORCH_CHECK(prop.cooperativeLaunch, "CUDA FPS requires cooperative launch support");
    const int blocks_per_batch = std::max(1, prop.multiProcessorCount / std::max(1, batch_size));
    const int grid_blocks = batch_size * blocks_per_batch;

    auto temp = torch::empty({batch_size, point_count}, float_opts);
    auto bucket_far_idx = torch::empty({batch_size, bucket_count}, int_opts);
    auto bucket_far_dist = torch::empty({batch_size, bucket_count}, float_opts);
    auto next_idx = torch::empty({batch_size}, int_opts);
    auto start_positions = torch::empty({batch_size}, int_opts);
    auto ret_indices = torch::empty({batch_size, k}, long_opts);
    findStartKernel<<<batch_size, 256, 0, stream>>>(
        src, batch_size, point_count, start, start_positions.data_ptr<int>());

    // cudaLaunchCooperativeKernel needs stable lvalue pointers for arguments.
    const int *bucket_offset_ptr = bucket_offset.data_ptr<int>();
    const int *bucket_length_ptr = bucket_length.data_ptr<int>();
    const float3 *bbox_max_ptr = reinterpret_cast<const float3 *>(bbox_max_t.data_ptr<float>());
    const float3 *bbox_min_ptr = reinterpret_cast<const float3 *>(bbox_min_t.data_ptr<float>());
    float *temp_ptr = temp.data_ptr<float>();
    int *bucket_far_idx_ptr = bucket_far_idx.data_ptr<int>();
    float *bucket_far_dist_ptr = bucket_far_dist.data_ptr<float>();
    int *next_idx_ptr = next_idx.data_ptr<int>();
    const int *start_positions_ptr = start_positions.data_ptr<int>();
    int64_t *ret_indices_ptr = ret_indices.data_ptr<int64_t>();
    int sample_count = static_cast<int>(k);
    void *kernel_args[] = {
        (void *)&src,
        (void *)&bucket_offset_ptr,
        (void *)&bucket_length_ptr,
        (void *)&bbox_max_ptr,
        (void *)&bbox_min_ptr,
        (void *)&batch_size,
        (void *)&point_count,
        (void *)&bucket_count,
        (void *)&sample_count,
        (void *)&start,
        (void *)&start_positions_ptr,
        (void *)&blocks_per_batch,
        (void *)&temp_ptr,
        (void *)&bucket_far_idx_ptr,
        (void *)&bucket_far_dist_ptr,
        (void *)&next_idx_ptr,
        (void *)&ret_indices_ptr,
    };
    check_cuda(cudaLaunchCooperativeKernel(
                   (void *)sampleKernel<kSampleBlock>, dim3(grid_blocks), dim3(kSampleBlock),
                   kernel_args, 0, stream),
               "launch CUDA FPS sample kernel failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto ret_tensor = torch::gather(
        x_reshaped_raw, 1,
        ret_indices.view({ret_indices.size(0), ret_indices.size(1), 1}).repeat({1, 1, x_reshaped_raw.size(2)}));

    auto ret_tensor_sizes = old_size;
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
    auto ret_indices_sizes = old_size;
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;
    return std::make_tuple(ret_tensor.view(ret_tensor_sizes), ret_indices.view(ret_indices_sizes));
}
