#pragma once

#include "fpn_types.h"

template<typename T, int TILE_SIZE, int CHANNELS_PER_THREAD>
__global__ void unified_fpn_kernel(
    const T* __restrict__ backbone_features,
    const T* __restrict__ lateral_weights,
    const T* __restrict__ output_weights,
    T* __restrict__ output_features,
    const FPNLevelConfig* level_configs,
    const FPNKernelConfig kernel_config,
    int level,
    int batch_size,
    int output_channels
);

template<typename T, int TILE_SIZE>
__global__ void lateral_conv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
);

template<typename T, int TILE_SIZE>
__global__ void bilinear_upsample_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x
);

template<typename T, int TILE_SIZE, int KERNEL_SIZE>
__global__ void depthwise_conv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int padding
);

template<typename T>
__global__ void batch_norm_kernel(
    T* __restrict__ input_output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ running_mean,
    const T* __restrict__ running_var,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
);

template<typename T, FPNActivation ACTIVATION>
__global__ void activation_kernel(
    T* __restrict__ input_output,
    int total_elements,
    float alpha = 0.01f
);

template<typename T>
__global__ void element_wise_add_kernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    T* __restrict__ output,
    int total_elements
);

__device__ __forceinline__ float4 load_float4_aligned(const float* ptr);
__device__ __forceinline__ void store_float4_aligned(float* ptr, float4 val);
__device__ __forceinline__ float bilinear_interpolate(
    const float* __restrict__ input,
    int height, int width, int channels,
    float y, float x, int c
);
__device__ __forceinline__ float4 warp_reduce_sum(float4 val);

template<int TILE_SIZE>
__device__ __forceinline__ void cooperative_copy_to_shared(
    const float* __restrict__ global_mem,
    float* __restrict__ shared_mem,
    int elements_per_thread,
    int total_elements
);

template<int TILE_SIZE, int CHANNELS_PER_THREAD>
__device__ __forceinline__ void prefetch_feature_tile(
    const float* __restrict__ input,
    int channels, int height, int width,
    int tile_y, int tile_x,
    int thread_y, int thread_x
);