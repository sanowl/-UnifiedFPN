#include "../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <texture_fetch_functions.h>

// Optimized bilinear upsampling with texture memory and cache-friendly access
template<typename T, int TILE_SIZE, int CHANNELS_PER_THREAD>
__global__ void __launch_bounds__(256, 4)
optimized_bilinear_upsample_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x) {
    
    // Cache-friendly shared memory layout for interpolation weights
    __shared__ __align__(16) float interpolation_cache[TILE_SIZE + 2][TILE_SIZE + 2][4];
    __shared__ __align__(16) T pixel_cache[TILE_SIZE + 2][TILE_SIZE + 2][CHANNELS_PER_THREAD];
    
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * CHANNELS_PER_THREAD;
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tc = threadIdx.z;
    
    const int out_y = tile_y * TILE_SIZE + ty;
    const int out_x = tile_x * TILE_SIZE + tx;
    const int channel = channel_block + tc;
    
    // Early exit for out-of-bounds
    if (out_y >= out_height || out_x >= out_width || channel >= channels) return;
    
    // Compute source coordinates with subpixel precision
    const float src_y = (out_y + 0.5f) * scale_y - 0.5f;
    const float src_x = (out_x + 0.5f) * scale_x - 0.5f;
    
    // Hardware-accelerated coordinate calculation
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    // Clamp to valid range
    const int y0_clamped = max(0, y0);
    const int x0_clamped = max(0, x0);
    
    // Compute interpolation weights using FMA
    const float dy = src_y - __int2float_rn(y0);
    const float dx = src_x - __int2float_rn(x0);
    const float dy_inv = 1.0f - dy;
    const float dx_inv = 1.0f - dx;
    
    // Precompute bilinear weights
    const float w00 = dx_inv * dy_inv;
    const float w01 = dx * dy_inv;
    const float w10 = dx_inv * dy;
    const float w11 = dx * dy;
    
    // Cache interpolation weights in shared memory
    if (tc == 0) {
        interpolation_cache[ty][tx][0] = w00;
        interpolation_cache[ty][tx][1] = w01;
        interpolation_cache[ty][tx][2] = w10;
        interpolation_cache[ty][tx][3] = w11;
    }
    
    // Vectorized memory access for better bandwidth utilization
    const int input_base = batch_idx * channels * in_height * in_width;
    const int stride = in_width * channels;
    
    // Load 4 neighboring pixels with coalesced access
    const int idx00 = input_base + y0_clamped * stride + x0_clamped * channels + channel;
    const int idx01 = input_base + y0_clamped * stride + min(x1, in_width - 1) * channels + channel;
    const int idx10 = input_base + min(y1, in_height - 1) * stride + x0_clamped * channels + channel;
    const int idx11 = input_base + min(y1, in_height - 1) * stride + min(x1, in_width - 1) * channels + channel;
    
    // Use texture cache-friendly loads
    float v00, v01, v10, v11;
    
    if constexpr (std::is_same_v<T, float>) {
        v00 = __ldg(&input[idx00]);
        v01 = __ldg(&input[idx01]);
        v10 = __ldg(&input[idx10]);
        v11 = __ldg(&input[idx11]);
    } else {
        v00 = __half2float(__ldg(&input[idx00]));
        v01 = __half2float(__ldg(&input[idx01]));
        v10 = __half2float(__ldg(&input[idx10]));
        v11 = __half2float(__ldg(&input[idx11]));
    }
    
    __syncthreads();
    
    // Retrieve cached weights
    const float cached_w00 = interpolation_cache[ty][tx][0];
    const float cached_w01 = interpolation_cache[ty][tx][1];
    const float cached_w10 = interpolation_cache[ty][tx][2];
    const float cached_w11 = interpolation_cache[ty][tx][3];
    
    // Optimized bilinear interpolation using FMA
    float result = 0.0f;
    result = __fmaf_rn(v00, cached_w00, result);
    result = __fmaf_rn(v01, cached_w01, result);
    result = __fmaf_rn(v10, cached_w10, result);
    result = __fmaf_rn(v11, cached_w11, result);
    
    // Write output with coalesced pattern
    const int output_idx = batch_idx * channels * out_height * out_width +
                          out_y * out_width * channels + out_x * channels + channel;
    
    output[output_idx] = static_cast<T>(result);
}

// Specialized kernel for channels that are multiples of 4 (vectorized)
template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(256, 4)
vectorized_bilinear_upsample_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x) {
    
    // Process 4 channels at once for better memory throughput
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * 4; // Process 4 channels per block
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    const int out_y = tile_y * TILE_SIZE + ty;
    const int out_x = tile_x * TILE_SIZE + tx;
    
    if (out_y >= out_height || out_x >= out_width || channel_block >= channels) return;
    
    // Compute interpolation coordinates
    const float src_y = (out_y + 0.5f) * scale_y - 0.5f;
    const float src_x = (out_x + 0.5f) * scale_x - 0.5f;
    
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    const int y0_clamped = max(0, y0);
    const int x0_clamped = max(0, x0);
    
    const float dy = src_y - __int2float_rn(y0);
    const float dx = src_x - __int2float_rn(x0);
    
    // Vectorized pixel loading (4 channels at once)
    const int input_base = batch_idx * channels * in_height * in_width;
    const int stride = in_width * channels;
    
    // Load 4 neighboring pixels for 4 channels
    if constexpr (std::is_same_v<T, float> && (channels % 4 == 0)) {
        const int base_idx = input_base + y0_clamped * stride + x0_clamped * channels + channel_block;
        
        float4 v00 = __ldg(reinterpret_cast<const float4*>(&input[base_idx]));
        float4 v01 = __ldg(reinterpret_cast<const float4*>(&input[base_idx + (min(x1, in_width - 1) - x0_clamped) * channels]));
        float4 v10 = __ldg(reinterpret_cast<const float4*>(&input[base_idx + (min(y1, in_height - 1) - y0_clamped) * stride]));
        float4 v11 = __ldg(reinterpret_cast<const float4*>(&input[base_idx + (min(y1, in_height - 1) - y0_clamped) * stride + (min(x1, in_width - 1) - x0_clamped) * channels]));
        
        // Bilinear interpolation for all 4 channels
        const float w00 = (1.0f - dx) * (1.0f - dy);
        const float w01 = dx * (1.0f - dy);
        const float w10 = (1.0f - dx) * dy;
        const float w11 = dx * dy;
        
        float4 result;
        result.x = __fmaf_rn(v00.x, w00, __fmaf_rn(v01.x, w01, __fmaf_rn(v10.x, w10, v11.x * w11)));
        result.y = __fmaf_rn(v00.y, w00, __fmaf_rn(v01.y, w01, __fmaf_rn(v10.y, w10, v11.y * w11)));
        result.z = __fmaf_rn(v00.z, w00, __fmaf_rn(v01.z, w01, __fmaf_rn(v10.z, w10, v11.z * w11)));
        result.w = __fmaf_rn(v00.w, w00, __fmaf_rn(v01.w, w01, __fmaf_rn(v10.w, w10, v11.w * w11)));
        
        // Write vectorized output
        const int output_idx = batch_idx * channels * out_height * out_width +
                              out_y * out_width * channels + out_x * channels + channel_block;
        
        *reinterpret_cast<float4*>(&output[output_idx]) = result;
    }
}

// Template instantiations
template __global__ void optimized_bilinear_upsample_kernel<float, 16, 8>(
    const float*, float*, int, int, int, int, int, int, float, float);

template __global__ void optimized_bilinear_upsample_kernel<half, 16, 8>(
    const half*, half*, int, int, int, int, int, int, float, float);

template __global__ void vectorized_bilinear_upsample_kernel<float, 16>(
    const float*, float*, int, int, int, int, int, int, float, float);