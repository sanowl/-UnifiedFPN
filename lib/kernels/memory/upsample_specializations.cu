#include "../../include/fpn_kernels.h"
#include "../../include/fpn_types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <vector_types.h>

// ============================================================================
// TEMPLATE SPECIALIZATIONS FOR DIFFERENT DATA TYPES
// ============================================================================

/**
 * Specialized kernel for bfloat16 data type (Ampere and later)
 */
template<int TILE_SIZE, int CHANNELS_PER_BLOCK, int VECTORS_PER_THREAD>
__global__ void __launch_bounds__(256, 4)
production_bilinear_upsample_bfloat16_kernel(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float scale_y,
    const float scale_x,
    const bool align_corners = false) {
    
    // Shared memory with optimal layout for bfloat16
    __shared__ __align__(16) nv_bfloat16 shared_input[TILE_SIZE + 2][TILE_SIZE + 2][CHANNELS_PER_BLOCK + 1];
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * CHANNELS_PER_BLOCK;
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x % TILE_SIZE;
    
    const int global_out_y = tile_y * TILE_SIZE + thread_y;
    const int global_out_x = tile_x * TILE_SIZE + thread_x;
    
    if (global_out_y >= out_height || global_out_x >= out_width) return;
    
    // Compute interpolation coordinates
    float src_y, src_x;
    if (align_corners) {
        src_y = (out_height > 1) ? (float)global_out_y * (in_height - 1) / (out_height - 1) : 0.0f;
        src_x = (out_width > 1) ? (float)global_out_x * (in_width - 1) / (out_width - 1) : 0.0f;
    } else {
        src_y = ((float)global_out_y + 0.5f) * scale_y - 0.5f;
        src_x = ((float)global_out_x + 0.5f) * scale_x - 0.5f;
    }
    
    src_y = fmaxf(0.0f, fminf(src_y, (float)(in_height - 1)));
    src_x = fmaxf(0.0f, fminf(src_x, (float)(in_width - 1)));
    
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    const float wy = src_y - (float)y0;
    const float wx = src_x - (float)x0;
    
    const float w00 = (1.0f - wy) * (1.0f - wx);
    const float w01 = (1.0f - wy) * wx;
    const float w10 = wy * (1.0f - wx);
    const float w11 = wy * wx;
    
    // Process channels with optimized bfloat16 operations
    for (int c_base = channel_block; 
         c_base < min(channels, channel_block + CHANNELS_PER_BLOCK); 
         c_base += VECTORS_PER_THREAD) {
        
        const int c_end = min(c_base + VECTORS_PER_THREAD, min(channels, channel_block + CHANNELS_PER_BLOCK));
        
        #pragma unroll
        for (int c_offset = 0; c_offset < VECTORS_PER_THREAD && (c_base + c_offset) < c_end; ++c_offset) {
            const int c = c_base + c_offset;
            
            if (c < channels) {
                const int base_idx = batch_idx * channels * in_height * in_width + c;
                
                // Load corner values
                nv_bfloat16 val00 = input[base_idx + y0 * in_width * channels + x0 * channels];
                nv_bfloat16 val01 = input[base_idx + y0 * in_width * channels + x1 * channels];
                nv_bfloat16 val10 = input[base_idx + y1 * in_width * channels + x0 * channels];
                nv_bfloat16 val11 = input[base_idx + y1 * in_width * channels + x1 * channels];
                
                // Convert to float for high-precision interpolation
                const float f00 = __bfloat162float(val00);
                const float f01 = __bfloat162float(val01);
                const float f10 = __bfloat162float(val10);
                const float f11 = __bfloat162float(val11);
                
                // High-precision bilinear interpolation
                const float result = __fmaf_rn(__fmaf_rn(f00, w00, f01 * w01), 1.0f,
                                              __fmaf_rn(f10, w10, f11 * w11));
                
                // Store result
                const int out_idx = batch_idx * channels * out_height * out_width +
                                   global_out_y * out_width * channels +
                                   global_out_x * channels + c;
                
                output[out_idx] = __float2bfloat16(result);
            }
        }
    }
}

/**
 * Optimized half precision kernel with vectorized loads
 */
template<int TILE_SIZE, int CHANNELS_PER_BLOCK>
__global__ void __launch_bounds__(256, 4)
vectorized_half_upsample_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float scale_y,
    const float scale_x,
    const bool align_corners = false) {
    
    // Use half2 vectorization for better memory bandwidth
    __shared__ __align__(16) half2 shared_input_vec[TILE_SIZE][TILE_SIZE][CHANNELS_PER_BLOCK/2 + 1];
    
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * CHANNELS_PER_BLOCK;
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    
    const int global_out_y = tile_y * TILE_SIZE + thread_y;
    const int global_out_x = tile_x * TILE_SIZE + thread_x;
    
    if (global_out_y >= out_height || global_out_x >= out_width) return;
    
    // Compute interpolation coordinates
    float src_y, src_x;
    if (align_corners) {
        src_y = (out_height > 1) ? (float)global_out_y * (in_height - 1) / (out_height - 1) : 0.0f;
        src_x = (out_width > 1) ? (float)global_out_x * (in_width - 1) / (out_width - 1) : 0.0f;
    } else {
        src_y = ((float)global_out_y + 0.5f) * scale_y - 0.5f;
        src_x = ((float)global_out_x + 0.5f) * scale_x - 0.5f;
    }
    
    src_y = fmaxf(0.0f, fminf(src_y, (float)(in_height - 1)));
    src_x = fmaxf(0.0f, fminf(src_x, (float)(in_width - 1)));
    
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    const half2 wy = __float2half2_rn(src_y - (float)y0);
    const half2 wx = __float2half2_rn(src_x - (float)x0);
    const half2 one = __float2half2_rn(1.0f);
    
    const half2 w00 = __hmul2(__hsub2(one, wy), __hsub2(one, wx));
    const half2 w01 = __hmul2(__hsub2(one, wy), wx);
    const half2 w10 = __hmul2(wy, __hsub2(one, wx));
    const half2 w11 = __hmul2(wy, wx);
    
    // Process channels in pairs using half2 vectorization
    for (int c_base = channel_block; c_base < min(channels, channel_block + CHANNELS_PER_BLOCK); c_base += 2) {
        if (c_base + 1 < channels) {
            const int base_idx = batch_idx * channels * in_height * in_width + c_base;
            
            // Load corner values as half2
            const half2* input_ptr = reinterpret_cast<const half2*>(&input[base_idx + y0 * in_width * channels + x0 * channels]);
            half2 val00 = *input_ptr;
            
            input_ptr = reinterpret_cast<const half2*>(&input[base_idx + y0 * in_width * channels + x1 * channels]);
            half2 val01 = *input_ptr;
            
            input_ptr = reinterpret_cast<const half2*>(&input[base_idx + y1 * in_width * channels + x0 * channels]);
            half2 val10 = *input_ptr;
            
            input_ptr = reinterpret_cast<const half2*>(&input[base_idx + y1 * in_width * channels + x1 * channels]);
            half2 val11 = *input_ptr;
            
            // Vectorized bilinear interpolation using half2 operations
            half2 result = __hfma2(__hfma2(val00, w00, __hmul2(val01, w01)), one,
                                  __hfma2(val10, w10, __hmul2(val11, w11)));
            
            // Store result as half2
            const int out_idx = batch_idx * channels * out_height * out_width +
                               global_out_y * out_width * channels +
                               global_out_x * channels + c_base;
            
            half2* output_ptr = reinterpret_cast<half2*>(&output[out_idx]);
            *output_ptr = result;
        } else if (c_base < channels) {
            // Handle single remaining channel
            const int c = c_base;
            const int base_idx = batch_idx * channels * in_height * in_width + c;
            
            half val00 = input[base_idx + y0 * in_width * channels + x0 * channels];
            half val01 = input[base_idx + y0 * in_width * channels + x1 * channels];
            half val10 = input[base_idx + y1 * in_width * channels + x0 * channels];
            half val11 = input[base_idx + y1 * in_width * channels + x1 * channels];
            
            const float f00 = __half2float(val00);
            const float f01 = __half2float(val01);
            const float f10 = __half2float(val10);
            const float f11 = __half2float(val11);
            
            const float wy_f = __half2float(wy.x);
            const float wx_f = __half2float(wx.x);
            
            const float w00_f = (1.0f - wy_f) * (1.0f - wx_f);
            const float w01_f = (1.0f - wy_f) * wx_f;
            const float w10_f = wy_f * (1.0f - wx_f);
            const float w11_f = wy_f * wx_f;
            
            const float result = __fmaf_rn(__fmaf_rn(f00, w00_f, f01 * w01_f), 1.0f,
                                          __fmaf_rn(f10, w10_f, f11 * w11_f));
            
            const int out_idx = batch_idx * channels * out_height * out_width +
                               global_out_y * out_width * channels +
                               global_out_x * channels + c;
            
            output[out_idx] = __float2half(result);
        }
    }
}

/**
 * Optimized float kernel with float4 vectorization
 */
template<int TILE_SIZE, int CHANNELS_PER_BLOCK>
__global__ void __launch_bounds__(256, 4)
vectorized_float_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float scale_y,
    const float scale_x,
    const bool align_corners = false) {
    
    // Use float4 vectorization for maximum bandwidth
    __shared__ __align__(16) float4 shared_input_vec[TILE_SIZE][TILE_SIZE][CHANNELS_PER_BLOCK/4 + 1];
    
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * CHANNELS_PER_BLOCK;
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    
    const int global_out_y = tile_y * TILE_SIZE + thread_y;
    const int global_out_x = tile_x * TILE_SIZE + thread_x;
    
    if (global_out_y >= out_height || global_out_x >= out_width) return;
    
    // Compute interpolation coordinates
    float src_y, src_x;
    if (align_corners) {
        src_y = (out_height > 1) ? (float)global_out_y * (in_height - 1) / (out_height - 1) : 0.0f;
        src_x = (out_width > 1) ? (float)global_out_x * (in_width - 1) / (out_width - 1) : 0.0f;
    } else {
        src_y = ((float)global_out_y + 0.5f) * scale_y - 0.5f;
        src_x = ((float)global_out_x + 0.5f) * scale_x - 0.5f;
    }
    
    src_y = fmaxf(0.0f, fminf(src_y, (float)(in_height - 1)));
    src_x = fmaxf(0.0f, fminf(src_x, (float)(in_width - 1)));
    
    const int y0 = __float2int_rd(src_y);
    const int x0 = __float2int_rd(src_x);
    const int y1 = min(y0 + 1, in_height - 1);
    const int x1 = min(x0 + 1, in_width - 1);
    
    const float wy = src_y - (float)y0;
    const float wx = src_x - (float)x0;
    
    const float w00 = (1.0f - wy) * (1.0f - wx);
    const float w01 = (1.0f - wy) * wx;
    const float w10 = wy * (1.0f - wx);
    const float w11 = wy * wx;
    
    // Process channels in groups of 4 using float4 vectorization
    for (int c_base = channel_block; c_base < min(channels, channel_block + CHANNELS_PER_BLOCK); c_base += 4) {
        const int remaining_channels = min(4, channels - c_base);
        
        if (remaining_channels == 4) {
            const int base_idx = batch_idx * channels * in_height * in_width + c_base;
            
            // Load corner values as float4
            const float4* input_ptr = reinterpret_cast<const float4*>(&input[base_idx + y0 * in_width * channels + x0 * channels]);
            float4 val00 = *input_ptr;
            
            input_ptr = reinterpret_cast<const float4*>(&input[base_idx + y0 * in_width * channels + x1 * channels]);
            float4 val01 = *input_ptr;
            
            input_ptr = reinterpret_cast<const float4*>(&input[base_idx + y1 * in_width * channels + x0 * channels]);
            float4 val10 = *input_ptr;
            
            input_ptr = reinterpret_cast<const float4*>(&input[base_idx + y1 * in_width * channels + x1 * channels]);
            float4 val11 = *input_ptr;
            
            // Vectorized bilinear interpolation
            float4 result;
            result.x = __fmaf_rn(__fmaf_rn(val00.x, w00, val01.x * w01), 1.0f, __fmaf_rn(val10.x, w10, val11.x * w11));
            result.y = __fmaf_rn(__fmaf_rn(val00.y, w00, val01.y * w01), 1.0f, __fmaf_rn(val10.y, w10, val11.y * w11));
            result.z = __fmaf_rn(__fmaf_rn(val00.z, w00, val01.z * w01), 1.0f, __fmaf_rn(val10.z, w10, val11.z * w11));
            result.w = __fmaf_rn(__fmaf_rn(val00.w, w00, val01.w * w01), 1.0f, __fmaf_rn(val10.w, w10, val11.w * w11));
            
            // Store result as float4
            const int out_idx = batch_idx * channels * out_height * out_width +
                               global_out_y * out_width * channels +
                               global_out_x * channels + c_base;
            
            float4* output_ptr = reinterpret_cast<float4*>(&output[out_idx]);
            *output_ptr = result;
        } else {
            // Handle remaining channels individually
            for (int c_offset = 0; c_offset < remaining_channels; ++c_offset) {
                const int c = c_base + c_offset;
                const int base_idx = batch_idx * channels * in_height * in_width + c;
                
                float val00 = input[base_idx + y0 * in_width * channels + x0 * channels];
                float val01 = input[base_idx + y0 * in_width * channels + x1 * channels];
                float val10 = input[base_idx + y1 * in_width * channels + x0 * channels];
                float val11 = input[base_idx + y1 * in_width * channels + x1 * channels];
                
                const float result = __fmaf_rn(__fmaf_rn(val00, w00, val01 * w01), 1.0f,
                                              __fmaf_rn(val10, w10, val11 * w11));
                
                const int out_idx = batch_idx * channels * out_height * out_width +
                                   global_out_y * out_width * channels +
                                   global_out_x * channels + c;
                
                output[out_idx] = result;
            }
        }
    }
}

// ============================================================================
// TEMPLATE INSTANTIATIONS AND SPECIALIZATIONS
// ============================================================================

// Explicit template instantiations for bfloat16
template __global__ void production_bilinear_upsample_bfloat16_kernel<16, 64, 4>(
    const nv_bfloat16*, nv_bfloat16*, int, int, int, int, int, int, float, float, bool);

template __global__ void production_bilinear_upsample_bfloat16_kernel<32, 64, 4>(
    const nv_bfloat16*, nv_bfloat16*, int, int, int, int, int, int, float, float, bool);

// Explicit template instantiations for vectorized half
template __global__ void vectorized_half_upsample_kernel<16, 64>(
    const half*, half*, int, int, int, int, int, int, float, float, bool);

template __global__ void vectorized_half_upsample_kernel<32, 128>(
    const half*, half*, int, int, int, int, int, int, float, float, bool);

// Explicit template instantiations for vectorized float
template __global__ void vectorized_float_upsample_kernel<16, 64>(
    const float*, float*, int, int, int, int, int, int, float, float, bool);

template __global__ void vectorized_float_upsample_kernel<32, 128>(
    const float*, float*, int, int, int, int, int, int, float, float, bool);

// ============================================================================
// ARCHITECTURE-SPECIFIC KERNEL SELECTION
// ============================================================================

/**
 * Runtime kernel selection based on GPU architecture and data type
 */
template<typename T>
cudaError_t launch_optimized_upsample_adaptive(
    const T* input,
    T* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x,
    bool align_corners,
    cudaStream_t stream = nullptr) {
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Calculate optimal grid and block dimensions
    constexpr int TILE_SIZE = 16;
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_per_col = (out_height + TILE_SIZE - 1) / TILE_SIZE;
    const int total_spatial_tiles = tiles_per_row * tiles_per_col;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid;
    
    // Architecture-specific optimizations
    if constexpr (std::is_same_v<T, float>) {
        // Use vectorized float kernel for better bandwidth
        constexpr int CHANNELS_PER_BLOCK = 64;
        const int channel_blocks = (channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
        grid = dim3(total_spatial_tiles, channel_blocks, batch_size);
        
        vectorized_float_upsample_kernel<TILE_SIZE, CHANNELS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(
                input, output, batch_size, channels, in_height, in_width,
                out_height, out_width, scale_y, scale_x, align_corners);
                
    } else if constexpr (std::is_same_v<T, half>) {
        // Use vectorized half kernel with half2 operations
        constexpr int CHANNELS_PER_BLOCK = 128; // Higher for half precision
        const int channel_blocks = (channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
        grid = dim3(total_spatial_tiles, channel_blocks, batch_size);
        
        vectorized_half_upsample_kernel<TILE_SIZE, CHANNELS_PER_BLOCK>
            <<<grid, block, 0, stream>>>(
                input, output, batch_size, channels, in_height, in_width,
                out_height, out_width, scale_y, scale_x, align_corners);
                
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        // Use bfloat16 kernel for Ampere and later
        if (props.major >= 8) { // Ampere and later
            constexpr int CHANNELS_PER_BLOCK = 64;
            constexpr int VECTORS_PER_THREAD = 4;
            const int channel_blocks = (channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
            grid = dim3(total_spatial_tiles, channel_blocks, batch_size);
            
            production_bilinear_upsample_bfloat16_kernel<TILE_SIZE, CHANNELS_PER_BLOCK, VECTORS_PER_THREAD>
                <<<grid, block, 0, stream>>>(
                    input, output, batch_size, channels, in_height, in_width,
                    out_height, out_width, scale_y, scale_x, align_corners);
        } else {
            return cudaErrorInvalidDevice; // bfloat16 not supported on older architectures
        }
    } else {
        return cudaErrorInvalidValue; // Unsupported data type
    }
    
    return cudaGetLastError();
}

// Explicit instantiations for adaptive launcher
template cudaError_t launch_optimized_upsample_adaptive<float>(
    const float*, float*, int, int, int, int, int, int, float, float, bool, cudaStream_t);

template cudaError_t launch_optimized_upsample_adaptive<half>(
    const half*, half*, int, int, int, int, int, int, float, float, bool, cudaStream_t);

template cudaError_t launch_optimized_upsample_adaptive<nv_bfloat16>(
    const nv_bfloat16*, nv_bfloat16*, int, int, int, int, int, int, float, float, bool, cudaStream_t);

// ============================================================================
// C-STYLE INTERFACE FOR ADDITIONAL DATA TYPES
// ============================================================================

extern "C" {

cudaError_t launch_production_upsample_bfloat16(
    const nv_bfloat16* input,
    nv_bfloat16* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int align_corners,
    float* kernel_time_ms,
    float* bandwidth_gb_s) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const float scale_y = (float)in_height / (float)out_height;
    const float scale_x = (float)in_width / (float)out_width;
    
    cudaEventRecord(start);
    
    cudaError_t result = launch_optimized_upsample_adaptive<nv_bfloat16>(
        input, output, batch_size, channels, in_height, in_width,
        out_height, out_width, scale_y, scale_x, align_corners != 0);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (result == cudaSuccess) {
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        if (kernel_time_ms) *kernel_time_ms = time_ms;
        
        if (bandwidth_gb_s) {
            const size_t input_bytes = batch_size * channels * in_height * in_width * sizeof(nv_bfloat16);
            const size_t output_bytes = batch_size * channels * out_height * out_width * sizeof(nv_bfloat16);
            const size_t total_bytes = input_bytes + output_bytes;
            *bandwidth_gb_s = (total_bytes / (time_ms * 1e-3)) / 1e9;
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

} // extern "C"