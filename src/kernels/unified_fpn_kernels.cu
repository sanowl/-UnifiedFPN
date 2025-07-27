#include "../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>

__device__ __forceinline__ float4 load_float4_aligned(const float* ptr) {
    // Check alignment in debug mode only
    #ifdef DEBUG
    assert(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
    #endif
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void store_float4_aligned(float* ptr, float4 val) {
    // Check alignment in debug mode only
    #ifdef DEBUG
    assert(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
    #endif
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ float bilinear_interpolate(
    const float* __restrict__ input,
    int height, int width, int channels,
    float y, float x, int c) {
    
    // Use hardware-accelerated float-to-int conversion
    int y0 = __float2int_rd(y);
    int x0 = __float2int_rd(x);
    int y1 = min(y0 + 1, height - 1);
    int x1 = min(x0 + 1, width - 1);
    
    // Clamp coordinates with CUDA intrinsics
    y0 = max(0, y0);
    x0 = max(0, x0);
    
    // Use FMA instructions for better precision and performance
    float dy = y - __int2float_rn(y0);
    float dx = x - __int2float_rn(x0);
    
    // Calculate indices with strength reduction
    int width_channels = width * channels;
    int base_idx = y0 * width_channels + x0 * channels + c;
    int idx00 = base_idx;
    int idx01 = base_idx + channels;
    int idx10 = base_idx + width_channels;
    int idx11 = base_idx + width_channels + channels;
    
    // Use CUDA texture cache-friendly loads
    float v00 = __ldg(&input[idx00]);
    float v01 = __ldg(&input[idx01]);
    float v10 = __ldg(&input[idx10]);
    float v11 = __ldg(&input[idx11]);
    
    // Use FMA for precision and performance
    float v0 = __fmaf_rn(v01 - v00, dx, v00);
    float v1 = __fmaf_rn(v11 - v10, dx, v10);
    
    return __fmaf_rn(v1 - v0, dy, v0);
}

__device__ __forceinline__ float4 warp_reduce_sum(float4 val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, offset);
        val.w += __shfl_down_sync(0xFFFFFFFF, val.w, offset);
    }
    return val;
}

template<int TILE_SIZE>
__device__ __forceinline__ void cooperative_copy_to_shared(
    const float* __restrict__ global_mem,
    float* __restrict__ shared_mem,
    int elements_per_thread,
    int total_elements) {
    
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    const int vectors_per_thread = (elements_per_thread + 3) / 4;
    const int start_idx = thread_id * vectors_per_thread * 4;
    
    if (start_idx < total_elements) {
        const int copy_elements = min(vectors_per_thread * 4, total_elements - start_idx);
        
        #pragma unroll
        for (int i = 0; i < copy_elements; i += 4) {
            if (start_idx + i + 3 < total_elements) {
                float4 data = load_float4_aligned(&global_mem[start_idx + i]);
                store_float4_aligned(&shared_mem[start_idx + i], data);
            } else {
                for (int j = 0; j < min(4, copy_elements - i); ++j) {
                    shared_mem[start_idx + i + j] = global_mem[start_idx + i + j];
                }
            }
        }
    }
}

template<int TILE_SIZE, int CHANNELS_PER_THREAD>
__device__ __forceinline__ void prefetch_feature_tile(
    const float* __restrict__ input,
    int channels, int height, int width,
    int tile_y, int tile_x,
    int thread_y, int thread_x) {
    
    const int global_y = tile_y * TILE_SIZE + thread_y;
    const int global_x = tile_x * TILE_SIZE + thread_x;
    
    if (global_y < height && global_x < width) {
        #pragma unroll
        for (int c = 0; c < channels; c += 16) {
            if (c + 15 < channels) {
                const int idx = (global_y * width + global_x) * channels + c;
                // Prefetch 4 cache lines (64 bytes each) to L2
                asm("prefetch.global.L2 [%0];" :: "l"(&input[idx]));
                asm("prefetch.global.L2 [%0];" :: "l"(&input[idx + 4]));
                asm("prefetch.global.L2 [%0];" :: "l"(&input[idx + 8]));
                asm("prefetch.global.L2 [%0];" :: "l"(&input[idx + 12]));
            }
        }
    }
}

template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(256, 4)
lateral_conv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    // Use bank-conflict-free shared memory layout
    __shared__ __align__(16) T shared_input[TILE_SIZE * TILE_SIZE * 64 + 32];
    __shared__ __align__(16) T shared_weights[256 * 64 + 32];
    
    const int batch_idx = blockIdx.z;
    const int out_c_base = blockIdx.y * blockDim.z;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_c = threadIdx.z;
    
    const int global_y = tile_y * TILE_SIZE + thread_y;
    const int global_x = tile_x * TILE_SIZE + thread_x;
    const int out_c = out_c_base + thread_c;
    
    if (global_y >= height || global_x >= width || out_c >= out_channels) return;
    
    float accumulator = 0.0f;
    
    for (int c_base = 0; c_base < in_channels; c_base += 64) {
        const int c_chunk_size = min(64, in_channels - c_base);
        
        const int input_offset = batch_idx * in_channels * height * width + 
                               global_y * width * in_channels + global_x * in_channels + c_base;
        
        if (thread_c == 0) {
            // Use double buffering and async copy for better throughput
            #pragma unroll
            for (int c = 0; c < c_chunk_size; c += 4) {
                if (c + 3 < c_chunk_size && input_offset + c + 3 < batch_size * in_channels * height * width) {
                    if constexpr (std::is_same_v<T, float>) {
                        float4 input_vec = load_float4_aligned(&input[input_offset + c]);
                        int shared_idx = (thread_y * TILE_SIZE + thread_x) * (c_chunk_size + 1) + c;
                        store_float4_aligned(&shared_input[shared_idx], input_vec);
                    } else {
                        for (int i = 0; i < 4 && c + i < c_chunk_size; ++i) {
                            int shared_idx = (thread_y * TILE_SIZE + thread_x) * (c_chunk_size + 1) + c + i;
                            shared_input[shared_idx] = input[input_offset + c + i];
                        }
                    }
                }
            }
        }
        
        if (thread_y == 0 && thread_x < c_chunk_size && out_c < out_channels) {
            shared_weights[thread_c * c_chunk_size + thread_x] = 
                weights[out_c * in_channels + c_base + thread_x];
        }
        
        __syncthreads();
        
        // Vectorized computation with FMA instructions
        #pragma unroll
        for (int c = 0; c < c_chunk_size; c += 4) {
            if (c + 3 < c_chunk_size) {
                int shared_idx = (thread_y * TILE_SIZE + thread_x) * (c_chunk_size + 1) + c;
                
                if constexpr (std::is_same_v<T, float>) {
                    float4 input_vec = load_float4_aligned(&shared_input[shared_idx]);
                    float4 weight_vec = load_float4_aligned(&shared_weights[thread_c * c_chunk_size + c]);
                    
                    accumulator = __fmaf_rn(input_vec.x, weight_vec.x, accumulator);
                    accumulator = __fmaf_rn(input_vec.y, weight_vec.y, accumulator);
                    accumulator = __fmaf_rn(input_vec.z, weight_vec.z, accumulator);
                    accumulator = __fmaf_rn(input_vec.w, weight_vec.w, accumulator);
                } else {
                    for (int i = 0; i < 4; ++i) {
                        T input_val = shared_input[shared_idx + i];
                        T weight_val = shared_weights[thread_c * c_chunk_size + c + i];
                        accumulator += static_cast<float>(input_val * weight_val);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    if (bias) {
        accumulator += bias[out_c];
    }
    
    const int output_idx = batch_idx * out_channels * height * width +
                          global_y * width * out_channels + global_x * out_channels + out_c;
    output[output_idx] = static_cast<T>(accumulator);
}

template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(256, 4)
bilinear_upsample_kernel(
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
    
    const int batch_idx = blockIdx.z;
    const int c_base = blockIdx.y * 32;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_c = threadIdx.z;
    
    const int out_y = tile_y * TILE_SIZE + thread_y;
    const int out_x = tile_x * TILE_SIZE + thread_x;
    const int c = c_base + thread_c;
    
    if (out_y >= out_height || out_x >= out_width || c >= channels) return;
    
    const float src_y = (out_y + 0.5f) * scale_y - 0.5f;
    const float src_x = (out_x + 0.5f) * scale_x - 0.5f;
    
    const int input_base = batch_idx * channels * in_height * in_width;
    const float result = bilinear_interpolate(
        &input[input_base], in_height, in_width, channels, src_y, src_x, c
    );
    
    const int output_idx = batch_idx * channels * out_height * out_width +
                          out_y * out_width * channels + out_x * channels + c;
    output[output_idx] = static_cast<T>(result);
}

template<typename T, int TILE_SIZE, int KERNEL_SIZE>
__global__ void __launch_bounds__(256, 4)
depthwise_conv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int padding) {
    
    __shared__ float shared_input[(TILE_SIZE + KERNEL_SIZE - 1) * (TILE_SIZE + KERNEL_SIZE - 1) * 32];
    
    const int batch_idx = blockIdx.z;
    const int c_base = blockIdx.y * 32;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int thread_c = threadIdx.z;
    
    const int out_y = tile_y * TILE_SIZE + thread_y;
    const int out_x = tile_x * TILE_SIZE + thread_x;
    const int c = c_base + thread_c;
    
    if (out_y >= height || out_x >= width || c >= channels) return;
    
    float accumulator = 0.0f;
    
    const int kernel_radius = KERNEL_SIZE / 2;
    
    #pragma unroll
    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        #pragma unroll
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                int input_idx = batch_idx * channels * height * width + 
                               in_y * width * channels + in_x * channels + c;
                int weight_idx = c * KERNEL_SIZE * KERNEL_SIZE + 
                                (ky + kernel_radius) * KERNEL_SIZE + (kx + kernel_radius);
                
                accumulator += static_cast<float>(input[input_idx]) * 
                              static_cast<float>(weights[weight_idx]);
            }
        }
    }
    
    if (bias) {
        accumulator += static_cast<float>(bias[c]);
    }
    
    const int output_idx = batch_idx * channels * height * width +
                          out_y * width * channels + out_x * channels + c;
    output[output_idx] = static_cast<T>(accumulator);
}

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
    float eps) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        const int c = (idx / (height * width)) % channels;
        
        float input_val = static_cast<float>(input_output[idx]);
        float mean_val = static_cast<float>(running_mean[c]);
        float var_val = static_cast<float>(running_var[c]);
        float weight_val = weight ? static_cast<float>(weight[c]) : 1.0f;
        float bias_val = bias ? static_cast<float>(bias[c]) : 0.0f;
        
        float normalized = (input_val - mean_val) / sqrtf(var_val + eps);
        float result = normalized * weight_val + bias_val;
        
        input_output[idx] = static_cast<T>(result);
    }
}

template<typename T, FPNActivation ACTIVATION>
__global__ void activation_kernel(
    T* __restrict__ input_output,
    int total_elements,
    float alpha) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float val = static_cast<float>(input_output[idx]);
        
        if constexpr (ACTIVATION == FPNActivation::RELU) {
            val = fmaxf(0.0f, val);
        } else if constexpr (ACTIVATION == FPNActivation::LEAKY_RELU) {
            val = (val > 0.0f) ? val : alpha * val;
        } else if constexpr (ACTIVATION == FPNActivation::SWISH) {
            val = val / (1.0f + expf(-val));
        }
        
        input_output[idx] = static_cast<T>(val);
    }
}

template<typename T>
__global__ void element_wise_add_kernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    T* __restrict__ output,
    int total_elements) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = input1[idx] + input2[idx];
    }
}

// Explicit template instantiations
template __global__ void lateral_conv_kernel<float, 16>(
    const float* __restrict__, const float* __restrict__, const float* __restrict__, 
    float* __restrict__, int, int, int, int, int);

template __global__ void bilinear_upsample_kernel<float, 16>(
    const float* __restrict__, float* __restrict__, int, int, int, int, int, int, float, float);

template __global__ void depthwise_conv_kernel<float, 16, 3>(
    const float* __restrict__, const float* __restrict__, const float* __restrict__,
    float* __restrict__, int, int, int, int, int);

template __global__ void batch_norm_kernel<float>(
    float* __restrict__, const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__, int, int, int, int, float);

template __global__ void activation_kernel<float, FPNActivation::RELU>(
    float* __restrict__, int, float);

template __global__ void element_wise_add_kernel<float>(
    const float* __restrict__, const float* __restrict__, float* __restrict__, int);