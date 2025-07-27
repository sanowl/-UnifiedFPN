#include "../../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(512, 2)
backward_lateral_conv_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weights,
    T* __restrict__ grad_input,
    T* __restrict__ grad_weights,
    T* __restrict__ grad_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    auto block = cg::this_thread_block();
    
    __shared__ __align__(16) T shared_grad_output[TILE_SIZE * TILE_SIZE * 64];
    __shared__ __align__(16) T shared_input[TILE_SIZE * TILE_SIZE * 64];
    
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
    
    T grad_weight_accumulator = static_cast<T>(0);
    
    for (int c_base = 0; c_base < in_channels; c_base += 64) {
        const int c_chunk_size = min(64, in_channels - c_base);
        
        if (thread_c == 0) {
            const int grad_offset = batch_idx * out_channels * height * width + 
                                   global_y * width * out_channels + global_x * out_channels;
            
            #pragma unroll
            for (int c = 0; c < min(64, out_channels - out_c_base); c += 4) {
                if (c + 3 < min(64, out_channels - out_c_base) && out_c_base + c + 3 < out_channels) {
                    if constexpr (std::is_same_v<T, float>) {
                        float4 grad_vec = *reinterpret_cast<const float4*>(&grad_output[grad_offset + c]);
                        *reinterpret_cast<float4*>(&shared_grad_output[(thread_y * TILE_SIZE + thread_x) * 64 + c]) = grad_vec;
                    } else {
                        for (int j = 0; j < 4 && c + j < min(64, out_channels - out_c_base); ++j) {
                            shared_grad_output[(thread_y * TILE_SIZE + thread_x) * 64 + c + j] = grad_output[grad_offset + c + j];
                        }
                    }
                }
            }
            
            const int input_offset = batch_idx * in_channels * height * width + 
                                   global_y * width * in_channels + global_x * in_channels + c_base;
            
            #pragma unroll
            for (int c = 0; c < c_chunk_size; c += 4) {
                if (c + 3 < c_chunk_size && input_offset + c + 3 < batch_size * in_channels * height * width) {
                    if constexpr (std::is_same_v<T, float>) {
                        float4 input_vec = *reinterpret_cast<const float4*>(&input[input_offset + c]);
                        *reinterpret_cast<float4*>(&shared_input[(thread_y * TILE_SIZE + thread_x) * c_chunk_size + c]) = input_vec;
                    } else {
                        for (int j = 0; j < 4 && c + j < c_chunk_size; ++j) {
                            shared_input[(thread_y * TILE_SIZE + thread_x) * c_chunk_size + c + j] = input[input_offset + c + j];
                        }
                    }
                }
            }
        }
        
        block.sync();
        
        if (out_c < out_channels) {
            #pragma unroll
            for (int c = 0; c < c_chunk_size; ++c) {
                T grad_out_val = shared_grad_output[(thread_y * TILE_SIZE + thread_x) * 64 + (out_c - out_c_base)];
                T input_val = shared_input[(thread_y * TILE_SIZE + thread_x) * c_chunk_size + c];
                
                if constexpr (std::is_same_v<T, float>) {
                    grad_weight_accumulator = __fmaf_rn(grad_out_val, input_val, grad_weight_accumulator);
                } else {
                    grad_weight_accumulator += grad_out_val * input_val;
                }
            }
        }
        
        block.sync();
    }
    
    if (out_c < out_channels) {
        for (int c_base = 0; c_base < in_channels; c_base += 64) {
            int c = c_base + threadIdx.z % min(64, in_channels - c_base);
            if (c < in_channels) {
                atomicAdd(&grad_weights[out_c * in_channels + c], grad_weight_accumulator);
            }
        }
    }
    
    if (out_c < out_channels && thread_y == 0 && thread_x == 0) {
        T grad_bias_val = shared_grad_output[thread_c * 64];
        atomicAdd(&grad_bias[out_c], grad_bias_val);
    }
    
    T grad_input_accumulator = static_cast<T>(0);
    
    for (int out_c_iter = 0; out_c_iter < out_channels; ++out_c_iter) {
        T grad_out_val = grad_output[batch_idx * out_channels * height * width + 
                                    global_y * width * out_channels + global_x * out_channels + out_c_iter];
        
        for (int c = 0; c < in_channels; ++c) {
            T weight_val = weights[out_c_iter * in_channels + c];
            
            if constexpr (std::is_same_v<T, float>) {
                grad_input_accumulator = __fmaf_rn(grad_out_val, weight_val, grad_input_accumulator);
            } else {
                grad_input_accumulator += grad_out_val * weight_val;
            }
        }
    }
    
    if (global_y < height && global_x < width && thread_c == 0) {
        for (int c = 0; c < in_channels; ++c) {
            int grad_input_idx = batch_idx * in_channels * height * width + 
                               global_y * width * in_channels + global_x * in_channels + c;
            atomicAdd(&grad_input[grad_input_idx], grad_input_accumulator);
        }
    }
}

template __global__ void backward_lateral_conv_kernel<float, 16>(
    const float*, const float*, const float*, float*, float*, float*, int, int, int, int, int);

template __global__ void backward_lateral_conv_kernel<half, 16>(
    const half*, const half*, const half*, half*, half*, half*, int, int, int, int, int);