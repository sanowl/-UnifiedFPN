#include "../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Fused lateral convolution + batch normalization + activation kernel
template<typename T, int TILE_SIZE, FPNActivation ACTIVATION>
__global__ void __launch_bounds__(256, 4)
fused_lateral_conv_bn_activation_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    const T* __restrict__ bn_weight,
    const T* __restrict__ bn_bias,
    const T* __restrict__ bn_mean,
    const T* __restrict__ bn_var,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    float bn_eps = 1e-5f,
    float activation_alpha = 0.01f) {
    
    // Optimized shared memory layout
    __shared__ __align__(16) T shared_input[TILE_SIZE][TILE_SIZE + 1][64];
    __shared__ __align__(16) T shared_weights[64][64 + 1];
    __shared__ __align__(16) T shared_bn_params[4][64]; // weight, bias, mean, var
    
    const int batch_idx = blockIdx.z;
    const int out_c_base = blockIdx.y * 64;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tc = threadIdx.z;
    
    const int global_y = tile_y * TILE_SIZE + ty;
    const int global_x = tile_x * TILE_SIZE + tx;
    const int out_c = out_c_base + tc;
    
    if (global_y >= height || global_x >= width || out_c >= out_channels) return;
    
    // Load batch normalization parameters into shared memory
    if (ty == 0 && tx < 64 && out_c_base + tx < out_channels) {
        shared_bn_params[0][tx] = bn_weight[out_c_base + tx];
        shared_bn_params[1][tx] = bn_bias[out_c_base + tx];
        shared_bn_params[2][tx] = bn_mean[out_c_base + tx];
        shared_bn_params[3][tx] = bn_var[out_c_base + tx];
    }
    
    float accumulator = 0.0f;
    
    // Process input channels in chunks
    for (int c_base = 0; c_base < in_channels; c_base += 64) {
        const int c_chunk_size = min(64, in_channels - c_base);
        
        // Cooperative loading of input tile
        if (tc == 0) {
            #pragma unroll
            for (int c = 0; c < c_chunk_size; c += 4) {
                if (c + 3 < c_chunk_size) {
                    const int input_idx = batch_idx * in_channels * height * width +
                                         global_y * width * in_channels + 
                                         global_x * in_channels + c_base + c;
                    
                    if constexpr (std::is_same_v<T, float>) {
                        float4 input_vec = __ldg(reinterpret_cast<const float4*>(&input[input_idx]));
                        *reinterpret_cast<float4*>(&shared_input[ty][tx][c]) = input_vec;
                    } else {
                        for (int i = 0; i < 4; ++i) {
                            shared_input[ty][tx][c + i] = input[input_idx + i];
                        }
                    }
                }
            }
        }
        
        // Load weight matrix
        if (ty == 0 && tx < c_chunk_size && out_c < out_channels) {
            shared_weights[tx][tc] = weights[out_c * in_channels + c_base + tx];
        }
        
        __syncthreads();
        
        // Convolution computation with FMA
        #pragma unroll
        for (int c = 0; c < c_chunk_size; ++c) {
            T input_val = shared_input[ty][tx][c];
            T weight_val = shared_weights[c][tc];
            
            if constexpr (std::is_same_v<T, float>) {
                accumulator = __fmaf_rn(static_cast<float>(input_val), 
                                       static_cast<float>(weight_val), 
                                       accumulator);
            } else {
                accumulator += static_cast<float>(input_val * weight_val);
            }
        }
        
        __syncthreads();
    }
    
    // Add convolution bias
    if (bias && out_c < out_channels) {
        accumulator += static_cast<float>(bias[out_c]);
    }
    
    __syncthreads();
    
    // Apply batch normalization
    if (out_c < out_channels) {
        const float bn_weight_val = static_cast<float>(shared_bn_params[0][tc]);
        const float bn_bias_val = static_cast<float>(shared_bn_params[1][tc]);
        const float bn_mean_val = static_cast<float>(shared_bn_params[2][tc]);
        const float bn_var_val = static_cast<float>(shared_bn_params[3][tc]);
        
        // Normalize
        const float normalized = (accumulator - bn_mean_val) / sqrtf(bn_var_val + bn_eps);
        accumulator = normalized * bn_weight_val + bn_bias_val;
    }
    
    // Apply activation function
    if constexpr (ACTIVATION == FPNActivation::RELU) {
        accumulator = fmaxf(0.0f, accumulator);
    } else if constexpr (ACTIVATION == FPNActivation::LEAKY_RELU) {
        accumulator = (accumulator > 0.0f) ? accumulator : activation_alpha * accumulator;
    } else if constexpr (ACTIVATION == FPNActivation::SWISH) {
        accumulator = accumulator / (1.0f + expf(-accumulator));
    }
    
    // Store final result
    const int output_idx = batch_idx * out_channels * height * width +
                          global_y * width * out_channels + 
                          global_x * out_channels + out_c;
    
    output[output_idx] = static_cast<T>(accumulator);
}

// Fused upsample + add + output convolution kernel
template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(256, 4)
fused_upsample_add_conv_kernel(
    const T* __restrict__ higher_level_input,
    const T* __restrict__ lateral_input,
    const T* __restrict__ conv_weights,
    const T* __restrict__ conv_bias,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale_y,
    float scale_x,
    int kernel_size = 3,
    int padding = 1) {
    
    // Shared memory for intermediate results and convolution weights
    __shared__ __align__(16) T shared_upsampled[TILE_SIZE + 2][TILE_SIZE + 2][32];
    __shared__ __align__(16) T shared_conv_weights[9][32][32]; // 3x3 kernel
    
    const int batch_idx = blockIdx.z;
    const int channel_block = blockIdx.y * 32;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tc = threadIdx.z;
    
    const int out_y = tile_y * TILE_SIZE + ty;
    const int out_x = tile_x * TILE_SIZE + tx;
    const int channel = channel_block + tc;
    
    if (out_y >= out_height || out_x >= out_width || channel >= channels) return;
    
    // Load convolution weights into shared memory
    if (ty == 0 && tx < 9 && tc < 32 && channel_block + tc < channels) {
        for (int in_c = 0; in_c < min(32, channels - channel_block); ++in_c) {
            const int weight_idx = (channel_block + tc) * channels * 9 + 
                                  (channel_block + in_c) * 9 + tx;
            shared_conv_weights[tx][tc][in_c] = conv_weights[weight_idx];
        }
    }
    
    // Step 1: Bilinear upsampling
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
    
    // Load 4 neighboring pixels for upsampling
    const int input_base = batch_idx * channels * in_height * in_width;
    const int stride = in_width * channels;
    
    const float v00 = static_cast<float>(__ldg(&higher_level_input[
        input_base + y0_clamped * stride + x0_clamped * channels + channel]));
    const float v01 = static_cast<float>(__ldg(&higher_level_input[
        input_base + y0_clamped * stride + min(x1, in_width - 1) * channels + channel]));
    const float v10 = static_cast<float>(__ldg(&higher_level_input[
        input_base + min(y1, in_height - 1) * stride + x0_clamped * channels + channel]));
    const float v11 = static_cast<float>(__ldg(&higher_level_input[
        input_base + min(y1, in_height - 1) * stride + min(x1, in_width - 1) * channels + channel]));
    
    // Bilinear interpolation
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 = dx * (1.0f - dy);
    const float w10 = (1.0f - dx) * dy;
    const float w11 = dx * dy;
    
    float upsampled_val = __fmaf_rn(v00, w00, __fmaf_rn(v01, w01, 
                                   __fmaf_rn(v10, w10, v11 * w11)));
    
    // Step 2: Add lateral connection
    const int lateral_idx = batch_idx * channels * out_height * out_width +
                           out_y * out_width * channels + out_x * channels + channel;
    
    float lateral_val = static_cast<float>(__ldg(&lateral_input[lateral_idx]));
    float combined_val = upsampled_val + lateral_val;
    
    // Store in shared memory for convolution (with padding)
    const int shared_y = ty + 1;
    const int shared_x = tx + 1;
    shared_upsampled[shared_y][shared_x][tc] = static_cast<T>(combined_val);
    
    // Load padding values
    if (ty == 0 || tx == 0 || ty == TILE_SIZE - 1 || tx == TILE_SIZE - 1) {
        // Handle boundary conditions for convolution padding
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int pad_y = out_y + dy;
                const int pad_x = out_x + dx;
                
                if (pad_y >= 0 && pad_y < out_height && pad_x >= 0 && pad_x < out_width) {
                    // Load from combined result (would need separate kernel pass)
                    shared_upsampled[shared_y + dy][shared_x + dx][tc] = combined_val;
                } else {
                    shared_upsampled[shared_y + dy][shared_x + dx][tc] = static_cast<T>(0);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Step 3: 3x3 Depthwise convolution
    float conv_result = 0.0f;
    
    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int kernel_idx = ky * 3 + kx;
            const float input_val = static_cast<float>(
                shared_upsampled[shared_y + ky - 1][shared_x + kx - 1][tc]);
            const float weight_val = static_cast<float>(
                shared_conv_weights[kernel_idx][tc][tc]); // Depthwise
            
            conv_result = __fmaf_rn(input_val, weight_val, conv_result);
        }
    }
    
    // Add bias
    if (conv_bias && channel < channels) {
        conv_result += static_cast<float>(conv_bias[channel]);
    }
    
    // Store final result
    const int output_idx = batch_idx * channels * out_height * out_width +
                          out_y * out_width * channels + out_x * channels + channel;
    
    output[output_idx] = static_cast<T>(conv_result);
}

// Unified FPN kernel that processes an entire pyramid level
template<typename T, int TILE_SIZE>
__global__ void __launch_bounds__(512, 2)
unified_fpn_level_kernel(
    const T* __restrict__ backbone_input,
    const T* __restrict__ higher_level_input,
    const T* __restrict__ lateral_weights,
    const T* __restrict__ lateral_bias,
    const T* __restrict__ output_weights,
    const T* __restrict__ output_bias,
    const T* __restrict__ bn_weights,
    const T* __restrict__ bn_bias,
    const T* __restrict__ bn_mean,
    const T* __restrict__ bn_var,
    T* __restrict__ output,
    T* __restrict__ intermediate_buffer,
    int batch_size,
    int backbone_channels,
    int output_channels,
    int height,
    int width,
    int higher_level_height,
    int higher_level_width,
    bool is_top_level) {
    
    // Large shared memory for multi-stage processing
    extern __shared__ T dynamic_shared_memory[];
    
    T* shared_lateral = dynamic_shared_memory;
    T* shared_upsampled = shared_lateral + TILE_SIZE * TILE_SIZE * output_channels;
    T* shared_combined = shared_upsampled + TILE_SIZE * TILE_SIZE * output_channels;
    
    const int batch_idx = blockIdx.z;
    const int tile_idx = blockIdx.x;
    const int channel_block = blockIdx.y * output_channels;
    
    const int tiles_per_row = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = tile_idx / tiles_per_row;
    const int tile_x = tile_idx % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tc = threadIdx.z;
    
    const int global_y = tile_y * TILE_SIZE + ty;
    const int global_x = tile_x * TILE_SIZE + tx;
    
    if (global_y >= height || global_x >= width) return;
    
    // Stage 1: Lateral 1x1 convolution
    float lateral_result = 0.0f;
    
    for (int c = 0; c < backbone_channels; ++c) {
        const int input_idx = batch_idx * backbone_channels * height * width +
                             global_y * width * backbone_channels + 
                             global_x * backbone_channels + c;
        
        const int weight_idx = tc * backbone_channels + c;
        
        float input_val = static_cast<float>(__ldg(&backbone_input[input_idx]));
        float weight_val = static_cast<float>(__ldg(&lateral_weights[weight_idx]));
        
        lateral_result = __fmaf_rn(input_val, weight_val, lateral_result);
    }
    
    if (lateral_bias) {
        lateral_result += static_cast<float>(lateral_bias[tc]);
    }
    
    shared_lateral[(ty * TILE_SIZE + tx) * output_channels + tc] = static_cast<T>(lateral_result);
    
    __syncthreads();
    
    // Stage 2: Upsampling and addition (if not top level)
    float combined_result = lateral_result;
    
    if (!is_top_level && higher_level_input) {
        const float scale_y = static_cast<float>(higher_level_height) / height;
        const float scale_x = static_cast<float>(higher_level_width) / width;
        
        // Bilinear upsampling from higher level
        const float src_y = (global_y + 0.5f) * scale_y - 0.5f;
        const float src_x = (global_x + 0.5f) * scale_x - 0.5f;
        
        // Implementation similar to previous upsample kernel
        // ... (upsampling code) ...
        
        // combined_result += upsampled_value;
    }
    
    shared_combined[(ty * TILE_SIZE + tx) * output_channels + tc] = static_cast<T>(combined_result);
    
    __syncthreads();
    
    // Stage 3: Output 3x3 convolution + batch norm + activation
    // Implementation similar to previous fused kernel
    // ... (convolution, batch norm, activation code) ...
    
    // Store final result
    const int output_idx = batch_idx * output_channels * height * width +
                          global_y * width * output_channels + 
                          global_x * output_channels + tc;
    
    output[output_idx] = static_cast<T>(combined_result); // Replace with final processed result
}

// Template instantiations
template __global__ void fused_lateral_conv_bn_activation_kernel<float, 16, FPNActivation::RELU>(
    const float*, const float*, const float*, const float*, const float*, 
    const float*, const float*, float*, int, int, int, int, int, float, float);

template __global__ void fused_upsample_add_conv_kernel<float, 16>(
    const float*, const float*, const float*, const float*, float*, 
    int, int, int, int, int, int, float, float, int, int);

template __global__ void unified_fpn_level_kernel<float, 16>(
    const float*, const float*, const float*, const float*, const float*, 
    const float*, const float*, const float*, const float*, const float*, 
    float*, float*, int, int, int, int, int, int, int, bool);