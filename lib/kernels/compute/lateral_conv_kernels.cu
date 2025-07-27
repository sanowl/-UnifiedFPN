#include "../../include/fpn_kernels.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// Optimized lateral convolution kernel with improved memory coalescing
template<typename T, int TILE_SIZE, int CHANNELS_PER_BLOCK, int VECTORS_PER_THREAD>
__global__ void __launch_bounds__(256, 4)
optimized_lateral_conv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    // Bank-conflict-free shared memory layout with padding
    __shared__ __align__(16) T shared_input[TILE_SIZE][TILE_SIZE + 1][CHANNELS_PER_BLOCK];
    __shared__ __align__(16) T shared_weights[CHANNELS_PER_BLOCK][CHANNELS_PER_BLOCK + 1];
    
    // Cooperative groups for better warp-level coordination
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Improved thread indexing for better coalescing
    const int batch_idx = blockIdx.z;
    const int out_c_block = blockIdx.y * CHANNELS_PER_BLOCK;
    const int spatial_block = blockIdx.x;
    
    const int tiles_per_row = (width + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_y = spatial_block / tiles_per_row;
    const int tile_x = spatial_block % tiles_per_row;
    
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    const int global_y = tile_y * TILE_SIZE + ty;
    const int global_x = tile_x * TILE_SIZE + tx;
    
    // Early exit for out-of-bounds threads
    if (global_y >= height || global_x >= width) return;
    
    // Register arrays for vectorized operations
    float accumulator[CHANNELS_PER_BLOCK / WARP_SIZE] = {0.0f};
    
    // Process input channels in blocks for better cache utilization
    for (int in_c_base = 0; in_c_base < in_channels; in_c_base += CHANNELS_PER_BLOCK) {
        const int c_chunk_size = min(CHANNELS_PER_BLOCK, in_channels - in_c_base);
        
        // Optimized cooperative loading with vectorization
        #pragma unroll
        for (int c_offset = lane_id; c_offset < c_chunk_size; c_offset += WARP_SIZE) {
            if (c_offset < c_chunk_size) {
                // Coalesced global memory access
                int input_idx = batch_idx * in_channels * height * width + 
                               global_y * width * in_channels + 
                               global_x * in_channels + in_c_base + c_offset;
                
                shared_input[ty][tx][c_offset] = input[input_idx];
            }
        }
        
        // Load weight matrix with better memory pattern
        if (warp_id == 0) {
            #pragma unroll
            for (int out_c_offset = lane_id; out_c_offset < CHANNELS_PER_BLOCK; out_c_offset += WARP_SIZE) {
                if (out_c_block + out_c_offset < out_channels) {
                    #pragma unroll
                    for (int in_c_offset = 0; in_c_offset < c_chunk_size; ++in_c_offset) {
                        int weight_idx = (out_c_block + out_c_offset) * in_channels + 
                                       in_c_base + in_c_offset;
                        shared_weights[in_c_offset][out_c_offset] = weights[weight_idx];
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Optimized computation with FMA and vectorization
        #pragma unroll
        for (int out_c_offset = 0; out_c_offset < min(CHANNELS_PER_BLOCK, out_channels - out_c_block); ++out_c_offset) {
            if (warp_id == out_c_offset / WARP_SIZE) {
                const int local_out_c = out_c_offset % WARP_SIZE;
                
                #pragma unroll
                for (int in_c_offset = 0; in_c_offset < c_chunk_size; ++in_c_offset) {
                    T input_val = shared_input[ty][tx][in_c_offset];
                    T weight_val = shared_weights[in_c_offset][out_c_offset];
                    
                    if constexpr (std::is_same_v<T, float>) {
                        accumulator[local_out_c] = __fmaf_rn(
                            static_cast<float>(input_val), 
                            static_cast<float>(weight_val), 
                            accumulator[local_out_c]
                        );
                    } else {
                        accumulator[local_out_c] += static_cast<float>(input_val * weight_val);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output with bias addition and coalesced access
    #pragma unroll
    for (int out_c_offset = 0; out_c_offset < min(CHANNELS_PER_BLOCK, out_channels - out_c_block); ++out_c_offset) {
        if (warp_id == out_c_offset / WARP_SIZE) {
            const int local_out_c = out_c_offset % WARP_SIZE;
            const int global_out_c = out_c_block + out_c_offset;
            
            if (global_out_c < out_channels) {
                float result = accumulator[local_out_c];
                
                if (bias) {
                    result += static_cast<float>(bias[global_out_c]);
                }
                
                int output_idx = batch_idx * out_channels * height * width +
                               global_y * width * out_channels + 
                               global_x * out_channels + global_out_c;
                
                output[output_idx] = static_cast<T>(result);
            }
        }
    }
}

// Architecture-specific tensor core optimization for Ampere/Ada Lovelace
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(256, 2)
ampere_optimized_tensor_core_conv(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    using namespace nvcuda::wmma;
    
    // Multiple fragments for better instruction-level parallelism
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[2];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag[2];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];
    
    // Double-buffered shared memory for better throughput
    __shared__ __align__(16) half shared_input[2][WMMA_M * WMMA_K + 16];
    __shared__ __align__(16) half shared_weights[2][WMMA_K * WMMA_N + 16];
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int batch_idx = blockIdx.z;
    const int m_block = blockIdx.y * WMMA_M;
    const int n_block = blockIdx.x * WMMA_N;
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        fill_fragment(acc_frag[i], 0.0f);
    }
    
    int buffer_idx = 0;
    
    // Pipelined computation with double buffering
    for (int k_base = 0; k_base < in_channels; k_base += WMMA_K) {
        const int next_buffer = 1 - buffer_idx;
        
        // Asynchronous data loading with better memory patterns
        if (warp_id == 0) {
            // Load input matrix A
            #pragma unroll
            for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
                int m_offset = i / WMMA_K;
                int k_offset = i % WMMA_K;
                
                if (m_block + m_offset < height * width && k_base + k_offset < in_channels) {
                    int input_idx = batch_idx * in_channels * height * width +
                                   (m_block + m_offset) * in_channels + k_base + k_offset;
                    shared_input[buffer_idx][i] = input[input_idx];
                } else {
                    shared_input[buffer_idx][i] = __float2half(0.0f);
                }
            }
        }
        
        if (warp_id == 1) {
            // Load weight matrix B
            #pragma unroll
            for (int i = lane_id; i < WMMA_K * WMMA_N; i += 32) {
                int k_offset = i / WMMA_N;
                int n_offset = i % WMMA_N;
                
                if (k_base + k_offset < in_channels && n_block + n_offset < out_channels) {
                    int weight_idx = (n_block + n_offset) * in_channels + k_base + k_offset;
                    shared_weights[buffer_idx][i] = weights[weight_idx];
                } else {
                    shared_weights[buffer_idx][i] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Load fragments and compute
        load_matrix_sync(a_frag[0], shared_input[buffer_idx], WMMA_K);
        load_matrix_sync(b_frag[0], shared_weights[buffer_idx], WMMA_N);
        
        mma_sync(acc_frag[0], a_frag[0], b_frag[0], acc_frag[0]);
        
        buffer_idx = next_buffer;
        __syncthreads();
    }
    
    // Add bias and store results with better memory pattern
    if (bias && warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < acc_frag[0].num_elements; ++i) {
            int n_offset = (lane_id * acc_frag[0].num_elements + i) % WMMA_N;
            if (n_block + n_offset < out_channels) {
                acc_frag[0].x[i] += __half2float(bias[n_block + n_offset]);
            }
        }
    }
    
    // Store with coalesced access pattern
    half* output_base = &output[batch_idx * out_channels * height * width +
                               m_block * out_channels + n_block];
    store_matrix_sync(output_base, acc_frag[0], out_channels, mem_row_major);
}

// Template instantiations for different architectures
template __global__ void optimized_lateral_conv_kernel<float, 16, 64, 4>(
    const float*, const float*, const float*, float*, int, int, int, int, int);

template __global__ void optimized_lateral_conv_kernel<half, 16, 64, 4>(
    const half*, const half*, const half*, half*, int, int, int, int, int);

template __global__ void ampere_optimized_tensor_core_conv<16, 16, 16>(
    const half*, const half*, const half*, half*, int, int, int, int, int);

template __global__ void ampere_optimized_tensor_core_conv<32, 8, 16>(
    const half*, const half*, const half*, half*, int, int, int, int, int);